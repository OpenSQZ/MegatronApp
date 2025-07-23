#include <c10/cuda/CUDACachingAllocator.h>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <cuda_runtime.h>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <nccl.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <regex>
#include <torch/extension.h>
#include <torch/torch.h>
#include <vector>

namespace py = pybind11;

int NUM_GPUS = 4;
ncclComm_t f_prev_comm = nullptr;
ncclComm_t f_next_comm = nullptr;
ncclComm_t b_prev_comm = nullptr;
ncclComm_t b_next_comm = nullptr;
ncclUniqueId f_next_id;
ncclUniqueId f_prev_id;
ncclUniqueId b_next_id;
ncclUniqueId b_prev_id;
cudaStream_t f_next_stream;
cudaStream_t f_prev_stream;
cudaStream_t b_next_stream;
cudaStream_t b_prev_stream;

// the "queue"s
std::vector<std::optional<torch::Tensor>> forward_ready_tensors;
std::mutex forward_ready_tensors_mutex;
std::vector<std::optional<torch::Tensor>> forward_finished_tensors;
std::mutex forward_finished_tensors_mutex;
std::vector<std::optional<torch::Tensor>> backward_ready_tensors;
std::mutex backward_ready_tensors_mutex;
std::vector<std::optional<torch::Tensor>> backward_finished_tensors;
std::mutex backward_finished_tensors_mutex;

// debugging
std::mutex print_mutex;

// threads
std::optional<std::thread> forward_send_thread;
std::optional<std::thread> forward_recv_thread;
std::optional<std::thread> backward_send_thread;
std::optional<std::thread> backward_recv_thread;

#define CUDA_CALL(cmd)                                                         \
  do {                                                                         \
    cudaError_t e = cmd;                                                       \
    if (e != cudaSuccess) {                                                    \
      std::cerr << "CUDA Error: " << cudaGetErrorString(e) << std::endl;       \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)

#define NCCL_CALL(cmd)                                                         \
  do {                                                                         \
    ncclResult_t r = cmd;                                                      \
    if (r != ncclSuccess) {                                                    \
      std::cerr << "NCCL Error: " << ncclGetErrorString(r) << std::endl;       \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)

void destroy_nccl_resources() {
  if (f_prev_comm) {
    NCCL_CALL(ncclCommDestroy(f_prev_comm));
    f_prev_comm = nullptr;
  }
  if (f_next_comm) {
    NCCL_CALL(ncclCommDestroy(f_next_comm));
    f_next_comm = nullptr;
  }
  if (b_prev_comm) {
    NCCL_CALL(ncclCommDestroy(b_prev_comm));
    b_prev_comm = nullptr;
  }
  if (b_next_comm) {
    NCCL_CALL(ncclCommDestroy(b_next_comm));
    b_next_comm = nullptr;
  }

  namespace fs = std::filesystem;
  std::regex f_id_pattern(R"(f_id_\d+_\d+)");
  std::regex b_id_pattern(R"(b_id_\d+_\d+)");

  for (const auto &entry : fs::directory_iterator(fs::current_path())) {
    if (!entry.is_regular_file())
      continue;
    const std::string filename = entry.path().filename().string();
    if (std::regex_match(filename, f_id_pattern) ||
        std::regex_match(filename, b_id_pattern)) {
      std::error_code ec;
      fs::remove(entry.path(), ec);
      if (ec)
        std::cerr << "Failed to remove file " << filename << ": "
                  << ec.message() << std::endl;
      else
        std::cout << "Removed file " << filename << std::endl;
    }
  }

  cudaStreamDestroy(f_next_stream);
  cudaStreamDestroy(f_prev_stream);
  cudaStreamDestroy(b_next_stream);
  cudaStreamDestroy(b_prev_stream);
}

void setup_nccl_pipeline_pair(int pipeline_rank, int world_size, int prev_rank,
                              int next_rank, int cur_rank,
                              int total_num_microbatches) {
  CUDA_CALL(cudaSetDevice(cur_rank % NUM_GPUS));

  if (pipeline_rank % 2 == 0) {
    NCCL_CALL(ncclGetUniqueId(&f_prev_id));
    std::ofstream ofs("f_id_" + std::to_string(prev_rank) + "_" +
                          std::to_string(cur_rank),
                      std::ios::binary);
    ofs.write(reinterpret_cast<char *>(&f_prev_id), sizeof(f_prev_id));
    ofs.close();
    NCCL_CALL(ncclCommInitRank(&f_prev_comm, 2, f_prev_id, 1));
    std::this_thread::sleep_for(std::chrono::milliseconds(500));
    std::ifstream ifs("f_id_" + std::to_string(cur_rank) + "_" +
                          std::to_string(next_rank),
                      std::ios::binary);
    ifs.read(reinterpret_cast<char *>(&f_next_id), sizeof(f_next_id));
    ifs.close();
    NCCL_CALL(ncclCommInitRank(&f_next_comm, 2, f_next_id, 0));
  } else {
    std::this_thread::sleep_for(std::chrono::milliseconds(500));
    std::ifstream ifs("f_id_" + std::to_string(cur_rank) + "_" +
                          std::to_string(next_rank),
                      std::ios::binary);
    ifs.read(reinterpret_cast<char *>(&f_next_id), sizeof(f_next_id));
    ifs.close();
    NCCL_CALL(ncclCommInitRank(&f_next_comm, 2, f_next_id, 0));
    NCCL_CALL(ncclGetUniqueId(&f_prev_id));
    std::ofstream ofs("f_id_" + std::to_string(prev_rank) + "_" +
                          std::to_string(cur_rank),
                      std::ios::binary);
    ofs.write(reinterpret_cast<char *>(&f_prev_id), sizeof(f_prev_id));
    ofs.close();
    NCCL_CALL(ncclCommInitRank(&f_prev_comm, 2, f_prev_id, 1));
  }
  if (pipeline_rank % 2 == 0) {
    NCCL_CALL(ncclGetUniqueId(&b_prev_id));
    std::ofstream ofs("b_id_" + std::to_string(next_rank) + "_" +
                          std::to_string(cur_rank),
                      std::ios::binary);
    ofs.write(reinterpret_cast<char *>(&b_prev_id), sizeof(b_prev_id));
    ofs.close();
    NCCL_CALL(ncclCommInitRank(&b_prev_comm, 2, b_prev_id, 1));
    std::this_thread::sleep_for(std::chrono::milliseconds(500));
    std::ifstream ifs("b_id_" + std::to_string(cur_rank) + "_" +
                          std::to_string(prev_rank),
                      std::ios::binary);
    ifs.read(reinterpret_cast<char *>(&b_next_id), sizeof(b_next_id));
    ifs.close();
    NCCL_CALL(ncclCommInitRank(&b_next_comm, 2, b_next_id, 0));
  } else {
    std::this_thread::sleep_for(std::chrono::milliseconds(500));
    std::ifstream ifs("b_id_" + std::to_string(cur_rank) + "_" +
                          std::to_string(prev_rank),
                      std::ios::binary);
    ifs.read(reinterpret_cast<char *>(&b_next_id), sizeof(b_next_id));
    ifs.close();
    NCCL_CALL(ncclCommInitRank(&b_next_comm, 2, b_next_id, 0));
    NCCL_CALL(ncclGetUniqueId(&b_prev_id));
    std::ofstream ofs("b_id_" + std::to_string(next_rank) + "_" +
                          std::to_string(cur_rank),
                      std::ios::binary);
    ofs.write(reinterpret_cast<char *>(&b_prev_id), sizeof(b_prev_id));
    ofs.close();
    NCCL_CALL(ncclCommInitRank(&b_prev_comm, 2, b_prev_id, 1));
  }
  cudaStreamCreate(&f_next_stream);
  cudaStreamCreate(&f_prev_stream);
  cudaStreamCreate(&b_next_stream);
  cudaStreamCreate(&b_prev_stream);
  forward_ready_tensors =
      std::vector<std::optional<torch::Tensor>>(total_num_microbatches);
  forward_finished_tensors =
      std::vector<std::optional<torch::Tensor>>(total_num_microbatches);
  backward_ready_tensors =
      std::vector<std::optional<torch::Tensor>>(total_num_microbatches);
  backward_finished_tensors =
      std::vector<std::optional<torch::Tensor>>(total_num_microbatches);
}

// retrieve tensor from queue
std::optional<torch::Tensor> get_forward_tensor(int k) {
  forward_ready_tensors_mutex.lock();
  auto result = std::move(forward_ready_tensors[k]);
  forward_ready_tensors[k].reset();
  forward_ready_tensors_mutex.unlock();
  return result;
}

// put tensor into queue
void put_forward_tensor(int k, const torch::Tensor &tensor) {
  forward_finished_tensors_mutex.lock();
  forward_finished_tensors[k] = std::move(tensor);
  forward_finished_tensors_mutex.unlock();
}

// retrieve tensor from queue
std::optional<torch::Tensor> get_backward_tensor(int k) {
  backward_ready_tensors_mutex.lock();
  auto result = std::move(backward_ready_tensors[k]);
  backward_ready_tensors[k].reset();
  backward_ready_tensors_mutex.unlock();
  return result;
}

// put tensor into queue
void put_backward_tensor(int k, const torch::Tensor &tensor) {
  backward_finished_tensors_mutex.lock();
  backward_finished_tensors[k] = std::move(tensor);
  backward_finished_tensors_mutex.unlock();
}

// send forward using nccl
void forward_send_nccl(const torch::Tensor &tensor, int index) {
  if (!tensor.is_cuda())
    throw std::runtime_error("Tensor must be on CUDA device!");

  int64_t numel = tensor.numel();

  auto index_tensor = torch::empty(
      {1}, torch::TensorOptions().dtype(torch::kInt32).device(tensor.device()));
  index_tensor[0] = index;

  NCCL_CALL(ncclGroupStart());
  NCCL_CALL(ncclSend(index_tensor.data_ptr(), 1, ncclInt32, 1, f_next_comm, 0));
  NCCL_CALL(ncclSend(tensor.data_ptr(), numel, ncclHalf, 1, f_next_comm, 0));
  NCCL_CALL(ncclGroupEnd());

  cudaStreamSynchronize(0);
}

// send backward using nccl
void backward_send_nccl(const torch::Tensor &tensor, int index) {
  if (!tensor.is_cuda())
    throw std::runtime_error("Tensor must be on CUDA device!");

  int64_t numel = tensor.numel();

  auto index_tensor = torch::empty(
      {1}, torch::TensorOptions().dtype(torch::kInt32).device(tensor.device()));
  index_tensor[0] = index;

  NCCL_CALL(ncclGroupStart());
  NCCL_CALL(ncclSend(index_tensor.data_ptr(), 1, ncclInt32, 1, b_next_comm, 0));
  NCCL_CALL(ncclSend(tensor.data_ptr(), numel, ncclHalf, 1, b_next_comm, 0));
  NCCL_CALL(ncclGroupEnd());

  cudaStreamSynchronize(0);
}

std::pair<torch::Tensor, int> forward_recv_nccl(size_t num_elements, int rank) {
  auto data_tensor =
      torch::empty({static_cast<long>(num_elements)},
                   torch::TensorOptions()
                       .dtype(torch::kFloat16)
                       .device(torch::Device(torch::kCUDA, rank % NUM_GPUS)));

  auto index_tensor = torch::empty(
      {1},
      torch::TensorOptions().dtype(torch::kInt32).device(data_tensor.device()));

  NCCL_CALL(ncclGroupStart());
  NCCL_CALL(ncclRecv(index_tensor.data_ptr(), 1, ncclInt32, 0, f_prev_comm, 0));
  NCCL_CALL(ncclRecv(data_tensor.data_ptr(), num_elements, ncclHalf, 0,
                     f_prev_comm, 0));
  NCCL_CALL(ncclGroupEnd());

  cudaStreamSynchronize(0);

  int index = index_tensor[0].item<int>();
  data_tensor.set_requires_grad(true);
  return {data_tensor, index};
}

std::pair<torch::Tensor, int> backward_recv_nccl(size_t num_elements,
                                                 int rank) {
  auto data_tensor =
      torch::empty({static_cast<long>(num_elements)},
                   torch::TensorOptions()
                       .dtype(torch::kFloat16)
                       .device(torch::Device(torch::kCUDA, rank % NUM_GPUS)));

  auto index_tensor = torch::empty(
      {1},
      torch::TensorOptions().dtype(torch::kInt32).device(data_tensor.device()));

  NCCL_CALL(ncclGroupStart());
  NCCL_CALL(ncclRecv(index_tensor.data_ptr(), 1, ncclInt32, 0, b_prev_comm, 0));
  NCCL_CALL(ncclRecv(data_tensor.data_ptr(), num_elements, ncclHalf, 0,
                     b_prev_comm, 0));
  NCCL_CALL(ncclGroupEnd());

  cudaStreamSynchronize(0);

  int index = index_tensor[0].item<int>();
  data_tensor.set_requires_grad(true);
  return {data_tensor, index};
}

void forward_send(int num_model_chunks, int pipeline_parallel_size,
                  int total_num_microbatches, bool is_last_stage,
                  int forward_next_rank,
                  int total_num_microbatches_to_send_forward) {
  std::vector<int> forward_sent_indices;
  int forward_sent_count = 0;
  while (1) {
    int k = 0;
    bool has_tensor_to_send = false;
    for (int i = 0; i < num_model_chunks; i++) {
      for (int j = i * pipeline_parallel_size; j < total_num_microbatches;
           j += pipeline_parallel_size * num_model_chunks) {
        for (k = j; k < j + pipeline_parallel_size; k++) {
          forward_finished_tensors_mutex.lock();
          if (forward_finished_tensors[k].has_value() &&
              std::find(forward_sent_indices.begin(),
                        forward_sent_indices.end(),
                        k) == forward_sent_indices.end() &&
              !(is_last_stage &&
                k % (num_model_chunks * pipeline_parallel_size) >=
                    (num_model_chunks - 1) * pipeline_parallel_size)) {
            forward_sent_indices.push_back(k);
            has_tensor_to_send = true;
            forward_finished_tensors_mutex.unlock();
            break;
          }
          forward_finished_tensors_mutex.unlock();
        }
        if (has_tensor_to_send)
          break;
      }
      if (has_tensor_to_send)
        break;
    }
    if (has_tensor_to_send) {
      int index = k + is_last_stage * pipeline_parallel_size;
      forward_send_nccl(forward_finished_tensors[k].value(), index);
      forward_finished_tensors[k].reset();
      forward_sent_count++;
      if (forward_sent_count == total_num_microbatches_to_send_forward)
        return;
    }
  }
}

void forward_recv(int cur_rank, int total_num_microbatches_to_recv_forward,
                  int tensor_shape) {
  int forward_recved_count = 0;
  while (1) {
    auto [tensor, index] = forward_recv_nccl(tensor_shape, cur_rank);
    forward_ready_tensors_mutex.lock();
    forward_ready_tensors[index] = std::move(tensor);
    forward_ready_tensors_mutex.unlock();
    forward_recved_count++;
    if (forward_recved_count == total_num_microbatches_to_recv_forward)
      return;
  }
}

void backward_send(int num_model_chunks, int pipeline_parallel_size,
                   int total_num_microbatches, bool is_first_stage,
                   int backward_next_rank,
                   int total_num_microbatches_to_send_backward) {
  std::vector<int> backward_sent_indices;
  int backward_sent_count = 0;
  while (1) {
    int k = 0;
    bool has_tensor_to_send = false;
    for (int i = 0; i < num_model_chunks; i++) {
      for (int j = i * pipeline_parallel_size; j < total_num_microbatches;
           j += pipeline_parallel_size * num_model_chunks) {
        for (k = j; k < j + pipeline_parallel_size; k++) {
          backward_finished_tensors_mutex.lock();
          if (backward_finished_tensors[k].has_value() &&
              std::find(backward_sent_indices.begin(),
                        backward_sent_indices.end(),
                        k) == backward_sent_indices.end() &&
              !(is_first_stage &&
                k % (num_model_chunks * pipeline_parallel_size) >=
                    (num_model_chunks - 1) * pipeline_parallel_size)) {
            backward_sent_indices.push_back(k);
            has_tensor_to_send = true;
            backward_finished_tensors_mutex.unlock();
            break;
          }
          backward_finished_tensors_mutex.unlock();
        }
        if (has_tensor_to_send)
          break;
      }
      if (has_tensor_to_send)
        break;
    }
    if (has_tensor_to_send) {
      auto index = k + is_first_stage * pipeline_parallel_size;
      backward_send_nccl(backward_finished_tensors[k].value(), index);
      backward_finished_tensors[k].reset();
      backward_sent_count++;
      if (backward_sent_count == total_num_microbatches_to_send_backward)
        return;
    }
  }
}

void backward_recv(int cur_rank, int total_num_microbatches_to_recv_backward,
                   int tensor_shape) {
  int backward_recved_count = 0;
  while (1) {
    auto [tensor, index] = backward_recv_nccl(tensor_shape, cur_rank);
    backward_ready_tensors_mutex.lock();
    backward_ready_tensors[index] = std::move(tensor);
    backward_ready_tensors_mutex.unlock();
    backward_recved_count++;
    if (backward_recved_count == total_num_microbatches_to_recv_backward)
      return;
  }
}

void thread_pool(int num_model_chunks, int pipeline_parallel_size,
                 int total_num_microbatches, bool is_last_stage,
                 bool is_first_stage, int forward_next_rank,
                 int backward_next_rank, int cur_rank,
                 int total_num_microbatches_to_send_forward,
                 int total_num_microbatches_to_recv_forward,
                 int total_num_microbatches_to_send_backward,
                 int total_num_microbatches_to_recv_backward, int tensor_shape,
                 bool forward_only) {
  forward_send_thread =
      std::thread(forward_send, num_model_chunks, pipeline_parallel_size,
                  total_num_microbatches, is_last_stage, forward_next_rank,
                  total_num_microbatches_to_send_forward);
  forward_recv_thread =
      std::thread(forward_recv, cur_rank,
                  total_num_microbatches_to_recv_forward, tensor_shape);
  if (!forward_only) {
    backward_send_thread =
        std::thread(backward_send, num_model_chunks, pipeline_parallel_size,
                    total_num_microbatches, is_first_stage, backward_next_rank,
                    total_num_microbatches_to_send_backward);
    backward_recv_thread =
        std::thread(backward_recv, cur_rank,
                    total_num_microbatches_to_recv_backward, tensor_shape);
  }
  // forward_send_thread.detach();
  // forward_recv_thread.detach();
}

void join_threads() {
  if (forward_send_thread && forward_send_thread->joinable()) {
    forward_send_thread->join();
    forward_send_thread.reset();
  }
  if (forward_recv_thread && forward_recv_thread->joinable()) {
    forward_recv_thread->join();
    forward_recv_thread.reset();
  }
  if (backward_send_thread && backward_send_thread->joinable()) {
    backward_send_thread->join();
    backward_send_thread.reset();
  }
  if (backward_recv_thread && backward_recv_thread->joinable()) {
    backward_recv_thread->join();
    backward_recv_thread.reset();
  }
}

PYBIND11_MODULE(send_nccl, m) {
  m.def("setup_nccl_pipeline_pair", &setup_nccl_pipeline_pair,
        py::arg("pipeline_rank"), py::arg("world_size"), py::arg("prev_rank"),
        py::arg("next_rank"), py::arg("cur_rank"),
        py::arg("total_num_microbatches"));
  m.def("put_forward_tensor", &put_forward_tensor,
        "Put forward tensor into slot", py::arg("k"), py::arg("tensor"));
  m.def("get_forward_tensor", &get_forward_tensor,
        "Get forward tensor from slot", py::arg("k"));
  m.def("put_backward_tensor", &put_backward_tensor,
        "Put backward tensor into slot", py::arg("k"), py::arg("tensor"));
  m.def("get_backward_tensor", &get_backward_tensor,
        "Get backward tensor from slot", py::arg("k"));
  m.def("thread_pool", &thread_pool, "Thread pool for pipeline parallelism",
        py::arg("num_model_chunks"), py::arg("pipeline_parallel_size"),
        py::arg("total_num_microbatches"), py::arg("is_last_stage"),
        py::arg("is_first_stage"), py::arg("forward_next_rank"),
        py::arg("backward_next_rank"), py::arg("cur_rank"),
        py::arg("total_num_microbatches_to_send_forward"),
        py::arg("total_num_microbatches_to_recv_forward"),
        py::arg("total_num_microbatches_to_send_backward"),
        py::arg("total_num_microbatches_to_recv_backward"),
        py::arg("tensor_shape"), py::arg("forward_only"));
  m.def("join_threads", &join_threads);
  m.def("destroy_nccl_resources", &destroy_nccl_resources);
}
