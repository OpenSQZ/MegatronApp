/*
 * Copyright 2025 Suanzhi Future Co., Ltd.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <arpa/inet.h>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h>
#include <dirent.h>
#include <fcntl.h>
#include <iostream>
#include <optional>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <rdma/rdma_cma.h>
#include <semaphore.h>
#include <sys/mman.h>
#include <thread>
#include <torch/extension.h>
#include <torch/torch.h>
#include <unistd.h>
#include <unordered_map>

namespace py = pybind11;

constexpr int NUM_RECV_WR = 10; // wait receive buffers

int NUM_GPUS = 0; // number of gpus per node

size_t WORKLOAD = 0; // workload of threads

// forward shared memory (current and next)
std::vector<int> f_shm_fds;
std::vector<void *> f_shm_ptrs;
std::vector<int> f_shm_fds_next;
std::vector<void *> f_shm_ptrs_next;

// backward shared memory (current and next)
std::vector<int> b_shm_fds;
std::vector<void *> b_shm_ptrs;
std::vector<int> b_shm_fds_next;
std::vector<void *> b_shm_ptrs_next;

// forward semaphores
std::vector<sem_t *> f_write_sems;
std::vector<sem_t *> f_read_sems;
std::vector<sem_t *> f_write_sems_next;
std::vector<sem_t *> f_read_sems_next;

// backward semaphores
std::vector<sem_t *> b_write_sems;
std::vector<sem_t *> b_read_sems;
std::vector<sem_t *> b_write_sems_next;
std::vector<sem_t *> b_read_sems_next;

std::vector<std::string> ips; // infiniband IPs

// event channels
rdma_event_channel *forward_send_ec = nullptr;
rdma_event_channel *forward_recv_ec = nullptr;
rdma_event_channel *backward_send_ec = nullptr;
rdma_event_channel *backward_recv_ec = nullptr;

// connections
rdma_cm_id *forward_send_conn = nullptr;
rdma_cm_id *forward_recv_conn = nullptr;
rdma_cm_id *backward_send_conn = nullptr;
rdma_cm_id *backward_recv_conn = nullptr;

// listeners (receiver)
rdma_cm_id *forward_recv_listener = nullptr;
rdma_cm_id *backward_recv_listener = nullptr;

// events
rdma_cm_event *forward_send_event = nullptr;
rdma_cm_event *forward_recv_event = nullptr;
rdma_cm_event *backward_send_event = nullptr;
rdma_cm_event *backward_recv_event = nullptr;

// protected domains
ibv_pd *forward_send_pd = nullptr;
ibv_pd *forward_recv_pd = nullptr;
ibv_pd *backward_send_pd = nullptr;
ibv_pd *backward_recv_pd = nullptr;

// mrs
ibv_mr *forward_send_mr = nullptr;
ibv_mr *forward_recv_mrs[NUM_RECV_WR] = {};
ibv_mr *backward_send_mr = nullptr;
ibv_mr *backward_recv_mrs[NUM_RECV_WR] = {};

// completion queues
ibv_cq *forward_send_cq = nullptr;
ibv_cq *forward_recv_cq = nullptr;
ibv_cq *backward_send_cq = nullptr;
ibv_cq *backward_recv_cq = nullptr;

// flags
bool forward_send_rdma = false;
bool forward_recv_rdma = false;
bool backward_send_rdma = false;
bool backward_recv_rdma = false;

// buffers
char *forward_send_buffer = nullptr;
char *forward_recv_buffers[NUM_RECV_WR] = {};
char *backward_send_buffer = nullptr;
char *backward_recv_buffers[NUM_RECV_WR] = {};

int rdma_size = 0;

// the "queue"s
std::vector<std::optional<torch::Tensor>> forward_ready_tensors;
std::mutex forward_ready_tensors_mutex;
std::vector<std::optional<torch::Tensor>> forward_finished_tensors;
std::condition_variable forward_tensor_cv;
std::queue<std::pair<int, int>> forward_indices;
std::queue<int> forward_indices_rdma;
std::mutex forward_indices_mutex;
std::vector<std::unique_ptr<std::atomic<int>>> forward_write_counters;
std::atomic<int> forward_sent_count = 0;
std::vector<std::optional<torch::Tensor>> backward_ready_tensors;
std::mutex backward_ready_tensors_mutex;
std::vector<std::optional<torch::Tensor>> backward_finished_tensors;
std::condition_variable backward_tensor_cv;
std::queue<std::pair<int, int>> backward_indices;
std::queue<int> backward_indices_rdma;
std::mutex backward_indices_mutex;
std::vector<std::unique_ptr<std::atomic<int>>> backward_write_counters;
std::atomic<int> backward_sent_count = 0;

// configurations
std::vector<size_t> shm_sizes;
int num_model_chunks_global = 0;
int pipeline_parallel_size_global = 0;
int total_num_microbatches_global = 0;
bool is_pipeline_last_stage_global = false;
bool is_pipeline_first_stage_global = false;
bool running = true;
int num_threads_global = 0;
std::vector<std::pair<torch::Tensor, std::unique_ptr<std::atomic<int>>>>
    f_pinned_buffers;
std::vector<std::pair<torch::Tensor, std::unique_ptr<std::atomic<int>>>>
    b_pinned_buffers;
int GRANULARITY = 1;
int NUM_GPU_BUFFERS = 4;
std::vector<torch::Tensor> gpu_buffers;
std::queue<int> ready_buffers;
std::queue<int> expired_buffers;
std::mutex ready_buffers_mutex;
std::mutex expired_buffers_mutex;
std::condition_variable ready_buffers_cv;
std::condition_variable expired_buffers_cv;

void init_pinned_buffer(size_t numel, int num_threads) {
  if (num_threads % GRANULARITY)
    GRANULARITY = num_threads;
  f_pinned_buffers.resize(num_threads / GRANULARITY);
  for (auto &buffer : f_pinned_buffers) {
    buffer.first =
        torch::empty({static_cast<long>(numel / f_pinned_buffers.size())},
                     torch::TensorOptions()
                         .dtype(torch::kFloat16)
                         .device(torch::kCPU)
                         .pinned_memory(true));
    buffer.second = std::make_unique<std::atomic<int>>(0);
  }
  b_pinned_buffers.resize(num_threads / GRANULARITY);
  for (auto &buffer : b_pinned_buffers) {
    buffer.first =
        torch::empty({static_cast<long>(numel / b_pinned_buffers.size())},
                     torch::TensorOptions()
                         .dtype(torch::kFloat16)
                         .device(torch::kCPU)
                         .pinned_memory(true));
    buffer.second = std::make_unique<std::atomic<int>>(0);
  }
}

void init_gpu_buffers(size_t numel, int rank) {
  gpu_buffers.resize(NUM_GPU_BUFFERS);
  for (int i = 0; i < NUM_GPU_BUFFERS; ++i) {
    gpu_buffers[i] =
        torch::empty({static_cast<long>(numel)},
                     torch::TensorOptions()
                         .dtype(torch::kFloat16)
                         .device(torch::Device(torch::kCUDA, rank % NUM_GPUS)));
    ready_buffers.push(i);
  }
}

// forward send with RDMA
void forward_send_with_rdma(const torch::Tensor &tensor, int index) {
  if (!tensor.is_cuda())
    throw std::runtime_error("Tensor must be on CUDA device!");

  static thread_local at::Tensor
      forward_cpu_tensor_cache; // allocate a CPU tensor for moving tensors from
                                // GPU to CPU
  if (!forward_cpu_tensor_cache.defined() ||
      forward_cpu_tensor_cache.sizes() != tensor.sizes()) {
    forward_cpu_tensor_cache = torch::empty(
        tensor.sizes(),
        torch::TensorOptions().dtype(torch::kFloat16).device(torch::kCPU));
  }
  forward_cpu_tensor_cache.copy_(tensor);
  std::memcpy(forward_send_buffer, &index, sizeof(int));
  std::memcpy(forward_send_buffer + sizeof(int),
              forward_cpu_tensor_cache.data_ptr<torch::Half>(),
              forward_cpu_tensor_cache.numel() * sizeof(torch::Half));
  ibv_sge sge{};
  sge.addr = (uintptr_t)forward_send_buffer;
  sge.length = rdma_size;
  sge.lkey = forward_send_mr->lkey;

  ibv_send_wr wr{}, *bad_wr = nullptr;
  wr.sg_list = &sge;
  wr.num_sge = 1;
  wr.opcode = IBV_WR_SEND;
  wr.send_flags = IBV_SEND_SIGNALED;

  if (ibv_post_send(forward_send_conn->qp, &wr, &bad_wr))
    throw std::runtime_error("Failed to post forward send.");

  ibv_wc wc{};
  while (ibv_poll_cq(forward_send_cq, 1, &wc) == 0)
    usleep(1000);

  if (wc.status != IBV_WC_SUCCESS || wc.opcode != IBV_WC_SEND) {
    std::cerr << "Sender: Forward send failed, status = " << wc.status << "\n";
    throw std::runtime_error("Forward send failed.");
  }
}

// backward send with RDMA
void backward_send_with_rdma(const torch::Tensor &tensor, int index) {
  if (!tensor.is_cuda())
    throw std::runtime_error("Tensor must be on CUDA device!");

  static thread_local at::Tensor
      backward_cpu_tensor_cache; // allocate a CPU tensor for moving tensors
                                 // from GPU to CPU
  if (!backward_cpu_tensor_cache.defined() ||
      backward_cpu_tensor_cache.sizes() != tensor.sizes()) {
    backward_cpu_tensor_cache = torch::empty(
        tensor.sizes(),
        torch::TensorOptions().dtype(torch::kFloat16).device(torch::kCPU));
  }
  backward_cpu_tensor_cache.copy_(tensor);
  std::memcpy(backward_send_buffer, &index, sizeof(int));
  std::memcpy(backward_send_buffer + sizeof(int),
              backward_cpu_tensor_cache.data_ptr<torch::Half>(),
              backward_cpu_tensor_cache.numel() * sizeof(torch::Half));
  ibv_sge sge{};
  sge.addr = (uintptr_t)backward_send_buffer;
  sge.length = rdma_size;
  sge.lkey = backward_send_mr->lkey;

  ibv_send_wr wr{}, *bad_wr = nullptr;
  wr.sg_list = &sge;
  wr.num_sge = 1;
  wr.opcode = IBV_WR_SEND;
  wr.send_flags = IBV_SEND_SIGNALED;

  if (ibv_post_send(backward_send_conn->qp, &wr, &bad_wr))
    throw std::runtime_error("Failed to post backward send.");

  ibv_wc wc{};
  while (ibv_poll_cq(backward_send_cq, 1, &wc) == 0)
    usleep(1000);

  if (wc.status != IBV_WC_SUCCESS || wc.opcode != IBV_WC_SEND) {
    std::cerr << "Sender: Backward send failed, status = " << wc.status << "\n";
    throw std::runtime_error("Backward send failed.");
  }
}

// forward receiving with RDMA
std::pair<torch::Tensor, int> forward_recv_with_rdma(size_t num_elements,
                                                     int rank) {
  ibv_wc wc{};
  int num_comp;
  do {
    num_comp = ibv_poll_cq(forward_recv_cq, 1, &wc);
  } while (num_comp == 0);

  if (wc.status != IBV_WC_SUCCESS || wc.opcode != IBV_WC_RECV) {
    std::cerr << "Receiver: Failed to receive, status = " << wc.status << "\n";
    throw std::runtime_error("Failed to receive.");
  }

  int idx = wc.wr_id;
  int index = 0;
  std::memcpy(&index, forward_recv_buffers[idx], sizeof(int));

  auto tensor =
      torch::from_blob(forward_recv_buffers[idx] + sizeof(int),
                       {static_cast<long>(num_elements)}, torch::kFloat16)
          .clone()
          .to(torch::Device(torch::kCUDA, rank % NUM_GPUS))
          .set_requires_grad(true);

  ibv_sge sge{};
  sge.addr = (uintptr_t)forward_recv_buffers[idx];
  sge.length = rdma_size;
  sge.lkey = forward_recv_mrs[idx]->lkey;

  ibv_recv_wr wr{}, *bad_wr = nullptr;
  wr.wr_id = idx;
  wr.sg_list = &sge;
  wr.num_sge = 1;

  if (ibv_post_recv(forward_recv_conn->qp, &wr, &bad_wr))
    throw std::runtime_error("Failed to post recv.");

  return {tensor, index};
}

// backward receiving with RDMA
std::pair<torch::Tensor, int> backward_recv_with_rdma(size_t num_elements,
                                                      int rank) {
  ibv_wc wc{};
  int num_comp;
  do {
    num_comp = ibv_poll_cq(backward_recv_cq, 1, &wc);
  } while (num_comp == 0);

  if (wc.status != IBV_WC_SUCCESS || wc.opcode != IBV_WC_RECV) {
    std::cerr << "Receiver: Failed to receive, status = " << wc.status << "\n";
    throw std::runtime_error("Failed to receive.");
  }

  int idx = wc.wr_id;
  int index = 0;
  std::memcpy(&index, backward_recv_buffers[idx], sizeof(int));

  auto tensor =
      torch::from_blob(backward_recv_buffers[idx] + sizeof(int),
                       {static_cast<long>(num_elements)}, torch::kFloat16)
          .clone()
          .to(torch::Device(torch::kCUDA, rank % NUM_GPUS))
          .set_requires_grad(true);

  ibv_sge sge{};
  sge.addr = (uintptr_t)backward_recv_buffers[idx];
  sge.length = rdma_size;
  sge.lkey = backward_recv_mrs[idx]->lkey;

  ibv_recv_wr wr{}, *bad_wr = nullptr;
  wr.wr_id = idx;
  wr.sg_list = &sge;
  wr.num_sge = 1;

  if (ibv_post_recv(backward_recv_conn->qp, &wr, &bad_wr))
    throw std::runtime_error("Failed to post recv.");

  return {tensor, index};
}

void start_forward_receiver_thread_rdma(size_t numel, int rank) {
  std::thread([=]() {
    while (running) {
      auto [tensor, index] = forward_recv_with_rdma(numel, rank);
      {
        std::lock_guard<std::mutex> lock(forward_ready_tensors_mutex);
        forward_ready_tensors[index] = std::move(tensor);
      }
    }
  }).detach();
}

void start_backward_receiver_thread_rdma(size_t numel, int rank) {
  std::thread([=]() {
    while (running) {
      auto [tensor, index] = backward_recv_with_rdma(numel, rank);
      {
        std::lock_guard<std::mutex> lock(backward_ready_tensors_mutex);
        backward_ready_tensors[index] = std::move(tensor);
      }
    }
  }).detach();
}

void start_initializer_thread(size_t numel, int rank) {
  std::thread([=]() {
    while (running) {
      int k = 0;

      {
        std::unique_lock<std::mutex> lock(expired_buffers_mutex);
        expired_buffers_cv.wait(
            lock, [] { return !running || !expired_buffers.empty(); });

        if (!running)
          break;

        k = expired_buffers.front();
        expired_buffers.pop();
      }

      gpu_buffers[k] = torch::empty(
          {static_cast<long>(numel)},
          torch::TensorOptions()
              .dtype(torch::kFloat16)
              .device(torch::Device(torch::kCUDA, rank % NUM_GPUS)));
      {
        std::unique_lock<std::mutex> lock2(ready_buffers_mutex);
        ready_buffers.push(k);
      }

      ready_buffers_cv.notify_one();
    }
  }).detach();
}

// debugging
std::mutex print_mutex;

using namespace std::chrono;

// recv forward using shm
int forward_read_shared_memory(int buffer_id, int rank, int threads_num,
                               size_t numel) {
  std::vector<std::thread> threads;
  int index = 0;
  for (int i = 0; i < threads_num; i++)
    threads.emplace_back([=, &index]() {
      if (!f_shm_ptrs[i])
        throw std::runtime_error(
            "Forward shared memory not initialized for this rank!");
      sem_wait(f_read_sems[i]);
      if (!running)
        return;
      if (i == threads_num - 1)
        std::memcpy(&index, f_shm_ptrs[i], sizeof(int));
      char *shm_src =
          (char *)f_shm_ptrs[i] + sizeof(int) * (i == threads_num - 1);
      size_t start = (i % GRANULARITY) * WORKLOAD;
      size_t end = std::min(start + WORKLOAD, numel / f_pinned_buffers.size());
      size_t length = end - start;
      void *dst =
          f_pinned_buffers[i / GRANULARITY].first.data_ptr<torch::Half>() +
          start;
      std::memcpy(dst, shm_src, length * sizeof(torch::Half));
      if (f_pinned_buffers[i / GRANULARITY].second->fetch_add(1) + 1 ==
          GRANULARITY) {
        auto gpu_slice = gpu_buffers[buffer_id].slice(
            0, (i / GRANULARITY) * GRANULARITY * WORKLOAD,
            (i / GRANULARITY + 1) * GRANULARITY * WORKLOAD);
        gpu_slice.copy_(f_pinned_buffers[i / GRANULARITY].first);
        f_pinned_buffers[i / GRANULARITY].second->store(0);
      }
      sem_post(f_write_sems[i]);
    });
  for (auto &t : threads)
    t.join();
  // gpu_tensor.copy_(f_pinned_buffer);
  gpu_buffers[buffer_id].set_requires_grad(true);
  return index;
}

// recv backward using shm
int backward_read_shared_memory(int buffer_id, int rank, int threads_num,
                                size_t numel) {
  std::vector<std::thread> threads;
  int index = 0;
  for (int i = 0; i < threads_num; i++)
    threads.emplace_back([=, &index]() {
      if (!b_shm_ptrs[i])
        throw std::runtime_error(
            "backward shared memory not initialized for this rank!");
      sem_wait(b_read_sems[i]);
      if (!running)
        return;
      if (i == threads_num - 1)
        std::memcpy(&index, b_shm_ptrs[i], sizeof(int));
      char *shm_src =
          (char *)b_shm_ptrs[i] + sizeof(int) * (i == threads_num - 1);
      size_t start = (i % GRANULARITY) * WORKLOAD;
      size_t end = std::min(start + WORKLOAD, numel / b_pinned_buffers.size());
      size_t length = end - start;
      void *dst =
          b_pinned_buffers[i / GRANULARITY].first.data_ptr<torch::Half>() +
          start;
      std::memcpy(dst, shm_src, length * sizeof(torch::Half));
      if (b_pinned_buffers[i / GRANULARITY].second->fetch_add(1) + 1 ==
          GRANULARITY) {
        auto gpu_slice = gpu_buffers[buffer_id].slice(
            0, (i / GRANULARITY) * GRANULARITY * WORKLOAD,
            (i / GRANULARITY + 1) * GRANULARITY * WORKLOAD);
        gpu_slice.copy_(b_pinned_buffers[i / GRANULARITY].first);
        b_pinned_buffers[i / GRANULARITY].second->store(0);
      }
      sem_post(b_write_sems[i]);
    });
  for (auto &t : threads)
    t.join();
  // gpu_tensor.copy_(b_pinned_buffer);
  gpu_buffers[buffer_id].set_requires_grad(true);
  return index;
}

void start_forward_sender_thread_rdma(int rank, int num_threads,
                                      bool is_pipeline_last_stage) {
  std::thread([=]() {
    while (running) {
      int k = 0; // Tensor index

      {
        std::unique_lock<std::mutex> lock(forward_indices_mutex);
        forward_tensor_cv.wait(
            lock, [] { return !running || !forward_indices_rdma.empty(); });

        if (!running)
          break;

        k = forward_indices_rdma.front();
        forward_indices_rdma.pop();
        if (k < 0 || k >= total_num_microbatches_global ||
            !forward_finished_tensors[k].has_value())
          throw std::runtime_error("CV signaled but value is invalid.");
      }

      forward_send_with_rdma(forward_finished_tensors[k].value(),
                             k + pipeline_parallel_size_global *
                                     is_pipeline_last_stage);
      forward_finished_tensors[k].reset();
      ++forward_sent_count;
    }
  }).detach();
}

void start_backward_sender_thread_rdma(int rank, int num_threads,
                                       bool is_pipeline_first_stage) {
  std::thread([=]() {
    while (running) {
      int k = 0; // Tensor index

      {
        std::unique_lock<std::mutex> lock(backward_indices_mutex);
        backward_tensor_cv.wait(
            lock, [] { return !running || !backward_indices_rdma.empty(); });

        if (!running)
          break;

        k = backward_indices_rdma.front();
        backward_indices_rdma.pop();
        if (k < 0 || k >= total_num_microbatches_global ||
            !backward_finished_tensors[k].has_value())
          throw std::runtime_error("CV signaled but value is invalid.");
      }

      backward_send_with_rdma(backward_finished_tensors[k].value(),
                              k + pipeline_parallel_size_global *
                                      is_pipeline_first_stage);
      backward_finished_tensors[k].reset();
      ++backward_sent_count;
    }
  }).detach();
}

void start_forward_sender_thread(int rank, int num_threads,
                                 bool is_pipeline_last_stage) {
  std::thread([=]() {
    while (running) {
      int k = 0;  // Tensor index
      int id = 0; // Chunk index in tensor

      {
        std::unique_lock<std::mutex> lock(forward_indices_mutex);
        forward_tensor_cv.wait(
            lock, [] { return !running || !forward_indices.empty(); });

        if (!running)
          break;

        std::tie(k, id) = forward_indices.front();
        forward_indices.pop();
        if (k < 0 || k >= total_num_microbatches_global ||
            !forward_finished_tensors[k].has_value())
          throw std::runtime_error("CV signaled but value is invalid.");
      }

      // initialize the link to the semaphore of next rank (forward target)
      if (!f_write_sems_next[id]) {
        std::string fw_write_sem_name = "/forward_write_sem_rank_" +
                                        std::to_string(rank) + "_" +
                                        std::to_string(id);
        f_write_sems_next[id] = sem_open(fw_write_sem_name.c_str(), 0);
      }

      // initialize the shm link of next rank (forward target)
      if (!f_shm_ptrs_next[id]) {
        std::string shm_name = "/forward_tensor_rank_" + std::to_string(rank) +
                               "_" + std::to_string(id);
        int shm_fd = shm_open(shm_name.c_str(), O_RDWR, 0666);
        if (shm_fd == -1)
          throw std::runtime_error(
              "Failed to open forward shared memory for writing.");

        void *shm_ptr =
            mmap(0, shm_sizes[id] + sizeof(int) * (id == num_threads - 1),
                 PROT_READ | PROT_WRITE, MAP_SHARED, shm_fd, 0);
        if (shm_ptr == MAP_FAILED) {
          close(shm_fd);
          throw std::runtime_error("Failed to map shared memory for writing.");
        }

        f_shm_fds_next[id] = shm_fd;
        f_shm_ptrs_next[id] = shm_ptr;
      }

      sem_wait(f_write_sems_next[id]);
      void *shm_ptr = f_shm_ptrs_next[id];
      torch::Tensor tensor = forward_finished_tensors[k].value().contiguous();
      auto index = k + pipeline_parallel_size_global * is_pipeline_last_stage;
      size_t total = tensor.numel();
      size_t start = id * WORKLOAD;
      size_t end = std::min(start + WORKLOAD, total);
      if (start >= end)
        continue;
      at::Tensor gpu_slice = tensor.view({-1}).narrow(0, start, end - start);
      cudaMemcpy((char *)shm_ptr + sizeof(int) * (id == num_threads - 1),
                 gpu_slice.data_ptr(), (end - start) * sizeof(torch::Half),
                 cudaMemcpyDeviceToHost);
      if (id == num_threads - 1)
        std::memcpy(shm_ptr, &index, sizeof(int));
      if (!f_read_sems_next[id]) {
        std::string fw_read_sem_name = "/forward_read_sem_rank_" +
                                       std::to_string(rank) + "_" +
                                       std::to_string(id);
        f_read_sems_next[id] = sem_open(fw_read_sem_name.c_str(), 0);
      }
      sem_post(f_read_sems_next[id]); // grant read access to the next rank

      if (forward_write_counters[k]->fetch_add(1) + 1 == num_threads) {
        forward_finished_tensors[k].reset();
        forward_write_counters[k]->store(0);
        ++forward_sent_count;
      }
    }
  }).detach();
}

void start_forward_receiver_thread(int rank, int num_threads, size_t numel) {
  std::thread([=]() {
    while (running) {
      int k = 0;

      {
        std::unique_lock<std::mutex> lock(ready_buffers_mutex);
        ready_buffers_cv.wait(
            lock, [] { return !running || !ready_buffers.empty(); });

        if (!running)
          break;

        k = ready_buffers.front();
        ready_buffers.pop();
      }
      int index = forward_read_shared_memory(k, rank, num_threads, numel);
      {
        std::lock_guard<std::mutex> lock2(forward_ready_tensors_mutex);
        forward_ready_tensors[index] = std::move(gpu_buffers[k]);
      }
      {
        std::lock_guard<std::mutex> lock3(expired_buffers_mutex);
        expired_buffers.push(k);
      }
      expired_buffers_cv.notify_one();
    }
  }).detach();
}

void start_backward_sender_thread(int rank, int num_threads,
                                  bool is_pipeline_first_stage) {
  std::thread([=]() {
    while (running) {
      int k = 0;
      int id = 0;

      {
        std::unique_lock<std::mutex> lock(backward_indices_mutex);
        backward_tensor_cv.wait(
            lock, [] { return !running || !backward_indices.empty(); });

        if (!running)
          break;

        std::tie(k, id) = backward_indices.front();
        backward_indices.pop();
        if (k < 0 || k >= total_num_microbatches_global ||
            !backward_finished_tensors[k].has_value())
          throw std::runtime_error("CV signaled but value is invalid.");
      }

      // initialize the link to the semaphore of next rank (backward target)
      if (!b_write_sems_next[id]) {
        std::string bw_write_sem_name = "/backward_write_sem_rank_" +
                                        std::to_string(rank) + "_" +
                                        std::to_string(id);
        b_write_sems_next[id] = sem_open(bw_write_sem_name.c_str(), 0);
      }

      // initialize the shm link of next rank (backward target)
      if (!b_shm_ptrs_next[id]) {
        std::string shm_name = "/backward_tensor_rank_" + std::to_string(rank) +
                               "_" + std::to_string(id);
        int shm_fd = shm_open(shm_name.c_str(), O_RDWR, 0666);
        if (shm_fd == -1)
          throw std::runtime_error(
              "Failed to open backward shared memory for writing.");

        void *shm_ptr =
            mmap(0, shm_sizes[id] + sizeof(int) * (id == num_threads - 1),
                 PROT_READ | PROT_WRITE, MAP_SHARED, shm_fd, 0);
        if (shm_ptr == MAP_FAILED) {
          close(shm_fd);
          throw std::runtime_error("Failed to map shared memory for writing.");
        }

        b_shm_fds_next[id] = shm_fd;
        b_shm_ptrs_next[id] = shm_ptr;
      }

      sem_wait(b_write_sems_next[id]);
      void *shm_ptr = b_shm_ptrs_next[id];
      torch::Tensor tensor = backward_finished_tensors[k].value();
      auto index = k + pipeline_parallel_size_global * is_pipeline_first_stage;
      size_t total = tensor.numel();
      size_t start = id * WORKLOAD;
      size_t end = std::min(start + WORKLOAD, total);
      if (start >= end)
        continue;
      at::Tensor gpu_slice = tensor.view({-1}).narrow(0, start, end - start);
      cudaMemcpy((char *)shm_ptr + sizeof(int) * (id == num_threads - 1),
                 gpu_slice.data_ptr(), (end - start) * sizeof(torch::Half),
                 cudaMemcpyDeviceToHost);
      if (id == num_threads - 1)
        std::memcpy(shm_ptr, &index, sizeof(int));
      if (!b_read_sems_next[id]) {
        std::string bw_read_sem_name = "/backward_read_sem_rank_" +
                                       std::to_string(rank) + "_" +
                                       std::to_string(id);
        b_read_sems_next[id] = sem_open(bw_read_sem_name.c_str(), 0);
      }
      sem_post(b_read_sems_next[id]); // grant read access to the next rank

      if (backward_write_counters[k]->fetch_add(1) + 1 == num_threads) {
        backward_finished_tensors[k].reset();
        backward_write_counters[k]->store(0);
        ++backward_sent_count;
      }
    }
  }).detach();
}

void start_backward_receiver_thread(int rank, int num_threads, size_t numel) {
  std::thread([=]() {
    while (running) {
      int k = 0;

      {
        std::unique_lock<std::mutex> lock(ready_buffers_mutex);
        ready_buffers_cv.wait(
            lock, [] { return !running || !ready_buffers.empty(); });

        if (!running)
          break;

        k = ready_buffers.front();
        ready_buffers.pop();
      }
      int index = backward_read_shared_memory(k, rank, num_threads, numel);
      {
        std::lock_guard<std::mutex> lock2(backward_ready_tensors_mutex);
        backward_ready_tensors[index] = std::move(gpu_buffers[k]);
      }
      {
        std::lock_guard<std::mutex> lock3(expired_buffers_mutex);
        expired_buffers.push(k);
      }
      expired_buffers_cv.notify_one();
    }
  }).detach();
}

// initialize shared memory and semaphores
void init_shared_memory(size_t numel, int rank, int total_num_microbatches,
                        int num_model_chunks, int pipeline_parallel_size,
                        const std::vector<std::string> &node_ips, int prev_rank,
                        int next_rank, bool is_pipeline_last_stage,
                        bool is_pipeline_first_stage, size_t workload,
                        int num_gpus) {
  py::gil_scoped_release release;
  WORKLOAD = workload;
  int num_threads = (numel + WORKLOAD - 1) / WORKLOAD;
  num_model_chunks_global = num_model_chunks;
  pipeline_parallel_size_global = pipeline_parallel_size;
  total_num_microbatches_global = total_num_microbatches;
  ips = node_ips;
  is_pipeline_last_stage_global = is_pipeline_last_stage;
  is_pipeline_first_stage_global = is_pipeline_first_stage;
  NUM_GPUS = num_gpus;

  shm_sizes.clear();
  size_t remaining = numel;
  for (int i = 0; i < num_threads; ++i) {
    size_t current_workload = std::min(remaining, WORKLOAD);
    shm_sizes.push_back(current_workload * sizeof(torch::Half));
    remaining -= current_workload;
  }

  forward_write_counters.reserve(total_num_microbatches_global);
  for (int i = 0; i < total_num_microbatches_global; ++i)
    forward_write_counters.emplace_back(std::make_unique<std::atomic<int>>(0));
  backward_write_counters.reserve(total_num_microbatches_global);
  for (int i = 0; i < total_num_microbatches_global; ++i)
    backward_write_counters.emplace_back(std::make_unique<std::atomic<int>>(0));

  forward_ready_tensors =
      std::vector<std::optional<torch::Tensor>>(total_num_microbatches_global);
  forward_finished_tensors =
      std::vector<std::optional<torch::Tensor>>(total_num_microbatches_global);
  backward_ready_tensors =
      std::vector<std::optional<torch::Tensor>>(total_num_microbatches_global);
  backward_finished_tensors =
      std::vector<std::optional<torch::Tensor>>(total_num_microbatches_global);

  f_shm_fds.resize(num_threads, -1);
  f_shm_ptrs.resize(num_threads, nullptr);
  f_shm_fds_next.resize(num_threads, -1);
  f_shm_ptrs_next.resize(num_threads, nullptr);

  b_shm_fds.resize(num_threads, -1);
  b_shm_ptrs.resize(num_threads, nullptr);
  b_shm_fds_next.resize(num_threads, -1);
  b_shm_ptrs_next.resize(num_threads, nullptr);

  f_write_sems.resize(num_threads, nullptr);
  f_read_sems.resize(num_threads, nullptr);
  f_write_sems_next.resize(num_threads, nullptr);
  f_read_sems_next.resize(num_threads, nullptr);

  b_write_sems.resize(num_threads, nullptr);
  b_read_sems.resize(num_threads, nullptr);
  b_write_sems_next.resize(num_threads, nullptr);
  b_read_sems_next.resize(num_threads, nullptr);

  num_threads_global = num_threads;

  for (int i = 0; i < num_threads; i++) {
    std::string shm_name = "/forward_tensor_rank_" + std::to_string(rank) +
                           "_" + std::to_string(i);
    size_t shm_size = shm_sizes[i] + sizeof(int) * (i == num_threads - 1);

    int shm_fd = shm_open(shm_name.c_str(), O_CREAT | O_RDWR, 0666);
    if (shm_fd == -1)
      throw std::runtime_error("Failed to create forward shared memory.");

    if (ftruncate(shm_fd, shm_size) == -1) {
      close(shm_fd);
      throw std::runtime_error("Failed to set forward shared memory size.");
    }

    void *shm_ptr =
        mmap(0, shm_size, PROT_READ | PROT_WRITE, MAP_SHARED, shm_fd, 0);
    if (shm_ptr == MAP_FAILED) {
      close(shm_fd);
      throw std::runtime_error("Failed to map forward shared memory.");
    }

    f_shm_fds[i] = shm_fd;
    f_shm_ptrs[i] = shm_ptr; // initialize forward shm

    shm_name = "/backward_tensor_rank_" + std::to_string(rank) + "_" +
               std::to_string(i);

    shm_fd = shm_open(shm_name.c_str(), O_CREAT | O_RDWR, 0666);
    if (shm_fd == -1)
      throw std::runtime_error("Failed to create backward shared memory");

    if (ftruncate(shm_fd, shm_size) == -1) {
      close(shm_fd);
      throw std::runtime_error("Failed to set backward shared memory size");
    }

    shm_ptr = mmap(0, shm_size, PROT_READ | PROT_WRITE, MAP_SHARED, shm_fd, 0);
    if (shm_ptr == MAP_FAILED) {
      close(shm_fd);
      throw std::runtime_error("Failed to map backward shared memory");
    }

    b_shm_fds[i] = shm_fd;
    b_shm_ptrs[i] = shm_ptr; // initialize backward shm

    std::string fw_write_sem_name = "/forward_write_sem_rank_" +
                                    std::to_string(rank) + "_" +
                                    std::to_string(i);
    std::string fw_read_sem_name = "/forward_read_sem_rank_" +
                                   std::to_string(rank) + "_" +
                                   std::to_string(i);
    std::string bw_write_sem_name = "/backward_write_sem_rank_" +
                                    std::to_string(rank) + "_" +
                                    std::to_string(i);
    std::string bw_read_sem_name = "/backward_read_sem_rank_" +
                                   std::to_string(rank) + "_" +
                                   std::to_string(i);

    sem_t *fw_write_sem =
        sem_open(fw_write_sem_name.c_str(), O_CREAT | O_EXCL, 0666, 1);
    if (fw_write_sem == SEM_FAILED) {
      perror("Failed to create forward write semaphore");
      throw std::runtime_error("sem_open failed: " + fw_write_sem_name);
    }

    sem_t *fw_read_sem =
        sem_open(fw_read_sem_name.c_str(), O_CREAT | O_EXCL, 0666, 0);
    if (fw_read_sem == SEM_FAILED) {
      perror("Failed to create forward read semaphore");
      throw std::runtime_error("sem_open failed: " + fw_read_sem_name);
    }

    sem_t *bw_write_sem =
        sem_open(bw_write_sem_name.c_str(), O_CREAT | O_EXCL, 0666, 1);
    if (bw_write_sem == SEM_FAILED) {
      perror("Failed to create backward write semaphore");
      throw std::runtime_error("sem_open failed: " + bw_write_sem_name);
    }

    sem_t *bw_read_sem =
        sem_open(bw_read_sem_name.c_str(), O_CREAT | O_EXCL, 0666, 0);
    if (bw_read_sem == SEM_FAILED) {
      perror("Failed to create backward read semaphore");
      throw std::runtime_error("sem_open failed: " + bw_read_sem_name);
    }

    f_write_sems[i] = fw_write_sem;
    f_read_sems[i] = fw_read_sem;
    b_write_sems[i] = bw_write_sem;
    b_read_sems[i] = bw_read_sem; // initialize semaphores
  }
  init_pinned_buffer(numel, num_threads_global);
  init_gpu_buffers(numel, rank);
  start_initializer_thread(numel, rank);
}

// establish forward RDMA connections
void init_forward_rdma(size_t numel, int rank, int next_rank, int prev_rank,
                       int pipeline_rank) {
  if (pipeline_rank % 2) {
    if (ips[rank] != ips[next_rank]) {
      forward_send_rdma = true;
      forward_send_ec = rdma_create_event_channel();

      rdma_create_id(forward_send_ec, &forward_send_conn, nullptr, RDMA_PS_TCP);

      sockaddr_in server_addr{};
      server_addr.sin_family = AF_INET;
      server_addr.sin_port = htons(20000 + next_rank);
      inet_pton(AF_INET, ips[next_rank].c_str(), &server_addr.sin_addr);

      rdma_resolve_addr(forward_send_conn, nullptr, (sockaddr *)&server_addr,
                        2000);
      rdma_get_cm_event(forward_send_ec, &forward_send_event);
      rdma_ack_cm_event(forward_send_event);

      rdma_resolve_route(forward_send_conn, 2000);
      rdma_get_cm_event(forward_send_ec, &forward_send_event);
      rdma_ack_cm_event(forward_send_event);
      forward_send_buffer = new char[numel * sizeof(torch::Half) + sizeof(int)];
      rdma_size = numel * sizeof(torch::Half) + sizeof(int);
      forward_send_pd = ibv_alloc_pd(forward_send_conn->verbs);
      forward_send_mr =
          ibv_reg_mr(forward_send_pd, forward_send_buffer,
                     numel * sizeof(torch::Half) + sizeof(int),
                     IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE);
      if (forward_send_mr == nullptr)
        throw std::runtime_error("Failed to map mr.");
      forward_send_cq =
          ibv_create_cq(forward_send_conn->verbs, 10, nullptr, nullptr, 0);

      ibv_qp_init_attr forward_qp_attr{};
      forward_qp_attr.send_cq = forward_send_cq;
      forward_qp_attr.recv_cq = forward_send_cq;
      forward_qp_attr.qp_type = IBV_QPT_RC;
      forward_qp_attr.cap.max_send_wr = 10;
      forward_qp_attr.cap.max_recv_wr = 1;
      forward_qp_attr.cap.max_send_sge = 1;
      forward_qp_attr.cap.max_recv_sge = 1;

      rdma_create_qp(forward_send_conn, forward_send_pd, &forward_qp_attr);

      rdma_conn_param forward_conn_param{};
      rdma_connect(forward_send_conn, &forward_conn_param);
      rdma_get_cm_event(forward_send_ec, &forward_send_event);
      rdma_ack_cm_event(forward_send_event);

      std::cout << "rank " << rank << " connected to forward next rank "
                << next_rank << " at port " << 20000 + next_rank << std::endl;
      start_forward_sender_thread_rdma(next_rank, num_threads_global,
                                       is_pipeline_last_stage_global);
    } else
      for (int i = 0; i < num_threads_global; i++)
        start_forward_sender_thread(next_rank, num_threads_global,
                                    is_pipeline_last_stage_global);
    if (ips[rank] != ips[prev_rank]) {
      forward_recv_rdma = true;
      rdma_event_channel *forward_recv_ec = rdma_create_event_channel();

      rdma_create_id(forward_recv_ec, &forward_recv_listener, nullptr,
                     RDMA_PS_TCP);

      sockaddr_in addr{};
      addr.sin_family = AF_INET;
      addr.sin_port = htons(20000 + rank);
      rdma_bind_addr(forward_recv_listener, (sockaddr *)&addr);
      rdma_listen(forward_recv_listener, 1);

      rdma_get_cm_event(forward_recv_ec, &forward_recv_event);
      forward_recv_conn = forward_recv_event->id;
      rdma_ack_cm_event(forward_recv_event);

      forward_recv_pd = ibv_alloc_pd(forward_recv_conn->verbs);
      for (int i = 0; i < NUM_RECV_WR; ++i) {
        forward_recv_buffers[i] =
            new char[numel * sizeof(torch::Half) + sizeof(int)];
        forward_recv_mrs[i] =
            ibv_reg_mr(forward_recv_pd, forward_recv_buffers[i],
                       numel * sizeof(torch::Half) + sizeof(int),
                       IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE);
        if (forward_recv_mrs[i] == nullptr)
          throw std::runtime_error("Failed to map mr.");
      }
      rdma_size = numel * sizeof(torch::Half) + sizeof(int);
      forward_recv_cq =
          ibv_create_cq(forward_recv_conn->verbs, 10, nullptr, nullptr, 0);

      ibv_qp_init_attr forward_qp_attr{};
      forward_qp_attr.send_cq = forward_recv_cq;
      forward_qp_attr.recv_cq = forward_recv_cq;
      forward_qp_attr.qp_type = IBV_QPT_RC;
      forward_qp_attr.cap.max_send_wr = 10;
      forward_qp_attr.cap.max_recv_wr = NUM_RECV_WR;
      forward_qp_attr.cap.max_send_sge = 1;
      forward_qp_attr.cap.max_recv_sge = 1;

      rdma_create_qp(forward_recv_conn, forward_recv_pd, &forward_qp_attr);

      rdma_conn_param forward_conn_param{};
      rdma_accept(forward_recv_conn, &forward_conn_param);

      rdma_get_cm_event(forward_recv_ec, &forward_recv_event);
      rdma_ack_cm_event(forward_recv_event);
      for (int i = 0; i < NUM_RECV_WR; ++i) {
        ibv_sge sge{};
        sge.addr = (uintptr_t)forward_recv_buffers[i];
        sge.length = rdma_size;
        sge.lkey = forward_recv_mrs[i]->lkey;

        ibv_recv_wr wr{}, *bad_wr = nullptr;
        wr.wr_id = i;
        wr.sg_list = &sge;
        wr.num_sge = 1;

        if (ibv_post_recv(forward_recv_conn->qp, &wr, &bad_wr))
          throw std::runtime_error("Failed to post recv.");
      }

      std::cout << "rank " << rank << " connected to forward prev rank "
                << prev_rank << " at port " << 20000 + rank << std::endl;
      start_forward_receiver_thread_rdma(numel, rank);
    } else
      start_forward_receiver_thread(rank, num_threads_global, numel);
  } else {
    if (ips[rank] != ips[prev_rank]) {
      forward_recv_rdma = true;
      rdma_event_channel *forward_recv_ec = rdma_create_event_channel();

      rdma_create_id(forward_recv_ec, &forward_recv_listener, nullptr,
                     RDMA_PS_TCP);

      sockaddr_in addr{};
      addr.sin_family = AF_INET;
      addr.sin_port = htons(20000 + rank);
      rdma_bind_addr(forward_recv_listener, (sockaddr *)&addr);
      rdma_listen(forward_recv_listener, 1);

      rdma_get_cm_event(forward_recv_ec, &forward_recv_event);
      forward_recv_conn = forward_recv_event->id;
      rdma_ack_cm_event(forward_recv_event);

      forward_recv_pd = ibv_alloc_pd(forward_recv_conn->verbs);
      for (int i = 0; i < NUM_RECV_WR; ++i) {
        forward_recv_buffers[i] =
            new char[numel * sizeof(torch::Half) + sizeof(int)];
        forward_recv_mrs[i] =
            ibv_reg_mr(forward_recv_pd, forward_recv_buffers[i],
                       numel * sizeof(torch::Half) + sizeof(int),
                       IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE);
        if (forward_recv_mrs[i] == nullptr)
          throw std::runtime_error("Failed to map mr.");
      }
      rdma_size = numel * sizeof(torch::Half) + sizeof(int);
      forward_recv_cq =
          ibv_create_cq(forward_recv_conn->verbs, 10, nullptr, nullptr, 0);

      ibv_qp_init_attr forward_qp_attr{};
      forward_qp_attr.send_cq = forward_recv_cq;
      forward_qp_attr.recv_cq = forward_recv_cq;
      forward_qp_attr.qp_type = IBV_QPT_RC;
      forward_qp_attr.cap.max_send_wr = 10;
      forward_qp_attr.cap.max_recv_wr = NUM_RECV_WR;
      forward_qp_attr.cap.max_send_sge = 1;
      forward_qp_attr.cap.max_recv_sge = 1;

      rdma_create_qp(forward_recv_conn, forward_recv_pd, &forward_qp_attr);

      rdma_conn_param forward_conn_param{};
      rdma_accept(forward_recv_conn, &forward_conn_param);

      rdma_get_cm_event(forward_recv_ec, &forward_recv_event);
      rdma_ack_cm_event(forward_recv_event);
      for (int i = 0; i < NUM_RECV_WR; ++i) {
        ibv_sge sge{};
        sge.addr = (uintptr_t)forward_recv_buffers[i];
        sge.length = rdma_size;
        sge.lkey = forward_recv_mrs[i]->lkey;

        ibv_recv_wr wr{}, *bad_wr = nullptr;
        wr.wr_id = i;
        wr.sg_list = &sge;
        wr.num_sge = 1;

        if (ibv_post_recv(forward_recv_conn->qp, &wr, &bad_wr))
          throw std::runtime_error("Failed to post recv.");
      }

      std::cout << "rank " << rank << " connected to forward prev rank "
                << prev_rank << " at port " << 20000 + rank << std::endl;
      start_forward_receiver_thread_rdma(numel, rank);
    } else
      start_forward_receiver_thread(rank, num_threads_global, numel);
    if (ips[rank] != ips[next_rank]) {
      forward_send_rdma = true;
      forward_send_ec = rdma_create_event_channel();

      rdma_create_id(forward_send_ec, &forward_send_conn, nullptr, RDMA_PS_TCP);

      sockaddr_in server_addr{};
      server_addr.sin_family = AF_INET;
      server_addr.sin_port = htons(20000 + next_rank);
      inet_pton(AF_INET, ips[next_rank].c_str(), &server_addr.sin_addr);

      rdma_resolve_addr(forward_send_conn, nullptr, (sockaddr *)&server_addr,
                        2000);
      rdma_get_cm_event(forward_send_ec, &forward_send_event);
      rdma_ack_cm_event(forward_send_event);

      rdma_resolve_route(forward_send_conn, 2000);
      rdma_get_cm_event(forward_send_ec, &forward_send_event);
      rdma_ack_cm_event(forward_send_event);
      forward_send_buffer = new char[numel * sizeof(torch::Half) + sizeof(int)];
      rdma_size = numel * sizeof(torch::Half) + sizeof(int);
      forward_send_pd = ibv_alloc_pd(forward_send_conn->verbs);
      forward_send_mr =
          ibv_reg_mr(forward_send_pd, forward_send_buffer,
                     numel * sizeof(torch::Half) + sizeof(int),
                     IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE);
      if (forward_send_mr == nullptr)
        throw std::runtime_error("Failed to map mr.");
      forward_send_cq =
          ibv_create_cq(forward_send_conn->verbs, 10, nullptr, nullptr, 0);

      ibv_qp_init_attr forward_qp_attr{};
      forward_qp_attr.send_cq = forward_send_cq;
      forward_qp_attr.recv_cq = forward_send_cq;
      forward_qp_attr.qp_type = IBV_QPT_RC;
      forward_qp_attr.cap.max_send_wr = 10;
      forward_qp_attr.cap.max_recv_wr = 1;
      forward_qp_attr.cap.max_send_sge = 1;
      forward_qp_attr.cap.max_recv_sge = 1;

      rdma_create_qp(forward_send_conn, forward_send_pd, &forward_qp_attr);

      rdma_conn_param forward_conn_param{};
      rdma_connect(forward_send_conn, &forward_conn_param);
      rdma_get_cm_event(forward_send_ec, &forward_send_event);
      rdma_ack_cm_event(forward_send_event);

      std::cout << "rank " << rank << " connected to forward next rank "
                << next_rank << " at port " << 20000 + next_rank << std::endl;
      start_forward_sender_thread_rdma(next_rank, num_threads_global,
                                       is_pipeline_last_stage_global);
    } else
      for (int i = 0; i < num_threads_global; i++)
        start_forward_sender_thread(next_rank, num_threads_global,
                                    is_pipeline_last_stage_global);
  }
}

// establish backward RDMA connections
void init_backward_rdma(size_t numel, int rank, int next_rank, int prev_rank,
                        int pipeline_rank) {
  if (pipeline_rank % 2) {
    if (ips[rank] != ips[prev_rank]) {
      backward_send_rdma = true;
      backward_send_ec = rdma_create_event_channel();

      rdma_create_id(backward_send_ec, &backward_send_conn, nullptr,
                     RDMA_PS_TCP);

      sockaddr_in server_addr{};
      server_addr.sin_family = AF_INET;
      server_addr.sin_port = htons(30000 + prev_rank);
      inet_pton(AF_INET, ips[prev_rank].c_str(), &server_addr.sin_addr);

      rdma_resolve_addr(backward_send_conn, nullptr, (sockaddr *)&server_addr,
                        2000);
      rdma_get_cm_event(backward_send_ec, &backward_send_event);
      rdma_ack_cm_event(backward_send_event);

      rdma_resolve_route(backward_send_conn, 2000);
      rdma_get_cm_event(backward_send_ec, &backward_send_event);
      rdma_ack_cm_event(backward_send_event);
      backward_send_buffer =
          new char[numel * sizeof(torch::Half) + sizeof(int)];
      rdma_size = numel * sizeof(torch::Half) + sizeof(int);
      backward_send_pd = ibv_alloc_pd(backward_send_conn->verbs);
      backward_send_mr =
          ibv_reg_mr(backward_send_pd, backward_send_buffer,
                     numel * sizeof(torch::Half) + sizeof(int),
                     IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE);
      if (backward_send_mr == nullptr)
        throw std::runtime_error("Failed to map mr.");
      backward_send_cq =
          ibv_create_cq(backward_send_conn->verbs, 10, nullptr, nullptr, 0);

      ibv_qp_init_attr backward_qp_attr{};
      backward_qp_attr.send_cq = backward_send_cq;
      backward_qp_attr.recv_cq = backward_send_cq;
      backward_qp_attr.qp_type = IBV_QPT_RC;
      backward_qp_attr.cap.max_send_wr = 10;
      backward_qp_attr.cap.max_recv_wr = 1;
      backward_qp_attr.cap.max_send_sge = 1;
      backward_qp_attr.cap.max_recv_sge = 1;

      rdma_create_qp(backward_send_conn, backward_send_pd, &backward_qp_attr);

      rdma_conn_param backward_conn_param{};
      rdma_connect(backward_send_conn, &backward_conn_param);
      rdma_get_cm_event(backward_send_ec, &backward_send_event);
      rdma_ack_cm_event(backward_send_event);

      std::cout << "rank " << rank << " connected to backward next rank "
                << prev_rank << " at port " << 30000 + prev_rank << std::endl;
      start_backward_sender_thread_rdma(prev_rank, num_threads_global,
                                        is_pipeline_first_stage_global);
    } else
      for (int i = 0; i < num_threads_global; i++)
        start_backward_sender_thread(prev_rank, num_threads_global,
                                     is_pipeline_first_stage_global);
    if (ips[rank] != ips[next_rank]) {
      backward_recv_rdma = true;
      rdma_event_channel *backward_recv_ec = rdma_create_event_channel();

      rdma_create_id(backward_recv_ec, &backward_recv_listener, nullptr,
                     RDMA_PS_TCP);

      sockaddr_in addr{};
      addr.sin_family = AF_INET;
      addr.sin_port = htons(30000 + rank);
      rdma_bind_addr(backward_recv_listener, (sockaddr *)&addr);
      rdma_listen(backward_recv_listener, 1);

      rdma_get_cm_event(backward_recv_ec, &backward_recv_event);
      backward_recv_conn = backward_recv_event->id;
      rdma_ack_cm_event(backward_recv_event);

      backward_recv_pd = ibv_alloc_pd(backward_recv_conn->verbs);
      for (int i = 0; i < NUM_RECV_WR; ++i) {
        backward_recv_buffers[i] =
            new char[numel * sizeof(torch::Half) + sizeof(int)];
        backward_recv_mrs[i] =
            ibv_reg_mr(backward_recv_pd, backward_recv_buffers[i],
                       numel * sizeof(torch::Half) + sizeof(int),
                       IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE);
        if (backward_recv_mrs[i] == nullptr)
          throw std::runtime_error("Failed to map mr.");
      }
      rdma_size = numel * sizeof(torch::Half) + sizeof(int);
      backward_recv_cq =
          ibv_create_cq(backward_recv_conn->verbs, 10, nullptr, nullptr, 0);

      ibv_qp_init_attr backward_qp_attr{};
      backward_qp_attr.send_cq = backward_recv_cq;
      backward_qp_attr.recv_cq = backward_recv_cq;
      backward_qp_attr.qp_type = IBV_QPT_RC;
      backward_qp_attr.cap.max_send_wr = 10;
      backward_qp_attr.cap.max_recv_wr = NUM_RECV_WR;
      backward_qp_attr.cap.max_send_sge = 1;
      backward_qp_attr.cap.max_recv_sge = 1;

      rdma_create_qp(backward_recv_conn, backward_recv_pd, &backward_qp_attr);

      rdma_conn_param backward_conn_param{};
      rdma_accept(backward_recv_conn, &backward_conn_param);

      rdma_get_cm_event(backward_recv_ec, &backward_recv_event);
      rdma_ack_cm_event(backward_recv_event);
      for (int i = 0; i < NUM_RECV_WR; ++i) {
        ibv_sge sge{};
        sge.addr = (uintptr_t)backward_recv_buffers[i];
        sge.length = rdma_size;
        sge.lkey = backward_recv_mrs[i]->lkey;

        ibv_recv_wr wr{}, *bad_wr = nullptr;
        wr.wr_id = i;
        wr.sg_list = &sge;
        wr.num_sge = 1;

        if (ibv_post_recv(backward_recv_conn->qp, &wr, &bad_wr))
          throw std::runtime_error("Failed to post recv.");
      }

      std::cout << "rank " << rank << " connected to backward prev rank "
                << next_rank << " at port " << 30000 + rank << std::endl;
      start_backward_receiver_thread_rdma(numel, rank);
    } else
      start_backward_receiver_thread(rank, num_threads_global, numel);
  } else {
    if (ips[rank] != ips[next_rank]) {
      backward_recv_rdma = true;
      rdma_event_channel *backward_recv_ec = rdma_create_event_channel();

      rdma_create_id(backward_recv_ec, &backward_recv_listener, nullptr,
                     RDMA_PS_TCP);

      sockaddr_in addr{};
      addr.sin_family = AF_INET;
      addr.sin_port = htons(30000 + rank);
      rdma_bind_addr(backward_recv_listener, (sockaddr *)&addr);
      rdma_listen(backward_recv_listener, 1);

      rdma_get_cm_event(backward_recv_ec, &backward_recv_event);
      backward_recv_conn = backward_recv_event->id;
      rdma_ack_cm_event(backward_recv_event);

      backward_recv_pd = ibv_alloc_pd(backward_recv_conn->verbs);
      for (int i = 0; i < NUM_RECV_WR; ++i) {
        backward_recv_buffers[i] =
            new char[numel * sizeof(torch::Half) + sizeof(int)];
        backward_recv_mrs[i] =
            ibv_reg_mr(backward_recv_pd, backward_recv_buffers[i],
                       numel * sizeof(torch::Half) + sizeof(int),
                       IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE);
        if (backward_recv_mrs[i] == nullptr)
          throw std::runtime_error("Failed to map mr.");
      }
      rdma_size = numel * sizeof(torch::Half) + sizeof(int);
      backward_recv_cq =
          ibv_create_cq(backward_recv_conn->verbs, 10, nullptr, nullptr, 0);

      ibv_qp_init_attr backward_qp_attr{};
      backward_qp_attr.send_cq = backward_recv_cq;
      backward_qp_attr.recv_cq = backward_recv_cq;
      backward_qp_attr.qp_type = IBV_QPT_RC;
      backward_qp_attr.cap.max_send_wr = 10;
      backward_qp_attr.cap.max_recv_wr = NUM_RECV_WR;
      backward_qp_attr.cap.max_send_sge = 1;
      backward_qp_attr.cap.max_recv_sge = 1;

      rdma_create_qp(backward_recv_conn, backward_recv_pd, &backward_qp_attr);

      rdma_conn_param backward_conn_param{};
      rdma_accept(backward_recv_conn, &backward_conn_param);

      rdma_get_cm_event(backward_recv_ec, &backward_recv_event);
      rdma_ack_cm_event(backward_recv_event);
      for (int i = 0; i < NUM_RECV_WR; ++i) {
        ibv_sge sge{};
        sge.addr = (uintptr_t)backward_recv_buffers[i];
        sge.length = rdma_size;
        sge.lkey = backward_recv_mrs[i]->lkey;

        ibv_recv_wr wr{}, *bad_wr = nullptr;
        wr.wr_id = i;
        wr.sg_list = &sge;
        wr.num_sge = 1;

        if (ibv_post_recv(backward_recv_conn->qp, &wr, &bad_wr))
          throw std::runtime_error("Failed to post recv.");
      }

      std::cout << "rank " << rank << " connected to backward prev rank "
                << next_rank << " at port " << 30000 + rank << std::endl;
      start_backward_receiver_thread_rdma(numel, rank);
    } else
      start_backward_receiver_thread(rank, num_threads_global, numel);
    if (ips[rank] != ips[prev_rank]) {
      backward_send_rdma = true;
      backward_send_ec = rdma_create_event_channel();

      rdma_create_id(backward_send_ec, &backward_send_conn, nullptr,
                     RDMA_PS_TCP);

      sockaddr_in server_addr{};
      server_addr.sin_family = AF_INET;
      server_addr.sin_port = htons(30000 + prev_rank);
      inet_pton(AF_INET, ips[prev_rank].c_str(), &server_addr.sin_addr);

      rdma_resolve_addr(backward_send_conn, nullptr, (sockaddr *)&server_addr,
                        2000);
      rdma_get_cm_event(backward_send_ec, &backward_send_event);
      rdma_ack_cm_event(backward_send_event);

      rdma_resolve_route(backward_send_conn, 2000);
      rdma_get_cm_event(backward_send_ec, &backward_send_event);
      rdma_ack_cm_event(backward_send_event);
      backward_send_buffer =
          new char[numel * sizeof(torch::Half) + sizeof(int)];
      rdma_size = numel * sizeof(torch::Half) + sizeof(int);
      backward_send_pd = ibv_alloc_pd(backward_send_conn->verbs);
      backward_send_mr =
          ibv_reg_mr(backward_send_pd, backward_send_buffer,
                     numel * sizeof(torch::Half) + sizeof(int),
                     IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE);
      if (backward_send_mr == nullptr)
        throw std::runtime_error("Failed to map mr.");
      backward_send_cq =
          ibv_create_cq(backward_send_conn->verbs, 10, nullptr, nullptr, 0);

      ibv_qp_init_attr backward_qp_attr{};
      backward_qp_attr.send_cq = backward_send_cq;
      backward_qp_attr.recv_cq = backward_send_cq;
      backward_qp_attr.qp_type = IBV_QPT_RC;
      backward_qp_attr.cap.max_send_wr = 10;
      backward_qp_attr.cap.max_recv_wr = 1;
      backward_qp_attr.cap.max_send_sge = 1;
      backward_qp_attr.cap.max_recv_sge = 1;

      rdma_create_qp(backward_send_conn, backward_send_pd, &backward_qp_attr);

      rdma_conn_param backward_conn_param{};
      rdma_connect(backward_send_conn, &backward_conn_param);
      rdma_get_cm_event(backward_send_ec, &backward_send_event);
      rdma_ack_cm_event(backward_send_event);

      std::cout << "rank " << rank << " connected to backward next rank "
                << prev_rank << " at port " << 30000 + prev_rank << std::endl;
      start_backward_sender_thread_rdma(prev_rank, num_threads_global,
                                        is_pipeline_first_stage_global);
    } else
      for (int i = 0; i < num_threads_global; i++)
        start_backward_sender_thread(prev_rank, num_threads_global,
                                     is_pipeline_first_stage_global);
  }
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
  forward_finished_tensors[k] = tensor;
  {
    std::lock_guard<std::mutex> lock(forward_indices_mutex);
    if (forward_send_rdma)
      forward_indices_rdma.push(k);
    else
      for (int i = 0; i < num_threads_global; i++)
        forward_indices.push({k, i});
  }
  forward_tensor_cv.notify_all();
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
  backward_finished_tensors[k] = tensor;
  {
    std::lock_guard<std::mutex> lock(backward_indices_mutex);
    if (backward_send_rdma)
      backward_indices_rdma.push(k);
    else
      for (int i = 0; i < num_threads_global; i++)
        backward_indices.push({k, i});
  }
  backward_tensor_cv.notify_all();
}

void join_threads() {
  running = false;
  for (auto &sem : f_read_sems)
    sem_post(sem);
  for (auto &sem : b_read_sems)
    sem_post(sem);
  forward_tensor_cv.notify_all();
  backward_tensor_cv.notify_all();
  std::this_thread::sleep_for(std::chrono::seconds(5));
}

bool check_send_finish(bool forward_only) {
  if (forward_only) {
    int expected =
        total_num_microbatches_global -
        pipeline_parallel_size_global * is_pipeline_last_stage_global *
            total_num_microbatches_global /
            (pipeline_parallel_size_global * num_model_chunks_global);
    return forward_sent_count.compare_exchange_strong(expected, 0);
  }
  forward_sent_count.store(0);
  int expected = total_num_microbatches_global -
                 pipeline_parallel_size_global *
                     is_pipeline_first_stage_global *
                     total_num_microbatches_global /
                     (pipeline_parallel_size_global * num_model_chunks_global);
  return backward_sent_count.compare_exchange_strong(expected, 0);
}

PYBIND11_MODULE(shm_tensor_new_rdma_pre_alloc, m) {
  m.def("init_shared_memory", &init_shared_memory, "Initialize shared memory",
        py::arg("numel"), py::arg("rank"), py::arg("total_num_microbatches"),
        py::arg("num_model_chunks"), py::arg("pipeline_parallel_size"),
        py::arg("node_ips"), py::arg("prev_rank"), py::arg("next_rank"),
        py::arg("is_pipeline_last_stage"), py::arg("is_pipeline_first_stage"),
        py::arg("workload"), py::arg("num_gpus"));
  m.def("init_forward_rdma", &init_forward_rdma,
        "Initialize forward shared rdma", py::arg("numel"), py::arg("rank"),
        py::arg("next_rank"), py::arg("prev_rank"), py::arg("pipeline_rank"));
  m.def("init_backward_rdma", &init_backward_rdma,
        "Initialize backward shared rdma", py::arg("numel"), py::arg("rank"),
        py::arg("next_rank"), py::arg("prev_rank"), py::arg("pipeline_rank"));
  m.def("put_forward_tensor", &put_forward_tensor,
        "Put forward tensor into slot", py::arg("k"), py::arg("tensor"));
  m.def("get_forward_tensor", &get_forward_tensor,
        "Get forward tensor from slot", py::arg("k"));
  m.def("put_backward_tensor", &put_backward_tensor,
        "Put backward tensor into slot", py::arg("k"), py::arg("tensor"));
  m.def("get_backward_tensor", &get_backward_tensor,
        "Get backward tensor from slot", py::arg("k"));
  m.def("join_threads", &join_threads, "Join the threads");
  m.def("check_send_finish", &check_send_finish,
        "Check whether the tensors are sent", py::arg("forward_only"));
}
