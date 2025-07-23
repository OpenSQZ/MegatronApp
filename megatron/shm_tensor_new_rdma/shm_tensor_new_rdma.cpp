#include <arpa/inet.h>
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

constexpr int NUM_RECV_WR = 20; // wait receive buffers

int NUM_GPUS = 0; // number of gpus per node

int f_shm_fd = 0;
void *f_shm_ptr = nullptr;
int f_shm_fd_next = 0;
void *f_shm_ptr_next = nullptr;
int b_shm_fd = 0;
void *b_shm_ptr = nullptr;
int b_shm_fd_next = 0;
void *b_shm_ptr_next = nullptr;
sem_t *f_write_sem = nullptr;
sem_t *f_read_sem = nullptr;
sem_t *f_write_sem_next = nullptr;
sem_t *f_read_sem_next = nullptr;
sem_t *b_write_sem = nullptr;
sem_t *b_read_sem = nullptr;
sem_t *b_write_sem_next = nullptr;
sem_t *b_read_sem_next = nullptr;

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
std::mutex forward_finished_tensors_mutex;
std::vector<std::optional<torch::Tensor>> backward_ready_tensors;
std::mutex backward_ready_tensors_mutex;
std::vector<std::optional<torch::Tensor>> backward_finished_tensors;
std::mutex backward_finished_tensors_mutex;

// configurations
size_t shm_size = 0;
int num_model_chunks_global = 0;
int pipeline_parallel_size_global = 0;
int total_num_microbatches_global = 0;
bool forward_only_global = false;
int RANK = 0;

// debugging
std::mutex print_mutex;

// threads
std::optional<std::thread> forward_send_thread;
std::optional<std::thread> forward_recv_thread;
std::optional<std::thread> backward_send_thread;
std::optional<std::thread> backward_recv_thread;

using namespace std::chrono;

// initialize shared memory and semaphores
void init_shared_memory(size_t numel, int rank, int total_num_microbatches,
                        int num_model_chunks, int pipeline_parallel_size,
                        const std::vector<std::string> &node_ips,
                        int num_gpus) {
  RANK = rank;
  shm_size = numel * sizeof(torch::Half) + sizeof(int);
  num_model_chunks_global = num_model_chunks;
  pipeline_parallel_size_global = pipeline_parallel_size;
  total_num_microbatches_global = total_num_microbatches;
  ips = node_ips;
  NUM_GPUS = num_gpus;
  std::string shm_name = "/forward_tensor_rank_" + std::to_string(rank);

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

  f_shm_fd = shm_fd;
  f_shm_ptr = shm_ptr; // initialize forward shm

  shm_name = "/backward_tensor_rank_" + std::to_string(rank);

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

  b_shm_fd = shm_fd;
  b_shm_ptr = shm_ptr; // initialize backward shm

  std::string fw_write_sem_name =
      "/forward_write_sem_rank_" + std::to_string(rank);
  std::string fw_read_sem_name =
      "/forward_read_sem_rank_" + std::to_string(rank);
  std::string bw_write_sem_name =
      "/backward_write_sem_rank_" + std::to_string(rank);
  std::string bw_read_sem_name =
      "/backward_read_sem_rank_" + std::to_string(rank);

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

  f_write_sem = fw_write_sem;
  f_read_sem = fw_read_sem;
  b_write_sem = bw_write_sem;
  b_read_sem = bw_read_sem; // initialize semaphores

  forward_ready_tensors =
      std::vector<std::optional<torch::Tensor>>(total_num_microbatches);
  forward_finished_tensors =
      std::vector<std::optional<torch::Tensor>>(total_num_microbatches);
  backward_ready_tensors =
      std::vector<std::optional<torch::Tensor>>(total_num_microbatches);
  backward_finished_tensors =
      std::vector<std::optional<torch::Tensor>>(total_num_microbatches);
}

// establish forward RDMA connections
void init_forward_rdma(size_t numel, int rank, int next_rank, int prev_rank,
                       int pipeline_rank) {
  if (pipeline_rank % 2) {
    if (ips[rank] != ips[next_rank]) {
      forward_send_rdma = true;
      forward_send_ec = rdma_create_event_channel();
      if (!forward_send_ec)
        throw std::runtime_error(
            "Rank " + std::to_string(rank) +
            ": forward send rdma_create_event_channel failed");

      if (rdma_create_id(forward_send_ec, &forward_send_conn, nullptr,
                         RDMA_PS_TCP))
        throw std::runtime_error("Rank " + std::to_string(rank) +
                                 ": forward send rdma_create_id failed");

      sockaddr_in server_addr{};
      server_addr.sin_family = AF_INET;
      server_addr.sin_port = htons(20000 + next_rank);
      if (inet_pton(AF_INET, ips[next_rank].c_str(), &server_addr.sin_addr) <=
          0)
        throw std::runtime_error(
            "Rank " + std::to_string(rank) +
            ": forward inet_pton failed to parse IP address");

      if (rdma_resolve_addr(forward_send_conn, nullptr,
                            (sockaddr *)&server_addr, 2000))
        throw std::runtime_error("Rank " + std::to_string(rank) +
                                 ": forward rdma_resolve_addr failed");
      if (rdma_get_cm_event(forward_send_ec, &forward_send_event))
        throw std::runtime_error(
            "Rank " + std::to_string(rank) +
            ": forward rdma_get_cm_event after addr failed");
      if (rdma_ack_cm_event(forward_send_event))
        throw std::runtime_error(
            "Rank " + std::to_string(rank) +
            ": forward rdma_ack_cm_event after addr failed");

      if (rdma_resolve_route(forward_send_conn, 2000))
        throw std::runtime_error("Rank " + std::to_string(rank) +
                                 ": forward rdma_resolve_route failed");
      if (rdma_get_cm_event(forward_send_ec, &forward_send_event))
        throw std::runtime_error(
            "Rank " + std::to_string(rank) +
            ": forward rdma_get_cm_event after route failed");
      if (rdma_ack_cm_event(forward_send_event))
        throw std::runtime_error(
            "Rank " + std::to_string(rank) +
            ": forward rdma_ack_cm_event after route failed");
      forward_send_buffer = new char[numel * sizeof(torch::Half) + sizeof(int)];
      rdma_size = numel * sizeof(torch::Half) + sizeof(int);
      forward_send_pd = ibv_alloc_pd(forward_send_conn->verbs);
      if (!forward_send_pd)
        throw std::runtime_error("Rank " + std::to_string(rank) +
                                 ": forward ibv_alloc_pd failed");
      forward_send_mr =
          ibv_reg_mr(forward_send_pd, forward_send_buffer,
                     numel * sizeof(torch::Half) + sizeof(int),
                     IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE);
      if (forward_send_mr == nullptr)
        throw std::runtime_error("Rank " + std::to_string(rank) +
                                 ": forward failed to map mr.");
      forward_send_cq =
          ibv_create_cq(forward_send_conn->verbs, 10, nullptr, nullptr, 0);
      if (!forward_send_cq)
        throw std::runtime_error("Rank " + std::to_string(rank) +
                                 ": forward ibv_create_cq failed");

      ibv_qp_init_attr forward_qp_attr{};
      forward_qp_attr.send_cq = forward_send_cq;
      forward_qp_attr.recv_cq = forward_send_cq;
      forward_qp_attr.qp_type = IBV_QPT_RC;
      forward_qp_attr.cap.max_send_wr = 10;
      forward_qp_attr.cap.max_recv_wr = 1;
      forward_qp_attr.cap.max_send_sge = 1;
      forward_qp_attr.cap.max_recv_sge = 1;

      if (rdma_create_qp(forward_send_conn, forward_send_pd, &forward_qp_attr))
        throw std::runtime_error("Rank " + std::to_string(rank) +
                                 ": forward rdma_create_qp failed");

      rdma_conn_param forward_conn_param{};
      if (rdma_connect(forward_send_conn, &forward_conn_param))
        throw std::runtime_error("Rank " + std::to_string(rank) +
                                 ": forward rdma_connect failed");
      if (rdma_get_cm_event(forward_send_ec, &forward_send_event))
        throw std::runtime_error(
            "Rank " + std::to_string(rank) +
            ": forward rdma_get_cm_event after connect failed");
      if (rdma_ack_cm_event(forward_send_event))
        throw std::runtime_error(
            "Rank " + std::to_string(rank) +
            ": forward rdma_ack_cm_event after connect failed");

      std::cout << "rank " << rank << " connected to forward next rank "
                << next_rank << " at port " << 20000 + next_rank << std::endl;
    }
    if (ips[rank] != ips[prev_rank]) {
      forward_recv_rdma = true;
      rdma_event_channel *forward_recv_ec = rdma_create_event_channel();
      if (!forward_recv_ec)
        throw std::runtime_error(
            "Rank " + std::to_string(rank) +
            ": forward recv rdma_create_event_channel failed");

      if (rdma_create_id(forward_recv_ec, &forward_recv_listener, nullptr,
                         RDMA_PS_TCP))
        throw std::runtime_error("Rank " + std::to_string(rank) +
                                 ": forward recv rdma_create_id failed");

      sockaddr_in addr{};
      addr.sin_family = AF_INET;
      addr.sin_port = htons(20000 + rank);
      if (rdma_bind_addr(forward_recv_listener, (sockaddr *)&addr))
        throw std::runtime_error("Rank " + std::to_string(rank) +
                                 ": forward rdma_bind_addr failed");
      if (rdma_listen(forward_recv_listener, 1))
        throw std::runtime_error("Rank " + std::to_string(rank) +
                                 ": forward rdma_listen failed");

      if (rdma_get_cm_event(forward_recv_ec, &forward_recv_event))
        throw std::runtime_error(
            "Rank " + std::to_string(rank) +
            ": forward rdma_get_cm_event after listen failed");
      forward_recv_conn = forward_recv_event->id;
      if (rdma_ack_cm_event(forward_recv_event))
        throw std::runtime_error(
            "Rank " + std::to_string(rank) +
            ": forward rdma_ack_cm_event after listen failed");

      forward_recv_pd = ibv_alloc_pd(forward_recv_conn->verbs);
      if (!forward_recv_pd)
        throw std::runtime_error("Rank " + std::to_string(rank) +
                                 ": forward ibv_alloc_pd failed");
      for (int i = 0; i < NUM_RECV_WR; ++i) {
        forward_recv_buffers[i] =
            new char[numel * sizeof(torch::Half) + sizeof(int)];
        forward_recv_mrs[i] =
            ibv_reg_mr(forward_recv_pd, forward_recv_buffers[i],
                       numel * sizeof(torch::Half) + sizeof(int),
                       IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE);
        if (forward_recv_mrs[i] == nullptr)
          throw std::runtime_error("Rank " + std::to_string(rank) +
                                   ": forward failed to map mr.");
      }
      rdma_size = numel * sizeof(torch::Half) + sizeof(int);
      forward_recv_cq =
          ibv_create_cq(forward_recv_conn->verbs, 10, nullptr, nullptr, 0);
      if (!forward_recv_cq)
        throw std::runtime_error("Rank " + std::to_string(rank) +
                                 ": forward ibv_create_cq failed");

      ibv_qp_init_attr forward_qp_attr{};
      forward_qp_attr.send_cq = forward_recv_cq;
      forward_qp_attr.recv_cq = forward_recv_cq;
      forward_qp_attr.qp_type = IBV_QPT_RC;
      forward_qp_attr.cap.max_send_wr = 10;
      forward_qp_attr.cap.max_recv_wr = NUM_RECV_WR;
      forward_qp_attr.cap.max_send_sge = 1;
      forward_qp_attr.cap.max_recv_sge = 1;

      if (rdma_create_qp(forward_recv_conn, forward_recv_pd, &forward_qp_attr))
        throw std::runtime_error("Rank " + std::to_string(rank) +
                                 ": forward rdma_create_qp failed");

      rdma_conn_param forward_conn_param{};
      if (rdma_accept(forward_recv_conn, &forward_conn_param))
        throw std::runtime_error("Rank " + std::to_string(rank) +
                                 ": forward rdma_accept failed");

      if (rdma_get_cm_event(forward_recv_ec, &forward_recv_event))
        throw std::runtime_error(
            "Rank " + std::to_string(rank) +
            ": forward rdma_get_cm_event after accept failed");
      if (rdma_ack_cm_event(forward_recv_event))
        throw std::runtime_error(
            "Rank " + std::to_string(rank) +
            ": forward rdma_ack_cm_event after accept failed");
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
          throw std::runtime_error("Rank " + std::to_string(rank) +
                                   ": forward failed to post recv.");
      }

      std::cout << "rank " << rank << " connected to forward prev rank "
                << prev_rank << " at port " << 20000 + rank << std::endl;
    }
  } else {
    if (ips[rank] != ips[prev_rank]) {
      forward_recv_rdma = true;
      rdma_event_channel *forward_recv_ec = rdma_create_event_channel();
      if (!forward_recv_ec)
        throw std::runtime_error(
            "Rank " + std::to_string(rank) +
            ": forward recv rdma_create_event_channel failed");

      if (rdma_create_id(forward_recv_ec, &forward_recv_listener, nullptr,
                         RDMA_PS_TCP))
        throw std::runtime_error("Rank " + std::to_string(rank) +
                                 ": forward recv rdma_create_id failed");

      sockaddr_in addr{};
      addr.sin_family = AF_INET;
      addr.sin_port = htons(20000 + rank);
      if (rdma_bind_addr(forward_recv_listener, (sockaddr *)&addr))
        throw std::runtime_error("Rank " + std::to_string(rank) +
                                 ": forward rdma_bind_addr failed");
      if (rdma_listen(forward_recv_listener, 1))
        throw std::runtime_error("Rank " + std::to_string(rank) +
                                 ": forward rdma_listen failed");

      if (rdma_get_cm_event(forward_recv_ec, &forward_recv_event))
        throw std::runtime_error(
            "Rank " + std::to_string(rank) +
            ": forward rdma_get_cm_event after listen failed");
      forward_recv_conn = forward_recv_event->id;
      if (rdma_ack_cm_event(forward_recv_event))
        throw std::runtime_error(
            "Rank " + std::to_string(rank) +
            ": forward rdma_ack_cm_event after listen failed");

      forward_recv_pd = ibv_alloc_pd(forward_recv_conn->verbs);
      if (!forward_recv_pd)
        throw std::runtime_error("Rank " + std::to_string(rank) +
                                 ": forward ibv_alloc_pd failed");
      for (int i = 0; i < NUM_RECV_WR; ++i) {
        forward_recv_buffers[i] =
            new char[numel * sizeof(torch::Half) + sizeof(int)];
        forward_recv_mrs[i] =
            ibv_reg_mr(forward_recv_pd, forward_recv_buffers[i],
                       numel * sizeof(torch::Half) + sizeof(int),
                       IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE);
        if (forward_recv_mrs[i] == nullptr)
          throw std::runtime_error("Rank " + std::to_string(rank) +
                                   ": forward failed to map mr.");
      }
      rdma_size = numel * sizeof(torch::Half) + sizeof(int);
      forward_recv_cq =
          ibv_create_cq(forward_recv_conn->verbs, 10, nullptr, nullptr, 0);
      if (!forward_recv_cq)
        throw std::runtime_error("Rank " + std::to_string(rank) +
                                 ": forward ibv_create_cq failed");

      ibv_qp_init_attr forward_qp_attr{};
      forward_qp_attr.send_cq = forward_recv_cq;
      forward_qp_attr.recv_cq = forward_recv_cq;
      forward_qp_attr.qp_type = IBV_QPT_RC;
      forward_qp_attr.cap.max_send_wr = 10;
      forward_qp_attr.cap.max_recv_wr = NUM_RECV_WR;
      forward_qp_attr.cap.max_send_sge = 1;
      forward_qp_attr.cap.max_recv_sge = 1;

      if (rdma_create_qp(forward_recv_conn, forward_recv_pd, &forward_qp_attr))
        throw std::runtime_error("Rank " + std::to_string(rank) +
                                 ": forward rdma_create_qp failed");

      rdma_conn_param forward_conn_param{};
      if (rdma_accept(forward_recv_conn, &forward_conn_param))
        throw std::runtime_error("Rank " + std::to_string(rank) +
                                 ": forward rdma_accept failed");

      if (rdma_get_cm_event(forward_recv_ec, &forward_recv_event))
        throw std::runtime_error(
            "Rank " + std::to_string(rank) +
            ": forward rdma_get_cm_event after accept failed");
      if (rdma_ack_cm_event(forward_recv_event))
        throw std::runtime_error(
            "Rank " + std::to_string(rank) +
            ": forward rdma_ack_cm_event after accept failed");
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
          throw std::runtime_error("Rank " + std::to_string(rank) +
                                   ": forward failed to post recv.");
      }

      std::cout << "rank " << rank << " connected to forward prev rank "
                << prev_rank << " at port " << 20000 + rank << std::endl;
    }
    if (ips[rank] != ips[next_rank]) {
      forward_send_rdma = true;
      forward_send_ec = rdma_create_event_channel();
      if (!forward_send_ec)
        throw std::runtime_error(
            "Rank " + std::to_string(rank) +
            ": forward send rdma_create_event_channel failed");

      if (rdma_create_id(forward_send_ec, &forward_send_conn, nullptr,
                         RDMA_PS_TCP))
        throw std::runtime_error("Rank " + std::to_string(rank) +
                                 ": forward send rdma_create_id failed");

      sockaddr_in server_addr{};
      server_addr.sin_family = AF_INET;
      server_addr.sin_port = htons(20000 + next_rank);
      if (inet_pton(AF_INET, ips[next_rank].c_str(), &server_addr.sin_addr) <=
          0)
        throw std::runtime_error(
            "Rank " + std::to_string(rank) +
            ": forward inet_pton failed to parse IP address");

      if (rdma_resolve_addr(forward_send_conn, nullptr,
                            (sockaddr *)&server_addr, 2000))
        throw std::runtime_error("Rank " + std::to_string(rank) +
                                 ": forward rdma_resolve_addr failed");
      if (rdma_get_cm_event(forward_send_ec, &forward_send_event))
        throw std::runtime_error(
            "Rank " + std::to_string(rank) +
            ": forward rdma_get_cm_event after addr failed");
      if (rdma_ack_cm_event(forward_send_event))
        throw std::runtime_error(
            "Rank " + std::to_string(rank) +
            ": forward rdma_ack_cm_event after addr failed");

      if (rdma_resolve_route(forward_send_conn, 2000))
        throw std::runtime_error("Rank " + std::to_string(rank) +
                                 ": forward rdma_resolve_route failed");
      if (rdma_get_cm_event(forward_send_ec, &forward_send_event))
        throw std::runtime_error(
            "Rank " + std::to_string(rank) +
            ": forward rdma_get_cm_event after route failed");
      if (rdma_ack_cm_event(forward_send_event))
        throw std::runtime_error(
            "Rank " + std::to_string(rank) +
            ": forward rdma_ack_cm_event after route failed");
      forward_send_buffer = new char[numel * sizeof(torch::Half) + sizeof(int)];
      rdma_size = numel * sizeof(torch::Half) + sizeof(int);
      forward_send_pd = ibv_alloc_pd(forward_send_conn->verbs);
      if (!forward_send_pd)
        throw std::runtime_error("Rank " + std::to_string(rank) +
                                 ": forward ibv_alloc_pd failed");
      forward_send_mr =
          ibv_reg_mr(forward_send_pd, forward_send_buffer,
                     numel * sizeof(torch::Half) + sizeof(int),
                     IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE);
      if (forward_send_mr == nullptr)
        throw std::runtime_error("Rank " + std::to_string(rank) +
                                 ": forward failed to map mr.");
      forward_send_cq =
          ibv_create_cq(forward_send_conn->verbs, 10, nullptr, nullptr, 0);
      if (!forward_send_cq)
        throw std::runtime_error("Rank " + std::to_string(rank) +
                                 ": forward ibv_create_cq failed");

      ibv_qp_init_attr forward_qp_attr{};
      forward_qp_attr.send_cq = forward_send_cq;
      forward_qp_attr.recv_cq = forward_send_cq;
      forward_qp_attr.qp_type = IBV_QPT_RC;
      forward_qp_attr.cap.max_send_wr = 10;
      forward_qp_attr.cap.max_recv_wr = 1;
      forward_qp_attr.cap.max_send_sge = 1;
      forward_qp_attr.cap.max_recv_sge = 1;

      if (rdma_create_qp(forward_send_conn, forward_send_pd, &forward_qp_attr))
        throw std::runtime_error("Rank " + std::to_string(rank) +
                                 ": forward rdma_create_qp failed");

      rdma_conn_param forward_conn_param{};
      if (rdma_connect(forward_send_conn, &forward_conn_param))
        throw std::runtime_error("Rank " + std::to_string(rank) +
                                 ": forward rdma_connect failed");
      if (rdma_get_cm_event(forward_send_ec, &forward_send_event))
        throw std::runtime_error(
            "Rank " + std::to_string(rank) +
            ": forward rdma_get_cm_event after connect failed");
      if (rdma_ack_cm_event(forward_send_event))
        throw std::runtime_error(
            "Rank " + std::to_string(rank) +
            ": forward rdma_ack_cm_event after connect failed");

      std::cout << "rank " << rank << " connected to forward next rank "
                << next_rank << " at port " << 20000 + next_rank << std::endl;
    }
  }
}

// establish backward RDMA connections
void init_backward_rdma(size_t numel, int rank, int next_rank, int prev_rank,
                        int pipeline_rank) {
  if (pipeline_rank % 2) {
    if (ips[rank] != ips[prev_rank]) {
      backward_send_rdma = true;
      backward_send_ec = rdma_create_event_channel();
      if (!backward_send_ec)
        throw std::runtime_error(
            "Rank " + std::to_string(rank) +
            ": backward send rdma_create_event_channel failed");

      if (rdma_create_id(backward_send_ec, &backward_send_conn, nullptr,
                         RDMA_PS_TCP))
        throw std::runtime_error("Rank " + std::to_string(rank) +
                                 ": backward send rdma_create_id failed");

      sockaddr_in server_addr{};
      server_addr.sin_family = AF_INET;
      server_addr.sin_port = htons(30000 + prev_rank);
      if (inet_pton(AF_INET, ips[prev_rank].c_str(), &server_addr.sin_addr) <=
          0)
        throw std::runtime_error(
            "Rank " + std::to_string(rank) +
            ": backward inet_pton failed to parse IP address");

      if (rdma_resolve_addr(backward_send_conn, nullptr,
                            (sockaddr *)&server_addr, 2000))
        throw std::runtime_error("Rank " + std::to_string(rank) +
                                 ": backward rdma_resolve_addr failed");
      if (rdma_get_cm_event(backward_send_ec, &backward_send_event))
        throw std::runtime_error(
            "Rank " + std::to_string(rank) +
            ": backward rdma_get_cm_event after addr failed");
      if (rdma_ack_cm_event(backward_send_event))
        throw std::runtime_error(
            "Rank " + std::to_string(rank) +
            ": backward rdma_ack_cm_event after addr failed");

      if (rdma_resolve_route(backward_send_conn, 2000))
        throw std::runtime_error("Rank " + std::to_string(rank) +
                                 ": backward rdma_resolve_route failed");
      if (rdma_get_cm_event(backward_send_ec, &backward_send_event))
        throw std::runtime_error(
            "Rank " + std::to_string(rank) +
            ": backward rdma_get_cm_event after route failed");
      if (rdma_ack_cm_event(backward_send_event))
        throw std::runtime_error(
            "Rank " + std::to_string(rank) +
            ": backward rdma_ack_cm_event after route failed");
      backward_send_buffer =
          new char[numel * sizeof(torch::Half) + sizeof(int)];
      rdma_size = numel * sizeof(torch::Half) + sizeof(int);
      backward_send_pd = ibv_alloc_pd(backward_send_conn->verbs);
      if (!backward_send_pd)
        throw std::runtime_error("Rank " + std::to_string(rank) +
                                 ": backward ibv_alloc_pd failed");
      backward_send_mr =
          ibv_reg_mr(backward_send_pd, backward_send_buffer,
                     numel * sizeof(torch::Half) + sizeof(int),
                     IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE);
      if (backward_send_mr == nullptr)
        throw std::runtime_error("Rank " + std::to_string(rank) +
                                 ": backward failed to map mr.");
      backward_send_cq =
          ibv_create_cq(backward_send_conn->verbs, 10, nullptr, nullptr, 0);
      if (!backward_send_cq)
        throw std::runtime_error("Rank " + std::to_string(rank) +
                                 ": backward ibv_create_cq failed");

      ibv_qp_init_attr backward_qp_attr{};
      backward_qp_attr.send_cq = backward_send_cq;
      backward_qp_attr.recv_cq = backward_send_cq;
      backward_qp_attr.qp_type = IBV_QPT_RC;
      backward_qp_attr.cap.max_send_wr = 10;
      backward_qp_attr.cap.max_recv_wr = 1;
      backward_qp_attr.cap.max_send_sge = 1;
      backward_qp_attr.cap.max_recv_sge = 1;

      if (rdma_create_qp(backward_send_conn, backward_send_pd,
                         &backward_qp_attr))
        throw std::runtime_error("Rank " + std::to_string(rank) +
                                 ": backward rdma_create_qp failed");

      rdma_conn_param backward_conn_param{};
      if (rdma_connect(backward_send_conn, &backward_conn_param))
        throw std::runtime_error("Rank " + std::to_string(rank) +
                                 ": backward rdma_connect failed");
      if (rdma_get_cm_event(backward_send_ec, &backward_send_event))
        throw std::runtime_error(
            "Rank " + std::to_string(rank) +
            ": backward rdma_get_cm_event after connect failed");
      if (rdma_ack_cm_event(backward_send_event))
        throw std::runtime_error(
            "Rank " + std::to_string(rank) +
            ": backward rdma_ack_cm_event after connect failed");

      std::cout << "rank " << rank << " connected to backward next rank "
                << prev_rank << " at port " << 30000 + prev_rank << std::endl;
    }
    if (ips[rank] != ips[next_rank]) {
      backward_recv_rdma = true;
      rdma_event_channel *backward_recv_ec = rdma_create_event_channel();
      if (!backward_recv_ec)
        throw std::runtime_error(
            "Rank " + std::to_string(rank) +
            ": backward recv rdma_create_event_channel failed");

      if (rdma_create_id(backward_recv_ec, &backward_recv_listener, nullptr,
                         RDMA_PS_TCP))
        throw std::runtime_error("Rank " + std::to_string(rank) +
                                 ": backward recv rdma_create_id failed");

      sockaddr_in addr{};
      addr.sin_family = AF_INET;
      addr.sin_port = htons(30000 + rank);
      if (rdma_bind_addr(backward_recv_listener, (sockaddr *)&addr))
        throw std::runtime_error("Rank " + std::to_string(rank) +
                                 ": backward rdma_bind_addr failed");
      if (rdma_listen(backward_recv_listener, 1))
        throw std::runtime_error("Rank " + std::to_string(rank) +
                                 ": backward rdma_listen failed");

      if (rdma_get_cm_event(backward_recv_ec, &backward_recv_event))
        throw std::runtime_error(
            "Rank " + std::to_string(rank) +
            ": backward rdma_get_cm_event after listen failed");
      backward_recv_conn = backward_recv_event->id;
      if (rdma_ack_cm_event(backward_recv_event))
        throw std::runtime_error(
            "Rank " + std::to_string(rank) +
            ": backward rdma_ack_cm_event after listen failed");

      backward_recv_pd = ibv_alloc_pd(backward_recv_conn->verbs);
      if (!backward_recv_pd)
        throw std::runtime_error("Rank " + std::to_string(rank) +
                                 ": backward ibv_alloc_pd failed");
      for (int i = 0; i < NUM_RECV_WR; ++i) {
        backward_recv_buffers[i] =
            new char[numel * sizeof(torch::Half) + sizeof(int)];
        backward_recv_mrs[i] =
            ibv_reg_mr(backward_recv_pd, backward_recv_buffers[i],
                       numel * sizeof(torch::Half) + sizeof(int),
                       IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE);
        if (backward_recv_mrs[i] == nullptr)
          throw std::runtime_error("Rank " + std::to_string(rank) +
                                   ": backward failed to map mr.");
      }
      rdma_size = numel * sizeof(torch::Half) + sizeof(int);
      backward_recv_cq =
          ibv_create_cq(backward_recv_conn->verbs, 10, nullptr, nullptr, 0);
      if (!backward_recv_cq)
        throw std::runtime_error("Rank " + std::to_string(rank) +
                                 ": backward ibv_create_cq failed");

      ibv_qp_init_attr backward_qp_attr{};
      backward_qp_attr.send_cq = backward_recv_cq;
      backward_qp_attr.recv_cq = backward_recv_cq;
      backward_qp_attr.qp_type = IBV_QPT_RC;
      backward_qp_attr.cap.max_send_wr = 10;
      backward_qp_attr.cap.max_recv_wr = NUM_RECV_WR;
      backward_qp_attr.cap.max_send_sge = 1;
      backward_qp_attr.cap.max_recv_sge = 1;

      if (rdma_create_qp(backward_recv_conn, backward_recv_pd,
                         &backward_qp_attr))
        throw std::runtime_error("Rank " + std::to_string(rank) +
                                 ": backward rdma_create_qp failed");

      rdma_conn_param backward_conn_param{};
      if (rdma_accept(backward_recv_conn, &backward_conn_param))
        throw std::runtime_error("Rank " + std::to_string(rank) +
                                 ": backward rdma_accept failed");

      if (rdma_get_cm_event(backward_recv_ec, &backward_recv_event))
        throw std::runtime_error(
            "Rank " + std::to_string(rank) +
            ": backward rdma_get_cm_event after accept failed");
      if (rdma_ack_cm_event(backward_recv_event))
        throw std::runtime_error(
            "Rank " + std::to_string(rank) +
            ": backward rdma_ack_cm_event after accept failed");
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
          throw std::runtime_error("Rank " + std::to_string(rank) +
                                   ": backward failed to post recv.");
      }

      std::cout << "rank " << rank << " connected to backward prev rank "
                << next_rank << " at port " << 30000 + rank << std::endl;
    }
  } else {
    if (ips[rank] != ips[next_rank]) {
      backward_recv_rdma = true;
      rdma_event_channel *backward_recv_ec = rdma_create_event_channel();
      if (!backward_recv_ec)
        throw std::runtime_error(
            "Rank " + std::to_string(rank) +
            ": backward recv rdma_create_event_channel failed");

      if (rdma_create_id(backward_recv_ec, &backward_recv_listener, nullptr,
                         RDMA_PS_TCP))
        throw std::runtime_error("Rank " + std::to_string(rank) +
                                 ": backward recv rdma_create_id failed");

      sockaddr_in addr{};
      addr.sin_family = AF_INET;
      addr.sin_port = htons(30000 + rank);
      if (rdma_bind_addr(backward_recv_listener, (sockaddr *)&addr))
        throw std::runtime_error("Rank " + std::to_string(rank) +
                                 ": backward rdma_bind_addr failed");
      if (rdma_listen(backward_recv_listener, 1))
        throw std::runtime_error("Rank " + std::to_string(rank) +
                                 ": backward rdma_listen failed");

      if (rdma_get_cm_event(backward_recv_ec, &backward_recv_event))
        throw std::runtime_error(
            "Rank " + std::to_string(rank) +
            ": backward rdma_get_cm_event after listen failed");
      backward_recv_conn = backward_recv_event->id;
      if (rdma_ack_cm_event(backward_recv_event))
        throw std::runtime_error(
            "Rank " + std::to_string(rank) +
            ": backward rdma_ack_cm_event after listen failed");

      backward_recv_pd = ibv_alloc_pd(backward_recv_conn->verbs);
      if (!backward_recv_pd)
        throw std::runtime_error("Rank " + std::to_string(rank) +
                                 ": backward ibv_alloc_pd failed");
      for (int i = 0; i < NUM_RECV_WR; ++i) {
        backward_recv_buffers[i] =
            new char[numel * sizeof(torch::Half) + sizeof(int)];
        backward_recv_mrs[i] =
            ibv_reg_mr(backward_recv_pd, backward_recv_buffers[i],
                       numel * sizeof(torch::Half) + sizeof(int),
                       IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE);
        if (backward_recv_mrs[i] == nullptr)
          throw std::runtime_error("Rank " + std::to_string(rank) +
                                   ": backward failed to map mr.");
      }
      rdma_size = numel * sizeof(torch::Half) + sizeof(int);
      backward_recv_cq =
          ibv_create_cq(backward_recv_conn->verbs, 10, nullptr, nullptr, 0);
      if (!backward_recv_cq)
        throw std::runtime_error("Rank " + std::to_string(rank) +
                                 ": backward ibv_create_cq failed");

      ibv_qp_init_attr backward_qp_attr{};
      backward_qp_attr.send_cq = backward_recv_cq;
      backward_qp_attr.recv_cq = backward_recv_cq;
      backward_qp_attr.qp_type = IBV_QPT_RC;
      backward_qp_attr.cap.max_send_wr = 10;
      backward_qp_attr.cap.max_recv_wr = NUM_RECV_WR;
      backward_qp_attr.cap.max_send_sge = 1;
      backward_qp_attr.cap.max_recv_sge = 1;

      if (rdma_create_qp(backward_recv_conn, backward_recv_pd,
                         &backward_qp_attr))
        throw std::runtime_error("Rank " + std::to_string(rank) +
                                 ": backward rdma_create_qp failed");

      rdma_conn_param backward_conn_param{};
      if (rdma_accept(backward_recv_conn, &backward_conn_param))
        throw std::runtime_error("Rank " + std::to_string(rank) +
                                 ": backward rdma_accept failed");

      if (rdma_get_cm_event(backward_recv_ec, &backward_recv_event))
        throw std::runtime_error(
            "Rank " + std::to_string(rank) +
            ": backward rdma_get_cm_event after accept failed");
      if (rdma_ack_cm_event(backward_recv_event))
        throw std::runtime_error(
            "Rank " + std::to_string(rank) +
            ": backward rdma_ack_cm_event after accept failed");
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
          throw std::runtime_error("Rank " + std::to_string(rank) +
                                   ": backward failed to post recv.");
      }

      std::cout << "rank " << rank << " connected to backward prev rank "
                << next_rank << " at port " << 30000 + rank << std::endl;
    }
    if (ips[rank] != ips[prev_rank]) {
      backward_send_rdma = true;
      backward_send_ec = rdma_create_event_channel();
      if (!backward_send_ec)
        throw std::runtime_error(
            "Rank " + std::to_string(rank) +
            ": backward send rdma_create_event_channel failed");

      if (rdma_create_id(backward_send_ec, &backward_send_conn, nullptr,
                         RDMA_PS_TCP))
        throw std::runtime_error("Rank " + std::to_string(rank) +
                                 ": backward send rdma_create_id failed");

      sockaddr_in server_addr{};
      server_addr.sin_family = AF_INET;
      server_addr.sin_port = htons(30000 + prev_rank);
      if (inet_pton(AF_INET, ips[prev_rank].c_str(), &server_addr.sin_addr) <=
          0)
        throw std::runtime_error(
            "Rank " + std::to_string(rank) +
            ": backward inet_pton failed to parse IP address");

      if (rdma_resolve_addr(backward_send_conn, nullptr,
                            (sockaddr *)&server_addr, 2000))
        throw std::runtime_error("Rank " + std::to_string(rank) +
                                 ": backward rdma_resolve_addr failed");
      if (rdma_get_cm_event(backward_send_ec, &backward_send_event))
        throw std::runtime_error(
            "Rank " + std::to_string(rank) +
            ": backward rdma_get_cm_event after addr failed");
      if (rdma_ack_cm_event(backward_send_event))
        throw std::runtime_error(
            "Rank " + std::to_string(rank) +
            ": backward rdma_ack_cm_event after addr failed");

      if (rdma_resolve_route(backward_send_conn, 2000))
        throw std::runtime_error("Rank " + std::to_string(rank) +
                                 ": backward rdma_resolve_route failed");
      if (rdma_get_cm_event(backward_send_ec, &backward_send_event))
        throw std::runtime_error(
            "Rank " + std::to_string(rank) +
            ": backward rdma_get_cm_event after route failed");
      if (rdma_ack_cm_event(backward_send_event))
        throw std::runtime_error(
            "Rank " + std::to_string(rank) +
            ": backward rdma_ack_cm_event after route failed");
      backward_send_buffer =
          new char[numel * sizeof(torch::Half) + sizeof(int)];
      rdma_size = numel * sizeof(torch::Half) + sizeof(int);
      backward_send_pd = ibv_alloc_pd(backward_send_conn->verbs);
      if (!backward_send_pd)
        throw std::runtime_error("Rank " + std::to_string(rank) +
                                 ": backward ibv_alloc_pd failed");
      backward_send_mr =
          ibv_reg_mr(backward_send_pd, backward_send_buffer,
                     numel * sizeof(torch::Half) + sizeof(int),
                     IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE);
      if (backward_send_mr == nullptr)
        throw std::runtime_error("Rank " + std::to_string(rank) +
                                 ": backward failed to map mr.");
      backward_send_cq =
          ibv_create_cq(backward_send_conn->verbs, 10, nullptr, nullptr, 0);
      if (!backward_send_cq)
        throw std::runtime_error("Rank " + std::to_string(rank) +
                                 ": backward ibv_create_cq failed");

      ibv_qp_init_attr backward_qp_attr{};
      backward_qp_attr.send_cq = backward_send_cq;
      backward_qp_attr.recv_cq = backward_send_cq;
      backward_qp_attr.qp_type = IBV_QPT_RC;
      backward_qp_attr.cap.max_send_wr = 10;
      backward_qp_attr.cap.max_recv_wr = 1;
      backward_qp_attr.cap.max_send_sge = 1;
      backward_qp_attr.cap.max_recv_sge = 1;

      if (rdma_create_qp(backward_send_conn, backward_send_pd,
                         &backward_qp_attr))
        throw std::runtime_error("Rank " + std::to_string(rank) +
                                 ": backward rdma_create_qp failed");

      rdma_conn_param backward_conn_param{};
      if (rdma_connect(backward_send_conn, &backward_conn_param))
        throw std::runtime_error("Rank " + std::to_string(rank) +
                                 ": backward rdma_connect failed");
      if (rdma_get_cm_event(backward_send_ec, &backward_send_event))
        throw std::runtime_error(
            "Rank " + std::to_string(rank) +
            ": backward rdma_get_cm_event after connect failed");
      if (rdma_ack_cm_event(backward_send_event))
        throw std::runtime_error(
            "Rank " + std::to_string(rank) +
            ": backward rdma_ack_cm_event after connect failed");

      std::cout << "rank " << rank << " connected to backward next rank "
                << prev_rank << " at port " << 30000 + prev_rank << std::endl;
    }
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

// send forward using shm
void forward_create_shared_memory(const torch::Tensor &tensor, int rank,
                                  int index) {
  if (!tensor.is_cuda())
    throw std::runtime_error("Tensor must be on CUDA device!");

  // initialize the link to the semaphore of next rank (forward target)
  if (!f_write_sem_next) {
    std::string fw_write_sem_name =
        "/forward_write_sem_rank_" + std::to_string(rank);
    f_write_sem_next = sem_open(fw_write_sem_name.c_str(), 0);
  }

  sem_wait(f_write_sem_next); // wait for write permission (granted by read)

  std::string shm_name = "/forward_tensor_rank_" + std::to_string(rank);

  // initialize the shm link of next rank (forward target)
  if (!f_shm_ptr_next) {
    int shm_fd = shm_open(shm_name.c_str(), O_RDWR, 0666);
    if (shm_fd == -1)
      throw std::runtime_error("Failed to open shared memory for writing.");

    void *shm_ptr = mmap(0, tensor.numel() * sizeof(torch::Half) + sizeof(int),
                         PROT_READ | PROT_WRITE, MAP_SHARED, shm_fd, 0);
    if (shm_ptr == MAP_FAILED) {
      close(shm_fd);
      throw std::runtime_error("Failed to map shared memory for writing.");
    }

    f_shm_fd_next = shm_fd;
    f_shm_ptr_next = shm_ptr;
  }

  void *shm_ptr = f_shm_ptr_next;

  const int num_threads = 10;
  const int64_t total_elems = tensor.numel();
  const int64_t chunk_size = (total_elems + num_threads - 1) / num_threads;

  std::vector<std::thread> threads;
  auto *shm_half_ptr =
      reinterpret_cast<torch::Half *>((char *)shm_ptr + sizeof(int));

  for (int i = 0; i < num_threads; ++i) {
    int64_t start = i * chunk_size;
    int64_t end = std::min(start + chunk_size, total_elems);
    if (start >= end)
      continue;

    threads.emplace_back([=, &tensor]() {
      at::Tensor gpu_slice = tensor.view({-1}).narrow(0, start, end - start);
      cudaMemcpy((char *)shm_ptr + sizeof(int), gpu_slice.data_ptr(),
                 (end - start) * sizeof(torch::Half), cudaMemcpyDeviceToHost);
    });
  }

  for (auto &t : threads)
    t.join();

  std::memcpy(shm_ptr, &index, sizeof(int));

  if (!f_read_sem_next) {
    std::string fw_read_sem_name =
        "/forward_read_sem_rank_" + std::to_string(rank);
    f_read_sem_next = sem_open(fw_read_sem_name.c_str(), 0);
  }
  sem_post(f_read_sem_next); // grant read access to the next rank
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
    throw std::runtime_error("Rank " + std::to_string(RANK) +
                             " failed to post forward send.");

  ibv_wc wc{};
  while (ibv_poll_cq(forward_send_cq, 1, &wc) == 0)
    usleep(1000);

  if (wc.status != IBV_WC_SUCCESS || wc.opcode != IBV_WC_SEND) {
    std::cerr << "Sender: Forward send failed, status = " << wc.status << "\n";
    throw std::runtime_error("Rank " + std::to_string(RANK) +
                             " failed to send forward.");
  }
}

// send backward using shm
void backward_create_shared_memory(const torch::Tensor &tensor, int rank,
                                   int index) {
  if (!tensor.is_cuda())
    throw std::runtime_error("Tensor must be on CUDA device!");

  if (!b_write_sem_next) {
    std::string bw_write_sem_name =
        "/backward_write_sem_rank_" + std::to_string(rank);
    b_write_sem_next = sem_open(bw_write_sem_name.c_str(), 0);
  }

  sem_wait(b_write_sem_next);

  std::string shm_name = "/backward_tensor_rank_" + std::to_string(rank);

  if (!b_shm_ptr_next) {
    int shm_fd = shm_open(shm_name.c_str(), O_RDWR, 0666);
    if (shm_fd == -1)
      throw std::runtime_error("Failed to open shared memory for writing.");

    void *shm_ptr = mmap(0, tensor.numel() * sizeof(torch::Half) + sizeof(int),
                         PROT_READ | PROT_WRITE, MAP_SHARED, shm_fd, 0);
    if (shm_ptr == MAP_FAILED) {
      close(shm_fd);
      throw std::runtime_error("Failed to map shared memory for writing.");
    }

    b_shm_fd_next = shm_fd;
    b_shm_ptr_next = shm_ptr;
  }

  void *shm_ptr = b_shm_ptr_next;

  const int num_threads = 10;
  const int64_t total_elems = tensor.numel();
  const int64_t chunk_size = (total_elems + num_threads - 1) / num_threads;

  std::vector<std::thread> threads;
  auto *shm_half_ptr =
      reinterpret_cast<torch::Half *>((char *)shm_ptr + sizeof(int));

  for (int i = 0; i < num_threads; ++i) {
    int64_t start = i * chunk_size;
    int64_t end = std::min(start + chunk_size, total_elems);
    if (start >= end)
      continue;

    threads.emplace_back([=, &tensor]() {
      at::Tensor gpu_slice = tensor.view({-1}).narrow(0, start, end - start);
      cudaMemcpy((char *)shm_ptr + sizeof(int), gpu_slice.data_ptr(),
                 (end - start) * sizeof(torch::Half), cudaMemcpyDeviceToHost);
    });
  }

  for (auto &t : threads)
    t.join();

  std::memcpy(shm_ptr, &index, sizeof(int));

  if (!b_read_sem_next) {
    std::string bw_read_sem_name =
        "/backward_read_sem_rank_" + std::to_string(rank);
    b_read_sem_next = sem_open(bw_read_sem_name.c_str(), 0);
  }
  sem_post(b_read_sem_next);
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
    throw std::runtime_error("Rank " + std::to_string(RANK) +
                             " failed to post backward send.");

  ibv_wc wc{};
  while (ibv_poll_cq(backward_send_cq, 1, &wc) == 0)
    usleep(1000);

  if (wc.status != IBV_WC_SUCCESS || wc.opcode != IBV_WC_SEND) {
    std::cerr << "Sender: Backward send failed, status = " << wc.status << "\n";
    throw std::runtime_error("Rank " + std::to_string(RANK) +
                             " failed to send backward.");
  }
}

// recv forward using shm
std::pair<torch::Tensor, int> forward_read_shared_memory(size_t num_elements,
                                                         int rank) {
  if (!f_shm_ptr)
    throw std::runtime_error(
        "Forward shared memory not initialized for this rank!");
  sem_wait(f_read_sem);

  int index = 0;
  std::memcpy(&index, f_shm_ptr, sizeof(int));

  auto cpu_tensor = torch::from_blob((char *)f_shm_ptr + sizeof(int),
                                     {static_cast<long>(num_elements)},
                                     torch::TensorOptions()
                                         .dtype(torch::kFloat16)
                                         .device(torch::kCPU)
                                         .pinned_memory(true));

  auto gpu_tensor =
      torch::empty({static_cast<long>(num_elements)},
                   torch::TensorOptions()
                       .dtype(torch::kFloat16)
                       .device(torch::Device(torch::kCUDA, rank % NUM_GPUS)));

  const int num_threads = 10;
  const int64_t chunk_size = (num_elements + num_threads - 1) / num_threads;
  std::vector<std::thread> threads;

  for (int i = 0; i < num_threads; ++i) {
    int64_t start = i * chunk_size;
    int64_t end = std::min(start + chunk_size, (int64_t)num_elements);
    if (start >= end)
      continue;

    threads.emplace_back([=, &cpu_tensor, &gpu_tensor]() {
      auto cpu_slice = cpu_tensor.slice(0, start, end).contiguous();
      auto gpu_slice = gpu_tensor.slice(0, start, end);
      gpu_slice.copy_(cpu_slice, true);
    });
  }

  for (auto &t : threads)
    t.join();

  gpu_tensor.set_requires_grad(true);
  sem_post(f_write_sem);
  return {gpu_tensor, index};
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
    std::cerr << "Receiver: Failed to forward receive, status = " << wc.status
              << "\n";
    throw std::runtime_error("Rank " + std::to_string(RANK) +
                             " failed to forward receive.");
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
    throw std::runtime_error("Rank " + std::to_string(RANK) +
                             " failed to post forward recv.");

  return {tensor, index};
}

// recv backward using shm
std::pair<torch::Tensor, int> backward_read_shared_memory(size_t num_elements,
                                                          int rank) {
  if (!b_shm_ptr)
    throw std::runtime_error(
        "Forward shared memory not initialized for this rank!");
  sem_wait(b_read_sem);

  int index = 0;
  std::memcpy(&index, b_shm_ptr, sizeof(int));

  auto cpu_tensor = torch::from_blob((char *)b_shm_ptr + sizeof(int),
                                     {static_cast<long>(num_elements)},
                                     torch::TensorOptions()
                                         .dtype(torch::kFloat16)
                                         .device(torch::kCPU)
                                         .pinned_memory(true));

  auto gpu_tensor =
      torch::empty({static_cast<long>(num_elements)},
                   torch::TensorOptions()
                       .dtype(torch::kFloat16)
                       .device(torch::Device(torch::kCUDA, rank % NUM_GPUS)));

  const int num_threads = 10;
  const int64_t chunk_size = (num_elements + num_threads - 1) / num_threads;
  std::vector<std::thread> threads;

  for (int i = 0; i < num_threads; ++i) {
    int64_t start = i * chunk_size;
    int64_t end = std::min(start + chunk_size, (int64_t)num_elements);
    if (start >= end)
      continue;

    threads.emplace_back([=, &cpu_tensor, &gpu_tensor]() {
      auto cpu_slice = cpu_tensor.slice(0, start, end).contiguous();
      auto gpu_slice = gpu_tensor.slice(0, start, end);
      gpu_slice.copy_(cpu_slice, true);
    });
  }

  for (auto &t : threads)
    t.join();

  gpu_tensor.set_requires_grad(true);
  sem_post(b_write_sem);
  return {gpu_tensor, index};
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
    std::cerr << "Receiver: Failed to backward receive, status = " << wc.status
              << "\n";
    throw std::runtime_error("Rank " + std::to_string(RANK) +
                             " failed to backward receive.");
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
    throw std::runtime_error("Rank " + std::to_string(RANK) +
                             " failed to post backward receive.");

  return {tensor, index};
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
      if (forward_send_rdma)
        forward_send_with_rdma(forward_finished_tensors[k].value(), index);
      else
        forward_create_shared_memory(forward_finished_tensors[k].value(),
                                     forward_next_rank, index);
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
    if (forward_recv_rdma) {
      auto [tensor, index] = forward_recv_with_rdma(tensor_shape, cur_rank);
      forward_ready_tensors_mutex.lock();
      forward_ready_tensors[index] = tensor;
      forward_ready_tensors_mutex.unlock();
    } else {
      auto [tensor, index] = forward_read_shared_memory(tensor_shape, cur_rank);
      forward_ready_tensors_mutex.lock();
      forward_ready_tensors[index] = tensor;
      forward_ready_tensors_mutex.unlock();
    }
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
      if (backward_send_rdma)
        backward_send_with_rdma(backward_finished_tensors[k].value(), index);
      else
        backward_create_shared_memory(backward_finished_tensors[k].value(),
                                      backward_next_rank, index);
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
    if (backward_recv_rdma) {
      auto [tensor, index] = backward_recv_with_rdma(tensor_shape, cur_rank);
      backward_ready_tensors_mutex.lock();
      backward_ready_tensors[index] = tensor;
      backward_ready_tensors_mutex.unlock();
    } else {
      auto [tensor, index] =
          backward_read_shared_memory(tensor_shape, cur_rank);
      backward_ready_tensors_mutex.lock();
      backward_ready_tensors[index] = tensor;
      backward_ready_tensors_mutex.unlock();
    }
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

void clean_rdma() {
  std::cout << "Cleaning RDMA" << std::endl;
  if (forward_send_mr)
    ibv_dereg_mr(forward_send_mr);
  delete[] forward_send_buffer;

  for (int i = 0; i < NUM_RECV_WR; ++i) {
    if (forward_recv_mrs[i])
      ibv_dereg_mr(forward_recv_mrs[i]);
    delete[] forward_recv_buffers[i];
  }

  if (forward_send_cq)
    ibv_destroy_cq(forward_send_cq);
  if (forward_recv_cq)
    ibv_destroy_cq(forward_recv_cq);

  if (forward_send_conn)
    rdma_destroy_id(forward_send_conn);
  if (forward_recv_conn)
    rdma_destroy_id(forward_recv_conn);
  if (forward_recv_listener)
    rdma_destroy_id(forward_recv_listener);

  if (forward_send_pd)
    ibv_dealloc_pd(forward_send_pd);
  if (forward_recv_pd)
    ibv_dealloc_pd(forward_recv_pd);

  if (forward_send_ec)
    rdma_destroy_event_channel(forward_send_ec);
  if (forward_recv_ec)
    rdma_destroy_event_channel(forward_recv_ec);

  if (backward_send_mr)
    ibv_dereg_mr(backward_send_mr);
  delete[] backward_send_buffer;

  for (int i = 0; i < NUM_RECV_WR; ++i) {
    if (backward_recv_mrs[i])
      ibv_dereg_mr(backward_recv_mrs[i]);
    delete[] backward_recv_buffers[i];
  }

  if (backward_send_cq)
    ibv_destroy_cq(backward_send_cq);
  if (backward_recv_cq)
    ibv_destroy_cq(backward_recv_cq);

  if (backward_send_conn)
    rdma_destroy_id(backward_send_conn);
  if (backward_recv_conn)
    rdma_destroy_id(backward_recv_conn);
  if (backward_recv_listener)
    rdma_destroy_id(backward_recv_listener);

  if (backward_send_pd)
    ibv_dealloc_pd(backward_send_pd);
  if (backward_recv_pd)
    ibv_dealloc_pd(backward_recv_pd);

  if (backward_send_ec)
    rdma_destroy_event_channel(backward_send_ec);
  if (backward_recv_ec)
    rdma_destroy_event_channel(backward_recv_ec);
}

PYBIND11_MODULE(shm_tensor_new_rdma, m) {
  m.def("init_shared_memory", &init_shared_memory, "Initialize shared memory",
        py::arg("numel"), py::arg("rank"), py::arg("total_num_microbatches"),
        py::arg("num_model_chunks"), py::arg("pipeline_parallel_size"),
        py::arg("node_ips"), py::arg("num_gpus"));
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
  m.def("clean_rdma", &clean_rdma);
}