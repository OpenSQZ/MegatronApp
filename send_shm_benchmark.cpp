#include <chrono>
#include <cstring>
#include <fcntl.h>
#include <iostream>
#include <semaphore.h>
#include <stdexcept>
#include <string>
#include <sys/mman.h>
#include <sys/wait.h>
#include <torch/torch.h>
#include <unistd.h>
#include <unordered_map>

#define SHM_NAME "/forward_tensor_rank_0"
#define WRITE_SEM_NAME "/forward_write_sem_rank_0"
#define READ_SEM_NAME "/forward_read_sem_rank_0"

void forward_create_shared_memory(const torch::Tensor &tensor, int rank,
                                  int index) {
  if (!tensor.is_cuda())
    throw std::runtime_error("Tensor must be on CUDA device!");

  // 创建并打开共享内存
  int shm_size = tensor.numel() * sizeof(torch::Half) + sizeof(int);
  int shm_fd = shm_open(SHM_NAME, O_CREAT | O_RDWR, 0666);
  if (shm_fd == -1)
    throw std::runtime_error("Failed to create shared memory.");

  ftruncate(shm_fd, shm_size);
  void *shm_ptr =
      mmap(nullptr, shm_size, PROT_READ | PROT_WRITE, MAP_SHARED, shm_fd, 0);
  if (shm_ptr == MAP_FAILED)
    throw std::runtime_error("Failed to map shared memory.");

  // 创建信号量
  sem_t *write_sem = sem_open(WRITE_SEM_NAME, O_CREAT, 0666, 1);
  sem_t *read_sem = sem_open(READ_SEM_NAME, O_CREAT, 0666, 0);

  sem_wait(write_sem); // 等待读者处理完上一次

  // CPU 缓冲区
  at::Tensor cpu_tensor = torch::empty(
      tensor.sizes(),
      torch::TensorOptions().dtype(torch::kFloat16).device(torch::kCPU));
  cpu_tensor.copy_(tensor);

  // 写 index 和数据
  std::memcpy(shm_ptr, &index, sizeof(int));
  std::memcpy((char *)shm_ptr + sizeof(int), cpu_tensor.data_ptr<torch::Half>(),
              cpu_tensor.numel() * sizeof(torch::Half));

  sem_post(read_sem); // 通知读者
}

std::pair<torch::Tensor, int> forward_read_shared_memory(size_t num_elements,
                                                         int rank) {
  int shm_size = num_elements * sizeof(torch::Half) + sizeof(int);
  int shm_fd = shm_open(SHM_NAME, O_RDWR, 0666);
  if (shm_fd == -1)
    throw std::runtime_error("Failed to open shared memory for reading.");

  void *shm_ptr =
      mmap(nullptr, shm_size, PROT_READ | PROT_WRITE, MAP_SHARED, shm_fd, 0);
  if (shm_ptr == MAP_FAILED)
    throw std::runtime_error("Failed to map shared memory for reading.");

  sem_t *write_sem = sem_open(WRITE_SEM_NAME, 0);
  sem_t *read_sem = sem_open(READ_SEM_NAME, 0);

  sem_wait(read_sem); // 等待写者写入

  int index;
  std::memcpy(&index, shm_ptr, sizeof(int));

  torch::Tensor cpu_tensor =
      torch::from_blob(
          (char *)shm_ptr + sizeof(int), {static_cast<long>(num_elements)},
          torch::TensorOptions().dtype(torch::kFloat16).device(torch::kCPU))
          .clone();

  torch::Tensor gpu_tensor =
      cpu_tensor.to(torch::Device(torch::kCUDA, rank)).set_requires_grad(true);

  sem_post(write_sem); // 通知写者可以继续下一次

  return {gpu_tensor, index};
}

int main() {
  const int rank = 0;
  const int tensor_mb = 10;
  const int numel = tensor_mb * 1024 * 1024 / 2; // 10MB float16
  const int index = 42;

  pid_t pid = fork();

  if (pid == 0) {
    // 子进程：receiver
    sleep(1); // 等 sender 初始化完成
    auto start = std::chrono::high_resolution_clock::now();
    auto [recv_tensor, recv_index] = forward_read_shared_memory(numel, rank);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;

    std::cout << "[Receiver] Received tensor index: " << recv_index << "\n";
    std::cout << "[Receiver] Elapsed time: " << duration.count() * 1000
              << " ms\n";
  } else {
    // 父进程：sender
    torch::Tensor t = torch::randn(
        {numel}, torch::dtype(torch::kFloat16).device(torch::kCUDA, rank));

    auto start = std::chrono::high_resolution_clock::now();
    forward_create_shared_memory(t, rank, index);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;

    std::cout << "[Sender] Sent tensor index: " << index << "\n";
    std::cout << "[Sender] Elapsed time: " << duration.count() * 1000
              << " ms\n";

    wait(nullptr); // 等子进程结束

    // 清理资源
    shm_unlink(SHM_NAME);
    sem_unlink(WRITE_SEM_NAME);
    sem_unlink(READ_SEM_NAME);
  }

  return 0;
}