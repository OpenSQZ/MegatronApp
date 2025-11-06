#include <chrono>
#include <cstring>
#include <fcntl.h>
#include <iostream>
#include <semaphore.h>
#include <stdexcept>
#include <string>
#include <sys/mman.h>
#include <sys/wait.h>
#include <torch/extension.h>
#include <unistd.h>

#define SHM_NAME "/forward_tensor_rank_0"
#define WRITE_SEM_NAME "/forward_write_sem_rank_0"
#define READ_SEM_NAME "/forward_read_sem_rank_0"

void forward_create_shared_memory(const at::Tensor &tensor, int index) {
  if (!tensor.is_cuda())
    throw std::runtime_error("Tensor must be on CUDA device!");

  int shm_size = tensor.numel() * sizeof(at::Half) + sizeof(int);
  int shm_fd = shm_open(SHM_NAME, O_CREAT | O_RDWR, 0666);
  if (shm_fd == -1)
    throw std::runtime_error("Failed to create shared memory.");

  ftruncate(shm_fd, shm_size);
  void *shm_ptr =
      mmap(nullptr, shm_size, PROT_READ | PROT_WRITE, MAP_SHARED, shm_fd, 0);
  if (shm_ptr == MAP_FAILED)
    throw std::runtime_error("Failed to map shared memory.");

  sem_t *write_sem = sem_open(WRITE_SEM_NAME, O_CREAT, 0666, 1);
  sem_t *read_sem = sem_open(READ_SEM_NAME, O_CREAT, 0666, 0);

  sem_wait(write_sem);

  at::Tensor cpu_tensor = torch::empty(
      tensor.sizes(),
      torch::TensorOptions().dtype(torch::kFloat16).device(torch::kCPU));
  cpu_tensor.copy_(tensor);

  std::memcpy(shm_ptr, &index, sizeof(int));
  std::memcpy((char *)shm_ptr + sizeof(int), cpu_tensor.data_ptr<at::Half>(),
              cpu_tensor.numel() * sizeof(at::Half));

  sem_post(read_sem);

  munmap(shm_ptr, shm_size);
  close(shm_fd);
  sem_close(write_sem);
  sem_close(read_sem);
}

std::pair<at::Tensor, int> forward_read_shared_memory(size_t num_elements) {
  int shm_size = num_elements * sizeof(at::Half) + sizeof(int);
  int shm_fd = shm_open(SHM_NAME, O_RDWR, 0666);
  if (shm_fd == -1)
    throw std::runtime_error("Failed to open shared memory for reading.");

  void *shm_ptr =
      mmap(nullptr, shm_size, PROT_READ | PROT_WRITE, MAP_SHARED, shm_fd, 0);
  if (shm_ptr == MAP_FAILED)
    throw std::runtime_error("Failed to map shared memory for reading.");

  sem_t *write_sem = sem_open(WRITE_SEM_NAME, 0);
  sem_t *read_sem = sem_open(READ_SEM_NAME, 0);

  sem_wait(read_sem);

  int index;
  std::memcpy(&index, shm_ptr, sizeof(int));

  at::Tensor cpu_tensor =
      torch::from_blob(
          (char *)shm_ptr + sizeof(int), {static_cast<long>(num_elements)},
          torch::TensorOptions().dtype(torch::kFloat16).device(torch::kCPU))
          .clone();

  at::Tensor gpu_tensor = cpu_tensor.to(torch::kCUDA).set_requires_grad(true);

  sem_post(write_sem);

  munmap(shm_ptr, shm_size);
  close(shm_fd);
  sem_close(write_sem);
  sem_close(read_sem);

  return {gpu_tensor, index};
}

void shm_unlink_resources() {
  shm_unlink(SHM_NAME);
  sem_unlink(WRITE_SEM_NAME);
  sem_unlink(READ_SEM_NAME);
}

PYBIND11_MODULE(shm_benchmark, m) {
  m.def("write_tensor", &forward_create_shared_memory,
        "Write tensor to shared memory", pybind11::arg("tensor"),
        pybind11::arg("index"));
  m.def("read_tensor", &forward_read_shared_memory,
        "Read tensor from shared memory", pybind11::arg("num_elements"));
  m.def("cleanup", &shm_unlink_resources,
        "Cleanup shared memory and semaphores");
}
