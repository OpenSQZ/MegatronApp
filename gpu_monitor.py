import pynvml
import time
from datetime import datetime

# Initialize NVML
pynvml.nvmlInit()
device_count = pynvml.nvmlDeviceGetCount()

# Optional: write to a log file
log_file = open("gpu_usage_log2.txt", "w")

try:
    while True:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"\n[{timestamp}] GPU Process Utilization")
        log_file.write(f"\n[{timestamp}] GPU Process Utilization\n")

        for i in range(device_count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            name = pynvml.nvmlDeviceGetName(handle).decode()
            utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)

            print(f"GPU {i} ({name}): GPU Utilization: {utilization.gpu}% | Memory Utilization: {utilization.memory}%")
            log_file.write(f"GPU {i} ({name}): GPU Utilization: {utilization.gpu}% | Memory Utilization: {utilization.memory}%\n")

            # try:
            #     processes = pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
            #     for p in processes:
            #         mem = p.usedGpuMemory / 1024**2 if p.usedGpuMemory is not None else 0
            #         print(f"  PID: {p.pid}, Memory Used: {mem:.2f} MB")
            #         log_file.write(f"  PID: {p.pid}, Memory Used: {mem:.2f} MB\n")
            # except pynvml.NVMLError_NotSupported:
            #     print("  Compute process info not supported on this device.")
            #     log_file.write("  Compute process info not supported on this device.\n")

        log_file.flush()
        time.sleep(0.1)  # Wait for 1 second before next reading

except KeyboardInterrupt:
    print("Monitoring stopped by user.")

finally:
    pynvml.nvmlShutdown()
    log_file.close()
