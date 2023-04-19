import psutil
import GPUtil
import time
import os
import torch

def show_resource_usage(PID):
    process = psutil.Process(PID)
    
    # CPU usage
    cpu_percent = process.cpu_percent()
    cpu_message = f"CPU {cpu_percent}%"
    
    # Memory usage
    mem_info = process.memory_info()
    rss = mem_info.rss / (1024 ** 2)  # Resident set size in MB
    mem_message = f"MEM {rss/1024:.2f} GB "
    
    # GPU usage
    gpu_messages = []
    gpus = GPUtil.getGPUs()
    if torch.cuda.is_available():
        for gpu in gpus:
            gpu_percent = gpu.load * 100
            gpu_memory_used = gpu.memoryUsed
            gpu_memory_total = gpu.memoryTotal
            gpu_message = f"| GPU{gpu.id}, {gpu_percent:.2f}% - {gpu_memory_used:.2f}/{gpu_memory_total:.2f} MB "
            gpu_messages.append(gpu_message)
    else:
        gpu_messages.append("| No GPU detected")
    
    return cpu_message, mem_message, gpu_messages

def monitor_resources(PID, interval, log_file):
    start_time = time.time()
    with open(log_file, 'w') as f:
        while True:
            cpu_message, mem_message, gpu_messages = show_resource_usage(PID)
            f.write(f"\n{(time.time()-start_time)/60:.2f} min | ")
            # f.write(cpu_message + "\n")
            f.write(mem_message)
            for gpu_message in gpu_messages:
                f.write(gpu_message)
            f.flush()
            time.sleep(interval)