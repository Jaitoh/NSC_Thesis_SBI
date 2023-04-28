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
        num_gpus = torch.cuda.device_count()
        for i in range(num_gpus):
            gpu_message = f"| GPU {torch.cuda.get_device_name(i)}, {round(torch.cuda.max_memory_allocated(i) / 1024 ** 3, 1)} GB "
            gpu_messages.append(gpu_message)
        
    else:
        gpu_messages.append("| No GPU detected")
    
    return cpu_message, mem_message, gpu_messages

def monitor_resources(PID, interval, log_file):
    
    start_time = time.time()
    
    with open(log_file, 'w') as f:
        print(f"Monitoring resources for PID {PID} every {interval} seconds and writing to {log_file}")
        while True:
            cpu_message, mem_message, gpu_messages = show_resource_usage(PID)
            f.write(f"\n{(time.time()-start_time)/60:.2f} min | ")
            # f.write(cpu_message + "\n")
            f.write(mem_message)
            for gpu_message in gpu_messages:
                f.write(gpu_message)
            f.flush()
            time.sleep(interval)