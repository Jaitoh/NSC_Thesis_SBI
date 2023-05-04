import psutil
# import GPUtil
import time
import os
import torch
import gpustat

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
            
def print_mem_info(prefix="", do_print=False):
    
    if do_print:
        print_cpu_memory_usage(prefix=prefix)
        print_gpu_memory_usage()
    
def print_cpu_memory_usage(prefix=""):
    
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    
    print(f"{prefix} - cpu memory usage: {mem_info.rss / (1024 * 1024):.2f} MB", end=' | ')
            
def print_gpu_memory_usage():
    try:
        gpu_stats = gpustat.GPUStatCollection.new_query()
        for gpu in gpu_stats:
            print(f"GPU {gpu.index} - Memory Used: {gpu.memory_used} MB / {gpu.memory_total} MB", end=' | ')
    except Exception as e:
        print(f"Error querying GPU memory usage: {e}")