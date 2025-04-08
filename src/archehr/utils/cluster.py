import os
import subprocess

import torch


def get_idle_gpus():
    # Run nvidia-smi to get GPU memory usage
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"],
        stdout=subprocess.PIPE,
        text=True,
    )
    lines = result.stdout.strip().split("\n")
    
    # Parse the output to find idle GPUs
    idle_gpus = []
    for line in lines:
        gpu_index, memory_used = map(int, line.split(","))
        if memory_used < 100:  # Consider GPUs with <100 MiB memory used as idle
            idle_gpus.append(gpu_index)
    
    return idle_gpus


def initialize_profiler(save_folder):
    
    profiler = torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CUDA],
        schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=2),
        on_trace_ready=torch.profiler.tensorboard_trace_handler(
            os.path.join(save_folder, 'profile'),
        ),
        with_stack=True,
        record_shapes=True,
        with_flops=True,
    )
        
    return profiler
