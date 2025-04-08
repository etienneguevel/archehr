import subprocess


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
