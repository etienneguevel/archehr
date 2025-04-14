import os

from archehr import BASE_DIR
from archehr.cross_encode.train_fsdp import parse_args
from archehr.utils.cluster import get_idle_gpus


def main():
    """
    Submit a job to the cluster.
    
    This function checks for idle GPUs, sets the CUDA_VISIBLE_DEVICES 
    environment variable, and then runs the training script with the specified 
    arguments.
    """
    # Parse command line arguments
    args = parse_args()

    # Check for idle GPUs
    # idle_gpus = get_idle_gpus()
    # if len(idle_gpus) < args.world_size:
        # If not enough idle GPUs are available, raise an error
    #    raise ValueError("The number of idle GPUs is less than asked.")

    # Set CUDA_VISIBLE_DEVICES to the idle GPUs
    # os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, idle_gpus))    
    # print(f"Using GPUs: {os.environ['CUDA_VISIBLE_DEVICES']}")

    # Search the idle GPUs
    idle_gpus = get_idle_gpus()
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, idle_gpus))
    print(f"Using GPUs: {os.environ['CUDA_VISIBLE_DEVICES']}")
    
    # Run the training script
    script_path = BASE_DIR / "cross_encode" / "train_fsdp.py"
    prompt = " ".join([f"--{arg} {val}" for arg, val in vars(args).items()])
    os.system(f"""
        torchrun --nproc_per_node={args.world_size} {script_path} {prompt}
    """)


if __name__ == "__main__":
    main()
