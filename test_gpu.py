import torch
import sys

def main():
    print(f"Python version: {sys.version}")
    print(f"PyTorch version: {torch.__version__}")
    
    cuda_available = torch.cuda.is_available()
    print(f"CUDA (ROCm) available: {cuda_available}")
    
    if cuda_available:
        device_count = torch.cuda.device_count()
        print(f"Number of GPUs detected: {device_count}")
        for i in range(device_count):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
            
        # Try allocating a test tensor on GPU
        device = torch.device("cuda:0")
        try:
            x = torch.randn(1000, 1000, device=device)
            y = torch.randn(1000, 1000, device=device)
            z = torch.matmul(x, y)
            print("Successfully allocated tensors and performed matrix multiplication on GPU!")
            print(f"Allocated memory: {torch.cuda.memory_allocated(device) / (1024**2):.2f} MB")
        except Exception as e:
            print(f"Error during GPU operations: {e}")
    else:
        print("GPU is not available to PyTorch. Please check ROCm/driver installation inside WSL.")

if __name__ == "__main__":
    main()
