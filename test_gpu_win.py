import sys

def main():
    print(f"Python version: {sys.version}")
    
    try:
        import torch
        print(f"PyTorch version: {torch.__version__}")
    except ImportError:
        print("PyTorch is not installed in the current environment!")
        return
        
    try:
        import torch_directml
        print("torch-directml package is installed!")
        dml_device = torch_directml.device()
        print(f"DirectML Device: {dml_device}")
        
        # Test tensor allocation and computation on DML GPU device
        x = torch.randn(1000, 1000).to(dml_device)
        y = torch.randn(1000, 1000).to(dml_device)
        z = torch.matmul(x, y)
        print("Successfully allocated tensors and performed matrix multiplication on DirectML GPU!")
        
    except ImportError:
        print("torch-directml is not installed in this environment.")
        print("For GPU acceleration on AMD in Windows, run: pip install torch-directml")
    except Exception as e:
        print(f"Error during DirectML operations: {e}")

if __name__ == "__main__":
    main()
