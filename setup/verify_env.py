"""Environment verification script for ML/LLM setup."""

import sys
import platform
import numpy as np
import pandas as pd
import sklearn
import torch
import transformers
import accelerate
import matplotlib.pyplot as plt
import seaborn as sns


def check_python():
    """Check Python and system info."""
    print("=" * 50)
    print("PYTHON & SYSTEM INFO")
    print("=" * 50)
    print(f"Python: {sys.version}")
    print(f"Platform: {platform.platform()}")
    print(f"Executable: {sys.executable}\n")


def check_core_libs():
    """Check core data science libraries."""
    print("=" * 50)
    print("CORE DATA LIBRARIES")
    print("=" * 50)
    libs = {
        'NumPy': np.__version__,
        'Pandas': pd.__version__,
        'Scikit-learn': sklearn.__version__
    }
    for name, ver in libs.items():
        print(f"{name:15} {ver}")
    print()


def check_pytorch():
    """Check PyTorch and hardware acceleration."""
    print("=" * 50)
    print("PYTORCH & HARDWARE")
    print("=" * 50)
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA: {torch.cuda.is_available()}")
    print(f"Apple MPS: {torch.backends.mps.is_available()}")
    
    sample = torch.randn(2, 3)
    print(f"\nTensor test:\n{sample}\n")


def check_transformers():
    """Check LLM libraries."""
    print("=" * 50)
    print("TRANSFORMERS & LLM TOOLS")
    print("=" * 50)
    print(f"Transformers: {transformers.__version__}")
    print(f"Accelerate: {accelerate.__version__}\n")


def check_data_viz():
    """Test data manipulation and visualization."""
    print("=" * 50)
    print("DATA TEST & VISUALIZATION")
    print("=" * 50)
    
    test_data = np.random.randn(50, 3)
    test_df = pd.DataFrame(test_data, columns=['X', 'Y', 'Z'])
    
    print(test_df.head(10))
    print(f"\nDataFrame shape: {test_df.shape}\n")
    
    fig, ax = plt.subplots(figsize=(8, 5))
    test_df.plot(kind='box', ax=ax)
    ax.set_title('Distribution Check')
    plt.tight_layout()
    plt.savefig('setup/distribution_check.png')
    print("Plot saved to: setup/distribution_check.png\n")


def main():
    """Run all verification checks."""
    try:
        check_python()
        check_core_libs()
        check_pytorch()
        check_transformers()
        check_data_viz()
        
        print("=" * 50)
        print("ALL CHECKS COMPLETE!")
        print("=" * 50)
        
    except Exception as e:
        print(f"\nError: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()