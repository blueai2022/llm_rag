# llm_rag

LLM Projects, including RAG with Reranking

## Enviroment

### Quick Start (Jupyter Notebook)
For quick experimentation and learning:

**Prerequisites:**
- Python 3.9+
- pip (Python package manager)
- (Optional) virtualenv or conda for environment isolation

**Setup:**
```bash
# Create and activate virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate  # On Mac/Linux

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Launch Jupyter
jupyter notebook
```

**Key dependencies:** `torch`, `transformers`, `numpy`, `pandas`, `scikit-learn`, `jupyter`

**Verify installation:**
```bash
python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
python -c "import transformers; print(f'Transformers version: {transformers.__version__}')"
```

