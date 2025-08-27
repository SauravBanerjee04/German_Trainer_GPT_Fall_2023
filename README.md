# German Trainer — GPT (Fall 2023)

Interactive notebook(s) for experimenting with German language training/evaluation using large language models (LLMs). This repo is designed to be run as a Jupyter notebook locally or in Google Colab.

> Repo contents include:
> - `German_Trainer_LLM.ipynb` — main notebook with code and experiments  
> - `German-Trainer GPT Report.pdf` — accompanying write-up/report  
> *(both visible in the repository root)*

---

## Quick Start (Google Colab — easiest)

1. Open the notebook directly in Colab:
   - Go to the repo, click **German_Trainer_LLM.ipynb**, then **Open in Colab** (or copy its URL into Colab’s “GitHub” tab).
2. In Colab, run cells top-to-bottom.
3. If a cell errors due to a missing package, install it in-notebook with:
   ```bash
   %pip install PACKAGE_NAME
   ```
4. (Optional) If the notebook expects API keys (e.g., OpenAI/Hugging Face), add them in a hidden cell:
   ```python
   import os
   os.environ["OPENAI_API_KEY"] = "sk-..."         # example only
   os.environ["HF_TOKEN"] = "hf_..."               # example only
   ```

---

## Local Setup (Python)

### Prerequisites
- Python 3.9–3.11
- `git`
- Jupyter (either `notebook` or `jupyterlab`) or VS Code with the Jupyter extension
- (Optional) GPU + CUDA if you plan to fine-tune or run larger transformer models locally

### 1) Clone the repo
```bash
git clone https://github.com/SauravBanerjee04/German_Trainer_GPT_Fall_2023.git
cd German_Trainer_GPT_Fall_2023
```

### 2) Create a virtual environment
```bash
# macOS/Linux
python -m venv .venv
source .venv/bin/activate

# Windows (PowerShell)
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### 3) Install dependencies
This project doesn’t ship a `requirements.txt`. Start with a common LLM/notebook stack:
```bash
pip install --upgrade pip
pip install jupyter jupyterlab ipywidgets
pip install pandas numpy scikit-learn matplotlib
pip install torch --index-url https://download.pytorch.org/whl/cu121  # if using NVIDIA CUDA 12.1; otherwise omit/adjust
pip install transformers datasets accelerate
# If using OpenAI or Hugging Face Hub APIs:
pip install openai huggingface_hub
```

> Tip: If the notebook imports anything else (e.g., `evaluate`, `sentencepiece`, `sacrebleu`, `langcodes`), just `pip install` it as you encounter the import.

### 4) Launch Jupyter and run the notebook
```bash
jupyter lab
# or
jupyter notebook
```
Then open `German_Trainer_LLM.ipynb` and run cells sequentially.

---

## Running in VS Code

1. Install the **Python** and **Jupyter** extensions.
2. Open the repo folder.
3. Select your `.venv` interpreter (⌘⇧P / Ctrl+Shift+P → “Python: Select Interpreter”).
4. Open `German_Trainer_LLM.ipynb` and run the cells.

---

## Configuration & Secrets

- **Model/API keys**: If the notebook uses hosted models (OpenAI, Hugging Face Inference), set relevant keys in environment variables before starting Jupyter:
  ```bash
  # macOS/Linux
  export OPENAI_API_KEY="sk-..."
  export HF_TOKEN="hf-..."
  jupyter lab
  ```
  ```powershell
  # Windows PowerShell
  $env:OPENAI_API_KEY="sk-..."
  $env:HF_TOKEN="hf-..."
  jupyter lab
  ```
- **Checkpoints/Models**: If the notebook references local models, update any file paths in the cells to point to your environment.

---

## Data

If the notebook downloads public datasets via `datasets` (🤗), they will be cached automatically at `~/.cache/huggingface/datasets` by default. If you use custom CSV/TSV files, place them in a `data/` folder in the repo and adjust paths in the notebook accordingly.

---

## Troubleshooting

- **ModuleNotFoundError**: Install the missing package with `pip install <name>` inside your active virtual environment (or a Colab cell).
- **CUDA/GPU not used**: Verify `torch.cuda.is_available()` returns `True`. Install a CUDA-compatible PyTorch build for your driver/CUDA version.
- **API quota/auth errors**: Ensure your keys are set correctly; re-run the cell that reads environment variables.
- **Long cell runtimes**: Use smaller models (e.g., `distilbert-base`, `t5-small`, `gpt2`) or run on Colab GPU (Runtime → Change runtime type → GPU).

---

## Repo Structure

```
German_Trainer_GPT_Fall_2023/
├── German_Trainer_LLM.ipynb        # main notebook
└── German-Trainer GPT Report.pdf   # project report / write-up
```

---

## License

If you plan to open-source this work, add a `LICENSE` file (e.g., MIT or Apache-2.0). If using third-party models/datasets, follow their licenses and terms.

---

## Acknowledgments

- Hugging Face `transformers`, `datasets`, and `accelerate`
- PyTorch
- Jupyter/Colab ecosystem
