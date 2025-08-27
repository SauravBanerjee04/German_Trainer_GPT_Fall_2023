# German Trainer — GPT (Fall 2023)

This project explores how Large Language Models (LLMs) like GPT can be used as **German language training assistants**.  
The focus was on building an interactive notebook that:
- Generates exercises for practicing German vocabulary and grammar
- Evaluates student responses
- Provides feedback in natural language
- Benchmarks GPT performance against traditional datasets and evaluation metrics

The repository includes:
- `German_Trainer_LLM.ipynb` — main notebook with implementation, experiments, and results  
- `German-Trainer GPT Report.pdf` — detailed write-up explaining methodology, experiments, and findings  

---

## 🔍 Project Overview

### Goal
To investigate whether modern LLMs can function as effective German language tutors — helping learners practice in a dynamic, adaptive way instead of relying only on static exercises.

### Process
1. **Notebook Setup**  
   - Implemented training and testing pipelines inside Jupyter/Colab.  
   - Integrated Hugging Face `transformers` and `datasets` for model access.  
   - Optionally connected to APIs (OpenAI GPT, Hugging Face Hub).  

2. **Exercise Generation**  
   - Created vocabulary quizzes, grammar fill-in-the-blanks, and translation tasks.  
   - Leveraged prompt engineering to control GPT outputs.  

3. **Evaluation**  
   - Compared generated exercises to standard datasets.  
   - Measured correctness, fluency, and usefulness of GPT’s explanations.  
   - Used BLEU/ROUGE and qualitative feedback as evaluation metrics.  

4. **Iteration & Analysis**  
   - Adjusted prompts and task formats to improve response quality.  
   - Ran multiple experiments, logging results in the notebook.  

### Results
- GPT performed **very well** on vocabulary/translation tasks, often generating creative and correct exercises.  
- Grammar explanations were useful but occasionally inconsistent.  
- Students benefitted from interactive, conversational practice compared to static drills.  
- The project showed LLMs can be a **viable supplement** to traditional German learning tools — especially for self-study.

---

## 🚀 How to Run

### Option 1: Google Colab (recommended)
1. Open the notebook in Colab:
   - In GitHub, click **German_Trainer_LLM.ipynb** → **Open in Colab**
2. Run the cells top-to-bottom.
3. If a dependency is missing, install it inside the notebook:
   ```bash
   %pip install PACKAGE_NAME
   ```
4. (Optional) If using APIs (OpenAI/Hugging Face), set your keys:
   ```python
   import os
   os.environ["OPENAI_API_KEY"] = "sk-..."   # example only
   os.environ["HF_TOKEN"] = "hf-..."
   ```

### Option 2: Local (Python)
```bash
# Clone repo
git clone https://github.com/SauravBanerjee04/German_Trainer_GPT_Fall_2023.git
cd German_Trainer_GPT_Fall_2023

# Create venv
python -m venv .venv
source .venv/bin/activate      # macOS/Linux
.venv\Scripts\Activate.ps1     # Windows PowerShell

# Install common deps
pip install jupyter jupyterlab ipywidgets
pip install pandas numpy scikit-learn matplotlib
pip install torch transformers datasets accelerate
pip install openai huggingface_hub
```

Then run:
```bash
jupyter lab
```
and open `German_Trainer_LLM.ipynb`.

---

## 📂 Repo Structure

```
German_Trainer_GPT_Fall_2023/
├── German_Trainer_LLM.ipynb        # main notebook (implementation + experiments)
└── German-Trainer GPT Report.pdf   # final report with detailed results
```

---

## ⚡ Key Takeaways

- GPT can generate **realistic, adaptive exercises** in German.  
- Evaluation shows promise for **AI-assisted language learning**.  
- Future work: consistency improvements, scaling to other languages, and more rigorous human evaluation.

---

## 🛠️ Built With
- Python, Jupyter, Google Colab
- Hugging Face `transformers`, `datasets`, `accelerate`
- PyTorch
- OpenAI API (optional)

---

## 📜 License
If you plan to make this public for reuse, add a `LICENSE` file (MIT or Apache-2.0 recommended).  

---

## 🙏 Acknowledgments
- Hugging Face ecosystem  
- PyTorch  
- OpenAI GPT models  
- German language learners who tested the exercises
