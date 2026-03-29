<div align="center">

# 🧠 T5 Dialogue Summarizer

### Fine-tuning T5-small for Abstractive Dialogue Summarization on the SAMSum Dataset

[![Python](https://img.shields.io/badge/Python-3.10-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black)](https://huggingface.co/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?style=for-the-badge&logo=jupyter&logoColor=white)](https://jupyter.org/)
[![License](https://img.shields.io/badge/License-MIT-22C55E?style=for-the-badge)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Active-22C55E?style=for-the-badge)]()

<br/>

> **Abstractive summarization** of real-world chat dialogues using a fine-tuned **T5-small** Transformer model — trained end-to-end with the HuggingFace `Trainer` API on the SAMSum corpus.

</div>

---

## 📌 Table of Contents

- [Overview](#-overview)
- [Dataset](#-dataset)
- [Project Architecture](#-project-architecture)
- [Pipeline](#-pipeline)
- [Model & Training Config](#-model--training-configuration)
- [Repository Structure](#-repository-structure)
- [Getting Started](#-getting-started)
- [Requirements](#-requirements)
- [Results](#-results)
- [Roadmap](#-roadmap)
- [Contributing](#-contributing)
- [License](#-license)

---

## 🔍 Overview

This project fine-tunes Google's **T5-small** (`t5-small`) model on the **SAMSum** dataset to perform **abstractive dialogue summarization** — the task of condensing multi-turn conversations into concise, human-readable summaries.

Unlike extractive summarization (which picks existing sentences), abstractive summarization **generates novel text**, closely mimicking how a human would summarize a chat.

| Feature | Detail |
|---|---|
| 🤖 **Base Model** | `t5-small` (60M parameters) |
| 📚 **Dataset** | SAMSum Corpus |
| 🏋️ **Task** | Abstractive Dialogue Summarization |
| 🧮 **Framework** | HuggingFace Transformers + PyTorch |
| 💻 **Platform** | CPU / CUDA / Apple MPS (auto-detected) |

---

## 📊 Dataset

The **SAMSum** dataset contains thousands of messenger-like conversations annotated with human-written summaries.

| Split | Total Samples | Samples Used |
|---|---|---|
| 🏋️ Train | 14,732 | 4,000 *(random seed=42)* |
| ✅ Validation | 818 | 500 *(random seed=42)* |
| 🧪 Test | ~820 | Full |

**Columns:** `id` · `dialogue` · `summary`

**Sample:**

```
Dialogue:
  Amanda: I baked cookies. Do you want some?
  Jerry: Sure!
  Amanda: I'll bring you tomorrow :-)

Summary:
  Amanda baked cookies and will bring Jerry some tomorrow.
```

> 📂 Dataset files are stored in the `samsum_dataset/` directory as CSV files.

---

## 🏗️ Project Architecture

```
Raw Dialogue (CSV)
       │
       ▼
┌─────────────────┐
│  Data Cleaning  │  ← Strip \r\n, extra spaces, HTML tags, lowercase
└────────┬────────┘
         │
         ▼
┌─────────────────────┐
│  T5 Tokenization    │  ← max_length=512 (input), 150 (summary)
│  T5Tokenizer        │  ← padding="max_length"
└────────┬────────────┘
         │
         ▼
┌──────────────────────────┐
│  T5ForConditionalGen.    │  ← Pre-trained: t5-small
│  Fine-tuned on SAMSum    │  ← 6 epochs, batch=8
└────────┬─────────────────┘
         │
         ▼
   Generated Summary
```

---

## 🔄 Pipeline

### 1️⃣ &nbsp; Data Loading & Exploration
```python
train_data = pd.read_csv("samsum-train.csv")
val_data   = pd.read_csv("samsum-validation.csv")
```

### 2️⃣ &nbsp; Random Sampling
```python
train_data = train_data.sample(n=4000, random_state=42).reset_index(drop=True)
val_data   = val_data.sample(n=500,  random_state=42).reset_index(drop=True)
```

### 3️⃣ &nbsp; Text Cleaning
```python
def clean_data(text):
    text = re.sub(r"\r\n", " ", text)   # remove line breaks
    text = re.sub(r"\s+", " ", text)    # collapse whitespace
    text = re.sub(r"<.*?>", " ", text)  # strip HTML tags
    return text.strip().lower()
```

### 4️⃣ &nbsp; Tokenization
```python
tokenizer = T5Tokenizer.from_pretrained("t5-small")

# Inputs  → dialogue tokens  (max_length=512)
# Targets → summary tokens   (max_length=150)
# Labels  → target input_ids added to inputs dict
```

### 5️⃣ &nbsp; Model Loading
```python
model = T5ForConditionalGeneration.from_pretrained("t5-small")
```

### 6️⃣ &nbsp; Training
```python
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=6,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    weight_decay=0.02,
    eval_strategy="epoch",
    save_strategy="epoch",
)
```

---

## ⚙️ Model & Training Configuration

| Hyperparameter | Value |
|---|---|
| 🏗️ Model | `t5-small` |
| 🔁 Epochs | `6` |
| 📦 Batch Size (Train) | `8` |
| 📦 Batch Size (Eval) | `8` |
| ⚖️ Weight Decay | `0.02` |
| 📏 Input Max Length | `512` tokens |
| 📝 Summary Max Length | `150` tokens |
| 💾 Save Strategy | `epoch` |
| 📊 Eval Strategy | `epoch` |
| 🖥️ Device | Auto (CUDA / MPS / CPU) |

---

## 📁 Repository Structure

```
t5-dialogue-summarizer/
│
├── 📓 Text_Summarizer.ipynb        # Main notebook — full pipeline
│
├── 📂 samsum_dataset/
│   ├── samsum-train.csv            # Training split      (14,732 samples)
│   ├── samsum-validation.csv       # Validation split    (818 samples)
│   └── samsum-test.csv             # Test split
│
├── 📂 results/                     # Model checkpoints (auto-generated)
│
├── 📄 requirements.txt             # Python dependencies
└── 📄 README.md                    # You are here
```

---

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/saifullah857/Transformer-T5-Dialogue-Summarizer.git
cd Transformer project
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Launch the Notebook

```bash
jupyter notebook Text_Summarizer.ipynb
```

### 4. Run Inference on a Custom Dialogue

```python
from transformers import T5Tokenizer, T5ForConditionalGeneration

tokenizer = T5Tokenizer.from_pretrained("./results/checkpoint-best")
model     = T5ForConditionalGeneration.from_pretrained("./results/checkpoint-best")

dialogue = "summarize: John: Are we meeting today? Sarah: Yes, at 3pm. John: Perfect, see you then!"
inputs   = tokenizer(dialogue, return_tensors="pt", max_length=512, truncation=True)
outputs  = model.generate(**inputs, max_length=60)

print(tokenizer.decode(outputs[0], skip_special_tokens=True))
# → "John and Sarah are meeting at 3pm today."
```

---

## 📦 Requirements

```txt
torch>=2.0.0
transformers>=4.35.0
pandas>=1.5.0
scikit-learn>=1.2.0
jupyter>=1.0.0
```

> Install all at once:
> ```bash
> pip install torch transformers pandas scikit-learn jupyter
> ```

---

## 📈 Results

> Training in progress — results will be updated after full fine-tuning.

| Metric | Score |
|---|---|
| 📊 ROUGE-1 | *TBD* |
| 📊 ROUGE-2 | *TBD* |
| 📊 ROUGE-L | *TBD* |
| 🔁 Epochs | 6 |

---

## 🗺️ Roadmap

- [x] Data loading & exploration
- [x] Text preprocessing pipeline
- [x] T5 tokenization
- [x] Model loading (T5-small)
- [x] Training argument configuration
- [ ] Full fine-tuning run
- [ ] ROUGE evaluation on test set
- [ ] Inference script / demo
- [ ] Upgrade to `t5-base` or `flan-t5`
- [ ] Gradio / Streamlit demo app

---

## 🤝 Contributing

Contributions are welcome! Feel free to open issues or submit pull requests.

```bash
# Fork → Clone → Create branch → Commit → Push → Pull Request
git checkout -b feature/your-feature-name
```

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

<div align="center">

Made with ❤️ using 🤗 HuggingFace Transformers & PyTorch

⭐ **Star this repo** if you found it useful!

</div>