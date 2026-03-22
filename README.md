# 🌐 English to Hindi Translator (NLP Project)

## 📌 Overview
This project is a Neural Machine Translation (NMT) model that translates English sentences into Hindi using the Hugging Face Transformers library.
The model is trained on a bilingual dataset and fine-tuned using a Seq2Seq Transformer architecture.

---

## 🚀 Features
- 🔤 Translate English text to Hindi
- 🤖 Uses Transformer-based deep learning model
- ⚡ GPU-accelerated training (PyTorch)
- 📊 Fine-tuned on real translation dataset
- 🧠 Supports custom input sentences

---

## 🛠️ Tech Stack
- Python
- Hugging Face Transformers
- PyTorch
- Datasets Library
- Google Colab

---

## 📂 Dataset
Dataset used:
cfilt/iitb-english-hindi

- Contains English-Hindi sentence pairs
- Used for training and validation

---

## 🧠 Model
Pretrained model:
Helsinki-NLP/opus-mt-en-hi

- Fine-tuned for improved translation performance
- Based on Seq2Seq Transformer architecture

---

## ⚙️ Installation

pip install transformers datasets sentencepiece accelerate
---
## ▶️ How to Run

### 1. Load model & tokenizer
```python
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
model_name = "Helsinki-NLP/opus-mt-en-hi"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

##For Translation
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
text = "How are you?"
inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
inputs = {k: v.to(device) for k, v in inputs.items()}
outputs = model.generate(**inputs)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
