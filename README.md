# Cross-Lingual Text Classification

This project implements a multilingual text classification pipeline using HuggingFace's `transformers` library. It supports training a classifier on combined multilingual datasets and deploying it via API or UI.

---

## 📁 Project Structure

```
.
├── data/                   # Raw & preprocessed datasets
├── saved_model/           # Trained model directory
├── app.py                 # FastAPI backend for prediction
├── gradio_ui.py           # Gradio interface (optional)
├── train.py               # Script to train the model
├── requirements.txt       # Python dependencies
└── README.md              # Project documentation
```

---

## 🧠 Model Details

- **Base Model**: XLM-RoBERTa (`xlm-roberta-base`)
- **Framework**: Transformers + Datasets (HuggingFace)
- **Task**: Multi-class text classification (e.g., Sentiment/Intent/Topic detection)

---

## 🧪 Training Pipeline

1. **Data Loading**:
    - English + Hindi → `train`, `val`
    - German → held out for testing

2. **Preprocessing**:
    - Tokenized using `AutoTokenizer`
    - Batched using `DataCollatorWithPadding`

3. **Training Configuration**:
```python
TrainingArguments(
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=2,
    learning_rate=2e-5,
    save_strategy="epoch",
    evaluation_strategy="epoch",
    logging_dir="./logs",
    load_best_model_at_end=True,
    metric_for_best_model="accuracy"
)
```

---

## 🚀 Deployment Options

### Option 1: FastAPI (app.py)

```bash
uvicorn app:app --reload
```

### Option 2: Gradio UI

```bash
python gradio_ui.py
```

---

## 🔧 Setup

```bash
pip install -r requirements.txt
```

Train the model:

```bash
python train.py
```

---

## 🧪 Sample JSON Request (API)

```json
POST /predict
{
    "text": "Je suis très stressé aujourd'hui."
}
```

---

## 📊 Metrics & Results

To be filled in based on your test run — accuracy, F1, confusion matrix on unseen (German) data.

---

## 📦 Requirements

- transformers
- datasets
- scikit-learn
- torch
- fastapi
- uvicorn
- gradio

---

## 📜 License

MIT