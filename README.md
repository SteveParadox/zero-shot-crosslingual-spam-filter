# Cross-Lingual Text Classification

This project implements a multilingual text classification pipeline using HuggingFace's `transformers` library. It supports training a classifier on combined multilingual datasets and deploying it via FastAPI.

---

## 📁 Project Structure

```
.
├── data/                   # Raw & preprocessed datasets
├── saved_model/           # Trained model directory
├── app.py                 # FastAPI backend for prediction
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
    output_dir="./results",
    eval_strategy="yes",                
    save_strategy="no",                     
    learning_rate=3e-5,                     
    per_device_train_batch_size=32,         
    num_train_epochs=3,                    
    weight_decay=0.01,
    logging_steps=50,                      
    logging_dir="./logs",
    report_to="wandb",                     
    disable_tqdm=False,                    
    save_total_limit=1                     
)
```

---

## 🚀 Deployment Options

### Option 1: FastAPI (app.py)

```bash
uvicorn app:app --reload
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

| Label            | Precision | Recall | F1-Score | Support |
| ---------------- | --------- | ------ | -------- | ------- |
| **Ham**          | 1.00      | 0.99   | 1.00     | 4825    |
| **Spam**         | 0.96      | 0.98   | 0.97     | 747     |
| **Accuracy**     | –         | –      | **0.99** | 5572    |
| **Macro Avg**    | 0.98      | 0.99   | 0.98     | 5572    |
| **Weighted Avg** | 0.99      | 0.99   | 0.99     | 5572    |

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
