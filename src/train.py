import argparse
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
from umap import UMAP
import os
import shutil

# Dataset class
class SpamDataset(Dataset):
    def __init__(self, encodings, labels, texts=None):
        self.encodings = encodings
        self.labels = labels
        self.texts = texts

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        if self.texts is not None:
            item['text'] = self.texts[idx]
        return item

# Tokenize helper
def tokenize_texts(tokenizer, texts):
    return tokenizer(texts, truncation=True, padding=True, max_length=128, return_tensors="pt")

# Metric computation
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds, average="weighted"),
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="xlm-roberta-base")
    parser.add_argument("--output_dir", type=str, default="./results")
    parser.add_argument("--num_train_epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()

    df = pd.read_csv("data-en-hi-de-fr.csv")
    df.columns = ['labels', 'english', 'hindi', 'german', 'french']
    df['labels'] = df['labels'].map({"ham": 0, "spam": 1})

    df_train = pd.concat([
        pd.DataFrame({'text': df['english'], 'label': df['labels'], 'lang': 'en'}),
        pd.DataFrame({'text': df['hindi'],   'label': df['labels'], 'lang': 'hi'})
    ], ignore_index=True)

    df_test = pd.DataFrame({'text': df['german'], 'label': df['labels'], 'lang': 'de'})

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    train_encodings = tokenize_texts(tokenizer, df_train['text'].tolist())
    test_encodings = tokenize_texts(tokenizer, df_test['text'].tolist())

    train_dataset = SpamDataset(train_encodings, df_train['label'].tolist(), df_train['text'].tolist())
    test_dataset = SpamDataset(test_encodings, df_test['label'].tolist(), df_test['text'].tolist())

    model = AutoModelForSequenceClassification.from_pretrained(args.model_name, num_labels=2)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        num_train_epochs=args.num_train_epochs,
        learning_rate=3e-5,
        weight_decay=0.01,
        logging_dir="./logs",
        eval_strategy="epoch",
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
    )

    trainer.train()
    predictions = trainer.predict(test_dataset)
    preds = predictions.predictions.argmax(axis=-1)
    labels = predictions.label_ids

    print(classification_report(labels, preds, target_names=["ham", "spam"]))

    cm = confusion_matrix(labels, preds)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["ham", "spam"], yticklabels=["ham", "spam"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.show()

    model.save_pretrained("./local_model")
    tokenizer.save_pretrained("./local_model")
    shutil.make_archive("spam_model", 'zip', "./local_model")

if __name__ == "__main__":
    main()
