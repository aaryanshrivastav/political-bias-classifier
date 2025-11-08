import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import os

# ✅ Load data
train_path = os.path.join(os.path.dirname(__file__), "data", "cleaned_train.parquet")
test_path  = os.path.join(os.path.dirname(__file__), "data", "cleaned_test.parquet")

train_df = pd.read_parquet(train_path)
test_df  = pd.read_parquet(test_path)

# ✅ Map all label values to numeric (0, 1, 2)
label_map = {
    "left": 0,
    "right": 1,
    "center": 2,
    0: 0,
    1: 1,
    2: 2
}

if "label" in train_df.columns:
    train_df["bias"] = train_df["label"].map(label_map)
elif "bias" in train_df.columns:
    train_df["bias"] = train_df["bias"].map(label_map)

if "label" in test_df.columns:
    test_df["bias"] = test_df["label"].map(label_map)
if "bias" in test_df.columns:
    test_df["bias"] = test_df["bias"].map(label_map)

# ✅ Drop rows with missing or unmapped labels
train_df = train_df.dropna(subset=["bias"])
test_df = test_df.dropna(subset=["bias"])

# ✅ Ensure bias is int
train_df["bias"] = train_df["bias"].astype(int)
test_df["bias"] = test_df["bias"].astype(int)

# ✅ Ensure we have the right text column (from transcript or content)
if "content" not in train_df.columns and "transcript" in train_df.columns:
    train_df["content"] = train_df["transcript"]

if "content" not in test_df.columns and "transcript" in test_df.columns:
    test_df["content"] = test_df["transcript"]

# ✅ Convert to Hugging Face Datasets
train_ds = Dataset.from_pandas(train_df[["content", "bias"]])
test_ds  = Dataset.from_pandas(test_df[["content", "bias"]])

# ✅ Rename for Trainer
train_ds = train_ds.rename_column("bias", "labels")
test_ds = test_ds.rename_column("bias", "labels")

# ✅ Model & Tokenizer
model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3)

# ✅ Tokenization
def tokenize(batch):
    return tokenizer(batch["content"], padding="max_length", truncation=True, max_length=256)

train_ds = train_ds.map(tokenize, batched=True)
test_ds = test_ds.map(tokenize, batched=True)

train_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
test_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

# ✅ Metrics
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted')
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc, "f1": f1, "precision": precision, "recall": recall}

# ✅ Training arguments
training_args = TrainingArguments(
    output_dir=os.path.join(os.path.dirname(__file__), "../models/saved_model"),
    do_eval=True,
    do_train=True,
    eval_strategy="steps",
    eval_steps=500,
    save_steps=500,
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir=os.path.join(os.path.dirname(__file__), "../models/logs"),
    logging_steps=100,
    load_best_model_at_end=True,
    metric_for_best_model="f1",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=test_ds,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

trainer.train()
trainer.evaluate()
