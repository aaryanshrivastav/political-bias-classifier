import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import os

# Load data
data_path = os.path.join(os.path.dirname(__file__), "data", "cleaned_train.parquet")
df = pd.read_parquet(data_path)

# Convert to Hugging Face Dataset and split
hf_dataset = Dataset.from_pandas(df)
hf_dataset = hf_dataset.train_test_split(test_size=0.2, seed=42)
train_ds = hf_dataset['train']
test_ds = hf_dataset['test']

# Keep only necessary columns
train_ds = train_ds.remove_columns([col for col in train_ds.column_names if col not in ['content', 'bias']])
test_ds  = test_ds.remove_columns([col for col in test_ds.column_names if col not in ['content', 'bias']])

# Rename label column for HF Trainer
train_ds = train_ds.rename_column("bias", "labels")
test_ds = test_ds.rename_column("bias", "labels")

# Model & Tokenizer
model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3)

# Tokenization
def tokenize(batch):
    return tokenizer(batch["content"], padding="max_length", truncation=True, max_length=256)

train_ds = train_ds.map(tokenize, batched=True)
test_ds = test_ds.map(tokenize, batched=True)

# Set format for PyTorch
train_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
test_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

# Metrics
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted')
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc, "f1": f1, "precision": precision, "recall": recall}

# ✅ Training arguments (fixed parameter name)
training_args = TrainingArguments(
    output_dir=os.path.join(os.path.dirname(__file__), "../models/saved_model"),
    do_eval=True,
    do_train=True,
    eval_strategy="steps",         # ✅ Changed from evaluation_strategy
    eval_steps=500,
    save_steps=500,
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir=os.path.join(os.path.dirname(__file__), "../models/logs"),
    logging_steps=100,
    load_best_model_at_end=True,  # ✅ Optional: load best checkpoint at end
    metric_for_best_model="f1",   # ✅ Optional: use F1 for best model selection
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=test_ds,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

# Train & evaluate
trainer.train()
trainer.evaluate()