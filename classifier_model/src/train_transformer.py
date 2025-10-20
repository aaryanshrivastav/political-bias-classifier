import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments

# Load data
df = pd.read_csv("../data/cleaned_dataset.csv")
hf_dataset = Dataset.from_pandas(df)
hf_dataset = hf_dataset.train_test_split(test_size=0.2)
train_ds = hf_dataset['train']
test_ds = hf_dataset['test']

# Model setup
model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3)

# Tokenization
def tokenize(batch):
    return tokenizer(batch['text'], padding=True, truncation=True, max_length=256)

train_ds = train_ds.map(tokenize, batched=True)
test_ds = test_ds.map(tokenize, batched=True)

# Training arguments
training_args = TrainingArguments(
    output_dir="../models/saved_model",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir='../models/logs',
    load_best_model_at_end=True
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=test_ds,
    tokenizer=tokenizer
)

trainer.train()
trainer.evaluate()
