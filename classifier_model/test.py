from transformers import AutoTokenizer, AutoModelForSequenceClassification
import shutil
import os

final_checkpoint = r"C:\Projects\political-bias-classifier-1\classifier_model\models\saved_model\checkpoint-5613"  # e.g. "C:\\Projects\\...\\checkpoint-1500"
export_dir = r"C:\Projects\political-bias-classifier-1\classifier_model\models\final_models_2"

# Save the final version
tokenizer = AutoTokenizer.from_pretrained(final_checkpoint)
model = AutoModelForSequenceClassification.from_pretrained(final_checkpoint)
model.save_pretrained(export_dir)
tokenizer.save_pretrained(export_dir)

print(f"✅ Final model exported to: {export_dir}")
