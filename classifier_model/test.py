from transformers import AutoTokenizer, AutoModelForSequenceClassification
import shutil
import os

final_checkpoint = r"C:\Projects\political-bias-classifier-1\classifier_model\models\saved_model\checkpoint-4389"  # e.g. "C:\\Projects\\...\\checkpoint-1500"
export_dir = r"C:\Projects\political-bias-classifier-1\classifier_model\models\final_models"

# Save the final version
tokenizer = AutoTokenizer.from_pretrained(final_checkpoint)
model = AutoModelForSequenceClassification.from_pretrained(final_checkpoint)
model.save_pretrained(export_dir)
tokenizer.save_pretrained(export_dir)

print(f"✅ Final model exported to: {export_dir}")

# (optional) remove other checkpoints to save space
for folder in os.listdir(os.path.dirname(final_checkpoint)):
    if folder.startswith("checkpoint") and folder != os.path.basename(final_checkpoint):
        shutil.rmtree(os.path.join(os.path.dirname(final_checkpoint), folder))
