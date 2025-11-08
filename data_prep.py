import pandas as pd
import re
import os
from datasets import load_dataset

def clean_text(text):
    """
    Cleans text: lowercases, removes URLs, brackets, special characters, extra spaces, and line breaks.
    """
    text = str(text).lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"\[.*?\]", "", text)
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
    text = re.sub(r'\s+', ' ', text)
    text = text.replace('\n', ' ').replace('\r', ' ')
    return text.strip()

def load_parquet_file(path):
    """
    Loads a Parquet file and cleans the 'text' column if present.
    """
    df = pd.read_parquet(path)
    if 'text' in df.columns:
        df['text'] = df['text'].apply(clean_text)
    return df

if __name__ == "__main__":
    data_dir = os.path.join(os.path.dirname(__file__), "data")
    os.makedirs(data_dir, exist_ok=True)

    # Example: parquet files already exist in ./data folder
    parquet_files = {
        "train": os.path.join(data_dir, "train.parquet"),
        "test": os.path.join(data_dir, "test.parquet"),
        "valid": os.path.join(data_dir, "valid.parquet"),
    }

    cleaned_dfs = {}

    for split_name, path in parquet_files.items():
        if os.path.exists(path):
            df = load_parquet_file(path)
            cleaned_dfs[split_name] = df
            output_path = os.path.join(data_dir, f"cleaned_{split_name}.parquet")
            df.to_parquet(output_path, index=False)
            print(f"💾 Saved cleaned {split_name} split to {output_path}")
        else:
            print(f"⚠️ Parquet file not found: {path}")

    print("✅ All existing Parquet files cleaned and saved successfully.")
