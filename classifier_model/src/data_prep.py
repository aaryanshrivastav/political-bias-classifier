import pandas as pd
import re

def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"\[.*?\]", "", text)
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
    return text

def load_dataset(path):
    df = pd.read_csv(path)
    df['text'] = df['text'].apply(clean_text)
    return df

if __name__ == "__main__":
    df = load_dataset("../data/dataset.csv")
    df.to_csv("../data/cleaned_dataset.csv", index=False)
    print("Dataset cleaned and saved.")
