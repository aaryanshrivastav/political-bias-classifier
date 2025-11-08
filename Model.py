from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification

# Load tokenizer & model directly from Hugging Face Hub
tokenizer = AutoTokenizer.from_pretrained("Arstacity/political-bias-classifier")
model = AutoModelForSequenceClassification.from_pretrained("Arstacity/political-bias-classifier")

# Label mapping
label_map = {
    "LABEL_0": "Left",
    "LABEL_1": "Right",
    "LABEL_2": "Center"
}

def classify_long_text(text, chunk_size=512, overlap=50):
    # Initialize classifier
    classifier = pipeline(
        "text-classification",
        model=model,
        tokenizer=tokenizer,
        truncation=True,
        max_length=chunk_size
    )

    # Split into overlapping chunks
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)

    # Run classification on all chunks
    results = classifier(chunks)

    # Aggregate label scores
    label_scores = {}
    for r in results:
        label = r['label']
        score = r['score']
        label_scores[label] = label_scores.get(label, 0) + score

    # Find final label
    final_label = max(label_scores, key=label_scores.get)
    readable_label = label_map.get(final_label, final_label)

    return {
        "final_label": readable_label,
        "label_scores": {label_map.get(k, k): v for k, v in label_scores.items()}
    }


if __name__=="__main__":
    transcript = """Enter Your Example Transcript"""
    result = classify_long_text(transcript)
    print(result)
