from transformers import pipeline

classifier = pipeline("text-classification", model="../models/saved_model")

example_text = "The government should invest more in renewable energy."
print(classifier(example_text))
