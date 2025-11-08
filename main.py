import os
from TranscriptGenerator import transcribe_youtube_video
from TranscriptTranslator import translate_to_english
from Model import classify_long_text
from langdetect import detect


def process_youtube_video(url: str):
    """
    Full pipeline:
    1. Transcribe YouTube video
    2. Detect language
    3. Translate to English (if needed)
    4. Classify political bias
    5. Return structured results
    """
    print("\nStep 1: Transcribing video")
    transcribed_text, language = transcribe_youtube_video(url)
    print(f"🗣️ Detected language: {language}")

    english_text = transcribed_text
    if language != "en":
        print("\nStep 2: Translating to English")
        try:
            english_text = translate_to_english(transcribed_text)
        except Exception as e:
            print("Proceeding with original transcript.")

    print("\nStep 3: Classifying Political Bias")
    result = classify_long_text(english_text)

    return {
        "language": language,
        "translated": english_text != transcribed_text,
        "final_label": result["final_label"],
        "label_scores": result["label_scores"],
    }


if __name__ == "__main__":
    youtube_url = input("Enter YouTube video URL: ").strip()
    results = process_youtube_video(youtube_url)

    print("\nPipeline Summary:")
    for key, value in results.items():
        print(f"{key}: {value}")
