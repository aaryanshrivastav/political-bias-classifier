import os
from dotenv import load_dotenv
import google.generativeai as genai


def load_gemini_api_key() -> str:
    """
    Load the Gemini API key from the .env file.
    Raises an error if not found.
    """
    load_dotenv()
    gemini_api_key = os.getenv("GEMINI_API_KEY")
    if not gemini_api_key:
        raise ValueError("GEMINI_API_KEY is not set. Please ensure it is defined in your .env file.")
    return gemini_api_key


def translate_to_english(text: str, model_name: str = "gemini-2.0-flash") -> str:
    """
    Translate the given text to English using the Gemini API SDK.
    """
    api_key = load_gemini_api_key()
    genai.configure(api_key=api_key)

    try:
        model = genai.GenerativeModel(model_name)
        prompt = f"""
            Translate the following text to neutral and accurate English. 
            Preserve factual content and political tone exactly as in the original language, without adding bias or interpretation.

            Text:
            {text}
            """
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        print(f"Translation failed: {e}")
        return text  

if __name__ == "__main__":
    user_text = input("Enter the text you want to translate to English: ").strip()
    translated = translate_to_english(user_text)
    print("\nTranslated Text:")
    print(translated)
