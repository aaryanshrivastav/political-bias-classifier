import os
import re
import whisper
from langdetect import detect
from pytubefix import YouTube


def startfile(fn: str) -> None:
    """
    Open a file with the default system application.
    Works on Windows, macOS, and Linux.
    """
    if os.name == 'nt':  # Windows
        os.startfile(fn)
    elif os.name == 'posix':  # macOS or Linux
        os.system(f'open "{fn}"' if sys.platform == 'darwin' else f'xdg-open "{fn}"')


def create_and_open_txt(text: str, filename: str) -> None:
    """
    Create a .txt file with the given text and open it.
    """
    with open(filename, "w", encoding="utf-8") as file:
        file.write(text)
    startfile(filename)


def transcribe_youtube_video(url: str, model_name: str = "base") -> tuple[str, str]:
    """
    Download the audio from a YouTube video, transcribe it using Whisper,
    and return the transcribed text and detected language.
    """
    # Create output directory if not exists
    output_path = "YoutubeAudios"
    os.makedirs(output_path, exist_ok=True)

    # Extract video ID safely
    video_id = re.search(r"v=([a-zA-Z0-9_-]+)", url)
    video_id = video_id.group(1) if video_id else "unknown"

    filename = f"{video_id}_audio.mp3"
    audio_file_path = os.path.join(output_path, filename)

    # Download audio
    yt = YouTube(url)
    audio_stream = yt.streams.filter(only_audio=True).first()
    audio_stream.download(output_path=output_path, filename=filename)
    print(f"✅ Audio downloaded to {audio_file_path}")

    # Transcribe using Whisper
    model = whisper.load_model(model_name)
    result = model.transcribe(audio_file_path)
    transcribed_text = result["text"]
    print("Transcription complete.")

    # Detect language
    language = detect(transcribed_text)
    print(f"Detected language: {language}")

    return transcribed_text, language


def generate_transcript_file(url: str, model_name: str = "base") -> None:
    """
    Generate and open a text transcript file for a given YouTube URL.
    """
    transcribed_text, language = transcribe_youtube_video(url, model_name)
    output_filename = f"output_{language}.txt"
    create_and_open_txt(transcribed_text, output_filename)
    print(f"📄 Transcript saved as {output_filename}")


if __name__ == "__main__":
    url = input("Enter the YouTube video URL: ").strip()
    generate_transcript_file(url)
