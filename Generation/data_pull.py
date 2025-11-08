import google.generativeai as genai
from youtube_transcript_api import YouTubeTranscriptApi
from googleapiclient.discovery import build
import pandas as pd
import time
import os
from tqdm import tqdm

class TranscriptDatasetBuilder:
    def __init__(self, gemini_api_key: str, youtube_api_key: str):
        """
        Build a labeled dataset using YouTube transcripts and Gemini API
        """
        # Configure Gemini
        genai.configure(api_key=gemini_api_key)
        self.model = genai.GenerativeModel('models/gemini-2.5-flash')

        # Configure YouTube
        self.youtube = build('youtube', 'v3', developerKey=youtube_api_key)

        # Classification prompt
        self.classification_prompt = """
You are a political bias classifier. Analyze the following video transcript and classify its political bias.

Transcript:
{transcript}

Instructions:
- Classify as: "left", "center", or "right"
- Base your decision on the political stance expressed in the content
- "left" = progressive, liberal viewpoints
- "center" = balanced, bipartisan, or non-partisan content
- "right" = conservative, traditional viewpoints
- Consider the overall tone and policy positions mentioned

Respond with ONLY ONE WORD: left, center, or right

Classification:"""

    def get_transcript(self, video_id: str) -> str:
        """Fetch transcript from YouTube video"""
        try:
            ytt_api = YouTubeTranscriptApi()
            transcript_list = ytt_api.list(video_id)
            transcript = transcript_list.find_transcript(['en', 'en-IN', 'en-US']).fetch()
            transcript_text = " ".join([entry.text for entry in transcript])
            return transcript_text
        except Exception as e:
            print(f"❌ Transcript error for {video_id}: {e}")
            return None

    def get_video_title(self, video_id: str) -> str:
        """Fetch video title from YouTube Data API"""
        try:
            response = self.youtube.videos().list(
                part="snippet",
                id=video_id
            ).execute()
            return response['items'][0]['snippet']['title']
        except Exception as e:
            print(f"❌ Title fetch error for {video_id}: {e}")
            return "Unknown Title"

    def classify_with_gemini(self, transcript: str, max_retries=3):
        """Classify transcript using Gemini API"""
        if not transcript:
            return None

        max_chars = 8000
        if len(transcript) > max_chars:
            transcript = transcript[:max_chars] + "..."

        for attempt in range(max_retries):
            try:
                response = self.model.generate_content(
                    self.classification_prompt.format(transcript=transcript)
                )
                label = response.text.strip().lower()
                if label not in ['left', 'center', 'right']:
                    print(f"⚠️ Invalid label '{label}', retrying...")
                    time.sleep(2)
                    continue
                return label
            except Exception as e:
                print(f"⚠️ Gemini API error (attempt {attempt+1}/{max_retries}): {e}")
                time.sleep(5)
        return None

    def build_dataset_from_csv(self, csv_path: str, output_path="data/labeled_dataset.parquet", delay=2.0):
        """
        Build labeled dataset from CSV containing video IDs per channel
        """
        df = pd.read_csv(csv_path)
        sources = list(df.columns)
        print(f"📄 Loaded {csv_path} with sources: {sources}")

        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        data = []
        total_videos = df.size
        print(f"\nProcessing {total_videos} videos total...\n")

        with tqdm(total=total_videos, desc="Processing videos") as pbar:
            for source in sources:
                video_ids = df[source].dropna().tolist()
                for video_id in video_ids:
                    video_id = str(video_id).strip()
                    transcript = self.get_transcript(video_id)
                    if not transcript:
                        pbar.update(1)
                        continue

                    title = self.get_video_title(video_id)
                    label = self.classify_with_gemini(transcript)
                    if not label:
                        pbar.update(1)
                        continue

                    data.append({
                        "video_title": title,
                        "video_id": video_id,
                        "transcript": transcript,
                        "label": label,
                        "source": source
                    })
                    pbar.update(1)
                    time.sleep(delay)

        df_out = pd.DataFrame(data)
        df_out.to_parquet(output_path)
        print(f"\n✅ Dataset saved to: {output_path}")
        print(df_out['label'].value_counts())
        return df_out


# =============================
# USAGE
# =============================
if __name__ == "__main__":
    GEMINI_API_KEY = "AIzaSyBHFb-hkGlduQ0RamXCGYcV_cT_cY94KR0"
    YOUTUBE_API_KEY = "AIzaSyCK97ISDl5InhPUpmGwhqOlQHDp0pGQtFY"

    builder = TranscriptDatasetBuilder(GEMINI_API_KEY, YOUTUBE_API_KEY)
    dataset = builder.build_dataset_from_csv(
        csv_path="video_ids.csv",
        output_path="data/labeled_dataset.parquet",
        delay=2.0  # seconds between API calls
    )

    print("\n🎯 Final dataset preview:")
    print(dataset.head())
