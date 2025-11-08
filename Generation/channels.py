from googleapiclient.discovery import build
import pandas as pd

def get_channel_videos(api_key: str, channel_id: str, max_results: int = 200):
    """
    Get video IDs from a YouTube channel
    
    Args:
        api_key: YouTube Data API key
        channel_id: YouTube channel ID
        max_results: Number of videos to fetch
    """
    youtube = build('youtube', 'v3', developerKey="AIzaSyCK97ISDl5InhPUpmGwhqOlQHDp0pGQtFY")
    
    video_ids = []
    next_page_token = None
    
    while len(video_ids) < max_results:
        # Get uploads playlist ID
        request = youtube.channels().list(
            part='contentDetails',
            id=channel_id
        )
        response = request.execute()
        
        playlist_id = response['items'][0]['contentDetails']['relatedPlaylists']['uploads']
        
        # Get videos from playlist
        request = youtube.playlistItems().list(
            part='contentDetails',
            playlistId=playlist_id,
            maxResults=min(50, max_results - len(video_ids)),
            pageToken=next_page_token
        )
        response = request.execute()
        
        for item in response['items']:
            video_ids.append(item['contentDetails']['videoId'])
        
        next_page_token = response.get('nextPageToken')
        if not next_page_token:
            break
    
    return video_ids[:max_results]


# Example usage
YOUTUBE_API_KEY = "AIzaSyCK97ISDl5InhPUpmGwhqOlQHDp0pGQtFY"

channels = {
    'Dhruv Rathee': 'UC-CSyyi47VX1lD9zyeABW3w',
    'Mohak Mangal': 'UCz4a7agVFr1TxU-mpAP8hkw',
    'Swarajya': 'UCvXXqqFxmyI0YFewhOMJqxQ'
}

all_video_ids = {}
for name, channel_id in channels.items():
    print(f"Fetching videos from {name}...")
    video_ids = get_channel_videos(YOUTUBE_API_KEY, channel_id, max_results=50)
    all_video_ids[name] = video_ids
    print(f"  Found {len(video_ids)} videos")

# Save for later use
pd.DataFrame(all_video_ids).to_csv('video_ids.csv', index=False)