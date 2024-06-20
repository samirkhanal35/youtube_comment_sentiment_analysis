import os
import re
from googleapiclient.discovery import build
from dotenv import load_dotenv


# Function to extract video id from youtube video url
def extract_video_id(youtube_url):
    # Define the regex pattern for a YouTube video ID in various URL formats
    pattern = r"(?:https?:\/\/)?(?:www\.)?(?:youtube\.com\/(?:[^\/\n\s]+\/\S+\/|(?:v|e(?:mbed)?)\/|\S*?[?&]v=)|youtu\.be\/)([a-zA-Z0-9_-]{11})"

    # Search for the pattern in the given YouTube URL
    match = re.search(pattern, youtube_url)

    # Return the video ID if found
    if match:
        return match.group(1)
    else:
        return None


# Load the .env file
load_dotenv()

# Retrieve the API key from the .env file
api_key = os.getenv("YOUTUBE_API_KEY")
if not api_key:
    raise ValueError("API key not found in .env file")
else:
    print("api key found")

# Test the function with a sample YouTube URL
youtube_url = "https://www.youtube.com/watch?v=FqpnI3CNLvE&list=RDMMZgYEINUZcU0&index=27"
video_id = extract_video_id(youtube_url)

if video_id:
    print(f"The video ID is: {video_id}")
else:
    print("Video ID not found in the URL")
