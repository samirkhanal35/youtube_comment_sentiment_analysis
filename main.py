import os
import re
import emoji
import pandas as pd
import streamlit as st
from googleapiclient.discovery import build
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
from langdetect import detect, LangDetectException
from googletrans import Translator
from dotenv import load_dotenv

# Function to validate YouTube URL


def is_valid_youtube_url(url):
    youtube_regex = (
        r'(https?://)?(www\.)?'
        '(youtube|youtu|youtube-nocookie)\.(com|be)/'
        '(watch\?v=|embed/|v/|.+\?v=)?([^&=%\?]{11})')
    youtube_regex_match = re.match(youtube_regex, url)
    return youtube_regex_match is not None

# Function to extract video id from youtube video url


def extract_video_id(youtube_url: int):
    # Define the regex pattern for a YouTube video ID in various URL formats
    pattern = r"(?:https?:\/\/)?(?:www\.)?(?:youtube\.com\/(?:[^\/\n\s]+\/\S+\/|(?:v|e(?:mbed)?)\/|\S*?[?&]v=)|youtu\.be\/)([a-zA-Z0-9_-]{11})"

    # Search for the pattern in the given YouTube URL
    match = re.search(pattern, youtube_url)

    # Return the video ID if found
    if match:
        return match.group(1)
    else:
        return None

# Function to extract and return comments


def get_comments(video_id: str, api_key: str):
    comments = []
    # Create a Youtube resource object
    myYoutube = build('youtube', 'v3', developerKey=api_key)

    # Retrieving youtube video comments
    request = myYoutube.commentThreads().list(
        part='snippet, replies',
        maxResults=100,
        videoId=video_id
    )

    while request:
        response = request.execute()

        for item in response['items']:
            comment = item['snippet']['topLevelComment']['snippet']['textDisplay']
            comments.append(comment)

        # Check if there's a next page
        if 'nextPageToken' in response:
            request = myYoutube.commentThreads().list(
                part="snippet",
                videoId=video_id,
                textFormat="plainText",
                maxResults=100,
                pageToken=response['nextPageToken']
            )
        else:
            break

    return comments

# Function to replace emojis


def clean_comments(comment):
    # Replace emojis with their textual descriptions
    cleaned_comment = emoji.demojize(comment, delimiters=(":", ":"))

    # Removing special characters and symbols
    cleaned_comment = re.sub(r'[^\w\s]', '', cleaned_comment)

    return cleaned_comment.strip()


def detect_language(comment):
    try:
        return detect(comment)
    except LangDetectException:
        return 'unknown'


def translate_comment(comment, target_language='en'):
    translator = Translator()
    try:
        translation = translator.translate(comment, dest=target_language)
        return translation.text
    except Exception as e:
        return comment


def text_blob_analyze_sentiment(comment):
    # Perform sentiment analysis using TextBlob
    analysis = TextBlob(comment)
    polarity = analysis.sentiment.polarity
    # Categorize the sentiment score
    if polarity <= -0.05:
        return -1
    elif polarity >= 0.05:
        return 1
    else:
        return 0


def vader_analyze_sentiment(comment):
    # Perform sentiment analysis using VADER
    analyzer = SentimentIntensityAnalyzer()
    sentiment_scores = analyzer.polarity_scores(comment)
    compound = sentiment_scores['compound']
    # Categorize the sentiment score
    if compound <= -0.05:
        return -1
    elif compound >= 0.05:
        return 1
    else:
        return 0


# Streamlit app
def main():
    st.title("YouTube comments sentiment analyzer")

    # Input box for YouTube URL
    url = st.text_input("Enter YouTube URL:")

    # Dropdown for selecting a number between 100 and 1000 with steps of 100
    number = st.selectbox(
        "Select number of top comments:", range(100, 1100, 100))

    # Button to submit the URL
    if st.button("Submit"):
        if is_valid_youtube_url(url):
            st.success("The URL '{url}' is a valid YouTube URL.")
            st.write("Selected number: {number}")
            st.write("Extracting comments...")

            # # Load the .env file
            load_dotenv()

            # # Retrieve the API key from the .env file
            api_key = os.getenv("YOUTUBE_API_KEY")
            if not api_key:
                raise ValueError("API key not found in .env file")
            # else:
            #     print("api key found")

            # # Test the function with a sample YouTube URL
            youtube_url = url
            video_id = extract_video_id(youtube_url)

            if video_id:
                st.write(f"The video ID is: {video_id}")
            else:
                st.write("Video ID not found in the URL")

            # # getting all comments
            # all_comments = get_comments(video_id, api_key)
            # # print("all comments:", all_comments)

            # # converting the comments into pandas dataframe
            # df = pd.DataFrame(all_comments, columns=['comments'])
            # print("df head with emojis:", df.head())

            # # Replace emojis with their meanings
            # df['comments'] = df['comments'].apply(clean_comments)
            # print("df head with emojis replaced with their meanings:", df.head())

            # # Detect language of each comment
            # df['Language'] = df['comments'].apply(detect_language)

            # # Translate comments to English if they are not in English
            # df['Translated_Comment'] = df.apply(lambda row: translate_comment(
            #     row['comments']) if row['Language'] != 'en' else row['comments'], axis=1)

            # # Perform sentiment analysis on each comment using VADER
            # df['sentiment_vader'] = df['comments'].apply(vader_analyze_sentiment)

            # # Perform sentiment analysis on each comment using VADER
            # df['sentiment_textblob'] = df['comments'].apply(text_blob_analyze_sentiment)

            # print("\nTranslated dataset with sentiments")
            # print(df[['Translated_Comment', 'sentiment_vader', 'sentiment_textblob']].head(10))

            # print("sentiment sums for vader sentiment analysis:")
            # vader_valuecounts = df['sentiment_vader'].value_counts()
            # total_count = len(df)
            # print(vader_valuecounts)
            # print("Sentiments:")
            # print("Neutral:", (vader_valuecounts[0]/total_count)*100, '%')
            # print("Positive:", (vader_valuecounts[1]/total_count)*100, '%')
            # print("Negative:", (vader_valuecounts[-1]/total_count)*100, '%')

            # print("sentiment sums for textblob sentiment analysis:")
            # textblob_valuecounts = df['sentiment_textblob'].value_counts()
            # print(textblob_valuecounts)
            # print("Sentiments:")
            # print("Neutral:", (textblob_valuecounts[0]/total_count)*100, '%')
            # print("Positive:", (textblob_valuecounts[1]/total_count)*100, '%')
            # print("Negative:", (textblob_valuecounts[-1]/total_count)*100, '%')

        else:
            st.error("Invalid YouTube URL. Please enter a valid URL.")


if __name__ == "__main__":
    main()
