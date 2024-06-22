import os
import re
import emoji
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import requests
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


# Function to check if video has comments enabled
def check_comments_enabled(video_id, api_key, url):
    url = url
    params = {
        'part': 'snippet,replies',
        'maxResults': 1,
        'videoId': video_id,
        'key': api_key
    }
    response = requests.get(url, params=params)
    if response.status_code == 403:
        return False
    return True

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
        videoId=video_id,
        textFormat="plainText",
        order="relevance"
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
                pageToken=response['nextPageToken'],
                order="relevance"
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


# Function to detect language in the comments
def detect_language(comment):
    try:
        return detect(comment)
    except LangDetectException:
        return 'unknown'

# Function to translate the comment to english


def translate_comment(comment, target_language='en'):
    translator = Translator()
    try:
        translation = translator.translate(comment, dest=target_language)
        return translation.text
    except Exception as e:
        return comment

# Function to Perform sentiment analysis using text blob


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

# Function to Perform sentiment analysis using vader


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
            st.write("Extracting video_id...")

            # # Load the .env file
            # load_dotenv()

            # # Retrieve the API key from the .env file
            # api_key = os.getenv("YOUTUBE_API_KEY")
            api_key = st.secrets("YOUTUBE_API_KEY")
            if not api_key:
                raise ValueError("API key not found in .env file")
            # else:
            #     print("api key found")

            # # Test the function with a sample YouTube URL
            youtube_url = url
            video_id = extract_video_id(youtube_url)

            if video_id:
                st.write("✔️")
                st.write(f"The video ID is: {video_id}")
            else:
                st.write("Video ID not found in the URL")
                return

            # # getting all comments
            st.write("Extracting comments...")
            try:
                all_comments = get_comments(video_id, api_key)
            except:
                st.error(
                    "The Youtube video has disabled the comments. So cannot scrape the comments.")
                return
            # # print("all comments:", all_comments)

            # # converting the comments into pandas dataframe
            st.write("✔️")
            st.write("Cleaning comments...")
            df = pd.DataFrame(all_comments, columns=['comments'])
            # print("df head with emojis:", df.head())

            # # Replace emojis with their meanings
            df['comments'] = df['comments'].apply(clean_comments)
            # print("df head with emojis replaced with their meanings:", df.head())

            # # Detect language of each comment
            st.write("✔️")
            st.write("Detecting language of comments...")
            df['Language'] = df['comments'].apply(detect_language)

            # # Translate comments to English if they are not in English
            st.write("✔️")
            st.write("Translating comments to english...")
            df['Translated_Comment'] = df.apply(lambda row: translate_comment(
                row['comments']) if row['Language'] != 'en' else row['comments'], axis=1)

            # # Perform sentiment analysis on each comment using VADER
            st.write("✔️")
            st.write("Performing sentiment analysis using VADER...")
            df['sentiment_vader'] = df['comments'].apply(
                vader_analyze_sentiment)

            # # Perform sentiment analysis on each comment using VADER
            st.write("✔️")
            st.write("Performing sentiment analysis using TextBlob...")
            df['sentiment_textblob'] = df['comments'].apply(
                text_blob_analyze_sentiment)

            # print("\nTranslated dataset with sentiments")
            # print(df[['Translated_Comment', 'sentiment_vader', 'sentiment_textblob']].head(10))

            # print("sentiment sums for vader sentiment analysis:")
            vader_valuecounts = df['sentiment_vader'].value_counts()
            total_count = len(df)

            # print("sentiment sums for textblob sentiment analysis:")
            textblob_valuecounts = df['sentiment_textblob'].value_counts()
            # print(textblob_valuecounts)

            # Data
            sentiments = ['Neutral Comments',
                          'Positive Comments', 'Negative Comments']
            vader_values = [round((vader_valuecounts[0]/total_count)*100), round(
                (vader_valuecounts[1]/total_count)*100), round((vader_valuecounts[-1]/total_count)*100)]
            textblob_values = [round((textblob_valuecounts[0]/total_count)*100), round(
                (textblob_valuecounts[1]/total_count)*100), round((textblob_valuecounts[-1]/total_count)*100)]

            # Position of the bars on the x-axis
            x = np.arange(len(sentiments))

            # Width of a bar
            width = 0.35

            # Create the bar chart
            fig, ax = plt.subplots()

            # Bars for Vader
            bars1 = ax.bar(x - width/2, vader_values, width, label='Vader')

            # Bars for TextBlob
            bars2 = ax.bar(x + width/2, textblob_values,
                           width, label='TextBlob')

            # Adding labels
            ax.set_xlabel('Sentiment Categories')
            ax.set_ylabel('Percentage')
            ax.set_title('Sentiment Analysis Using Vader and TextBlob')
            ax.set_xticks(x)
            ax.set_xticklabels(sentiments)
            ax.legend()

            # Adding the percentage labels inside the bars
            for i, v in enumerate(vader_values):
                ax.text(x[i] - width/2, v + 1,
                        f'{v:.2f}%', ha='center', va='bottom')

            for i, v in enumerate(textblob_values):
                ax.text(x[i] + width/2, v + 1,
                        f'{v:.2f}%', ha='center', va='bottom')

            # Display the plot in Streamlit
            st.pyplot(fig)

        else:
            st.error("Invalid YouTube URL. Please enter a valid URL.")


if __name__ == "__main__":
    main()
