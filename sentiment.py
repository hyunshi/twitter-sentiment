import pandas as pd
import streamlit as st
import cleantext
import numpy as np
import re
import nltk
import gensim.utils
import time
import seaborn as sns
import matplotlib.pyplot as plt
import string
from nltk.sentiment import SentimentIntensityAnalyzer
nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('vader_lexicon')
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from nltk.sentiment import SentimentIntensityAnalyzer
from gensim.utils import simple_preprocess
from textblob import TextBlob
from streamlit_option_menu import option_menu
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from PIL import Image
from tqdm import tqdm
from wordcloud import WordCloud
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

# Define the sid variable (SentimentIntensityAnalyzer) in the global scope
sid = SentimentIntensityAnalyzer()

im = Image.open("image/carat.ico")
st.set_page_config(page_title="FeelTech", page_icon=im, layout="wide")
st.header('Twitter Sentiment Analysis')
st.markdown("##")

# side bar
st.sidebar.image("image/carat.png", caption="Developed and Maintained by: Hidayah Athira")

# switcher
st.sidebar.header("Twitter Analysis")
with st.expander('Analyze CSV'):
    upl = st.file_uploader('Upload file')
    if upl:
        df = pd.read_csv(upl, encoding='latin-1')
        st.dataframe(df, use_container_width=True)
        positive_percentage = 0  # Initialize the variables before the if block
        negative_percentage = 0
        neutral_percentage = 0
        if st.button('Clean Data'):
            # convert all tweet into lowercase
            df['tweets'] = df['tweets'].str.lower()

            # Removing Twitter Handles(@User)
            def remove_users(tweets):
                remove_user = re.compile(r"@[A-Za-z0-9]+")
                return remove_user.sub(r"", tweets)

            df['tweets'] = df['tweets'].apply(remove_users)

            # Remove links
            def remove_links(tweets):
                remove_no_link = re.sub(r"http\S+", "", tweets)
                return remove_no_link

            df['tweets'] = df['tweets'].apply(remove_links)
            df['tweets'].tail()

            # Remove Punctuations, Numbers, and Special Characters
            english_punctuations = string.punctuation
            punctuations_list = english_punctuations

            def cleaning_punctuations(tweets):
                translator = str.maketrans('', '', punctuations_list)
                return tweets.translate(translator)

            df['tweets'] = df['tweets'].apply(lambda x: cleaning_punctuations(x))
            df['tweets'].tail()

            # Repeating characters
            def cleaning_repeating_char(tweets):
                return re.sub(r'(.)1+', r'1', tweets)

            df['tweets'] = df['tweets'].apply(lambda x: cleaning_repeating_char(x))
            df['tweets'].tail()

            # Remove Number
            def cleaning_numbers(data):
                return re.sub('[0-9]+', '', data)

            df['tweets'] = df['tweets'].apply(lambda x: cleaning_numbers(x))
            df['tweets'].tail()

            # Remove short words
            df['tweets'] = df['tweets'].apply(lambda x: ' '.join([w for w in x.split() if len(w) > 3]))

            # Remove hashtag
            def remove_hashtags(tweets, pattern):
                r = re.findall(pattern, tweets)
                for i in r:
                    text = re.sub(i, '', tweets)
                return tweets

            df['tweets'] = np.vectorize(remove_hashtags)(df['tweets'], "#[\W]*")

            # Emoji removal
            def remove_emojis(string):
                remove_emoji = re.compile("["u"\U0001F600-\U0001F64F"  # emoticons
                                          u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                                          u"\U0001F680-\U0001F6FF"  # transport & map symbols
                                          u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                                          "]+", flags=re.UNICODE)
                return remove_emoji.sub(r'', string).encode("utf-8").decode("utf-8")

            df['tweets'] = df['tweets'].apply(remove_emojis)

            # Lemmatization
            lemmatizer = WordNetLemmatizer()
            wordnet_map = {"N": wordnet.NOUN, "V": wordnet.VERB, "R": wordnet.ADV}

            def lemmatize_words(tweets):
                pos_tagged_text = nltk.pos_tag(tweets.split())
                return " ".join([lemmatizer.lemmatize(word, wordnet_map.get(pos[0], wordnet.NOUN)) for word, pos in
                                 pos_tagged_text])

            # Prepare Stop words
            stop_words = stopwords.words('english')
            stop_words = ['from', 'https', 'twitter', 'still', "no", "nor", "aren't", 'couldn', "couldn't", 'didn',
                          "didn't", "doesn", "doesn't", "don", "don't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven',
                          "haven't", 'isn', "isn't", 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'no',
                          'not', "shan't", 'shan', "shan't", 'shouldn', "shouldn't", "that'll", 'wasn', "wasn't", 'weren',
                          "weren't", "won't", 'wouldn', "wouldn't"]

            def remove_stopwords(tweets):
                return [[word for word in simple_preprocess(str(tweets)) if word not in stop_words] for tweets in
                        tweets]

            df['stop_word'] = remove_stopwords(df['tweets'])
            df.head()

            # Tokenize Word
            def tokenize(tweets):
                tokenizer = RegexpTokenizer(r'\w+')
                return tokenizer.tokenize(tweets)

            df['tweets'] = df['tweets'].apply(tokenize)
            df['tweets'].head()

            df.drop_duplicates(subset='tweets', keep='first', inplace=True)

            # Display the count of unique rows after removing duplicates
            st.write(f"Number of unique tweets after removing duplicates: {len(df)}")

            # Initialize sentiment counts
            sentiment_counts = {"Positive": 0, "Negative": 0, "Neutral": 0}

            # Function to get sentiment label using TextBlob
            def label_sentiment(tokens):
                # Join the list of tokens into a single string
                text = ' '.join(tokens)

                # Create a TextBlob object
                blob = TextBlob(text)

                # Get the sentiment score
                blob_sentiment_score = blob.sentiment.polarity

                # Get the VADER sentiment label
                sid = SentimentIntensityAnalyzer()
                vader_sentiment_score = sid.polarity_scores(text)['compound']

                # Choose the sentiment label based on the higher absolute sentiment score
                if abs(blob_sentiment_score) > abs(vader_sentiment_score):
                    blob_sentiment_label = "positive" if blob_sentiment_score > 0 else "negative"
                    vader_sentiment_label = "positive" if vader_sentiment_score > 0 else "negative"
                elif abs(blob_sentiment_score) < abs(vader_sentiment_score):
                    blob_sentiment_label = "positive" if blob_sentiment_score > 0 else "negative"
                    vader_sentiment_label = "positive" if vader_sentiment_score > 0 else "negative"
                else:
                    blob_sentiment_label = "neutral"
                    vader_sentiment_label = "neutral"

                return blob_sentiment_label, vader_sentiment_label

            # Apply the labeling function to your DataFrame
            df['sentiment'] = df['tweets'].apply(lambda x: label_sentiment(x))

            def calculate_vader_sentiment(tweet_list):
                sid = SentimentIntensityAnalyzer()
                sentiments = []

                for tweet in tweet_list:
                    # Join the list of tokens into a single string
                    text = ' '.join(tweet)

                    sentiment_scores = sid.polarity_scores(text)
                    compound_score = sentiment_scores['compound']

                    if compound_score >= 0.05:
                        sentiments.append('positive')
                    elif compound_score <= -0.05:
                        sentiments.append('negative')
                    else:
                        sentiments.append('neutral')

                return sentiments

            # Apply the modified function to the 'tweets' column
            df['vader_sentiment_label'] = calculate_vader_sentiment(df['tweets'])
            df['vader_compound_score'] = df['tweets'].apply(lambda x: sid.polarity_scores(' '.join(x))['compound'])

            # Calculate percentages
            if df is not None:
                # Initialize sentiment counts
                sentiment_counts = {"Positive": 0, "Negative": 0, "Neutral": 0}

                # Create Streamlit progress bar
                total_progress = st.progress(0)

                # Loop through each text and calculate the sentiment
                for i, tokens in enumerate(df['tweets']):
                    # Join the list of tokens into a single string
                    text = ' '.join(tokens)

                    # Create a TextBlob object
                    analysis = TextBlob(text)

                    # Get the sentiment score
                    sentiment_score = analysis.sentiment.polarity

                    if sentiment_score > 0:
                        sentiment_counts["Positive"] += 1
                    elif sentiment_score < 0:
                        sentiment_counts["Negative"] += 1
                    else:
                        sentiment_counts["Neutral"] += 1

                    # Update Streamlit total progress bar
                    total_progress.progress((i + 1) / len(df))

                # Close Streamlit total progress bar
                st.success("Sentiment analysis completed!")

                # Display sentiment percentages
                total_tweets = len(df)
                positive_percentage = (sentiment_counts["Positive"] / total_tweets) * 100
                negative_percentage = (sentiment_counts["Negative"] / total_tweets) * 100
                neutral_percentage = (sentiment_counts["Neutral"] / total_tweets) * 100

                total_tweets = len(df)
                vader_positive_percentage = (df[df['vader_sentiment_label'] == 'positive'].shape[0] / total_tweets) * 100
                vader_negative_percentage = (df[df['vader_sentiment_label'] == 'negative'].shape[0] / total_tweets) * 100
                vader_neutral_percentage = (df[df['vader_sentiment_label'] == 'neutral'].shape[0] / total_tweets) * 100
                st.write("Sentiment Analysis Results:")

                # Display individual progress bars for positive, negative, and neutral
                st.write("Progress by Sentiment:")
                st.write("Positive Percentage: {:.2f}%".format(vader_positive_percentage))
                st.progress(vader_positive_percentage / 100)
                st.write("Negative Percentage: {:.2f}%".format(vader_negative_percentage))
                st.progress(vader_negative_percentage / 100)
                st.write("Neutral Percentage: {:.2f}%".format(vader_neutral_percentage))
                st.progress(vader_neutral_percentage / 100)
                st.dataframe(df, use_container_width=True)
            return df


def visualize(df):
    # ... (your existing visualize function code)

    def tune_hyperparameters_bnb(X_train, y_train):
        # ... (your existing tune_hyperparameters_bnb function code)

    # ... (rest of your functions)

    def sideBar():
        with st.sidebar:
            selected = option_menu(
                menu_title="Main Menu",
                options=["Home", "Visualization"],
                icons=["house", "eye"],
                menu_icon="cast",
                default_index=0
            )
        if selected == "Home":
            df = Home()
        elif selected == "Visualization":
            df = Home()
            visualize(df)

    sideBar()

# Call the sideBar function to start the application
sideBar()
