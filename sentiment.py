
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
nltk.download('wordnet')
nltk.download('stopwords')
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
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


im = Image.open("image/carat.ico")
st.set_page_config(page_title="FeelTech",page_icon=im,layout="wide")
st.header('Twitter Sentiment Analysis')
st.markdown("##")

#side bar
st.sidebar.image("image/sev.jpg",caption="Developed and Maintaned by: Hidayah Athira")
         
#switcher
st.sidebar.header("Twitter Analysis")
if __name__ == "__main__":
    with st.expander('Analyze tweets'):
            tweets = st.text_input('tweets here: ')
            if tweets:
                blob = TextBlob(tweets)
                st.write('Polarity: ', round(blob.sentiment.polarity,2))
                st.write('Subjectivity: ', round(blob.sentiment.subjectivity,2))


            pre = st.text_input('Clean tweets: ')
            if pre:
                st.write(cleantext.clean(pre, clean_all= False, extra_spaces=True ,
                                        stopwords=True ,lowercase=True ,numbers=True , punct=True))
    with st.expander('Analyze CSV'):
        upl = st.file_uploader('Upload file')
        if upl:
                df = pd.read_csv(upl)
                st.dataframe(df, use_container_width=True)
                positive_percentage = 0  # Initialize the variables before the if block
                negative_percentage = 0
                neutral_percentage = 0
                if st.button('Clean Data'):
                    
                    #convert all tweet into lowercase
                    df['tweets'] = df['tweets'].str.lower()


                    #Removing Twitter Handles(@User)
                    def remove_users(tweets):
                        remove_user = re.compile(r"@[A-Za-z0-9]+")
                        return remove_user.sub(r"", tweets)


                    df['tweets'] = df['tweets'].apply(remove_users)


                    #Remove links
                    def remove_links(tweets):
                        remove_no_link = re.sub(r"http\S+","",tweets)
                        return remove_no_link


                    df['tweets'] = df['tweets'].apply(remove_links)
                    df['tweets'].tail()

                    #Remove Punctuations, Numbers, and Special Characters
                    english_punctuations = string.punctuation
                    punctuations_list = english_punctuations

                    def cleaning_punctuations(tweets):
                        translator = str.maketrans('', '', punctuations_list)
                        return tweets.translate(translator)
                    df['tweets']= df['tweets'].apply(lambda x: cleaning_punctuations(x))
                    df['tweets'].tail()

                    #Repeating characters
                    def cleaning_repeating_char(tweets):
                        return re.sub(r'(.)1+', r'1', tweets)
                    df['tweets'] = df['tweets'].apply(lambda x: cleaning_repeating_char(x))
                    df['tweets'].tail()

                    #Remove Number
                    def cleaning_numbers(data):
                        return re.sub('[0-9]+', '', data)
                    df['tweets'] = df['tweets'].apply(lambda x: cleaning_numbers(x))
                    df['tweets'].tail()

                    #Remove short words
                    df['tweets'] = df['tweets'].apply(lambda x: ' '.join([w for w in x.split() if len(w)>3]))


                    #Remove hashtag
                    def remove_hashtags(tweets, pattern):
                        r = re.findall(pattern, tweets)
                        for i in r:
                            text = re.sub(i, '', tweets)
                        return tweets

                    df['tweets'] = np.vectorize(remove_hashtags)(df['tweets'],"#[\W]*")

                    #Emoji removal
                    def remove_emojis(string):
                        remove_emoji = re.compile("["u"\U0001F600-\U0001F64F"  # emoticons
                                                u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                                                u"\U0001F680-\U0001F6FF"  # transport & map symbols
                                                u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                                                "]+", flags=re.UNICODE)
                        return remove_emoji.sub(r'', string)

                    df['tweets'] = df['tweets'].apply(remove_emojis)


                    #Lemmatization
                    lemmatizer = WordNetLemmatizer()
                    wordnet_map = {"N":wordnet.NOUN, "V":wordnet.VERB, "R":wordnet.ADV}

                    def lemmatize_words(tweets):
                        pos_tagged_text = nltk.pos_tag(tweets.split())
                        return " ".join([lemmatizer.lemmatize(word, wordnet_map.get(pos[0], wordnet.NOUN)) for word, pos in pos_tagged_text])


                    #Prepare Stop words
                    stop_words = stopwords.words('english')
                    stop_words = ['from', 'https', 'twitter', 'still', "no","nor","aren't", 'couldn', "couldn't", 'didn', "didn't", "doesn", "doesn't", "don",
                                        "don't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 
                                        'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'no', 'not', "shan't", 
                                        'shan', "shan't", 'shouldn', "shouldn't","that'll", 'wasn', "wasn't", 'weren', 
                                        "weren't", "won't", 'wouldn', "wouldn't"]

                    def remove_stopwords(tweets):
                        return [[word for word in simple_preprocess(str(tweets)) if word not in stop_words]for tweets in tweets]

                    df['stop_word'] = remove_stopwords(df['tweets'])
                    df.head()

                    #Tokenize Word
                    def tokenize(tweets):
                        tokenizer = RegexpTokenizer(r'\w+')
                        return tokenizer.tokenize(tweets)

                    df['tweets'] = df['tweets'].apply(tokenize)
                    df['tweets'].head()


                    # Initialize sentiment counts
                    sentiment_counts = {"Positive": 0, "Negative": 0, "Neutral": 0}

                    # Function to get sentiment label using TextBlob
                    def label_sentiment(tokens):
                        # Join the list of tokens into a single string
                        text = ' '.join(tokens)
                        
                        # Create a TextBlob object
                        analysis = TextBlob(text)
                        
                        # Get the sentiment score
                        sentiment_score = analysis.sentiment.polarity
                        
                        if sentiment_score > 0:
                            return "positive"
                        elif sentiment_score < 0:
                            return "negative"
                        else:
                            return "neutral"

                    # Apply the labeling function to your DataFrame
                    df['sentiment'] = df['tweets'].apply(lambda x: label_sentiment(x))

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

                        st.write("Sentiment Analysis Results:")
                        
                        # Display individual progress bars for positive, negative, and neutral
                        st.write("Progress by Sentiment:")
                        st.write("Positive Percentage: {:.2f}%".format(positive_percentage))
                        st.progress(positive_percentage / 100)
                        st.write("Negative Percentage: {:.2f}%".format(negative_percentage))
                        st.progress(negative_percentage / 100)
                        st.write("Neutral Percentage: {:.2f}%".format(neutral_percentage))
                        st.progress(neutral_percentage / 100)
                        st.dataframe(df, use_container_width=True)
                        
                        # Filter tweets related to election, pru, and pilihanraya
                        election_keywords = ['general', 'pru15', 'malaysia']
                        election_related_tweets = df[df['tweets'].apply(lambda x: any(keyword in x for keyword in election_keywords))]

                        # Check if there are positive and negative tweets related to election
                        positive_tweets_election = election_related_tweets[election_related_tweets['sentiment'] == 'positive']['tweets']
                        negative_tweets_election = election_related_tweets[election_related_tweets['sentiment'] == 'negative']['tweets']

                        if not positive_tweets_election.empty and not negative_tweets_election.empty:
                            # Generate WordClouds for positive and negative sentiments
                            positive_tweets_election = ' '.join(positive_tweets_election.apply(lambda x: ' '.join(x)))
                            negative_tweets_election = ' '.join(negative_tweets_election.apply(lambda x: ' '.join(x)))

                            # WordCloud for Positive Sentiment related to election
                            st.write("WordCloud for Positive Sentiment Related to Election:")
                            wordcloud_positive_election = WordCloud(width=800, height=400, background_color='black').generate(positive_tweets_election)
                            plt.figure(figsize=(10, 5))
                            plt.imshow(wordcloud_positive_election, interpolation='bilinear')
                            plt.axis('off')
                            st.pyplot(plt)

                            # WordCloud for Negative Sentiment related to election
                            st.write("WordCloud for Negative Sentiment Related to Election:")
                            wordcloud_negative_election = WordCloud(width=800, height=400, background_color='black').generate(negative_tweets_election)
                            plt.figure(figsize=(10, 5))
                            plt.imshow(wordcloud_negative_election, interpolation='bilinear')
                            plt.axis('off')
                            st.pyplot(plt)
                            
                            # Create empty lists to store percentages for each text
                            positive_percentages = []
                            negative_percentages = []
                            neutral_percentages = []
                            
                            # Loop through each text and calculate the sentiment
                            for tokens in df['tweets']:
                                # Join the list of tokens into a single string
                                text = ' '.join(tokens)
                                
                                # Create a TextBlob object
                                analysis = TextBlob(text)
                                
                                # Get the sentiment score
                                sentiment_score = analysis.sentiment.polarity
                                
                                if sentiment_score > 0:
                                    positive_percentages.append(sentiment_score * 100)
                                    negative_percentages.append(0)
                                    neutral_percentages.append(0)
                                elif sentiment_score < 0:
                                    positive_percentages.append(0)
                                    negative_percentages.append(-sentiment_score * 100)
                                    neutral_percentages.append(0)
                                else:
                                    positive_percentages.append(0)
                                    negative_percentages.append(0)
                                    neutral_percentages.append(0)

                            # Add the percentages as new columns in the DataFrame
                            df['positive_percentage'] = positive_percentages
                            df['negative_percentage'] = negative_percentages
                            df['neutral_percentage'] = neutral_percentages

                            vectorizer = TfidfVectorizer(max_features=500000)

                            # Join the lists of tokens into strings
                            df['tweets_str'] = df['tweets'].apply(lambda x: ' '.join(x))

                            # Vectorize the text data
                            X = vectorizer.fit_transform(df['tweets_str'])
                            y = df['sentiment']

                            # Convert sentiment labels to numerical values
                            y = y.map({'positive': 0, 'negative': 1, 'neutral': 2})

                            # Split the data into training and testing sets
                            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

                                
                            def model_Evaluate(model):
                                # Predict values for Test dataset
                                y_pred = model.predict(X_test)

                                # Print the evaluation metrics for the dataset.
                                classification_rep = classification_report(y_test, y_pred)
                                st.write("Classification Report:")
                                st.text(classification_rep)

                                # Compute and plot the Confusion matrix
                                cf_matrix = confusion_matrix(y_test, y_pred)
                                categories = ['Positive', 'Negative', 'Neutral']
                                group_percentages = ['{0:.2%}'.format(value) for value in cf_matrix.flatten() / np.sum(cf_matrix)]
                                group_names = ['TPpos', 'Eneg', 'Eneu', 'Epos', 'TPneg', 'Eneu', 'Epos', 'Eneg', 'TPneu']
                                labels = [f'{v1}\n{v2}' for v1, v2 in zip(group_names, group_percentages)]
                                labels = np.asarray(labels).reshape(3, 3)
                                
                                # Display the Confusion Matrix
                                st.write("Confusion Matrix:")
                                plt.figure(figsize=(8, 6))
                                sns.heatmap(cf_matrix, annot=labels, fmt='', cmap="Blues", cbar=False,
                                            xticklabels=categories, yticklabels=categories)
                                plt.xlabel("Predicted values", fontdict={'size':14}, labelpad=10)
                                plt.ylabel("Actual values", fontdict={'size':14}, labelpad=10)
                                plt.title("Confusion Matrix", fontdict={'size':18}, pad=20)
                                st.pyplot(plt)

                            # Create a Multinomial Naive Bayes classifier
                            BNBmodel = BernoulliNB()
                            BNBmodel.fit(X_train, y_train)
                            model_Evaluate(BNBmodel)
                            y_pred1 = BNBmodel.predict(X_test)
                                
                            

                         
  
