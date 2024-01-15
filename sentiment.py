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
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from gensim.utils import simple_preprocess
from imblearn.over_sampling import SMOTE
from streamlit_option_menu import option_menu
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.naive_bayes import MultinomialNB, BernoulliNB, GaussianNB
from sklearn.preprocessing import label_binarize
from PIL import Image
from tqdm import tqdm
from wordcloud import WordCloud
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

nltk.download('vader_lexicon')



# Define the sid variable (SentimentIntensityAnalyzer) in the global scope
sid = SentimentIntensityAnalyzer()

# Load image for page icon
im = Image.open("image/carat.ico")

# Set Streamlit page configuration
st.set_page_config(page_title="FeelTech", page_icon=im, layout="wide")

# Main header
st.header('Twitter Sentiment Analysis')
st.markdown("##")

# Sidebar image and information
st.sidebar.image("image/carat.png", caption="Developed and Maintained by: Hidayah Athira")

# Sidebar switcher
st.sidebar.header("Twitter Analysis")

def Home():
    # Add this line to download WordNet data
    nltk.download('wordnet')
    nltk.download('stopwords')
    nltk.download('vader_lexicon')

    upl = st.file_uploader('Upload file')
    if upl:
        df = pd.read_csv(upl, encoding='latin-1')
        st.dataframe(df, use_container_width=True)

        # Data cleaning and sentiment analysis code
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
            remove_emoji = re.compile(
                "["u"\U0001F600-\U0001F64F"  # emoticons
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

        df['tweets'] = df['tweets'].apply(tokenize).tolist()
        df['tweets'].head()

        df.drop_duplicates(subset='tweets', keep='first', inplace=True)

        # Display the count of unique rows after removing duplicates
        st.write(f"Number of unique tweets after removing duplicates: {len(df)}")

        # Initialize sentiment counts
        sentiment_counts = {"Positive": 0, "Negative": 0}

        def calculate_vader_sentiment(tweet_list, threshold=0.05):
            sentiments = []
        
            for tweet in tweet_list:
                # Join the list of tokens into a single string
                text = ' '.join(tweet)
        
                sentiment_scores = sid.polarity_scores(text)
                compound_score = sentiment_scores['compound']
        
                if compound_score >= threshold:
                    sentiments.append('positive')
                else:
                    sentiments.append('negative')
        
            return sentiments
            
        # Apply the modified function to the 'tweets' column
        df['vader_sentiment_label'] = calculate_vader_sentiment(df['tweets'])
        df['vader_compound_score'] = df['tweets'].apply(lambda x: sid.polarity_scores(' '.join(x))['compound'])

        # Calculate percentages
        if df is not None:
            # Initialize sentiment counts
            sentiment_counts = {"Positive": 0, "Negative": 0}

            # Create Streamlit progress bar
            total_progress = st.progress(0)

            # Loop through each text and calculate the sentiment
            for i, tokens in enumerate(df['tweets']):

                # Join the list of tokens into a single string
                text = ' '.join(tokens)

                # Calculate the VADER sentiment label
                vader_sentiment_label = sid.polarity_scores(text)
                compound_score = vader_sentiment_label['compound']

                if compound_score >= 0.05:
                    sentiment_counts["Positive"] += 1
                elif compound_score <= -0.05:
                    sentiment_counts["Negative"] += 1

                # Update Streamlit total progress bar
                total_progress.progress((i + 1) / len(df))

            # Close Streamlit total progress bar
            st.success("Sentiment analysis completed!")

            # Display sentiment percentages
            total_tweets = len(df)
            vader_positive_percentage = (sentiment_counts["Positive"] / total_tweets) * 100
            vader_negative_percentage = (sentiment_counts["Negative"] / total_tweets) * 100
            st.write("Sentiment Analysis Results:")

            # Display individual progress bars for positive, negative, and neutral
            st.write("Progress by Sentiment:")
            st.write("Positive Percentage: {:.2f}%".format(vader_positive_percentage))
            st.progress(vader_positive_percentage / 100)
            st.write("Negative Percentage: {:.2f}%".format(vader_negative_percentage))
            st.progress(vader_negative_percentage / 100)
            return df

def visualize(df):
    # Filter tweets related to election, pru, and pilihanraya
    election_keywords = ['general', 'pru15', 'malaysia']
    election_related_tweets = df[df['tweets'].apply(lambda x: any(keyword in ' '.join(x) for keyword in election_keywords))]

    # Check if there are positive and negative tweets related to election
    positive_tweets_election = election_related_tweets[election_related_tweets['vader_sentiment_label'] == 'positive']['tweets']
    negative_tweets_election = election_related_tweets[election_related_tweets['vader_sentiment_label'] == 'negative']['tweets']

    if not positive_tweets_election.empty and not negative_tweets_election.empty:
        # Generate WordClouds for positive and negative sentiments
        positive_tweets_election = ' '.join(positive_tweets_election.apply(lambda x: ' '.join(x)))
        negative_tweets_election = ' '.join(negative_tweets_election.apply(lambda x: ' '.join(x)))

        # Create a two-column layout
        col1, col2 = st.columns(2)

        # WordCloud for Positive Sentiment related to election
        with col1:
            st.write("WordCloud for Positive Sentiment Related to Election:")
            wordcloud_positive_election = WordCloud(width=400, height=400, background_color='black').generate(
                positive_tweets_election)
            image_positive = wordcloud_positive_election.to_image()
            st.image(image_positive, caption='Positive Sentiment WordCloud', use_column_width=True)

        # WordCloud for Negative Sentiment related to election
        with col2:
            st.write("WordCloud for Negative Sentiment Related to Election:")
            wordcloud_negative_election = WordCloud(width=400, height=400, background_color='black').generate(
                negative_tweets_election)
            image_negative = wordcloud_negative_election.to_image()
            st.image(image_negative, caption='Negative Sentiment WordCloud', use_column_width=True)

        # Generate WordCloud based on Vader sentiment scores
        vader_positive_tweets = election_related_tweets[election_related_tweets['vader_sentiment_label'] == 'positive'][
            'tweets']
        vader_negative_tweets = election_related_tweets[election_related_tweets['vader_sentiment_label'] == 'negative'][
            'tweets']

        vader_positive_tweets = ' '.join(vader_positive_tweets.apply(lambda x: ' '.join(x)))
        vader_negative_tweets = ' '.join(vader_negative_tweets.apply(lambda x: ' '.join(x)))

        # Create a two-column layout for Vader sentiment WordClouds
        col3, col4 = st.columns(2)

        # WordCloud for Positive Sentiment based on Vader
        with col3:
            st.write("WordCloud for Positive Sentiment Based on Vader:")
            wordcloud_positive_vader = WordCloud(width=400, height=400, background_color='black').generate(
                vader_positive_tweets)
            image_positive_vader = wordcloud_positive_vader.to_image()
            st.image(image_positive_vader, caption='Positive Sentiment WordCloud (Vader)', use_column_width=True)

        # WordCloud for Negative Sentiment based on Vader
        with col4:
            st.write("WordCloud for Negative Sentiment Based on Vader:")
            wordcloud_negative_vader = WordCloud(width=400, height=400, background_color='black').generate(
                vader_negative_tweets)
            image_negative_vader = wordcloud_negative_vader.to_image()
            st.image(image_negative_vader, caption='Negative Sentiment WordCloud (Vader)', use_column_width=True)

        def tune_hyperparameters_bnb(X_train, y_train):
            # Define the parameter grid for Bernoulli Naive Bayes
            param_grid = {'alpha': [0.1, 0.5, 1.0, 1.5, 2.0]}

            # Create the Bernoulli Naive Bayes classifier
            bnb_model = BernoulliNB()

            # Instantiate the GridSearchCV object
            grid_search = GridSearchCV(estimator=bnb_model, param_grid=param_grid, scoring='accuracy', cv=5)

            # Fit the GridSearchCV to the data
            grid_search.fit(X_train, y_train)

            return grid_search.best_estimator_
        
        def tune_hyperparameters_mnb(X_train, y_train):
            # Define the parameter grid for Multinomial Naive Bayes
            param_grid = {'alpha': [0.1, 0.5, 1.0, 1.5, 2.0]}
        
            # Create the Multinomial Naive Bayes classifier
            mnb_model = MultinomialNB()
        
            # Instantiate the GridSearchCV object
            grid_search = GridSearchCV(estimator=mnb_model, param_grid=param_grid, scoring='accuracy', cv=5)
        
            # Fit the GridSearchCV to the data
            grid_search.fit(X_train, y_train)
        
            return grid_search.best_estimator_

        vectorizer = TfidfVectorizer(max_features=5000000, stop_words='english', norm='l2', sublinear_tf=True)

        # Convert the list of arrays to a 2D NumPy array
        X = vectorizer.fit_transform(df['tweets'].apply(lambda x: ' '.join(x)))
        y = df['vader_sentiment_label']

        # Convert sentiment labels to numerical values
        y_numerical = y.map({'positive': 0, 'negative':1, 'neutral':2})

        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y_numerical, test_size=0.2, random_state=42)

        def model_Evaluate(model):
            # Predict values for Test dataset
            y_pred = model.predict(X_test)

            # Convert sentiment labels to numerical values
            y_numerical = label_binarize(y_test, classes=[0, 1, 2])

            # Print the evaluation metrics for the dataset.
            classification_rep = classification_report(y_test, y_pred)
            st.write("Classification Report:")
            st.text(classification_rep)

            # Compute and plot the Confusion matrix
            cf_matrix = confusion_matrix(y_test, y_pred)
            categories = ['Positive', 'Negative']
            group_percentages = ['{0:.2%}'.format(value) for value in cf_matrix.flatten() / np.sum(cf_matrix)]
            group_names = ['True Positive', 'False Positive', 'False Negative', 'True Negative']
            labels = [f'{v1}\n{v2}' for v1, v2 in zip(group_names, group_percentages)]
            labels = np.asarray(labels).reshape(len(categories), len(categories))
            
            # Display the Confusion Matrix
            st.write("Confusion Matrix:")
            plt.figure(figsize=(8, 6))
            sns.heatmap(cf_matrix, annot=labels, fmt='', cmap="Blues", cbar=False,
                        xticklabels=categories, yticklabels=categories)
            plt.xlabel("Predicted values", fontdict={'size':14}, labelpad=10)
            plt.ylabel("Actual values", fontdict={'size':14}, labelpad=10)
            plt.title("Confusion Matrix", fontdict={'size':18}, pad=20)
            st.pyplot(plt)

            # Compute ROC curve and ROC area for each class
            fpr = dict()
            tpr = dict()
            roc_auc = dict()
        
            for i in range(2):
                fpr[i], tpr[i], _ = roc_curve(y_numerical[:, i], model.predict_proba(X_test)[:, i])
                roc_auc[i] = auc(fpr[i], tpr[i])
        
            # Display the ROC curve
            st.write("Receiver Operating Characteristic (ROC) Curve:")
            plt.figure(figsize=(8, 6))
        
            for i in range(2):
                plt.plot(fpr[i], tpr[i], label=f'Class {i} (AUC = {roc_auc[i]:.2f})')
        
            plt.plot([0, 1], [0, 1], 'k--', color='grey', linestyle='--', label='Random')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('ROC Curve')
            plt.legend(loc='lower right')
            plt.grid(True)
            
            # Display the ROC curve plot
            st.pyplot(plt)
             
        # Create a Best Bernoulli Naive Bayes classifier
        best_bnb_model = tune_hyperparameters_bnb(X_train, y_train)
        cv_scores = cross_val_score(best_bnb_model, X_train, y_train, cv=5, scoring='accuracy')
        st.write("Cross-Validation Scores:")
        st.write(cv_scores)
        st.write(f"Mean Accuracy: {np.mean(cv_scores)}")
        st.write(f"Standard Deviation: {np.std(cv_scores)}")

        # Create a Best Multinomial Naive Bayes classifier
        best_mnb_model = tune_hyperparameters_mnb(X_train, y_train)
        st.subheader("Evaluation for Multinomial Naive Bayes Model:")
        model_Evaluate(best_mnb_model)
        
        st.subheader("Evaluation for Bernoulli Naive Bayes Model:")
        model_Evaluate(best_bnb_model)
        y_pred_original = best_bnb_model.predict(X_test)

def sideBar():
    with st.sidebar:
        selected = option_menu(
            menu_title="Main Menu",
            options=["Home"],
            icons=["house"],
            menu_icon="cast",
            default_index=0
        )
    if selected == "Home":
        df = Home()
        if df is not None:  # Check if the dataframe is not empty before calling visualize
            visualize(df)

# Run the sidebar function
sideBar()
