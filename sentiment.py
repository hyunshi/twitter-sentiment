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
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, roc_auc_score
from sklearn.naive_bayes import MultinomialNB, BernoulliNB, GaussianNB
from sklearn.preprocessing import label_binarize
from PIL import Image
from tqdm import tqdm
from wordcloud import WordCloud
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from gensim.models import Word2Vec

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
        df['tweets'] = df['tweets'].astype(str).str.lower()

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

        # Remove Punctuations, Numbers, and Special Characters
        english_punctuations = string.punctuation
        punctuations_list = english_punctuations

        def cleaning_punctuations(tweets):
            translator = str.maketrans('', '', punctuations_list)
            return tweets.translate(translator)

        df['tweets'] = df['tweets'].apply(lambda x: cleaning_punctuations(x))

        # Repeating characters
        def cleaning_repeating_char(tweets):
            return re.sub(r'(.)1+', r'1', tweets)

        df['tweets'] = df['tweets'].apply(lambda x: cleaning_repeating_char(x))

        # Remove Number
        def cleaning_numbers(data):
            return re.sub('[0-9]+', '', data)

        df['tweets'] = df['tweets'].apply(lambda x: cleaning_numbers(x))

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
        stop_words = ['from', 'https', 'twitter', 'still', "nor", "aren't", 'couldn', "couldn't", 'didn',
                      "didn't", "doesn", "doesn't", "don", "don't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven',
                      "haven't", 'isn', "isn't", 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't",
                      "shan't", 'shan', "shan't", 'shouldn', "shouldn't", "that'll", 'wasn', "wasn't", 'weren',
                      "weren't", "won't", 'wouldn', "wouldn't"]

        def remove_stopwords(tweets):
            return [[word for word in simple_preprocess(str(tweets)) if word not in stop_words] for tweets in
                    tweets]

        df['stop_word'] = remove_stopwords(df['tweets'])

        # Tokenize Word
        def tokenize(tweets):
            tokenizer = RegexpTokenizer(r'\w+')
            return tokenizer.tokenize(tweets)

        df['tweets'] = df['tweets'].apply(tokenize).tolist()

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
    # Calculate the number of positive and negative tweets
    positive_count = df[df['vader_sentiment_label'] == 'positive'].shape[0]
    negative_count = df[df['vader_sentiment_label'] == 'negative'].shape[0]

    st.write(f"Total number of positive tweets: {positive_count}")
    st.write(f"Total number of negative tweets: {negative_count}")

    # Create a bar chart showing the number of positive and negative tweets
    st.write("Number of Positive and Negative Tweets:")
    fig, ax = plt.subplots()
    ax.bar(['Positive', 'Negative'], [positive_count, negative_count])
    ax.set_xlabel("Sentiment")
    ax.set_ylabel("Number of Tweets")
    ax.set_title("Number of Positive and Negative Tweets")
    st.pyplot(fig)

    # Separate positive and negative tweets
    positive_tweets = df[df['vader_sentiment_label'] == 'positive']['tweets'].tolist()
    negative_tweets = df[df['vader_sentiment_label'] == 'negative']['tweets'].tolist()

    # Flatten the list of lists
    flat_positive_tweets = [item for sublist in positive_tweets for item in sublist]
    flat_negative_tweets = [item for sublist in negative_tweets for item in sublist]

    # Create WordCloud for positive words
    st.write("WordCloud for Positive Words:")
    wordcloud_positive = WordCloud(width=400, height=400, background_color='black').generate(' '.join(flat_positive_tweets))
    image_positive = wordcloud_positive.to_image()
    st.image(image_positive, caption='Positive Words WordCloud', use_column_width=True)

    # Create WordCloud for negative words
    st.write("WordCloud for Negative Words:")
    wordcloud_negative = WordCloud(width=400, height=400, background_color='black').generate(' '.join(flat_negative_tweets))
    image_negative = wordcloud_negative.to_image()
    st.image(image_negative, caption='Negative Words WordCloud', use_column_width=True)


    def tune_hyperparameters_bnb(X_train, y_train):
        # Define the parameter grid for Bernoulli Naive Bayes
        param_grid = {'alpha': [0.1, 0.5, 1.0]}

        # Create the Bernoulli Naive Bayes classifier
        bnb_model = BernoulliNB()

        # Instantiate the GridSearchCV object
        grid_search = GridSearchCV(estimator=bnb_model, param_grid=param_grid, scoring='accuracy', cv=5)

        # Fit the GridSearchCV to the data
        grid_search.fit(X_train, y_train)

        return grid_search.best_estimator_
        
    vectorizer = TfidfVectorizer(max_features=50000, stop_words='english', norm='l2', sublinear_tf=True)

    # Convert the list of arrays to a 2D NumPy array
    X = vectorizer.fit_transform(df['tweets'].apply(lambda x: ' '.join(x)))
    y = df['vader_sentiment_label']

    num_features = X.shape[1]

    # Convert sentiment labels to numerical values
    y_numerical = y.map({'positive': 0, 'negative': 1})

    # Apply SMOTE to balance the classes
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y_numerical)
    
    # Split the resampled data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)
    st.write(f"Number of Features (Vocabulary Size): {num_features}")
    
    def model_Evaluate(model):
        # Predict values for Test dataset
        y_pred = model.predict(X_test)

        # Convert sentiment labels to numerical values
        y_numerical = label_binarize(y_test, classes=['positive', 'negative'])

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
        
        # Compute and plot the ROC-AUC curve for positive class
        proba_positive_class = model.predict_proba(X_test)[:, 1]
        fpr_positive, tpr_positive, thresholds_positive = roc_curve(y_test, proba_positive_class, pos_label=1)
        
        # Compute and plot the ROC-AUC curve for negative class
        proba_negative_class = model.predict_proba(X_test)[:, 0]
        fpr_negative, tpr_negative, thresholds_negative = roc_curve(y_test, proba_negative_class, pos_label=0)
        
        # Display the ROC-AUC Curve
        st.write("ROC-AUC Curve:")
        plt.figure(figsize=(8, 6))
        
        # Plot the ROC curve for the positive class
        plt.plot(fpr_positive, tpr_positive, color='darkorange', lw=2, label='ROC curve (area = {:.2f}) for Positive Class'.format(auc(fpr_positive, tpr_positive)))
        
        # Plot the ROC curve for the negative class
        plt.plot(fpr_negative, tpr_negative, color='navy', lw=2, linestyle='--', label='ROC curve (area = {:.2f}) for Negative Class'.format(auc(fpr_negative, tpr_negative)))
        
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc='lower right')
        st.pyplot(plt)
    def model_Evaluate_svm(model):
        # Predict values for Test dataset
        y_pred = model.predict(X_test)
    
        # Convert sentiment labels to numerical values
        y_numerical = label_binarize(y_test, classes=['positive', 'negative'])
    
        # Print the evaluation metrics for the dataset.
        classification_rep = classification_report(y_test, y_pred)
        st.write("SVM Model Classification Report:")
        st.text(classification_rep)
    
        # Compute and plot the Confusion matrix
        cf_matrix = confusion_matrix(y_test, y_pred)
        categories = ['Positive', 'Negative']
        group_percentages = ['{0:.2%}'.format(value) for value in cf_matrix.flatten() / np.sum(cf_matrix)]
        group_names = ['True Positive', 'False Positive', 'False Negative', 'True Negative']
        labels = [f'{v1}\n{v2}' for v1, v2 in zip(group_names, group_percentages)]
        labels = np.asarray(labels).reshape(len(categories), len(categories))
    
        # Display the Confusion Matrix
        st.write("SVM Model Confusion Matrix:")
        plt.figure(figsize=(8, 6))
        sns.heatmap(cf_matrix, annot=labels, fmt='', cmap="Blues", cbar=False,
                    xticklabels=categories, yticklabels=categories)
        plt.xlabel("Predicted values", fontdict={'size':14}, labelpad=10)
        plt.ylabel("Actual values", fontdict={'size':14}, labelpad=10)
        plt.title("SVM Model Confusion Matrix", fontdict={'size':18}, pad=20)
        st.pyplot(plt)
    
        # Compute and plot the ROC-AUC curve for positive class
        proba_positive_class = model.decision_function(X_test)
        fpr_positive, tpr_positive, thresholds_positive = roc_curve(y_test, proba_positive_class)
    
        # Display the ROC-AUC Curve for SVM Model
        st.write("SVM Model ROC-AUC Curve:")
        plt.figure(figsize=(8, 6))
        plt.plot(fpr_positive, tpr_positive, color='darkorange', lw=2, label='SVM ROC curve (area = {:.2f})'.format(auc(fpr_positive, tpr_positive)))
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('SVM Model Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc='lower right')
        st.pyplot(plt)
                 
    # Create a Best Bernoulli Naive Bayes classifier
    best_bnb_model = tune_hyperparameters_bnb(X_train, y_train)

    st.subheader("Evaluation for Bernoulli Naive Bayes Model:")
    cv_scores = cross_val_score(best_bnb_model, X_train, y_train, cv=5, scoring='accuracy')
    st.write("Cross-Validation Scores:")
    st.write(cv_scores)
    st.write(f"Mean Accuracy: {np.mean(cv_scores)}")
    st.write(f"Standard Deviation: {np.std(cv_scores)}")
    model_Evaluate(best_bnb_model)
    y_pred_original = best_bnb_model.predict(X_test)

    st.subheader("Evaluation for SVM Model:")
    svm_model = SVC(kernel='linear', C=1)
    svm_model.fit(X_train, y_train)
    model_Evaluate_svm(svm_model)
    y_pred_original = svm_model.predict(X_test)

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
