import re
import string
import matplotlib.pyplot as plt
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tag import pos_tag
from nltk.tokenize import word_tokenize
from nltk import FreqDist, classify, NaiveBayesClassifier
import random

# Function to remove noise from tokens
def remove_noise(tweet_tokens, stop_words=()):
    cleaned_tokens = []
    for token, tag in pos_tag(tweet_tokens):
        token = re.sub('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+#]|[!*\(\),]|' \
                       '(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', token)
        token = re.sub("(@[A-Za-z0-9_]+)", "", token)

        # Lemmatization based on POS tagging
        if tag.startswith("NN"):
            pos = 'n'
        elif tag.startswith('VB'):
            pos = 'v'
        else:
            pos = 'a'

        lemmatizer = WordNetLemmatizer()
        token = lemmatizer.lemmatize(token, pos)

        # Add token to cleaned tokens if it's not punctuation or stop word
        if len(token) > 0 and token not in string.punctuation and token.lower() not in stop_words:
            cleaned_tokens.append(token.lower())
    return cleaned_tokens

# Function to get all words from cleaned tokens
def get_all_words(cleaned_tokens_list):
    for tokens in cleaned_tokens_list:
        for token in tokens:
            yield token

# Function to generate tokens for model
def get_tweets_for_model(cleaned_tokens_list):
    for tweet_tokens in cleaned_tokens_list:
        yield dict([token, True] for token in tweet_tokens)

# Function to extract emojis from text using Unicode ranges
def extract_emojis(text):
    emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"  # emoticons
                               u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                               u"\U0001F680-\U0001F6FF"  # transport & map symbols
                               u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                               u"\U00002500-\U00002BEF"  # chinese char
                               u"\U00002702-\U000027B0"
                               u"\U00002702-\U000027B0"
                               u"\U000024C2-\U0001F251"
                               u"\U0001f926-\U0001f937"
                               u"\U00010000-\U0010ffff"
                               u"\u2640-\u2642"
                               u"\u2600-\u2B55"
                               u"\u200d"
                               u"\u23cf"
                               u"\u23e9"
                               u"\u231a"
                               u"\ufe0f"  # dingbats
                               u"\u3030"
                               "]+", flags=re.UNICODE)
    return ''.join(emoji_pattern.findall(text))

# Function to map emojis to sentiment scores (you'd need to define this mapping)
def map_emoji_sentiment(emoji):
    # Example mapping
    if emoji in ['😊', '😄', '🙂']:
        return 'Positive'
    elif emoji in ['😞', '😔', '😕']:
        return 'Negative'
    else:
        return 'Neutral'

# Function to predict sentiment for a single sentence
def predict_sentiment_single(custom_sentence):
    custom_tokens = remove_noise(word_tokenize(custom_sentence))
    emojis = extract_emojis(custom_sentence)
    emoji_sentiments = [map_emoji_sentiment(emoji) for emoji in emojis]
    return classifier.classify(dict([token, True] for token in custom_tokens)), emoji_sentiments

# Function to predict sentiment for a paragraph
def predict_sentiment(custom_paragraph):
    sentiments = []
    emoji_sentiments = []
    for sentence in custom_paragraph.split("."):
        if sentence.strip():
            text_sentiment, emojis = predict_sentiment_single(sentence)
            sentiments.append(text_sentiment)
            emoji_sentiments.extend(emojis)

    # Calculate the percentage of positive and negative sentiments
    positive_percentage = (sentiments.count('Positive') / len(sentiments)) * 100
    negative_percentage = 100 - positive_percentage

    return positive_percentage, negative_percentage, emoji_sentiments

# Load positive and negative tweets
stop_words = stopwords.words('english')

# Tokenize positive and negative tweets
positive_tweet_tokens = ["I had pad thai and it was the best meal in a while. Loved the atmosphere. The service was excellent, and I would definitely recommend this place to my friends."]
negative_tweet_tokens = ["I am dissatisfied with your e-commerce product 😕😕😕😕."]

# Clean and preprocess the tokens
positive_cleaned_tokens_list = [remove_noise(word_tokenize(tokens), stop_words) for tokens in positive_tweet_tokens]
negative_cleaned_tokens_list = [remove_noise(word_tokenize(tokens), stop_words) for tokens in negative_tweet_tokens]

# Combine positive and negative cleaned tokens for the model
all_pos_words = get_all_words(positive_cleaned_tokens_list)
freq_dist_pos = FreqDist(all_pos_words)

positive_tokens_for_model = get_tweets_for_model(positive_cleaned_tokens_list)
negative_tokens_for_model = get_tweets_for_model(negative_cleaned_tokens_list)

# Prepare dataset for training the model
positive_dataset = [(tweet_dict, "Positive") for tweet_dict in positive_tokens_for_model]
negative_dataset = [(tweet_dict, "Negative") for tweet_dict in negative_tokens_for_model]

# Combine positive and negative datasets
dataset = positive_dataset + negative_dataset
random.shuffle(dataset)

# Split dataset into training and testing data
train_data = dataset[:1]  # Take one example for training (adjust as needed)
test_data = dataset[1:]   # Take the rest for testing

# Train the Naive Bayes classifier
classifier = NaiveBayesClassifier.train(train_data)

# Calculate accuracy on the test data
print("Accuracy is:", classify.accuracy(classifier, test_data))

# Show the most informative features
print(classifier.show_most_informative_features(10))

# Define a custom paragraph
custom_paragraph = """
I am dissatisfied with your e-commerce product 😕😕😕😕 .
"""

# Predict sentiment for the custom paragraph
positive_percentage, negative_percentage, emoji_sentiments = predict_sentiment(custom_paragraph)

# Plotting the pie chart for sentiment distribution
labels = ['Positive', 'Negative']
sizes = [positive_percentage, negative_percentage]
colors = ['#66b3ff', '#ff9999']  # Blue for Positive, Red for Negative
explode = (0.1, 0)

plt.figure(figsize=(8, 6))
plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
plt.axis('equal')
plt.title('Sentiment Distribution')
plt.show()

print("Emoji Sentiments:", emoji_sentiments)
