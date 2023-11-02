import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import pandas as pd

# Load the data and preprocess it
data = pd.read_csv(r'Flask Server\Model Train\English YouTube Hate Speech Corpus\IMSyPP_EN_YouTube_comments_train.csv')
data = data[['Text', 'Type']]
data = data[data['Type'] != '0']
data['Text'] = data['Text'].str.lower()
data = data.dropna()

X_train_initial = data['Text']

# Define stopwords and lemmatizer
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# Custom tokenizer function for stop words removal and lemmatization
def custom_tokenizer(text):
    words = nltk.word_tokenize(text)
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    return words

# Create and fit the TF-IDF vectorizer
vectorizer = TfidfVectorizer(tokenizer=custom_tokenizer)
X_train_tfidf = vectorizer.fit_transform(X_train_initial)

# Save the vectorizer and custom tokenizer to a pickle file
with open('vectorizer.pkl', 'wb') as file:
    pickle.dump((vectorizer), file)