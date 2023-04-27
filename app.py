from flask import Flask, request, render_template
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import string

app = Flask(__name__)

# Load the dataset into a pandas DataFrame
df = pd.read_csv('product_review.csv')

# Preprocess the text data
nltk.download('stopwords')
nltk.download('wordnet')


def preprocess_text(text):
    # Convert text to lowercase
    text = text.lower()

    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))

    # Tokenize text
    tokens = nltk.word_tokenize(text)

    # Remove stop words
    tokens = [token for token in tokens if token not in stopwords.words('english')]

    # Perform lemmatization
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]

    # Join tokens back into a string
    text = ' '.join(tokens)

    return text


df['review'] = df['review'].apply(preprocess_text)

# Vectorize the preprocessed text data using TF-IDF
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['review'])
y = df['sentiment']

# Train a logistic regression model on the full dataset
clf = LogisticRegression()
clf.fit(X, y)


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Get review from request form
        review = request.form['review']

        # Preprocess the review
        review = preprocess_text(review)

        # Vectorize the preprocessed review using the same vectorizer
        review = vectorizer.transform([review])

        # Predict the sentiment of the review using the trained model
        sentiment = clf.predict(review)[0]

        # Predict the score of the review using the trained model
        score = clf.predict_proba(review)[0][1] * 5

        return render_template('index.html', sentiment=sentiment, score=score)
    else:
        return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)
