from flask import Flask, render_template, request
import pickle
import requests
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer

app = Flask(__name__)

# Load the pickled model and vectorizer
with open('best_classifier.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('vectorizer.pkl', 'rb') as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

# Custom Tokenizer class for web scraping
class MyTokenizer:
    def transform(self, X):
        return [' '.join(doc) if isinstance(doc, list) else doc for doc in X]

# Function to get text content from a URL
def get_text_from_url(url):
    try:
        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')
        paragraphs = soup.find_all('p')
        text_content = ' '.join([p.get_text() for p in paragraphs])
        return text_content
    except Exception as e:
        return str(e)

# Predict category based on the URL
def predict_category(url):
    content = get_text_from_url(url)
    content_list = MyTokenizer().transform([content])
    content_tfidf = vectorizer.transform(content_list)
    prediction = model.predict(content_tfidf)
    return prediction[0]

# Flask route to handle the input form
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        url = request.form['url']
        category = predict_category(url)
        return render_template('index.html', url=url, category=category)
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
