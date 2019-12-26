import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))
tfidf = pickle.load(open('tfidf.pkl', 'rb'))
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    if request.method == 'POST':
        message = request.form['message']
        d = [message]
        vect = tfidf.transform(d).toarray()
        my_prediction = model.predict(vect)
        if (my_prediction == [0]):
            predicted = 'This is NOT a Hatred Tweet'
        else:
            predicted = 'This is a Hatred Tweet'
    return render_template('result.html', prediction_text = predicted)

if __name__ == "__main__":
    app.run(debug=True)