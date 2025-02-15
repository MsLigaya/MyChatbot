import json
import random
import pickle
import numpy as np
import nltk
import tensorflow as tf

from flask import Flask, render_template, request
from nltk.stem import WordNetLemmatizer

# Load trained model
model = tf.keras.models.load_model("chatbot_model/chatbot_model.h5")

# Load tokenizer data
words = pickle.load(open("chatbot_model/words.pkl", "rb"))
classes = pickle.load(open("chatbot_model/classes.pkl", "rb"))

lemmatizer = WordNetLemmatizer()

app = Flask(__name__)

def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

def bag_of_words(sentence):
    sentence_words = clean_up_sentence(sentence)
    bag = [1 if w in sentence_words else 0 for w in words]
    return np.array(bag)

def predict_class(sentence):
    bow = bag_of_words(sentence)
    res = model.predict(np.array([bow]))[0]
    return classes[np.argmax(res)]

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/get", methods=["GET", "POST"])
def chatbot_response():
    msg = request.form["msg"]
    intent = predict_class(msg)
    response = random.choice([i["responses"] for i in json.load(open("intents.json"))["intents"] if i["tag"] == intent][0])
    return response

if __name__ == "__main__":
    app.run(debug=True)
