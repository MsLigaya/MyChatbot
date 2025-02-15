import json
import random
import numpy as np
import tensorflow as tf
import nltk
import pickle

from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

# Initialize lemmatizer
lemmatizer = WordNetLemmatizer()

# Load intents dataset
with open("intents.json") as file:
    data = json.load(file)

words = []
classes = []
documents = []

# Preprocess data
for intent in data["intents"]:
    for pattern in intent["patterns"]:
        word_list = nltk.word_tokenize(pattern)
        words.extend(word_list)
        documents.append((word_list, intent["tag"]))
    
    if intent["tag"] not in classes:
        classes.append(intent["tag"])

words = sorted(set([lemmatizer.lemmatize(w.lower()) for w in words]))
classes = sorted(set(classes))

# Save words and classes
pickle.dump(words, open("chatbot_model/words.pkl", "wb"))
pickle.dump(classes, open("chatbot_model/classes.pkl", "wb"))

training = []
output_empty = [0] * len(classes)

for doc in documents:
    bag = []
    word_patterns = [lemmatizer.lemmatize(w.lower()) for w in doc[0]]

    for w in words:
        bag.append(1 if w in word_patterns else 0)

    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1

    training.append([bag, output_row])

random.shuffle(training)
X_train, y_train = np.array([i[0] for i in training]), np.array([i[1] for i in training])

# Build the model
model = Sequential([
    Dense(128, activation="relu", input_shape=(len(X_train[0]),)),
    Dropout(0.5),
    Dense(64, activation="relu"),
    Dropout(0.5),
    Dense(len(y_train[0]), activation="softmax")
])

model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

# Train and save the model
model.fit(X_train, y_train, epochs=200, batch_size=5, verbose=1)
model.save("chatbot_model/chatbot_model.h5")

print("Training complete. Model saved.")

total_samples = 0
for intent in data["intents"]:
    total_samples += len(intent["patterns"])

print(f"Total number of samples in the dataset: {total_samples}")
