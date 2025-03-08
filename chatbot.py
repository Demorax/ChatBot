import random
import json
import pickle
from typing import Any

import numpy as np
import nltk

from nltk.stem import WordNetLemmatizer
from keras.models import load_model
from numpy import ndarray, dtype

lemmatizer = WordNetLemmatizer()
model = load_model('chatbot_model.keras')

json_file = json.loads(open('intents_larger.json').read())
words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))

# Function to process user input
def preprocess_input(sentence) -> list[str]:
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word) for word in sentence_words]
    return sentence_words

# Function to convert sentence into a bag of words
def bag_of_words(sentence) -> ndarray[Any, dtype[str]]:
    sentence_words = preprocess_input(sentence)
    bag = [1 if word in sentence_words else 0 for word in words]
    return np.array(bag)

# Function to predict the class of user input
def predict_class(sentence) -> list[dict[str, int | Any]]:
    bow = bag_of_words(sentence)
    res = model.predict(np.array([bow]))[0]

    ERROR_THRESHOLD = 0.25
    result = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    result.sort(key=lambda x: x[1], reverse=True)
    return_result = []
    for r in result:
        return_result.append({'intent': classes[r[0]], 'probability': r[1]})

    return return_result

# Function to get a response from the intents file
def get_response(sentence: str, file) -> str:
    intents_list = file['intents']
    tag = sentence[0]["intent"]
    for i in intents_list:
        if i["tag"] == tag:
            result = random.choice(i["responses"])
            break

    return result

# Chat loop
def chatbot():
    print("Bot is ready (Type 'quit' to exit)")
    while True:
        user_input = input("Enter a sentence: ")
        if user_input == 'quit':
            break
        intent = predict_class(user_input)
        response = get_response(intent, json_file)
        print(response)


if __name__ == '__main__':
    chatbot()