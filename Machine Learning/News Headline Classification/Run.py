import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow import keras
import pickle

with open('tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)

while True:
    print("CLASSIFIER STARTED")
    sentence = input("Enter your sentence:")
    model = keras.models.load_model("Text_Classification.h5", compile=True)

    if len(sentence.split())==1: 
        print('You are entered wrong word') 
        break
    # Function to preprocess and predict
    def predict_class(sentence, tokenizer, model, max_length):
        # Tokenize the input sentence
        sequence = tokenizer.texts_to_sequences([sentence])
    
        # Pad the sequence to match training input length
        padded_sequence = pad_sequences(sequence, maxlen=max_length, padding='post', truncating='post')
    
        # Get prediction probabilities
        prediction = model.predict(padded_sequence)
    
        # Get class with highest probability
        predicted_class = np.argmax(prediction)
    
        return predicted_class


    encode_dict = {'Business': 0, 'Entertainment': 1, 'Science': 2, 'Health': 3}


    key_list = list(encode_dict.keys())
    value_list = list(encode_dict.values())

    output = predict_class(sentence,tokenizer,model,19)

    if output in value_list:
        key = key_list[value_list.index(output)]


    print(key)

    