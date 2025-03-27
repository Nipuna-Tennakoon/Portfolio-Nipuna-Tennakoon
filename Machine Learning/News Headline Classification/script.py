import numpy as np
from tensorflow import keras
import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense

print('Wait')
#Import dataset
df = pd.read_csv('newsCorpora.csv', sep ='\t', names = ['ID','Title','URL', 'Publisher', 'Category', 'Story', 'Hostname','Timestamp'])
df=df[['Title','Category']]
dict = {'b':'Business', 't':'Science', 'e':'Entertainment', 'm':'Health'}

#Apply category names to category column
def update_category(x):
    return dict[x]
df['Category'] = df['Category'].apply(lambda x: update_category(x))

#Select small piece of a dataset
df = df.sample(frac=0.05,random_state=1)
df = df.reset_index(drop=True)

#Encoding the category
encode_dict = {}

def encode_category(x):
    if x not in encode_dict.keys():
        encode_dict[x]=len(encode_dict)
    return encode_dict[x]

df['Encode Category'] = df['Category'].apply(lambda x:encode_category(x))


X_train,X_test,Y_train,Y_test = train_test_split(df['Title'],df['Encode Category'],stratify=df['Encode Category'])

print(encode_dict)

# Define tokenizer (limiting vocab size if needed)
MAX_VOCAB_SIZE = 20000  # Adjust based on dataset size
tokenizer = Tokenizer(num_words=MAX_VOCAB_SIZE, oov_token="<OOV>")


# Fit tokenizer on the text data
tokenizer.fit_on_texts(X_train)

# Convert text to sequences
X_train_seq = tokenizer.texts_to_sequences(X_train)
X_test_seq = tokenizer.texts_to_sequences(X_test)

# Determine max sequence length
MAX_SEQ_LENGTH = max(len(seq) for seq in X_train_seq)  # or set a fixed value

# Pad sequences to ensure uniform input size
X_train_pad = pad_sequences(X_train_seq, maxlen=MAX_SEQ_LENGTH, padding='post', truncating='post')
X_test_pad = pad_sequences(X_test_seq, maxlen=MAX_SEQ_LENGTH, padding='post', truncating='post')


# One-hot encode labels
Y_train_cat = to_categorical(Y_train, num_classes=4)
Y_test_cat = to_categorical(Y_test, num_classes=4)

embedding_dim = 100  # Adjust based on complexity

model = Sequential([
    Embedding(input_dim=MAX_VOCAB_SIZE, output_dim=embedding_dim, input_length=MAX_SEQ_LENGTH),
    LSTM(64),
    Dense(4, activation='softmax')  # 4 output classes
])

model.compile(
    loss='categorical_crossentropy',  # Entropy-based loss function
    optimizer='adam',
    metrics=['accuracy']
)
# Train the model
history = model.fit(
    X_train_pad, Y_train_cat,  # Tokenized and padded input & one-hot labels
    validation_data=(X_test_pad, Y_test_cat),  # Validation set
    epochs=2,  # Number of iterations (adjustable)
    batch_size=32,  # Number of samples per batch
    verbose=1  # Display training progress
)
loss, accuracy = model.evaluate(X_test_pad, Y_test_cat)
print(f"Test Accuracy: {accuracy:.4f}")

model.save("Text_Classification.h5", save_format="h5")

import pickle

# Save tokenizer
with open("tokenizer.pkl", "wb") as f:
    pickle.dump(tokenizer, f)
