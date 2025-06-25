import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv('project_2/IMDB Dataset.csv')
print('Data loaded:', df.shape)
print(df.head())

# Encode labels
df['sentiment'] = df['sentiment'].map({'positive': 1, 'negative': 0})

# Split data
X_train, X_test, y_train, y_test = train_test_split(df['review'], df['sentiment'], test_size=0.2, random_state=42)

# Tokenization and padding
max_words = 10000
max_len = 200
tokenizer = Tokenizer(num_words=max_words, oov_token='<OOV>')
tokenizer.fit_on_texts(X_train)

X_train_seq = tokenizer.texts_to_sequences(X_train)
X_test_seq = tokenizer.texts_to_sequences(X_test)

X_train_pad = pad_sequences(X_train_seq, maxlen=max_len, padding='post', truncating='post')
X_test_pad = pad_sequences(X_test_seq, maxlen=max_len, padding='post', truncating='post')

# Model definition
model = Sequential([
    Embedding(max_words, 64, input_length=max_len),
    Bidirectional(LSTM(64, return_sequences=True)),
    Dropout(0.5),
    Bidirectional(LSTM(32)),
    Dropout(0.5),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

# Training
history = model.fit(
    X_train_pad, y_train,
    epochs=3,
    batch_size=128,
    validation_split=0.2
)

# Evaluation
loss, acc = model.evaluate(X_test_pad, y_test)
print(f'\nTest Loss: {loss:.4f}, Test Accuracy: {acc:.4f}')

# Visualizations
plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.title('Accuracy')
plt.legend()
plt.subplot(1,2,2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Loss')
plt.legend()
plt.tight_layout()
plt.show()

# Sample predictions
sample_texts = X_test.sample(5, random_state=42)
sample_seqs = tokenizer.texts_to_sequences(sample_texts)
sample_pad = pad_sequences(sample_seqs, maxlen=max_len, padding='post', truncating='post')
preds = model.predict(sample_pad)
for text, pred in zip(sample_texts, preds):
    print(f'\nReview: {text[:100]}...')
    print(f'Predicted Sentiment: {"positive" if pred[0] > 0.5 else "negative"} (score: {pred[0]:.2f})') 