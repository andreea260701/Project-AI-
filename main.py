import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Flatten, Dense
import numpy as np

# Citirea și prelucrarea datelor
file_path = "SMSSpamCollection"
df = pd.read_csv(file_path, sep='\t', header=None, names=['label', 'message'])

# Maparea etichetelor la valori binare
df['label'] = df['label'].map({'ham': 0, 'spam': 1})

# Împărțirea datelor în seturi de antrenament și testare
X_train, X_test, y_train, y_test = train_test_split(df['message'], df['label'], test_size=0.2, random_state=42)

# Tokenizarea textelor
max_words = 10000
tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(X_train)
X_train_seq = tokenizer.texts_to_sequences(X_train)
X_test_seq = tokenizer.texts_to_sequences(X_test)

# Padding sequences
max_length = 256
X_train_pad = pad_sequences(X_train_seq, maxlen=max_length)
X_test_pad = pad_sequences(X_test_seq, maxlen=max_length)

print(f"Padded training sample: {X_train_pad[0]}")

# One-hot encoding
def one_hot_encode(sequences, dimension=10000):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1.0
    return results

x_train = one_hot_encode(X_train_pad)
x_test = one_hot_encode(X_test_pad)

print(f"One-hot encoded training sample: {x_train[0]}")

# Construirea modelului
model = Sequential()
model.add(Embedding(input_dim=max_words, output_dim=32, input_length=max_length))
model.add(Flatten())
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
print(model.summary())

# Antrenarea modelului
history = model.fit(X_train_pad, y_train, epochs=10, batch_size=32, validation_split=0.2)

# Plotarea acurateței și pierderilor pe epoci
plt.figure()
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

plt.figure()
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Evaluarea modelului
train_loss, train_acc = model.evaluate(X_train_pad, y_train)
print(f'Train Accuracy: {train_acc:.4f}')

test_loss, test_acc = model.evaluate(X_test_pad, y_test)
print(f'Test Accuracy: {test_acc:.4f}')

# Predictii si matricea de confuzie
predictions = model.predict(X_test_pad)
binary_predictions = np.where(predictions > 0.5, 1, 0)
conf_matrix = confusion_matrix(y_test, binary_predictions)
print("Confusion Matrix:")
print(conf_matrix)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Predicted Ham', 'Predicted Spam'],
            yticklabels=['Actual Ham', 'Actual Spam'])
plt.xlabel('Prediction')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# Calcularea si afisarea AUC-ROC
fpr, tpr, _ = roc_curve(y_test, predictions)
auc_roc = roc_auc_score(y_test, predictions)
print(f"AUC ROC Score: {auc_roc:.2f}")

