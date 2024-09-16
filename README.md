# SMS Spam Classification using Neural Networks

This project demonstrates the use of a Neural Network to classify SMS messages as either **spam** or **ham** (not spam). The model is trained using the `SMSSpamCollection` dataset and utilizes TF-IDF for text vectorization.

## Project Structure

- **Data**: The dataset is a collection of SMS messages labeled as either `ham` or `spam`. It is loaded from a tab-separated file named `SMSSpamCollection`.
- **Model**: A Sequential Neural Network model is used, implemented with Keras.
- **Text Vectorization**: TF-IDF (Term Frequency-Inverse Document Frequency) is applied to convert the text messages into numerical features.
- **Evaluation**: The model's performance is measured using the ROC-AUC score and visualized through an ROC curve.

## Files

- **SMSSpamCollection**: The dataset used for training and testing the model. It consists of two columns: `label` (either 'spam' or 'ham') and `message` (the SMS content).
- **`spam_classification.py`**: The main script that loads the data, builds the neural network, and evaluates its performance.
