
# Deep Learning Applications for Image Classification

This repository contains code and resources for applying deep learning techniques to image classification tasks. The project demonstrates how to build and train deep learning models for classifying images into different categories.

## Repository Structure

```
Deep-Learning-Applications-for-Image-Classification/
│
├── .gitattributes
├── Assignment 3.xlsx
├── Code File.ipynb
├── Problem Statement.pdf
├── Project Report.docx
└── Project Report.pdf
```

- `.gitattributes`: Configuration file to ensure consistent handling of files across different operating systems.
- `Assignment 3.xlsx`: Excel file containing data or results related to the project.
- `Code File.ipynb`: Jupyter Notebook containing the code for building, training, and evaluating the image classification models.
- `Problem Statement.pdf`: PDF file detailing the problem statement for the project.
- `Project Report.docx`: Word document containing the project report.
- `Project Report.pdf`: PDF version of the project report.

## Getting Started

To get started with this project, follow the steps below:

### Prerequisites

Make sure you have the following installed:

- Python 3.x
- Jupyter Notebook
- Required Python libraries (listed in `requirements.txt`)

### Installation

1. Clone this repository to your local machine:

```bash
git clone https://github.com/Harshraj1301/Deep-Learning-Applications-for-Image-Classification.git
```

2. Navigate to the project directory:

```bash
cd Deep-Learning-Applications-for-Image-Classification
```

3. Install the required Python libraries:

```bash
pip install -r requirements.txt
```

### Usage

1. Open the Jupyter Notebook:

```bash
jupyter notebook "Code File.ipynb"
```

2. Follow the instructions in the notebook to run the code cells and perform image classification using deep learning models.

### Code Explanation

The notebook `Code File.ipynb` includes the following steps:

1. **Data Preprocessing**: Loading and preprocessing the image data to make it suitable for training the deep learning models.
2. **Model Building**: Constructing deep learning models using Keras and TensorFlow.
3. **Model Training**: Training the models on the preprocessed image data.
4. **Model Evaluation**: Evaluating the performance of the trained models on test data.
5. **Image Classification**: Using the trained models to classify new images.

Here are the contents of the notebook:

# Part 1 - Step 1

# Part 1 - Step 2

# Part 1 - Step 3

# Part 1 - Step 4

# Part 1 - Step 5

# Part 2 - Step 1

# Part 2 - Step 2

# Part 2 - Step 3

# Part 2 - Step 4

# Part 2 - Step 5

## Code Cells

```python
import pandas as pd
```

```python
data = pd.read_excel('Assignment 3.xlsx')

# Building the training and test datasets as specified

# Filtering the first 400 restaurant reviews and the first 400 movie reviews
train_restaurant = data[(data['label'] == 'restaurant') & (data['id'] <= 400)]
train_movie = data[(data['label'] == 'movie') & (data['id'] >= 501) & (data['id'] <= 900)]

# Combining the two datasets to form the training dataset
train_dataset = pd.concat([train_restaurant, train_movie])

# The rest of the data will be used as the test dataset
test_dataset = data.drop(train_dataset.index)

# Checking the first few rows of each dataset to ensure they are correct
train_dataset.head(), test_dataset.head()
```

```python
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import string

# Define a custom tokenizer function
def custom_tokenizer(text):
    # Initialize WordNetLemmatizer
    lemmatizer = WordNetLemmatizer()

    # Split the text into words
    words = text.split()

    # Lemmatize, remove stop words and punctuations
    tokens = [lemmatizer.lemmatize(word) for word in words 
              if word.lower() not in stopwords.words('english') 
              and word not in string.punctuation]

    return tokens

# Initialize TfidfVectorizer
tfidf_vectorizer = TfidfVectorizer(tokenizer=custom_tokenizer, min_df=5, ngram_range=(1, 2))

# Apply TF-IDF transformation to the training dataset
tfidf_matrix = tfidf_vectorizer.fit_transform(train_dataset['review'])

# Example: To view the shape of the TF-IDF matrix and feature names
print(tfidf_matrix.shape)
print(tfidf_vectorizer.get_feature_names_out()[:10])

```

```python
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(tfidf_matrix, train_dataset['label'], test_size=0.3, random_state=42)

# Initialize models
naive_bayes_model = MultinomialNB()
logit_model = LogisticRegression()
random_forest_model = RandomForestClassifier(n_estimators=50)
svm_model = SVC()
ann_model = MLPClassifier(hidden_layer_sizes=(4,), max_iter=1000)

# Dictionary to store models
models = {
    "Naive Bayes": naive_bayes_model,
    "Logistic Regression": logit_model,
    "Random Forest": random_forest_model,
    "SVM": svm_model,
    "ANN": ann_model
}

# Train each model and calculate accuracy
for name, model in models.items():
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    print(f"{name} Accuracy: {accuracy:.2f}")

# Note: This code assumes that the tfidf_matrix and train_dataset['label'] are already created and available.

```

```python
#pip install keras
```

```python
#pip install --upgrade keras tensorflow
```

```python
#pip install tensorflow
```

```python
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

# Number of words to consider as features
max_features = 10000  # This is an arbitrary number, you can adjust it based on your vocabulary size

# Maximum length of each document
maxlen = 100

# Initialize the tokenizer with a maximum number of words
tokenizer = Tokenizer(num_words=max_features)

# Fit the tokenizer on the training data
tokenizer.fit_on_texts(train_dataset['review'])

# Convert the texts to sequences
train_sequences = tokenizer.texts_to_sequences(train_dataset['review'])
test_sequences = tokenizer.texts_to_sequences(test_dataset['review'])

# Pad the sequences so they all have the same length
X_train = pad_sequences(train_sequences, maxlen=maxlen)
X_test = pad_sequences(test_sequences, maxlen=maxlen)

# Example: To view the shape of the sequences
print(X_train.shape)
print(X_test.shape)
```

```python
from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dropout, Dense
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

# Assuming max_features and maxlen are defined as before
max_features = 10000
maxlen = 100

# Initialize and fit the tokenizer
tokenizer = Tokenizer(num_words=max_features)
tokenizer.fit_on_texts(train_dataset['review'])

# Convert texts to sequences
train_sequences = tokenizer.texts_to_sequences(train_dataset['review'])
test_sequences = tokenizer.texts_to_sequences(test_dataset['review'])

# Pad sequences
X_train = pad_sequences(train_sequences, maxlen=maxlen)
X_test = pad_sequences(test_sequences, maxlen=maxlen)

# Convert labels to numeric
y_train = np.array(train_dataset['label'].apply(lambda x: 1 if x == 'movie' else 0))
y_test = np.array(test_dataset['label'].apply(lambda x: 1 if x == 'movie' else 0))

# Define the model
model = Sequential()
model.add(Embedding(max_features, 20, input_length=maxlen))
model.add(LSTM(40, dropout=0.2, recurrent_dropout=0.2))
model.add(Dropout(0.1))
model.add(Dense(1, activation='sigmoid'))

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, batch_size=100, epochs=10, validation_data=(X_test, y_test))

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print(f'Accuracy: {accuracy:.2f}')

```

```python
from keras.datasets import cifar10
import matplotlib.pyplot as plt

# Load CIFAR-10 dataset
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Define the CIFAR-10 classes
classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
           'dog', 'frog', 'horse', 'ship', 'truck']

# Visualize the first 20 images of the test set
plt.figure(figsize=(10, 4))
for i in range(20):
    plt.subplot(2, 10, i+1)
    plt.imshow(x_test[i])
    plt.title(classes[y_test[i][0]])
    plt.axis('off')
plt.tight_layout()
plt.show()
```

```python
pip install tensorflow

```

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dropout, MaxPooling2D, Flatten, Dense
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.datasets import cifar10

# Load CIFAR-10 dataset
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Normalize the data
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

# Convert class vectors to binary class matrices
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# Define the CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=x_train.shape[1:]),  # a
    Dropout(0.2),                                                        # b
    Conv2D(32, (3, 3), activation='relu'),                               # c
    MaxPooling2D(pool_size=(2, 2)),                                      # d
    Conv2D(64, (3, 3), activation='relu'),                               # e
    Dropout(0.2),                                                        # f
    Conv2D(64, (3, 3), activation='relu'),                               # g
    MaxPooling2D(pool_size=(2, 2)),                                      # h
    Flatten(),                                                           # i
    Dense(256, activation='relu'),                                       # j
    Dropout(0.2),                                                        # k
    Dense(10, activation='softmax')                                      # l
])

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
batch_size = 500  # Smaller than 500 to prevent overheating
epochs = 5
model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(x_test, y_test), verbose=1)

# Evaluate the model
accuracy = model.evaluate(x_test, y_test, verbose=0)[1]
print(f'Accuracy: {accuracy * 100:.2f}%')
```

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dropout, MaxPooling2D, Flatten, Dense
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.datasets import cifar10

# Load CIFAR-10 dataset
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Normalize the data
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

# Convert class vectors to binary class matrices
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# Define the modified CNN model
model_modified = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=x_train.shape[1:]),  # a
    Dropout(0.2),                                                          # b
    Conv2D(32, (3, 3), activation='relu'),                                 # c
    MaxPooling2D(pool_size=(2, 2)),                                        # d
    Conv2D(64, (3, 3), activation='relu'),                                 # e
    Dropout(0.2),                                                          # f
    Conv2D(64, (3, 3), activation='relu'),                                 # g
    MaxPooling2D(pool_size=(2, 2)),                                        # h
    Conv2D(128, (3, 3), activation='relu'),                                # step 3 a
    Dropout(0.2),                                                          # step 3 b
    Conv2D(128, (3, 3), activation='relu'),                                # step 3 c
    Flatten(),                                                             # i
    Dense(256, activation='relu'),                                         # j
    Dropout(0.2),                                                          # k
    Dense(10, activation='softmax')                                        # l
])

# Compile the modified model
model_modified.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the modified model
batch_size = 500  # Smaller than 500 to prevent overheating
epochs = 5
model_modified.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(x_test, y_test), verbose=1)

# Evaluate the modified model
accuracy_modified = model_modified.evaluate(x_test, y_test, verbose=0)[1]
print(f'Accuracy: {accuracy_modified * 100:.2f}%')

```

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dropout, MaxPooling2D, Flatten, Dense
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.datasets import cifar10

# Load and preprocess the CIFAR-10 dataset
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# Define the modified CNN model (from step 3)
model_modified = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=x_train.shape[1:]),  # a
    Dropout(0.2),                                                          # b
    Conv2D(32, (3, 3), activation='relu'),                                 # c
    MaxPooling2D(pool_size=(2, 2)),                                        # d
    Conv2D(64, (3, 3), activation='relu'),                                 # e
    Dropout(0.2),                                                          # f
    Conv2D(64, (3, 3), activation='relu'),                                 # g
    MaxPooling2D(pool_size=(2, 2)),                                        # h
    Conv2D(128, (3, 3), activation='relu'),                                # step 3 a
    Dropout(0.2),                                                          # step 3 b
    Conv2D(128, (3, 3), activation='relu'),                                # step 3 c
    Flatten(),                                                             # i
    Dense(256, activation='relu'),                                         # j
    Dropout(0.2),                                                          # k
    Dense(10, activation='softmax')                                        # l
])

# Compile the model
model_modified.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model for 20 epochs
batch_size = 500  # Smaller than 500 to prevent overheating
epochs_extended = 20
model_modified.fit(x_train, y_train, batch_size=batch_size, epochs=epochs_extended, validation_data=(x_test, y_test), verbose=1)

# Evaluate the model
accuracy_extended = model_modified.evaluate(x_test, y_test, verbose=0)[1]
print(f'Accuracy after 20 epochs: {accuracy_extended * 100:.2f}%')

```

```python
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier

# Flatten the image data
x_train_flat = x_train.reshape(x_train.shape[0], -1)
x_test_flat = x_test.reshape(x_test.shape[0], -1)

# Reshape the labels for Naïve Bayes
y_train_flat = y_train.argmax(axis=1)
y_test_flat = y_test.argmax(axis=1)

# Train a Naïve Bayes model
naive_bayes_model = MultinomialNB()
naive_bayes_model.fit(x_train_flat, y_train_flat)

# Train a Random Forest model with 100 trees and max depth of 10
random_forest_model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
random_forest_model.fit(x_train_flat, y_train_flat)

# Predictions using Naïve Bayes
y_pred_nb = naive_bayes_model.predict(x_test_flat)
accuracy_nb = accuracy_score(y_test_flat, y_pred_nb)

# Predictions using Random Forest
y_pred_rf = random_forest_model.predict(x_test_flat)
accuracy_rf = accuracy_score(y_test_flat, y_pred_rf)

accuracy_nb, accuracy_rf

```

## Results

The notebook includes the results of the image classification tasks, showcasing the performance of different deep learning models on the given dataset.

## Contributing

If you'd like to contribute to this project, please follow these steps:

1. Fork the repository.
2. Create a new branch: `git checkout -b feature-branch-name`
3. Make your changes and commit them: `git commit -m 'Add some feature'`
4. Push to the branch: `git push origin feature-branch-name`
5. Submit a pull request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgements

- This project was created as part of an assignment by Harshraj Jadeja.
- Thanks to the open-source community for providing valuable resources and libraries for deep learning.

---

Feel free to modify this `README.md` file as per your specific requirements and project details.
