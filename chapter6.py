# ------------------------------------------------------------
# Listing 6-1. Install, import, and prepare NLTK.
# ------------------------------------------------------------

# Install nltk.
!pip install nltk

# Import nltk.
import nltk

# Download all datasets and models
nltk.download('all')


# ------------------------------------------------------------
# Listing 6-2. Download HTML and extract text.
# ------------------------------------------------------------

from urllib.request import urlopen
from bs4 import BeautifulSoup

# Define url string.
url = 'https://www.sec.gov/Archives/edgar/data/1286973/000156459020025868/d934487dex991.htm'

# Send GET request.
html = urlopen(url)

# Parse HTML tree.
soup = BeautifulSoup(html.read())

# Identify all paragraphs.
paragraphs = soup.findAll('p')

# Create list of text attribute of paragraphs.
paragraphs = [p.text for p in paragraphs]


# ------------------------------------------------------------
# Listing 6-3. Join paragraphs into single string.
# ------------------------------------------------------------

# Join paragraphs into single string.
corpus = " ".join(paragraphs)

# Print contents.
print(corpus)


# ------------------------------------------------------------
# Listing 6-4. Tokenize text into sentences using NLTK.
# ------------------------------------------------------------

import nltk

# Instantiate sentence tokenizer.
sentTokenizer = nltk.sent_tokenize

# Identify sentences.
sentences = sentTokenizer(corpus)

# Print the number of sentences.
print(len(sentences))

# Print a sentence.
print(sentences[7])


# ------------------------------------------------------------
# Listing 6-5. Convert characters to lower case and 
# remove stop words.
# ------------------------------------------------------------

from nltk.corpus import stopwords

# Convert all characters to lowercase.
sentences = [s.lower() for s in sentences]

# Define stop words as a set.
stops = set(stopwords.words('english'))

# Instantiate word tokenizer.
wordTokenizer = nltk.word_tokenize

# Divide corpus into list of lists.
words = [wordTokenizer(s) for s in sentences]

# Remove stop words.
for j in range(len(words)):
	words[j] = [w for w in words[j] if 
w not in stops]

# Print first five words in first sentence.
print(words[0][:5])


# ------------------------------------------------------------
# Listing 6-6. Replacing words with their stems.
# ------------------------------------------------------------

from nltk.stem.porter import PorterStemmer

# Instantiate Porter stemmer.
stemmer = PorterStemmer()

# Apply Porter stemmer.
for j in range(len(words)):
	words[j] = [stemmer.stem(w) for w in words[j]]

# Print first five words in first sentence.
print(words[0][:5])


# ------------------------------------------------------------
# Listing 6-7. Remove special characters and join words into
# sentences.
# ------------------------------------------------------------

import re

# Remove special characters, punctuation, and numbers.
for j in range(len(words)):
	words[j] = [re.sub('[^a-z]+', '', w)
for w in words[j]]

# Rejoin words into sentences.
for j in range(len(words)):
	words[j] = " ".join(words[j]).strip()

# Print sentence.
print(words[7])


# ------------------------------------------------------------
# Listing 6-8. Construct the document-term matrix.
# ------------------------------------------------------------

from sklearn.feature_extraction import text

# Instantiate vectorizer.
vectorizer = text.CountVectorizer(max_features = 10)

# Construct C matrix.
C = vectorizer.fit_transform(words)

# Print document-term matrix.
print(C.toarray())

# Print feature names.
print(vectorizer.get_feature_names())


# ------------------------------------------------------------
# Listing 6-9. Adjust the parameters of CountVectorizer().
# ------------------------------------------------------------

# Instantiate vectorizer.
vectorizer = text.CountVectorizer(
	max_features = 1000,
	max_df = 0.50,
	min_df = 0.05
)

# Construct C matrix.
C = vectorizer.fit_transform(words)

# Print shape of C matrix.
print(C.toarray().shape)

# Print terms.
print(vectorizer.get_feature_names()[:10])


# ------------------------------------------------------------
# Listing 6-10. Compute inverse document frequencies for all columns.
# ------------------------------------------------------------

# Instantiate vectorizer.
vectorizer = text.TfidfVectorizer(max_features = 10)

# Construct C matrix.
C = vectorizer.fit_transform(words)

# Print inverse document frequencies.
print(vectorizer.idf_)


# ------------------------------------------------------------
# Listing 6-11. Compute the document-term matrix for 
# unigrams and bigrams.
# ------------------------------------------------------------

# Instantiate vectorizer.
vectorizer = text.TfidfVectorizer(
max_features = 10,
ngram_range = (2,2)
)

# Construct C matrix.
C = vectorizer.fit_transform(words)

# Print feature names.
print(vectorizer.get_feature_names())


# ------------------------------------------------------------
# Listing 6-12. Compute the Loughran-McDonald measure of 
# positive sentiment.
# ------------------------------------------------------------

import pandas as pd

# Define data directory path.
data_path = '../data/chapter6/'

# Load the Loughran-McDonald dictionary.
lm = pd.read_excel(data_path+'LM2018.xlsx', 
sheet_name = 'Positive',
header = None)

# Convert series to DataFrame.
lm = pd.DataFrame(lm.values, columns = ['Positive'])

# Convert to lower case.
lm = lm['Positive'].apply(lambda x: x.lower())

# Convert DataFrame to list.
lm = lm.tolist()

# Print list of positive words.
print(lm)


# ------------------------------------------------------------
# Listing 6-13. Stem the LM dictionary.
# ------------------------------------------------------------

from nltk.stem.porter import PorterStemmer

# Instantiate Porter stemmer.
stemmer = PorterStemmer()

# Apply Porter stemmer.
slm = [stemmer.stem(word) for word in lm]

# Print length of list.
print(len(slm))

# Drop duplicates by converting to set.
slm = list(set(slm))

# Print length of list.
print(len(slm))


# ------------------------------------------------------------
# Listing 6-14. Count positive words.
# ------------------------------------------------------------

# Define empty array to hold counts.
counts = []

# Iterate through all sentences.
for w in words:
	# Set initial count to 0.
	count = 0
# Iterate over all dictionary words.
for i in slm:
	count += w.count(i)
	# Append counts.
	counts.append(count)


# ------------------------------------------------------------
# Listing 6-15. Perform LDA on 6-K filing text data.
# ------------------------------------------------------------

from sklearn.decomposition import LatentDirichletAllocation

# Set number of topics.
k = 5

# Instantiate LDA model.
lda = LatentDirichletAllocation(n_components = k)

# Recover feature names from vectorizer.
feature_names = vectorizer.get_feature_names()

# Train model on document-term matrix.
lda.fit(C)

# Recover word distribution for each topic.
wordDist = lda.components_

# Define empty topic list.
topics = []

# Recover topics.
for i in range(k):
	topics.append([feature_names[name] for 
name in wordDist[i].argsort()[-5:][::-1]])

# Print list of topics.
print(topics)


# ------------------------------------------------------------
# Listing 6-16. Assign topic probabilities to sentences.
# ------------------------------------------------------------

# Transform C matrix into topic probabilities.
topic_probs = lda.transform(C)

# Print topic probabilities.
print(topic_probs)


# ------------------------------------------------------------
# Listing 6-17. Preparing the data and model for a LAD regression 
# in TensorFlow.
# ------------------------------------------------------------

import tensorflow as tf
import numpy as np

# Draw initial values randomly.
alpha = tf.random.normal([1], stddev=1.0)
beta = tf.random.normal([25,1], stddev=1.0)

# Define variables.
alpha = tf.Variable(alpha, tf.float32)
beta = tf.Variable(beta, tf.float32)

# Convert data to numpy arrays.
x_train = np.array(x_train, np.float32)
y_train = np.array(y_train, np.float32)

# Define LAD model.
def LAD(alpha, beta, x_train):
	prediction = alpha + tf.matmul(x_train, beta)
	return prediction


# ------------------------------------------------------------
# Listing 6-18. Define an MAE loss function and perform optimization.
# ------------------------------------------------------------

# Define number of observations.
N = len(x_train)

# Define function to compute MAE loss.
def maeLoss(alpha, beta, x_train, y_train):
	y_hat = LAD(alpha, beta, x_train)
	y_hat = tf.reshape(y_hat, (N,))
	return tf.losses.mae(y_train, y_hat)

# Instantiate optimizer.
opt = tf.optimizers.Adam()

# Perform optimization.
for i in range(1000):
	opt.minimize(lambda: maeLoss(alpha, beta,
	x_train, y_train), 
	var_list = [alpha, beta])


# ------------------------------------------------------------
# Listing 6-19. Generate predicted values from model.
# ------------------------------------------------------------

# Generate predicted values.
y_pred = LAD(alpha, beta, x_train)


# ------------------------------------------------------------
# Listing 6-20. Generate predicted values from model.
# ------------------------------------------------------------

# Get feature names from vectorizer.
feature_names = vectorizer.get_feature_names()

# Print feature names.
print(feature_names)


# ------------------------------------------------------------
# Listing 6-21. Convert a LAD regression into a LASSO regression.
# ------------------------------------------------------------

# Re-define coefficient vector.
beta = tf.random.normal([25,1], stddev=1.0)
beta = tf.Variable(beta, tf.float32)

# Set value of lambda parameter.
lam = tf.constant(0.10, tf.float32)

# Modify the loss function.
def lassoLoss(alpha, beta, x_train, y_train, 
lam = lam):
	y_hat = LAD(alpha, beta, x_train)
	y_hat = tf.reshape(y_hat, (N,))
	loss = tf.losses.mae(y_train, y_hat) + \
	lam * tf.norm(beta, 1)
	return loss


# ------------------------------------------------------------
# Listing 6-22. Train a LASSO model.
# ------------------------------------------------------------

# Perform optimization.
for i in range(1000):
	opt.minimize(lambda: lassoLoss(alpha, beta, 
	x_train, y_train), var_list = [alpha, beta])

# Generate predicted values.
y_pred = LAD(alpha, beta, x_train)


# ------------------------------------------------------------
# Listing 6-23. Define a deep learning model for text using 
# the Keras API.
# ------------------------------------------------------------

import tensorflow as tf

# Define input layer.
inputs = tf.keras.Input(shape=(1000,))

# Define dense layer.
dense0 = tf.keras.layers.Dense(64,
activation="relu")(inputs)

# Define dropout layer.
dropout0 = tf.keras.layers.Dropout(0.20)(dense0)

# Define dense layer.
dense1 = tf.keras.layers.Dense(32,
activation="relu")(dropout0)

# Define dropout layer.
dropout1 = tf.keras.layers.Dropout(0.20)(dense1)

# Define output layer.
outputs = tf.keras.layers.Dense(1,
	activation="linear")(dropout1)

# Define model using inputs and outputs.
model = tf.keras.Model(inputs=inputs,
	outputs=outputs)


# ------------------------------------------------------------
# Listing 6-24. Summarize the architecture of a Keras model.
# ------------------------------------------------------------

# Print model architecture.
print(model.summary())


# ------------------------------------------------------------
# Listing 6-25. Compile and train the Keras model.
# ------------------------------------------------------------

# Compile the model.
model.compile(loss="mae", optimizer="adam")

# Train the model.
model.fit(x_train, y_train, epochs=20, 
batch_size=32, validation_split = 0.30)


# ------------------------------------------------------------
# Listing 6-26. Define a logistic model to perform 
# classification in TensorFlow.
# ------------------------------------------------------------

# Define a logistic model.
def logitModel(x_train, beta, alpha):
	prediction = tf.nn.softmax(tf.matmul(
	x_train, beta) + alpha)
	return prediction


# ------------------------------------------------------------
# Listing 6-27. Define a loss function for the logistic model 
# to perform classification in TensorFlow.
# ------------------------------------------------------------

# Define number of observations.
N = len(x_train)

# Define function to compute MAE loss.
def logisticLoss(alpha, beta, x_train, y_train):
	y_hat = LogitModel(alpha, beta, x_train)
	y_hat = tf.reshape(y_hat, (N,))
	loss = tf.losses.binary_crossentropy(
	y_train, y_hat)
	return loss


# ------------------------------------------------------------
# Listing 6-28. Modify a neural network to perform classification.
# ------------------------------------------------------------

# Change output layer to use sigmoid activation.
outputs = tf.keras.layers.Dense(1,
	activation="sigmoid")(dropout1)

# Use categorical cross entropy loss in compilation.
model.compile(loss="binary_crossentropy", optimizer="adam")