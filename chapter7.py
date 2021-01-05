# ------------------------------------------------------------
# Listing 7-1. Instantiate a sequence generator for inflation.
# ------------------------------------------------------------

import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator

# Set data path.
data_path = '../data/chapter7/'

# Load data.
inflation = pd.read_csv(data_path+'inflation.csv')

# Convert to numpy array.
inflation = np.array(inflation['Inflation'])

# Instantiate time series generator.
generator = TimeseriesGenerator(inflation, inflation,
	length = 4, batch_size = 12)


# ------------------------------------------------------------
# Listing 7-2. Train a neural network using generated sequences.
# ------------------------------------------------------------

# Define sequential model.
model = tf.keras.models.Sequential()

# Add input layer.
model.add(tf.keras.Input(shape=(4,)))

# Define dense layer.
model.add(tf.keras.layers.Dense(2, activation="relu"))

# Define output layer.
model.add(tf.keras.layers.Dense(1, activation="linear"))

# Compile the model.
model.compile(loss="mse", optimizer="adam")

# Train the model.
model.fit_generator(generator, epochs=100)


# ------------------------------------------------------------
# Listing 7-3. Summarize model architecture.
# ------------------------------------------------------------

# Print model architecture.
print(model.summary())


# ------------------------------------------------------------
# Listing 7-4. Instantiate a sequence generator for inflation.
# ------------------------------------------------------------

import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator

# Load data.
inflation = pd.read_csv(data_path+'inflation.csv')

# Convert to numpy array.
inflation = np.array(inflation['Inflation'])

# Add dimension.
inflation = np.expand_dims(inflation, 1)

# Instantiate time series generator.
train_generator = TimeseriesGenerator(
inflation[:211], inflation[:211], 
length = 4, batch_size = 12)


# ------------------------------------------------------------
# Listing 7-5. Define an RNN model in Keras.
# ------------------------------------------------------------

# Define sequential model.
model = tf.keras.models.Sequential()

# Define recurrent layer.
model.add(tf.keras.layers.SimpleRNN(2, input_shape=(4, 1)))

# Define output layer.
model.add(tf.keras.layers.Dense(1, activation="linear"))


# ------------------------------------------------------------
# Listing 7-6. Compile and train an RNN model in Keras.
# ------------------------------------------------------------

# Compile the model.
model.compile(loss="mse", optimizer="adam")

# Fit model to data using generator.
model.fit_generator(train_generator, epochs=100)


# ------------------------------------------------------------
# Listing 7-7. Summarize RNN architecture in a Keras model.
# ------------------------------------------------------------

# Print model summary.
print(model.summary())


# ------------------------------------------------------------
# Listing 7-8. Train an LSTM model in Keras.
# ------------------------------------------------------------

# Define sequential model.
model = tf.keras.models.Sequential()

# Define recurrent layer.
model.add(tf.keras.layers.LSTM(2, input_shape=(4, 1)))

# Define output layer.
model.add(tf.keras.layers.Dense(1, activation="linear"))

# Compile the model.
model.compile(loss="mse", optimizer="adam")

# Train the model.
model.fit_generator(train_generator, epochs=100)


# ------------------------------------------------------------
# Listing 7-9. Summarize LSTM architecture in a Keras model.
# ------------------------------------------------------------

# Print model architecture.
print(model.summary())


# ------------------------------------------------------------
# Listing 7-10. Incorrect use of LSTM hidden states.
# ------------------------------------------------------------

# Define sequential model.
model = tf.keras.models.Sequential()

# Define recurrent layer to return hidden states.
model.add(tf.keras.layers.LSTM(2, return_sequences=True, input_shape=(4, 1)))

# Define output layer.
model.add(tf.keras.layers.Dense(1, activation="linear"))

# Summarize model architecture.
model.summary()


# ------------------------------------------------------------
# Listing 7-11. Define a stacked LSTM model.
# ------------------------------------------------------------

# Define sequential model.
model = tf.keras.models.Sequential()

# Define recurrent layer to return hidden states.
model.add(tf.keras.layers.LSTM(3, return_sequences=True, input_shape=(4, 1)))

# Define second recurrent layer.
model.add(tf.keras.layers.LSTM(2))

# Define output layer.
model.add(tf.keras.layers.Dense(1, activation="linear"))


# ------------------------------------------------------------
# Listing 7-12. Summarize stacked LSTM architecture.
# ------------------------------------------------------------

# Summarize model architecture.
model.summary()


# ------------------------------------------------------------
# Listing 7-13. Load and preview inflation forecast data.
# ------------------------------------------------------------

import pandas as pd

# Load data.
macroData = pd.read_csv(data_path+'macrodata.csv', index_col = 'Date')

# Preview data.
print(macroData.round(1).tail())


# ------------------------------------------------------------
# Listing 7-14. Prepare data for use in LSTM model.
# ------------------------------------------------------------

import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator

# Define target and features.
target = np.array(macroData['Inflation'])
features = np.array(macroData)

# Define train generator.
train_generator = TimeseriesGenerator(features[:393], target[:393], length = 12, batch_size = 6)

# Define test generator.
test_generator = TimeseriesGenerator(features[393:], target[393:], length = 12, batch_size = 6)


# ------------------------------------------------------------
# Listing 7-15. Define and train LSTM model with multiple features.
# ------------------------------------------------------------

# Define sequential model.
model = tf.keras.models.Sequential()

# Define LSTM model with two cells.
model.add(tf.keras.layers.LSTM(2, input_shape=(12, 5)))

# Define output layer.
model.add(tf.keras.layers.Dense(1, activation="linear"))

# Compile the model.
model.compile(loss="mse", optimizer="adam")

# Train the model.
model.fit_generator(train_generator, epochs=100)


# ------------------------------------------------------------
# Listing 7-16. Use MSE to evaluate train and test sets.
# ------------------------------------------------------------

# Evaluate training set using MSE.
model.evaluate_generator(train_generator)

# Evaluate test set using MSE.
model.evaluate_generator(test_generator)


# ------------------------------------------------------------
# Listing 7-17. Define feature columns.
# ------------------------------------------------------------

# Define lagged inflation feature column.
inflation = tf.feature_column.numeric_column(
"inflation")

# Define unemployment feature column.
unemployment = tf.feature_column.numeric_column(
"unemployment")

# Define hours feature column.
hours = tf.feature_column.numeric_column(
"hours")

# Define earnings feature column.
earnings = tf.feature_column.numeric_column(
"earnings")

# Define M1 feature column.
m1 = tf.feature_column.numeric_column(
"m1")

# Define feature list.
feature_list = [inflation, unemployment, 
hours, earnings, m1]


# ------------------------------------------------------------
# Listing 7-18. Define the data generation functions.
# ------------------------------------------------------------

# Define input function for training data.
def train_data():
	train = macroData.iloc[:392]
	features = {"inflation": train["Inflation"],
	"unemployment": train["Unemployment"],
	"hours": train["Hours"],
	"earnings": train["Earnings"],
	"m1": train["M1"]}
	labels = macroData["Inflation"].iloc[1:393]
	return features, labels

# Define input function for test data.
def test_data():
	test = macroData.iloc[393:-1]
	features = {"inflation": test["Inflation"],
	"unemployment": test["Unemployment"],
	"hours": test["Hours"],
	"earnings": test["Earnings"],
	"m1": test["M1"]}
	labels = macroData["Inflation"].iloc[394:]
	return features, labels


# ------------------------------------------------------------
# Listing 7-19. Train and evaluate model. Print results.
# ------------------------------------------------------------

# Instantiate boosted trees regressor.
model = tf.estimator.BoostedTreesRegressor(feature_columns = 
feature_list, n_batches_per_layer = 1)
                                      
# Train model.             
model.train(train_data, steps=100)

# Evaluate train and test set.
train_eval = model.evaluate(train_data, steps = 1)
test_eval = model.evaluate(test_data, steps = 1)

# Print results.
print(pd.Series(train_eval))
print(pd.Series(test_eval))