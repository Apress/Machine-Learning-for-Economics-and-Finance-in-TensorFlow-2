# ------------------------------------------------------------
# Listing 5-1. Prepare image data for use in TensorFlow. 
# ------------------------------------------------------------

import matplotlib.image as mpimg
import numpy as np
import os

# Set image directory.
data_path = '../data/chapter5/'

# Generate file list.
images = os.listdir(data_path)

# Create list of ship images.
ships = [np.array(mpimg.imread(data_path+image)) 
for image in images if image[0] == '1']

# Create list of no-ship images.
noShips = [np.array(mpimg.imread(data_path+image)) 
for image in images if image[0] == '0']


# ------------------------------------------------------------
# Listing 5-2. Explore image data.
# ------------------------------------------------------------

import matplotlib.pyplot as plt

# Print item in list of ships.
print(np.shape(ships[0]))

(80, 80, 3)

# Print pixel intensies in [0,0] position.
print(ships[0][0,0])

# Show image of ship.
plt.imshow(ships[0])


# ------------------------------------------------------------
# Listing 5-3. Implement simple neural network in Keras.
# ------------------------------------------------------------

import tensorflow as tf

# Define sequential model.
model = tf.keras.Sequential()

# Add input layer.
model.add(tf.keras.Input(shape=(8,)))

# Define hidden layer.
model.add(tf.keras.layers.Dense(4,
activation="sigmoid"))

# Define output layer.
model.add(tf.keras.layers.Dense(1,
activation="sigmoid"))


# ------------------------------------------------------------
# Listing 5-4. Reshape images for use in a neural network 
# with dense layers.
# ------------------------------------------------------------

import numpy as np

# Reshape list of ship images.
ships = [ship.reshape(19200,) for ship in ships]

# Reshape list of non-ship images.
noShips = [noShip.reshape(19200,) for noShip in 
noShips]

# Define class labels.
labels = np.vstack([np.ones((len(ships), 1)),
		np.zeros((len(noShips), 1))])
	
# Stack flattened images into numpy array.
features = np.vstack([ships, noShips])	


# ------------------------------------------------------------
# Listing 5-5. Shuffle and split data into train and test samples.
# ------------------------------------------------------------

from sklearn.model_selection import train_test_split

# Shuffle and split sample.
X_train, X_test, y_train, y_test = \
	train_test_split(features, labels,
	test_size = 0.20, random_state=0
)


# ------------------------------------------------------------
# Listing 5-6. Modify a neural network to fit the input shape.
# ------------------------------------------------------------

import tensorflow as tf

# Define sequential model.
model = tf.keras.Sequential()

# Add input layer.
model.add(tf.keras.Input(shape=(19200,)))

# Define hidden layer.
model.add(tf.keras.layers.Dense(4,
activation="sigmoid"))

# Define output layer.
model.add(tf.keras.layers.Dense(1,
activation="sigmoid"))


# ------------------------------------------------------------
# Listing 5-7. Print a model summary in Keras.
# ------------------------------------------------------------

print(model.summary())


# ------------------------------------------------------------
# Listing 5-8. Compile and train the model in Keras.
# ------------------------------------------------------------

# Compile the model.
model.compile(loss='binary_crossentropy',
optimizer='adam', metrics=['accuracy'])

# Train the model.
model.fit(X_train, y_train, epochs=100, 
batch_size=32, validation_split = 0.20)


# ------------------------------------------------------------
# Listing 5-9. Evaluate the model on the test sample.
# ------------------------------------------------------------

# Evaluate the model.
model.evaluate(X_test, y_test)


# ------------------------------------------------------------
# Listing 5-10. Evaluate the confusion matrix.
# ------------------------------------------------------------

from sklearn.metrics import confusion_matrix

# Generate predictions.
y_pred = model.predict(X_test)>0.5

# Print confusion matrix.
print(confusion_matrix(y_test, y_pred))


# ------------------------------------------------------------
# Listing 5-11. Train the model with class weights.
# ------------------------------------------------------------

# Compute class weights.
cw0 = np.mean(y_train)
cw1 = 1.0 - cw0
class_weights = {0: cw0, 1: cw1}

# Train the model using class weights.
model.fit(X_train, y_train, epochs=100, 
class_weight = class_weights,
batch_size=32, 
validation_split = 0.20)


# ------------------------------------------------------------
# Listing 5-12. Evaluate the impact of class weights on the 
# confusion matrix.
# ------------------------------------------------------------

# Generate predictions.
y_pred = model.predict(X_test)>0.5

# Print confusion matrix.
print(confusion_matrix(y_test, y_pred))


# ------------------------------------------------------------
# Listing 5-13. Define a model in Keras with the functional API.
# ------------------------------------------------------------

import tensorflow as tf

# Define input layer.
inputs = tf.keras.Input(shape=(19200,))

# Define dense layer.
dense = tf.keras.layers.Dense(4,
activation="sigmoid")(inputs)

# Define output layer.
outputs = tf.keras.layers.Dense(1,
	activation="sigmoid")(dense)

# Define model using inputs and outputs.
model = tf.keras.Model(inputs=inputs,
	outputs=outputs)


# ------------------------------------------------------------
# Listing 5-14. Define a multi-input model in Keras with the 
# functional API.
# ------------------------------------------------------------

import tensorflow as tf

# Define input layer.
img_inputs = tf.keras.Input(shape=(19200,))
meta_inputs = tf.keras.Input(shape=(20,))

# Define dense layers.
img_dense = tf.keras.layers.Dense(4,
activation="sigmoid")(img_inputs)
meta_dense = tf.keras.layers.Dense(4,
activation="sigmoid")(meta_inputs)

# Concatenate layers.
merged = tf.keras.layers.Concatenate(axis=1)([
img_dense, meta_dense])

# Define output layer.
outputs = tf.keras.layers.Dense(1,
	activation="sigmoid")(merged)

# Define model using inputs and outputs.
model = tf.keras.Model(inputs=
[img_inputs, meta_inputs], 
outputs=outputs)


# ------------------------------------------------------------
# Listing 5-15. Define a deep neural network classifier 
# using Estimators.
# ------------------------------------------------------------

# Define numeric feature columns for image.
features_list = [tf.feature_column.numeric_column("image", 
	shape=(19200,))]

# Define input function.
def input_fn():
	features = {"image": X_train}
	return features, y_train

# Define a deep neural network classifier.
model = tf.estimator.DNNClassifier(
feature_columns=features_list,
hidden_units=[256, 128, 64, 32])

# Train the model.
model.train(input_fn, steps=20)


# ------------------------------------------------------------
# Listing 5-16. Evaluate deep neural network classifiers 
# using Estimators.
# ------------------------------------------------------------

# Evaluate model in-sample.
result = model.evaluate(input_fn, steps = 1)


# ------------------------------------------------------------
# Listing 5-17. Define a convolutional neural network.
# ------------------------------------------------------------

import tensorflow as tf

# Define sequential model.
model = tf.keras.Sequential()

# Add first convolutional layer.
model.add(tf.keras.layers.Conv2D(8,
kernel_size=3, activation='relu', 
input_shape=(80,80,3)))

# Add second convolutional layer.
model.add(tf.keras.layers.Conv2D(4,
kernel_size=3, activation='relu'))

# Flatten feature maps.
model.add(tf.keras.layers.Flatten())

# Define output layer.
model.add(tf.keras.layers.Dense(1, 
activation='sigmoid'))


# ------------------------------------------------------------
# Listing 5-18. Summarize the model architecture.
# ------------------------------------------------------------

# Print summary of model architecture.
print(model.summary())


# ------------------------------------------------------------
# Listing 5-19. Prepare image data for training in a CNN.
# ------------------------------------------------------------

# Define class labels.
labels = np.vstack([np.ones((len(ships), 1)),
		np.zeros((len(noShips), 1))])
	
# Stack flattened images into numpy array.
features = np.vstack([ships, noShips])	

# Shuffle and split sample.
X_train, X_test, y_train, y_test = \
	train_test_split(features, labels,
	test_size = 0.20, random_state=0
)

# Compute class weights.
w0 = np.mean(y_train)
w1 = 1.0 - w0
class_weights = {0: w0, 1: w1}


# ------------------------------------------------------------
# Listing 5-20. Train and evaluate the model.
# ------------------------------------------------------------

# Compile the model.
model.compile(loss='binary_crossentropy',
optimizer='adam', metrics=['accuracy'])

# Train the model using class weights.
model.fit(X_train, y_train, epochs = 10, 
	class_weight = class_weights,
	batch_size = 32, 
	validation_split = 0.20)

# Evaluate model.
model.evaluate(X_test, y_test)


# ------------------------------------------------------------
# Listing 5-21. Load a pretrained model using Keras applications.
# ------------------------------------------------------------

# Load model.
model = tf.keras.applications.resnet50.ResNet50(
	weights='imagenet',
	include_top=False
	)


# ------------------------------------------------------------
# Listing 5-22. Train the classification head of a pretrained 
# model in Keras.
# ------------------------------------------------------------

# Set convolutional base to be untrainable.
model.trainable = False

# Define input layer.
inputs = tf.keras.Input(shape=(80, 80, 3))
x = model(inputs, training=False)

# Define pooling and output layers, and model. 
x = tf.keras.layers.GlobalAveragePooling2D()(x)
outputs = tf.keras.layers.Dense(1)(x)
model = tf.keras.Model(inputs, outputs)

# Compile and train the model.
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(X_train, y_train, epochs = 10, 
	class_weight = class_weights,
	batch_size = 32, 
	validation_split = 0.20)


# ------------------------------------------------------------
# Listing 5-23. Fine-tune a pretrained model in Keras.
# ------------------------------------------------------------

# Set convolutional base to be untrainable.
model.trainable = True

# Compile model with a low learning rate.
model.compile(loss='binary_crossentropy', 
optimizer=tf.keras.optimizers.Adam(
learning_rate=1e-5), 
	metrics=['accuracy'])

# Perform fine-tuning.
model.fit(X_train, y_train, epochs = 10, 
	class_weight = class_weights)