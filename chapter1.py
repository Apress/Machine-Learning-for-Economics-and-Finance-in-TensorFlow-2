# ------------------------------------------------------------
# Listing 1-1. Implement OLS in TensorFlow 1.
# ------------------------------------------------------------

import tensorflow as tf

print(tf.__version__)

# Define the data as constants.
X = tf.constant([[1, 0], [1, 2]], tf.float32)
Y = tf.constant([[2], [4]], tf.float32)

# Matrix multiply X by X’s transpose and invert.
beta_0 = tf.linalg.inv(tf.matmul(tf.transpose(X), X))

# Matrix multiply beta_0 by X’s transpose.
beta_1 = tf.matmul(beta_0, tf.transpose(X))

# Matrix multiply beta_1 by Y.
beta = tf.matmul(beta_1, Y)

# Perform computation in context of session.
with tf.Session() as sess:
	sess.run(beta)
	print(beta.eval())


# ------------------------------------------------------------
# Listing 1-2. Implement OLS in TensorFlow 2.
# ------------------------------------------------------------

import tensorflow as tf

print(tf.__version__)

# Define the data as constants.
X = tf.constant([[1, 0], [1, 2]], tf.float32)
Y = tf.constant([[2], [4]], tf.float32)

# Matrix multiply X by X’s transpose and invert.
beta_0 = tf.linalg.inv(tf.matmul(tf.transpose(X), X))

# Matrix multiply beta_0 by X’s transpose.
beta_1 = tf.matmul(beta_0, tf.transpose(X))

# Matrix multiply beta_1 by Y.
beta = tf.matmul(beta_1, Y)

# Print coefficient vector.
print(beta.numpy())


# ------------------------------------------------------------
# Listing 1-3. Print tensors in TensorFlow 1.
# ------------------------------------------------------------

# Print the feature matrix.
print(X)

# Print the coefficient vector.
print(beta)


# ------------------------------------------------------------
# Listing 1-4. Print tensors in TensorFlow 2.
# ------------------------------------------------------------

# Print the feature matrix.
print(X)

# Print the coefficient vector.
print(beta.numpy())


# ------------------------------------------------------------
# Listing 1-5. Generate logs for a TensorBoard visualization 
# in TensorFlow 1.
# ------------------------------------------------------------

# Export static graph to log file.
with tf.Session() as sess:
	tf.summary.FileWriter('/logs', sess.graph)


# ------------------------------------------------------------
# Listing 1-6. Generate OLS predictions with static graphs 
# in TensorFlow 2.
# ------------------------------------------------------------

# Define OLS prediction function as static graph.
@tf.function
def ols_predict(X, beta):
	y_hat = tf.matmul(X, beta)
	return y_hat

# Predict Y using X and beta.
predictions = ols_predict(X, beta)


# ------------------------------------------------------------
# Listing 1-7. Solve an OLS model with tf.keras().
# ------------------------------------------------------------

# Define sequential model.
ols = tf.keras.Sequential()

# Add dense layer with linear activation.
ols.add(tf.keras.layers.Dense(1, input_shape = (2,),
use_bias = False , activation = ‘linear’))

# Set optimizer and loss.
ols.compile(optimizer = 'SGD', loss = 'mse')

# Train model.
ols.fit(X, Y, epochs = 500)

# Print parameter estimates.
print(ols.weights[0].numpy())


# ------------------------------------------------------------
# Listing 1-8. Solve an OLS model with tf.estimator().
# ------------------------------------------------------------

# Define feature columns.
features = [
tf.feature_column.numeric_column("constant"),
tf.feature_column.numeric_column("x1")
]

# Define model.
ols = tf.estimator.LinearRegressor(features)

# Define function to feed data to model.
def train_input_fn():
	features = {"constant": [1, 1], "x1": [0, 2]}
	target = [2, 4]
	return features, target

# Train OLS model.
ols.train(train_input_fn, steps = 100)


# ------------------------------------------------------------
# Listing 1-9. Make predictions with an OLS model with tf.estimator().
# ------------------------------------------------------------

# Define feature columns.
def test_input_fn():
    features = {"constant": [1, 1], "x1": [3, 5]}
    return features

# Define prediction generator.
predict_gen = ols.predict(input_fn=test_input_fn)

# Generate predictions.
predictions = [next(predict_gen) for j in range(2)]

# Print predictions.
print(predictions)


# ------------------------------------------------------------
# Listing 1-10. List all available devices, select CPU, and 
# then switch to GPU.
# ------------------------------------------------------------

import tensorflow as tf

# Print list of devices.
devices = tf.config.list_physical_devices()
print(devices)

# Set device to CPU.
tf.config.experimental.set_visible_devices(
devices[0], 'CPU')

# Change device to GPU.
tf.config.experimental.set_visible_devices(
devices[3], 'GPU')


# ------------------------------------------------------------
# Listing 1-11. Define constants and variables for OLS.
# ------------------------------------------------------------

import tensorflow as tf

# Define the data as constants.
X = tf.constant([[1, 0], [1, 2]], tf.float32)
Y = tf.constant([[2], [4]], tf.float32)

# Initialize beta.
beta = tf.Variable([[0.01],[0.01]], tf.float32)

# Compute the residual.
residuals = Y - tf.matmul(X, beta)


# ------------------------------------------------------------
# Listing 1-12. Perform scalar addition and multiplication in TensorFlow.
# ------------------------------------------------------------

import tensorflow as tf

# Define two scalars as constants.
s1 = tf.constant(5, tf.float32)
s2 = tf.constant(15, tf.float32)

# Add and multiply using tf.add() and tf.multiply().
s1s2_sum = tf.add(s1, s2)
s1s2_product = tf.multiply(s1, s2)

# Add and multiply using operator overloading.
s1s2_sum = s1+s2
s1s2_product = s1*s2

# Print sum.
print(s1s2_sum)

# Print product.
print(s1s2_product)


# ------------------------------------------------------------
# Listing 1-13. Perform tensor addition in TensorFlow.
# ------------------------------------------------------------

import tensorflow as tf

# Print the shapes of the two tensors.
print(images.shape)
(32, 64, 64, 3)
Print(transform.shape)
(32, 64, 64, 3)

# Convert numpy arrays into tensorflow constants.
images = tf.constant(images, tf.float32)
transform = tf.constant(transform, tf.float32)

# Perform tensor addition with tf.add().
images = tf.add(images, transform)

# Perform tensor addition with operator overloading.
images = images + transform


# ------------------------------------------------------------
# Listing 1-14. Perform elementwise multiplication in TensorFlow.
# ------------------------------------------------------------

import tensorflow as tf

# Generate 6-tensors from normal distribution draws.
A = tf.random.normal([5, 10, 7, 3, 2, 15])
B = tf.random.normal([5, 10, 7, 3, 2, 15])

# Perform elementwise multiplication.
C = tf.multiply(A, B)
C = A*B


# ------------------------------------------------------------
# Listing 1-15. Perform dot product in TensorFlow.
# ------------------------------------------------------------

import tensorflow as tf

# Set random seed to generate reproducible results.
tf.random.set_seed(1)

# Use normal distribution draws to generate tensors.
A = tf.random.normal([200])
B = tf.random.normal([200])

# Perform dot product.
c = tf.tensordot(A, B, axes = 1)

# Print numpy argument of c.
print(c.numpy())


# ------------------------------------------------------------
# Listing 1-16. Perform matrix multiplication in TensorFlow.
# ------------------------------------------------------------

import tensorflow as tf

# Use normal distribution draws to generate tensors.
A = tf.random.normal([200, 50])
B = tf.random.normal([50, 10])

# Perform matrix multiplication.
C = tf.matmul(A, B)

# Print shape of C.
print(C.shape)


# ------------------------------------------------------------
# Listing 1-17. Perform scalar-tensor addition and multiplication.
# ------------------------------------------------------------

import tensorflow as tf

# Define scalar term as a constant.
gamma = tf.constant(1/255.0)
mu = tf.constant(-0.50)

# Perform tensor-scalar multiplication.
images = gamma * images

# Perform tensor-scalar addition.
images = mu + images


# ------------------------------------------------------------
# Listing 1-18. Define random tensors.
# ------------------------------------------------------------

import tensorflow as tf

# Define random 3-tensor of images.
images = tf.random.uniform((64, 256, 256))

# Define random 2-tensor image transformation.
transform = tf.random.normal((256, 256))


# ------------------------------------------------------------
# Listing 1-19. Perform batch matrix multiplication.
# ------------------------------------------------------------

# Perform batch matrix multiplication.
batch_matmul = tf.matmul(images, transform)

# Perform batch elementwise multiplication.
batch_elementwise = tf.multiply(images, transform)


# ------------------------------------------------------------
# Listing 1-20. Compute a derivative in TensorFlow.
# ------------------------------------------------------------

import tensorflow as tf

# Define x as a constant.
x = tf.constant(2.0)

# Define f(g(x)) within an instance of gradient tape.
with tf.GradientTape() as t:
	t.watch(x)
	y = x**3
	f = 5*y**2

# Compute gradient of f with respect to x. 
df_dx = t.gradient(f, x)
print(df_dx.numpy())


# ------------------------------------------------------------
# Listing 1-21. Import image data with numpy.
# ------------------------------------------------------------

import numpy as np

# Import image data using numpy.
images = np.load('images.npy')

# Normalize pixel values to [0,1] interval.
images = images / 255.0

# Print the tensor shape.
print(images.shape)


# ------------------------------------------------------------
# Listing 1-22. Perform division in TensorFlow using 
# constant tensors.
# ------------------------------------------------------------

import tensorflow as tf

# Import image data using numpy.
images = np.load('images.npy')

# Convert the numpy array into a TensorFlow constant.
images = tf.constant(images)

# Normalize pixel values to [0,1] interval.
images = images / 255.0


# ------------------------------------------------------------
# Listing 1-23. Perform division in TensorFlow using the 
# division operation.
# ------------------------------------------------------------

import tensorflow as tf

# Import image data using numpy.
images = np.load('images.npy')

# Normalize pixel values to [0,1] interval.
images = tf.division(images, 255.0)


# ------------------------------------------------------------
# Listing 1-24. Load data in pandas for use in TensorFlow.
# ------------------------------------------------------------

import pandas as pd

# Import data using pandas.
data = np.load('data.csv')

# Convert data to a TensorFlow constant.
data_tensorflow = tf.constant(data)

# Convert data to a numpy array.
data_numpy = np.array(data)