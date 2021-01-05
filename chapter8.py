# ------------------------------------------------------------
# Listing 8-1. Define variables for PCA in TensorFlow.
# ------------------------------------------------------------

import tensorflow as tf
import pandas as pd
import numpy as np

# Define data path.
data_path = '../data/chapter8/'

# Load data.
C = pd.read_csv(data_path+'gdp_growth.csv', index_col = 'Date')

# Convert data to constant object.
C = tf.constant(np.array(C), tf.float32)

# Set number of principal components.
k = 5

# Get shape of feature matrix.
n, p = C.shape

# Define variable for gamma matrix.
G = tf.Variable(tf.random.normal((n, k)), tf.float32)

# Define variable for beta matrix.
B = tf.Variable(tf.random.normal((p, k)), tf.float32)


# ------------------------------------------------------------
# Listing 8-2. Perform PCA in TensorFlow.
# ------------------------------------------------------------

# Define PCA loss.
def pcaLoss(C, G, B):
	D = C - tf.matmul(G, tf.transpose(B))
	DT = tf.transpose(D)
	DDT = tf.matmul(D, DT)
	return tf.linalg.trace(DDT)

# Instantiate optimizer.
opt = tf.optimizers.Adam()

# Perform train model.
for i in range(1000):
	opt.minimize(lambda: pcaLoss(C, G, B), 
	var_list = [G, B])


# ------------------------------------------------------------
# Listing 8-3. Import the PCA library from sklearn and 
# prepare the data.
# ------------------------------------------------------------

from sklearn.decomposition import PCA

# Load data.
C = pd.read_csv(data_path+'gdp_growth.csv',
index_col = 'Date')

# Transform feature matrix into numpy array.
C = np.array(C)


# ------------------------------------------------------------
# Listing 8-4. Perform PCA with sklearn.
# ------------------------------------------------------------

# Set number of components.
k = 25

# Instantiate PCA model with k components.
pca = PCA(n_components=k)

# Fit model.
pca.fit(C)

# Return B matrix.
B = pca.components_.T

# Return G matrix.
G = pca.transform(C)

# Return variance shares.
S = pca.explained_variance_ratio_


# ------------------------------------------------------------
# Listing 8-5. Prepare data for use in a principal components 
# regression.
# ------------------------------------------------------------

import tensorflow as tf
import numpy as np
import pandas as pd

# Load data.
gdp = pd.read_csv(data_path+'gdp_growth.csv', 
index_col = 'Date')

# Copy Canada from C.
Y = gdp['CAN'].copy()

# Copy gdp to C and drop LUX.
C = gdp.copy()
del C['CAN']

# Convert data to numpy arrays.
Y = np.array(Y)
C = np.array(C)


# ------------------------------------------------------------
# Listing 8-6. Perform PCA and PCR.
# ------------------------------------------------------------

# Set number of components.
k = 5

# Instantiate PCA model with k components.
pca = PCA(n_components=k)

# Fit model and return principal components.
pca.fit(C)
G = tf.cast(pca.transform(C), tf.float32)

# Initialize model parameters.
beta = tf.Variable(tf.random.normal([k,1]), tf.float32)
alpha = tf.Variable(tf.random.normal([1,1]), tf.float32)

# Define prediction function.
def PCR(G, beta, alpha):
	predictions = alpha + tf.reshape(tf.matmul(G, beta), (236,))
	return predictions

# Define loss function.
def mseLoss(Y, G, beta, alpha):
	return tf.losses.mse(Y, PCR(G, beta, alpha))

# Instantiate an optimizer and minimize loss.
opt = tf.optimizers.Adam(0.1)
for j in range(100):
	opt.minimize(lambda: mseLoss(Y, G, beta, 
	alpha), var_list = [beta, alpha])


# ------------------------------------------------------------
# Listing 8-7. Perform PLS.
# ------------------------------------------------------------

from sklearn.cross_decomposition import PLSRegression

# Set number of components.
k = 5

# Instantiate PLS model with k components.
pls = PLSRegression(n_components = k)

# Train PLS model.
pls.fit(C, Y)

# Generate predictions.
pls.predict(C)


# ------------------------------------------------------------
# Listing 8-8. Train an autoencoder using the Keras API.
# ------------------------------------------------------------

# Set number of countries.
nCountries = 24

# Set number of nodes in latent state.
latentNodes = 5

# Define input layer for encoder.
encoderInput = tf.keras.layers.Input(shape = (nCountries))

# Define latent state.
latent = tf.keras.layers.Input(shape = (latentNodes))

# Define dense output layer for encoder.
encoded = tf.keras.layers.Dense(latentNodes, activation = 'tanh')(encoderInput)

# Define dense output layer for decoder.
decoded = tf.keras.layers.Dense(nCountries, activation = 'linear')(latent)

# Define separate models for encoder and decoder.
encoder = tf.keras.Model(encoderInput, encoded)
decoder = tf.keras.Model(latent, decoded)

# Define functional model for autoencoder.
autoencoder = tf.keras.Model(encoderInput, decoder(encoded))

# Compile model
autoencoder.compile(loss = 'mse', optimizer='adam')

# Train model
autoencoder.fit(C, C, epochs = 200)



# ------------------------------------------------------------
# Listing 8-9. Autoencoder model architecture summary.
# ------------------------------------------------------------

# Print summary of model architecture.
print(autoencoder.summary())


# ------------------------------------------------------------
# Listing 8-10. Generate latent state time series.
# ------------------------------------------------------------

# Generate latent state time series.
latentState = encoder.predict(C)

# Print shape of latent state series.
print(latentState.shape)


# ------------------------------------------------------------
# Listing 8-11. Perform dimensionality reduction in a regression 
# setting with an autoencoder latent state.
# ------------------------------------------------------------

# Initialize model parameters.
beta = tf.Variable(tf.random.normal([latentNodes,1]))
alpha = tf.Variable(tf.random.normal([1,1]))

# Define prediction function.
def LSR(latentState, beta, alpha):
	predictions = alpha + tf.reshape(
tf.matmul(latentState, beta), (236,))
	return predictions

# Define loss function.
def mseLoss(Y, latentState, beta, alpha):
	return tf.losses.mse(Y, LSR(latentState, 
beta, alpha))

# Instantiate an optimizer and minimize loss.
opt = tf.optimizers.Adam(0.1)
for j in range(100):
	opt.minimize(lambda: mseLoss(Y, 
latentState, beta, 
alpha), var_list = [beta, alpha])