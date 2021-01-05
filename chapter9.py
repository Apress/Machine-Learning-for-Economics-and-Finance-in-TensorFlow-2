# ------------------------------------------------------------
# Listing 9-1. Prepare GDP growth data for use in a VAE.
# ------------------------------------------------------------

import tensorflow as tf
import pandas as pd
import numpy as np

# Define data path.
data_path = '../data/chapter9/'

# Load and transpose data.
GDP = pd.read_csv(data_path+'gdp_growth.csv', 
index_col = 'Date').T

# Print data preview.
print(GDP.head())

# Convert data to numpy array.
GDP = np.array(GDP)

# Set number of countries and quarters.
nCountries, nQuarters = GDP.shape

# Set number of latent nodes and batch size.
latentNodes = 2
batchSize = 1


# ------------------------------------------------------------
# Listing 9-2. Define function to perform sampling task in VAE.
# ------------------------------------------------------------

# Define function for sampling layer.
def sampling(params, batchSize = batchSize, latentNodes = latentNodes):
	mean, lvar = params
	epsilon = tf.random.normal(shape=(
	batchSize, latentNodes))
	return mean + tf.exp(lvar / 2.0) * epsilon


# ------------------------------------------------------------
# Listing 9-3. Define encoder model for VAE.
# ------------------------------------------------------------

# Define input layer for encoder.
encoderInput = tf.keras.layers.Input(shape = (nQuarters))

# Define latent state.
latent = tf.keras.layers.Input(shape = (latentNodes))

# Define mean layer.
mean = tf.keras.layers.Dense(latentNodes)(encoderInput)

# Define log variance layer.
lvar = tf.keras.layers.Dense(latentNodes)(encoderInput)

# Define sampling layer.
encoded = tf.keras.layers.Lambda(sampling, output_shape=(latentNodes,))([mean, lvar])

# Define model for encoder.
encoder = tf.keras.Model(encoderInput, [mean, lvar, encoded])


# ------------------------------------------------------------
# Listing 9-4. Define decoder model for VAE.
# ------------------------------------------------------------

# Define output for decoder.
decoded = tf.keras.layers.Dense(nQuarters, activation = 'linear')(latent)

# Define the decoder model.
decoder = tf.keras.Model(latent, decoded)

# Define functional model for autoencoder.
vae = tf.keras.Model(encoderInput, decoder(encoded))


# ------------------------------------------------------------
# Listing 9-5. Define VAE loss.
# ------------------------------------------------------------

# Compute the reconstruction component of the loss.
reconstruction = tf.keras.losses.binary_crossentropy(
vae.inputs[0], vae.outputs[0])

# Compute the KL loss component.
kl = -0.5 * tf.reduce_mean(1 + lvar - tf.square(mean) - tf.exp(lvar), axis = -1)

# Combine the losses and add them to the model.
combinedLoss = reconstruction + kl
vae.add_loss(combinedLoss)


# ------------------------------------------------------------
# Listing 9-6. Compile and fit VAE.
# ------------------------------------------------------------

# Compile the model.
vae.compile(optimizer='adam')

# Fit model.
vae.fit(GDP, batch_size = batchSize, epochs = 100)


# ------------------------------------------------------------
# Listing 9-7. Generate latent states and time series with 
# trained VAE.
# ------------------------------------------------------------

# Generate series reconstruction.
prediction = vae.predict(GDP[0,:].reshape(1,236))

# Generate (random) latent state from inputs.
latentState = encoder.predict(GDP[0,:].reshape(1,236))

# Perturb latent state.
latentState[0] = latentState[0] + np.random.normal(1)

# Pass perturbed latent state to decoder.
decoder.predict(latentState)


# ------------------------------------------------------------
# Listing 9-8. Prepare GDP growth data for use in a GAN.
# ------------------------------------------------------------

import tensorflow as tf
import pandas as pd
import numpy as np

# Load and transpose data.
GDP = pd.read_csv(data_path+'gdp_growth.csv', 
index_col = 'Date').T

# Convert pandas DataFrame to numpy array.
GDP = np.array(GDP)


# ------------------------------------------------------------
# Listing 9-9. Define the generative model of a GAN.
# ------------------------------------------------------------

# Set dimension of latent state vector.
nLatent = 2

# Set number of countries and quarters.
nCountries, nQuarters = GDP.shape

# Define input layer.
generatorInput = tf.keras.layers.Input(shape = (nLatent,))

# Define hidden layer.
generatorHidden = tf.keras.layers.Dense(16, activation='relu')(generatorInput)

# Define generator output layer.
generatorOutput = tf.keras.layers.Dense(236, activation='linear')(generatorHidden)

# Define generator model.
generator = tf.keras.Model(inputs = generatorInput, outputs = generatorOutput)


# ------------------------------------------------------------
# Listing 9-10. Define and compile the discriminator 
# model of a GAN.
# ------------------------------------------------------------

# Define input layer.
discriminatorInput = tf.keras.layers.Input(shape = (nQuarters,))

# Define hidden layer.
discriminatorHidden = tf.keras.layers.Dense(16, activation='relu')(discriminatorInput)

# Define discriminator output layer.
discriminatorOutput = tf.keras.layers.Dense(1, activation='sigmoid')(discriminatorHidden)

# Define discriminator model.
discriminator = tf.keras.Model(inputs = discriminatorInput, outputs = discriminatorOutput)

# Compile discriminator.
discriminator.compile(loss='binary_crossentropy', optimizer=tf.optimizers.Adam(0.0001))


# ------------------------------------------------------------
# Listing 9-11. Define and compile the adversarial
# model of a GAN.
# ------------------------------------------------------------

# Define input layer for adversarial network.
adversarialInput = tf.keras.layers.Input(shape=(nLatent))

# Define generator output as generated time series.
timeSeries = generator(adversarialInput)

# Set discriminator to be untrainable.
discriminator.trainable = False

# Compute predictions from discriminator.
adversarialOutput = discriminator(timeSeries)

# Define adversarial model.
adversarial = tf.keras.Model(adversarialInput, adversarialOutput)

# Compile adversarial network.
adversarial.compile(loss='binary_crossentropy', optimizer=tf.optimizers.Adam(0.0001))


# ------------------------------------------------------------
# Listing 9-12. Train the discriminator and the adversarial network.
# ------------------------------------------------------------

# Set batch size.
batch, halfBatch = 12, 6

for j in range(1000):
	# Draw real training data.
	idx = np.random.randint(nCountries, size = halfBatch)
	real_gdp_series = GDP[idx, :]
	
	# Generate fake training data.
	latentState = np.random.normal(size=[halfBatch, nLatent])
	fake_gdp_series = generator.predict(latentState)
	
	# Combine input data.
	features = np.concatenate((real_gdp_series, fake_gdp_series))
	
	# Create labels.
	labels = np.ones([batch,1])
	labels[halfBatch:, :] = 0
	
	# Train discriminator.
	discriminator.train_on_batch(features, labels)
	
	# Generate latent state for adversarial net.
	latentState = np.random.normal(size=[batch, nLatent])
	
	# Generate labels for adversarial network.
	labels = np.ones([batch, 1])
	
	# Train adversarial network.
	adversarial.train_on_batch(latentState, labels)