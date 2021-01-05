# ------------------------------------------------------------
# Listing 3-1. Implement OLS in TensorFlow 2.
# ------------------------------------------------------------

import tensorflow as tf

# Define the data as constants.
X = tf.constant([[1, 0], [1, 2]], tf.float32)
Y = tf.constant([[2], [4]], tf.float32)

# Compute vector of parameters.
XT = tf.transpose(X)
XTX = tf.matmul(XT,X)
beta = tf.matmul(tf.matmul(tf.linalg.inv(XTX),XT),Y)


# ------------------------------------------------------------
# Listing 3-2. Generate input data for a linear regression.
# ------------------------------------------------------------

import tensorflow as tf

# Set number of observations and samples
S = 100
N = 10000

# Set true values of parameters.
alpha = tf.constant([1.], tf.float32)
beta = tf.constant([3.], tf.float32)

# Draw independent variable and error.
X = tf.random.normal([N, S])
epsilon = tf.random.normal([N, S], stddev=0.25)

# Compute dependent variable.
Y = alpha + beta*X + epsilon


# ------------------------------------------------------------
# Listing 3-3. Initialize variables and define the loss.
# ------------------------------------------------------------

# Draw initial values randomly.
alphaHat0 = tf.random.normal([1], stddev=5.0)
betaHat0 = tf.random.normal([1], stddev=5.0)

# Define variables.
alphaHat = tf.Variable(alphaHat0, tf.float32)
betaHat = tf.Variable(betaHat0, tf.float32)

# Define function to compute MAE loss.
def maeLoss(alphaHat, betaHat, xSample, ySample):
	prediction = alphaHat + betaHat*xSample
	error = ySample – prediction
	absError = tf.abs(error)
	return tf.reduce_mean(absError)


# ------------------------------------------------------------
# Listing 3-4. Define an optimizer and minimize the loss function.
# ------------------------------------------------------------

# Define optimizer.
opt = tf.optimizers.SGD()

# Define empty lists to hold parameter values.
alphaHist, betaHist = [], []

# Perform minimization and retain parameter updates.
for j in range(1000):

# Perform minimization step.
	opt.minimize(lambda: maeLoss(alphaHat, betaHat,
	X[:,0], Y[:,0]), var_list = [alphaHat, 
betaHat])

# Update list of parameters.
	alphaHist.append(alphaHat.numpy()[0])
	betaHist.append(betaHat.numpy()[0])


# ------------------------------------------------------------
# Listing 3-5. Plot the parameter training histories.
# ------------------------------------------------------------

# Define DataFrame of parameter histories.
params = pd.DataFrame(np.hstack([alphaHist, 
betaHist]), columns = ['alphaHat', 'betaHat'])

# Generate plot.
params.plot(figsize=(10,7))

# Set x axis label.
plt.xlabel('Epoch')

# Set y axis label.
plt.ylabel('Parameter Value')


# ------------------------------------------------------------
# Listing 3-6. Generate data for partially linear 
# regression experiment.
# ------------------------------------------------------------

import tensorflow as tf

# Set number of observations and samples
S = 100
N = 10000

# Set true values of parameters.
alpha = tf.constant([1.], tf.float32)
beta = tf.constant([3.], tf.float32)
theta = tf.constant([0.05], tf.float32)

# Draw independent variable and error.
X = tf.random.normal([N, S])
Z = tf.random.normal([N, S])
epsilon = tf.random.normal([N, S], stddev=0.25)

# Compute dependent variable.
Y = alpha + beta*X + tf.exp(theta*Z) + epsilon


# ------------------------------------------------------------
# Listing 3-7. Initialize variables and compute the loss.
# ------------------------------------------------------------

# Draw initial values randomly.
alphaHat0 = tf.random.normal([1], stddev=5.0)
betaHat0 = tf.random.normal([1], stddev=5.0)
thetaHat0 = tf.random.normal([1], mean = 0.05,                 
            stddev=0.10)

# Define variables.
alphaHat = tf.Variable(alphaHat0, tf.float32)
betaHat = tf.Variable(betaHat0, tf.float32)
thetaHat = tf.Variable(thetaHat0, tf.float32)

# Compute prediction.
def plm(alphaHat, betaHat, thetaHat, xS, zS):
	prediction = alphaHat + betaHat*xS + \
			tf.exp(thetaHat*zS)
	return prediction


# ------------------------------------------------------------
# Listing 3-8. Define a loss function for a partially 
# linear regression.
# ------------------------------------------------------------

# Define function to compute MAE loss.
def maeLoss(alphaHat, betaHat, thetaHat, xS, zS, yS):
	yHat = plm(alphaHat, betaHat, thetaHat, xS, zS)
	return tf.losses.mae(yS, yHat)


# ------------------------------------------------------------
# Listing 3-9. Train a partially linear regression model.
# ------------------------------------------------------------

# Instantiate optimizer.
opt = tf.optimizers.SGD()

# Perform optimization.
for i in range(1000):
	opt.minimize(lambda: maeLoss(alphaHat, betaHat,
	thetaHat, X[:,0], Z[:,0], Y[:,0]), 
	var_list = [alphaHat, betaHat, thetaHat])


# ------------------------------------------------------------
# Listing 3-10. Prepare the data for a TAR model of the 
# USD-GBP exchange rate.
# ------------------------------------------------------------

import pandas as pd
import numpy as np
import tensorflow as tf

# Define data path.
data_path = '../data/chapter3/'

# Load data.
data = pd.read_csv(data_path+'exchange_rate.csv')

# Convert log exchange rate to numpy array.
e = np.array(data['log_USD_GBP'])

# Identify exchange decreases greater than 2%.
de = tf.cast(np.diff(e[:-1]) < -0.02, tf.float32)

# Define the lagged exchange rate as a constant.
le = tf.constant(e[1:-1], tf.float32)

# Define the exchange rate as a constant.
e = tf.constant(e[2:], tf.float32)

# ------------------------------------------------------------
# Listing 3-11. Define parameters for a TAR model of the 
# USD-GBP exchange rate.
# ------------------------------------------------------------

# Define variables.
rho0Hat = tf.Variable(0.80, tf.float32)
rho1Hat = tf.Variable(0.80, tf.float32)


# ------------------------------------------------------------
# Listing 3-12. Define model and loss function for TAR model 
# of USD-GBP exchange rate.
# ------------------------------------------------------------

# Define model.
def tar(rho0Hat, rho1Hat, le, de):
	# Compute regime-specific prediction.
	regime0 = rho0Hat*le
	regime1 = rho1Hat*le
	# Compute prediction for regime.
	prediction = regime0*de + regime1*(1-de)
	return prediction

# Define loss.
def maeLoss(rho0Hat, rho1Hat, e, le, de):
	ehat = tar(rho0Hat, rho1Hat, le, de)
	return tf.losses.mae(e, ehat)

# ------------------------------------------------------------
# Listing 3-13. Train TAR model of the USD-GBP exchange rate.
# ------------------------------------------------------------

# Define optimizer.
opt = tf.optimizers.SGD()

# Perform minimization.
for i in range(20000):
	opt.minimize(lambda: maeLoss(
	rho0Hat, rho1Hat, e, le, de), 
	var_list = [rho0Hat, rho1Hat]
	)


# ------------------------------------------------------------
# Listing 3-14. Instantiate optimizers.
# ------------------------------------------------------------

# Instantiate optimizers.
sgd = tf.optimizers.SGD(learning_rate = 0.001, 
momentum = 0.5)
rms = tf.optimizers.RMSprop(learning_rate = 0.001,
	rho = 0.8, momentum = 0.9)
agrad = tf.optimizers.Adagrad(learning_rate = 0.001,
	initial_accumulator_value = 0.1)
adelt = tf.optimizers.Adadelta(learning_rate = 0.001,
	rho = 0.95)
adam = tf.optimizers.Adam(learning_rate = 0.001,
	beta_1 = 0.9, beta_2 = 0.999)