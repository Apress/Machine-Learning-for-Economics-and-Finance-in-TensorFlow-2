# ------------------------------------------------------------
# Listing 10-1. Define the constants and variables for the 
# cake-eating problem.
# ------------------------------------------------------------

import tensorflow as tf

# Define policy rule parameter.
theta = tf.Variable(0.1, tf.float32)

# Define discount factor.
beta = tf.constant(1.0, tf.float32)

# Define state at t = 0.
s0 = tf.constant(1.0, tf.float32)


# ------------------------------------------------------------
# Listing 10-2. Define a function for the policy rule.
# ------------------------------------------------------------

# Define policy rule.
def policyRule(theta, s0 = s0, beta = beta):
	s1 = tf.clip_by_value(theta*s0,
	clip_value_min = 0.01, clip_value_max = 0.99)
	return s1


# ------------------------------------------------------------
# Listing 10-3. Define the loss function.
# ------------------------------------------------------------

# Define the loss function.
def loss(theta, s0 = s0, beta = beta):
	s1 = policyRule(theta)
	v1 = tf.math.log(s1)
	v0 = tf.math.log(s0-s1) + beta*v1
	return -v0


# ------------------------------------------------------------
# Listing 10-4. Perform optimization.
# ------------------------------------------------------------

# Instantiate an optimizer.
opt = tf.optimizers.Adam(0.1)

# Perform minimization.
for j in range(500):
	opt.minimize(lambda: loss(theta), 
	var_list = [theta])


# ------------------------------------------------------------
# Listing 10-5. Define model parameters.
# ------------------------------------------------------------

import tensorflow as tf

# Define production function parameter.
alpha = tf.constant(0.33, tf.float32)

# Define discount factor.
beta = tf.constant(0.95, tf.float32)

# Define params for decision rules.
thetaK = tf.Variable(0.1, tf.float32)

# Define state grid.
k0 = tf.linspace(0.001, 1.00, 10000)


# ------------------------------------------------------------
# Listing 10-6. Define the loss function.
# ------------------------------------------------------------

# Define the loss function.
def loss(thetaK, k0 = k0, beta = beta):	
	# Define period t+1 capital.
	k1 = thetaK*k0**alpha
	
	# Define Euler equation residual.
	error = k1**alpha-beta*alpha*k0**alpha*k1**(alpha-1)
	
	return tf.reduce_mean(tf.multiply(error,error))


# ------------------------------------------------------------
# Listing 10-7. Perform optimization and evaluate results.
# ------------------------------------------------------------

# Instantiate an optimizer.
opt = tf.optimizers.Adam(0.1)

# Perform minimization.
for j in range(1000):
	opt.minimize(lambda: loss(thetaK), 
	var_list = [thetaK])

# Print thetaK.
print(thetaK)

# Compare analytical solution and thetaK.
print(alpha*beta)


# ------------------------------------------------------------
# Listing 10-8. Compute transition path.
# ------------------------------------------------------------

# Set initial value of capital.
k0 = 0.05

# Define empty lists.
y, k, c = [], [], []
	
# Perform transition.
for j in range(10):
	# Update variables.
	k1 = thetaK*k0**alpha
	c0 = (1-thetaK)*k0**alpha
	
	# Update lists.
	y.append(k0**alpha)
	k.append(k1)
	c.append(c0)
	
	# Update state.
	k0 = k1


# ------------------------------------------------------------
# Listing 10-9. Compute the Euler equation residuals.
# ------------------------------------------------------------

# Define state grid.
k0 = tf.linspace(0.001, 1.00, 10000)

# Define function to return Euler equation residuals.
def eer(k0, thetaK = thetaK, beta = beta):	
	# Define period t+1 capital.
	k1 = thetaK*k0**alpha
	
	# Define Euler equation residual.
	residuals = k1**alpha-beta*alpha*k0**alpha*k1**(alpha-1)
	
	return residuals

# Generate residuals.
resids = eer(k0)

# Print largest residual.
print(resids.numpy().max())


# ------------------------------------------------------------
# Listing 10-10. Install and import modules to 
# perform deep Q-learning.
# ------------------------------------------------------------

# Install keras-rl2.
!pip install keras-rl2

# Import numpy and tensorflow.
import numpy as np
import tensorflow as tf

# Import reinforcement learning modules from keras.
from rl.agents.dqn import DQNAgent
from rl.policy import EpsGreedyQPolicy
from rl.memory import SequentialMemory

# Import module for comparing RL algorithms.
import gym


# ------------------------------------------------------------
# Listing 10-11. Define custom reinforcement learning environment.
# ------------------------------------------------------------

# Define number of capital nodes.
n_capital = 1000

# Define environment.
class planner(gym.Env):
	def __init__(self):
		self.k = np.linspace(0.01, 1.0, n_capital)
		self.action_space = \
		gym.spaces.Discrete(n_capital)
		self.observation_space = \
		gym.spaces.Discrete(n_capital)
		self.decision_count = 0
		self.decision_max = 100
		self.observation = 500
		self.alpha = 0.33
	def step(self, action):
		assert self.action_space.contains(action)
		self.decision_count += 1
		done = False
		if(self.observation**self.alpha-action) > 0:
			reward = np.log(self.k[self.observation]**self.alpha-self.k[action])
		else:
			reward = -1000
		self.observation = action
		if (self.decision_count >= self.decision_max) or reward == -1000:
			done = True
		return self.observation, reward, done,\
		{"decisions": self.decision_count}
	def reset(self):
		self.decision_count = 0
		self.observation = 500
		return self.observation


# ------------------------------------------------------------
# Listing 10-12. Instantiate enviroment and define 
# model in TensorFlow.
# ------------------------------------------------------------

# Instantiate planner environment.
env = planner()

# Define model in TensorFlow.
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten(input_shape=(1,) + env.observation_space.shape))
model.add(tf.keras.layers.Dense(32, activation='relu'))
model.add(tf.keras.layers.Dense(n_capital, activation='linear'))


# ------------------------------------------------------------
# Listing 10-13. Set model hyperparameters and train.
# ------------------------------------------------------------

# Specify replay buffer.
memory = SequentialMemory(limit=10000, window_length=1)

# Define policy used to make training-time decisions.
policy = EpsGreedyQPolicy(0.30)

# Define deep Q-learning network (DQN).
dqn = DQNAgent(model=model, nb_actions=n_capital, 
	memory=memory, nb_steps_warmup=100,
	gamma=0.95, target_model_update=1e-2, 
	policy=policy)

# Compile and train model.               
dqn.compile(tf.keras.optimizers.Adam(0.005), metrics=['mse'])
dqn.fit(env, nb_steps=10000)