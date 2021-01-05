# ------------------------------------------------------------
# Listing 4-1. Prepare data for use in gradient boosted 
# classification trees.
# ------------------------------------------------------------

import pandas as pd
import tensorflow as tf

# Define data path.
data_path = '../chapter4/hmda.csv'

# Load hmda data using pandas.
hmda = pd.read_csv(data_path+"hmda.csv")

# Define applicant income feature column.
applicantIncome = tf.feature_column.numeric_column("applicantIncome")

# Define applicant msa relative income.
areaIncome = tf.feature_column.numeric_column("areaIncome")

# Combine features into list.
feature_list = [applicantIncome, areaIncome]


# ------------------------------------------------------------
# Listing 4-2. Define function to generate input data function.
# ------------------------------------------------------------

# Define input data function.
def input_fn():
	# Define dictionary of features.
features = {"applicantIncome": hmda['income’],
     "areaIncome": hmda['area_income’]}
	
	# Define labels.
labels = hmda['accepted'].copy()
	
	# Return features and labels.
return features, labels


# ------------------------------------------------------------
# Listing 4-3. Define and train a boosted trees classifier.
# ------------------------------------------------------------

# Define boosted trees classifier.
model = tf.estimator.BoostedTreesClassifier(
feature_columns = feature_list,
	n_batches_per_layer = 1)
					
# Train model using 100 epochs.
model.train(input_fn, steps=100)


# ------------------------------------------------------------
# Listing 4-4. Evaluate a boosted trees classifier.
# ------------------------------------------------------------

# Evaluate model in-sample.
result = model.evaluate(input_fn, steps = 1)

# Print results.
print(pd.Series(result))


# ------------------------------------------------------------
# Listing 4-5. Define and train a boosted trees regressor.
# ------------------------------------------------------------

# Define input data function.
def input_fn():
	features = {"applicantIncome": data['income],
	"msaIncome": data['area_income']}
	targets = data['loan_amount’].copy()
	return features, targets

# Define model.	
model = tf.estimator.BoostedTreesRegressor(
feature_columns = feature_list,   
n_batches_per_layer = 1)


# ------------------------------------------------------------
# Listing 4-6. Evaluate a boosted trees regressor.
# ------------------------------------------------------------

# Evaluate model in-sample.
result = model.evaluate(input_fn, steps = 1)

# Print results.
print(pd.Series(result))
