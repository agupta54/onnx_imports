import numpy as np
from sklearn import linear_model as lm
import pandas as pd

data = np.loadtxt(open('insurance_numeric.csv', 'rb'), delimiter=',', skiprows=1)

# Retreive feature vectors and outcomes:
datX = data[:, [0,1]] # Input features (age, bmi)
datY = data[:, [2]] # Charges

# Standardize the inputs:
barX = np.mean(datX, 0) # Mean for each of the inputs
sdX = np.std(datX, 0) # Sd for each of the inputs
datZ = (datX - barX) / sdX

# Log transform the output
logY = np.log(datY)

# Fit a linear model
lin_mod = lm.LinearRegression()
lin_mod.fit(datZ, logY)

# retrieve intercept and fitted coefficients:
intercept = lin_mod.intercept_
beta = lin_mod.coef_

## Storing what we need for inference in our processing pipeline.
print(" — — Values retrieved from training — — ")
print("For input statdardization / pre-processing we need:")
print(" — The column means {}".format(barX))
print(" — The column sds {}".format(sdX))
print("For the prediction we need:")
print("— The estimated coefficients: {}".format(beta))
print(" — The intercept: {}".format(intercept))
# store the training results in an object to make the code more readable later on:
training_results = {
 "barX" : barX.astype(np.float32),
 "sdX" : sdX.astype(np.float32),
 "beta" : beta.astype(np.float32),
 "intercept" : intercept.astype(np.float32),
}
# And, also creating the constraints (for usage in block 3):
constraints = {
 "maxprice" : np.array([400000]),
 "minyard" : np.array([1]),
}

# Get the data from a single house
first_row_example = data[1,:]
input_example = first_row_example[[0,1]]  # The features
output_example = first_row_example[2]  # The observed price
# 1. Standardize input for input to the model:
standardized_input_example = (input_example - training_results['barX'])/ training_results['sdX']
# 2. Predict the *log* price (using a dot product and the intercept)
predicted_log_price_example = training_results['intercept'] + np.dot(standardized_input_example, training_results['beta'].flatten())
# Compute the actual prediction on the original scale
predicted_price_example = np.exp(predicted_log_price_example)
print("Observed price: {}, predicted price: {}".format(output_example, predicted_price_example))
# See if it is interesting according to our simple decision rules:
interesting = input_example[1] > 0 and predicted_price_example < 400000
print("Interesting? {}".format(interesting))