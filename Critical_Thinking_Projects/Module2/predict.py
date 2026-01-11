import pandas as pd
from keras.models import load_model

model = load_model('trained_model.h5')

X = pd.read_csv("proposed_new_product_scaled.csv").values
prediction = model.predict(X, verbose=0)

# Grab just the first element of the first prediction (since we only have one)
prediction = prediction[0][0]

# Re-scale the data from the 0-to-1 range back to dollars
# These constants are from when the data was originally scaled down to the 0-to-1 range
MIN_FOR_TOTAL_EARNINGS = -0.115913      # Value from scale_data.py scaler.min_[8]
SCALE_FOR_TOTAL_EARNINGS = 0.0000036968 # Value from scale_data.py scaler.scale_[8]

prediction = prediction - MIN_FOR_TOTAL_EARNINGS
prediction = prediction / SCALE_FOR_TOTAL_EARNINGS

print("Earnings Prediction for Proposed Product - ${}".format(prediction))
