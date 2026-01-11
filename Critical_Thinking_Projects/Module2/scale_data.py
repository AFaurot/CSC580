import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Load training data set from CSV file
training_data_df = pd.read_csv('sales_data_training.csv')

# Load testing data set from CSV file
test_data_df = pd.read_csv('sales_data_test.csv')
# Data needs to be scaled to a small range like 0 to 1 for the neural network to work well
scaler = MinMaxScaler(feature_range=(0, 1))
# Scale both training inputs and outputs
scaled_training = scaler.fit_transform(training_data_df)
scaled_testing = scaler.transform(test_data_df)
# Print out the adjustment that the scaler applied to the total_earning column of data
print("Note: total_earnings values were scaled by multiplying by {:.10f}: and adding {:.6f}"
      .format(scaler.scale_[8], scaler.min_[8]))
# Create new pandas Data frames from the scaled data
scaled_training_df = pd.DataFrame(scaled_training, columns=training_data_df.columns.values)
scaled_testing_df = pd.DataFrame(scaled_testing, columns=test_data_df.columns.values)
# Save the scaled data frames to new CSV files
scaled_training_df.to_csv('sales_data_training_scaled.csv', index=False)
scaled_testing_df.to_csv('sales_data_test_scaled.csv', index=False)

# Scale the proposed new product (using same scaler as above)

# Load proposed new product data (RAW values)
proposed_product_df = pd.read_csv('proposed_new_product.csv')

# Feature names need to match, so add dummy column for total_earnings
proposed_product_df['total_earnings'] = 0
# Reorder columns to match training data
proposed_product_df = proposed_product_df[training_data_df.columns.values]
# Scale using the already-fit scaler
scaled_proposed_product = scaler.transform(proposed_product_df)

# Create DataFrame and save
scaled_proposed_product_df = pd.DataFrame(
    scaled_proposed_product,
    columns=proposed_product_df.columns.values
).drop('total_earnings', axis=1)  # Drop dummy column before saving

# Save to CSV
scaled_proposed_product_df.to_csv(
    'proposed_new_product_scaled.csv',
    index=False
)

