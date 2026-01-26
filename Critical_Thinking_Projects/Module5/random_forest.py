# Load the library with the iris dataset
from sklearn.datasets import load_iris
# Load scikit's random forest classifier library
from sklearn.ensemble import RandomForestClassifier
# Load pandas
import pandas as pd
# Load numpy
import numpy as np
# Evaluation helpers for analysis/writeup
from sklearn.metrics import accuracy_score, classification_report

# Set random seed
np.random.seed(0)

# Create an object called iris with the iris data
iris = load_iris()

# Create a dataframe with the four feature variables
df = pd.DataFrame(iris.data, columns=iris.feature_names)

# View the top 5 rows
print(df.head())

# Add a new column with the species names; this is what we are going to try to predict
df["species"] = pd.Categorical.from_codes(iris.target, iris.target_names)

# View the top 5 rows
print(df.head())
print("-"*60)
# Create a new column that, for each row, generates a random number between 0 and 1, and
# if that value is less than or equal to .75, then sets the value of that cell as True
# and false otherwise. This is a quick and dirty way of randomly assigning some rows to
# be used as the training data and some as the test data.

# Note for the instructor: here the assignment template has a typo. It says less than or equal to .75 but template code
# is formatted like below:
# df['is_train'] = np.random.uniform(0, 1, len(df)) = .75
df["is_train"] = np.random.uniform(0, 1, len(df)) <= 0.75

# View the top 5 rows
print(df.head())

# Create two new dataframes, one with the training rows and one with the test rows
train, test = df[df["is_train"] == True], df[df["is_train"] == False]

# Show the number of observations for the test and training dataframes
print("Number of observations in the training data:", len(train))
print("Number of observations in the test data:", len(test))
print("-"*60)
# Create a list of the feature column's names
features = df.columns[:4]

# View features
print("Features:", list(features))

# train['species'] contains the actual species names. Before we can use it,
# we need to convert each species name into a digit. So, in this case, there
# are three species, which have been coded as 0, 1, or 2.
y, species_names = pd.factorize(train["species"])

# View target array
print("Encoded target y (first 20 shown):", y[:20])
print("Mapping (label -> species):", dict(enumerate(species_names)))
print("-"*60)
# Create a random forest Classifier. By convention, clf means 'Classifier'
clf = RandomForestClassifier(n_jobs=2, random_state=0)

# Train the Classifier to take the training features and learn how they relate
# to the training y (the species)
clf.fit(train[features], y)

print(clf.get_params())
print("-"*60)

# Predicted probabilities  (rows = observations, columns = classes 0/1/2)
# I think this is what the assignment actually wants in step 5
# The original sample code below will only show the predicted class labels, not the probabilities
# clf.predict(test[features])
probs = clf.predict_proba(test[features])

# Show first 10 rows
probs_first_10 = pd.DataFrame(probs[:10], columns=species_names)
print("Predicted probabilities (first 10 test observations):")
print(probs_first_10)


# Also compute predicted class labels for later steps
pred_label_nums = clf.predict(test[features])
preds = species_names[pred_label_nums]  # convert numeric labels back to species names


# Step 6. Evaluate the classifier by comparing the predicted and actual species for the first five observations.

comparison_first_5 = pd.DataFrame({
    "Actual Species": test["species"].values[:5],
    "Predicted Species": preds[:5]
})
print("-"*60)

print("Actual vs Predicted (first 5 test observations):")
print(comparison_first_5)


# Step 7 . Create a confusion matrix and use it to interpret the classification method.

# Confusion matrix as a crosstab
cm_table = pd.crosstab(
    test["species"],
    preds,
    rownames=["Actual Species"],
    colnames=["Predicted Species"]
)


print("Confusion Matrix (crosstab):")
print(cm_table)
print("-"*60)
# Convert actual test species names to numeric labels using the same mapping as training
actual_label_nums = pd.Categorical(test["species"], categories=species_names).codes

acc = accuracy_score(actual_label_nums, pred_label_nums)
print("Test Accuracy:", acc)

print("Classification Report:")
print(classification_report(actual_label_nums, pred_label_nums, target_names=species_names))


# Step 8. View the list of features and their importance scores.

# View a list of the features and their importance scores
feature_importances = list(zip(features, clf.feature_importances_))

# Print feature importances neatly
print("Feature importances (feature, importance):")
for f, imp in feature_importances:
    print(f, imp)
