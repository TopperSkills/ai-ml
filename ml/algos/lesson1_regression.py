# Lesson 1: Supervised Learning -> REGRESSION
# Goal: predict the median house value of a California district (a NUMBER)
# from features like income, house age, rooms, location, etc.

from sklearn.datasets import fetch_california_housing   # built-in sample dataset
from sklearn.model_selection import train_test_split    # splits data into train/test sets
from sklearn.linear_model import LinearRegression       # the algorithm we will train
from sklearn.metrics import mean_absolute_error         # measures how wrong predictions are

# 1. Load the dataset
# Downloaded from the internet on the first run, then cached in ~/scikit_learn_data.
# as_frame=True returns pandas objects instead of plain numpy arrays.
data = fetch_california_housing(as_frame=True)
df = data.frame          # one DataFrame: 8 feature columns + 1 target column
print(df.head())         # quick look at the first 5 rows (always explore your data!)

# 2. Separate Features (X) and Target (y)
# X = inputs the model learns FROM, y = the answer the model learns to PREDICT.
X = df.drop(columns=["MedHouseVal"])  # all columns except the target
y = df["MedHouseVal"]                 # house value in $100,000s (e.g. 2.5 = $250,000)

# 3. Split into Training set (80%) and Testing set (20%)
# The model only learns from the training set. The test set stays hidden
# so we can check how well it performs on data it has never seen.
# random_state=42 makes the random split the same every run (reproducible results).
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Create and TRAIN the model
# LinearRegression learns one weight per feature, i.e. the best-fitting equation:
#   price = w1*MedInc + w2*HouseAge + ... + w8*Longitude + bias
# .fit() is where the learning happens: it finds the weights that minimise error.
model = LinearRegression().fit(X_train, y_train)

# 5. PREDICT on the unseen test data
# The trained model applies its learned equation to each test row.
preds = model.predict(X_test)

# 6. EVALUATE the model
# MAE (Mean Absolute Error) = average of |actual - predicted|.
# It is in the same unit as y, so MAE 0.53 means predictions are off
# by about $53,000 on average. Lower is better; 0 would be perfect.
print("MAE:", mean_absolute_error(y_test, preds))
