# Lesson: Ridge & Lasso Regression (Linear Regression + "brakes")
#
# Story: we invent data for 40 students. Their marks depend on only 3 things
# (hours studied, sleep, attendance), but we also add 12 junk columns (shoe size,
# bus number...). Then we check which model is smart enough to ignore the junk:
#   - LinearRegression -> no brake, trusts every column (even junk)
#   - Ridge            -> brake that SHRINKS every column's importance
#   - Lasso            -> brake that can REMOVE useless columns (importance = 0)

import numpy as np    
import pandas as pd   

# train_test_split: randomly divides data into a training part (to learn from)
# and a testing part (hidden, used at the end to check the model honestly)
from sklearn.model_selection import train_test_split

# StandardScaler: puts every column on the same scale (average 0, similar spread)
# so the brake treats all columns fairly. Needed for Ridge and Lasso.
from sklearn.preprocessing import StandardScaler

# make_pipeline: joins several steps into one object. Here: scale -> model.
# Calling .fit() / .predict() on it runs the steps in order automatically.
from sklearn.pipeline import make_pipeline

# The three algorithms we compare (all predict a NUMBER, i.e. regression)
from sklearn.linear_model import LinearRegression, Ridge, Lasso

# mean_absolute_error (MAE): "on average, how far off were the guesses?" Lower = better
from sklearn.metrics import mean_absolute_error

# ---------------------------------------------------------------- 1. create fake data
# Random-number generator with a fixed seed (0). Same seed -> same "random"
# numbers on every run and every computer, so results are reproducible.
rng = np.random.default_rng(0)

n = 40  # number of students (rows)

# Build a table with the 3 USEFUL columns.
# rng.uniform(low, high, n) -> n random decimal numbers between low and high.
# .round(1) -> keep 1 decimal place (e.g. 6.3 hours); .round(0) -> whole number.
df = pd.DataFrame({
    # 3 USEFUL features
    "hours_studied": rng.uniform(1, 10, n).round(1),     # 1 to 10 hours
    "sleep_hours": rng.uniform(4, 9, n).round(1),        # 4 to 9 hours
    "attendance_pct": rng.uniform(50, 100, n).round(0),  # 50% to 100%
})

# 12 JUNK features: random numbers with no link to marks
junk = ["shoe_size", "fav_number", "phone_last_digit", "birth_month", "height_cm", "siblings",
        "bus_number", "house_number", "fav_color_code", "pet_count", "tv_hours_last_sunday",
        "fav_cricket_player"]
for col in junk:
    # rng.integers(0, 50, n) -> n random whole numbers from 0 to 49.
    # Adds a new column of pure noise to the table.
    df[col] = rng.integers(0, 50, n)

# Real marks depend ONLY on the useful features (+ a bit of luck/noise)
# Recipe in words: start at 10, +5 per hour studied, +2 per hour of sleep,
# +0.2 per attendance %, then add random "luck" so data looks realistic.
#   rng.normal(0, 5, n) -> n random luck values, usually between about -5 and +5
#   .clip(0, 100)       -> keep marks inside 0..100 (no negative or 105 marks)
#   .round(0)           -> whole-number marks
# Notice: junk columns are NOT used here, so they truly have no effect on marks.
df["marks"] = (10 + 5 * df.hours_studied + 2 * df.sleep_hours + 0.2 * df.attendance_pct
               + rng.normal(0, 5, n)).clip(0, 100).round(0)

# ---------------------------------------------------------------- 2. features, target, split
# X = inputs (all 15 columns except marks); y = answer we want to predict (marks)
X = df.drop(columns="marks")
y = df["marks"]

# Split rows into train and test sets:
#   test_size=0.5    -> 50% for testing, so only 20 students to learn from
#                       (few rows + many columns = easy to overfit, on purpose)
#   random_state=42  -> same split every run (reproducible)
# Returns 4 parts: train inputs, test inputs, train answers, test answers.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

# ---------------------------------------------------------------- 3. train & compare
# Dictionary: name -> untrained algorithm. Lets us loop over all three the same way.
models = {
    "Linear": LinearRegression(),  # no brake
    "Ridge": Ridge(alpha=4),       # brake strength 5: shrinks all importances
    "Lasso": Lasso(alpha=0.5),       # brake strength 1: can set useless ones to 0
}

importances = {}  # will store each model's learned importances, for the table at the end

# .items() gives (key, value) pairs -> name = "Linear", algo = LinearRegression(), ...
for name, algo in models.items():
    # StandardScaler first (explained below), then the model
    # make_pipeline(StandardScaler(), algo) -> a 2-step pipeline: scale, then regress.
    # .fit(X_train, y_train) -> TRAINING: scaler learns each column's average/spread
    #   from train data only, then the algorithm learns importance per column.
    model = make_pipeline(StandardScaler(), algo).fit(X_train, y_train)

    # .predict(...) -> the trained model guesses marks (pipeline scales inputs first).
    # MAE compares real answers with guesses.
    # Train MAE = score on students it studied (practice paper it has seen).
    train_mae = mean_absolute_error(y_train, model.predict(X_train))
    # Test MAE = score on unseen students (the real exam). THIS is the score that matters.
    test_mae = mean_absolute_error(y_test, model.predict(X_test))

    # f-string formatting:
    #   {name:7}       -> pad name to 7 characters so the columns line up
    #   {train_mae:.2f} -> show 2 decimal places
    print(f"{name:7} train MAE {train_mae:.2f}  test MAE {test_mae:.2f}")

    # model[-1] = last step of the pipeline (the regression model itself).
    # .coef_ = learned importance of each column, in the same order as X.columns.
    # (Trailing "_" in sklearn = something learned during .fit().)
    importances[name] = model[-1].coef_   # model[-1] = the regression step

# ---------------------------------------------------------------- 4. compare importances
# What importance did each model give each feature?
# Build a table: rows = feature names, columns = Linear / Ridge / Lasso.
# Big number (+ or -) = model thinks this column matters a lot; 0 = ignored.
# Expect: Linear gives junk noticeable importance, Ridge shrinks it, Lasso zeroes most of it.
print(pd.DataFrame(importances, index=X.columns).round(2))
