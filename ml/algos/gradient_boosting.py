# Lesson: Gradient Boosting, part 1 - how it works + house prices
#
# Random Forest: build MANY deep trees INDEPENDENTLY, then let them vote.
# Gradient Boosting: build SMALL trees ONE AFTER ANOTHER. Each new tree studies the
# MISTAKES (errors) of all the trees before it, and tries to fix them.
#
# Like learning to play darts:
#   throw 1 -> you land 30 cm too far left
#   throw 2 -> you correct a bit to the right, now only 10 cm off
#   throw 3 -> a small correction again ... and after many throws you hit the centre.
#
# Part 1 (this file): watch boosting fix errors step by step, then compare it with
#                     Random Forest on 20,640 real houses.
# Part 2 (gradient_boosting_fraud.py): XGBoost on the large card-fraud data.

# ---------------------------------------------------------------- imports
import time                        # measure training time
import pandas as pd
import matplotlib.pyplot as plt    # line chart: error vs number of trees

# fetch_california_housing: real house prices, downloaded once (~400 KB) into ml/data
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split

# DecisionTreeRegressor: one small tree (we use it to build boosting BY HAND first)
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor   # the library companies actually use (pip install xgboost)

# MAE = on average, how far off are the guesses? Lower is better (see readme)
from sklearn.metrics import mean_absolute_error

# ================================================================ PART A: boosting by hand
# A tiny made-up table: house size -> price (in lakhs), so we can SEE every number.
houses = pd.DataFrame({
    "size_sqft": [600, 800, 1000, 1200, 1500, 1800, 2200, 2600],
    "price":     [30,  38,   45,   60,   72,   80,  105,  120],
})
X_small = houses[["size_sqft"]]
y_small = houses["price"]

# Step 0: start with the simplest guess of all -> the average price, for every house
prediction = pd.Series(y_small.mean(), index=houses.index)
history = {"start (average)": prediction.copy()}

# learning_rate = how much of each new tree's correction we accept.
# 0.5 -> "fix only half the mistake each time". Small steps = safer, needs more trees.
learning_rate = 0.5

for step in range(1, 4):
    # The error that is LEFT after all previous trees. Called the "residual".
    # +ve residual -> we guessed too low, -ve -> we guessed too high.
    residual = y_small - prediction

    # A tiny tree (max_depth=1 -> just ONE yes/no question, called a "stump").
    # IMPORTANT: it learns to predict the RESIDUAL (the mistake), NOT the price.
    tree = DecisionTreeRegressor(max_depth=1)
    tree.fit(X_small, residual)

    # New prediction = old prediction + a small part of the correction
    prediction = prediction + learning_rate * tree.predict(X_small)
    history[f"after tree {step}"] = prediction.copy()
    print(f"Tree {step} asked: size_sqft <= {tree.tree_.threshold[0]:.0f}?   "
          f"average error now: {mean_absolute_error(y_small, prediction):.1f} lakhs")

table = pd.DataFrame(history).round(1)
table.insert(0, "real price", y_small)
table.insert(0, "size_sqft", houses["size_sqft"])
print("\nPredictions getting closer to the real price, tree by tree:")
print(table.to_string(index=False))
# Expect: the average error drops with every tree (one tree may overshoot a few houses,
# and the next tree corrects that). With ~100 trees the guesses get very close.
# That's ALL gradient boosting is: keep adding small trees that predict the remaining error.
# ("Gradient" is the maths name for the direction that reduces the error the most.)

# ================================================================ PART B: real house prices
# 20,640 districts in California. Columns:
#   MedInc     -> median income of people in the area (in $10,000s)
#   HouseAge   -> median age of the houses
#   AveRooms, AveBedrms -> average rooms / bedrooms per house
#   Population, AveOccup -> people in the area / people per house
#   Latitude, Longitude  -> location on the map
# Target: MedHouseVal -> median house price (in $100,000s, so 2.5 = $250,000)
housing = fetch_california_housing(as_frame=True, data_home="ml/data")
X = housing.data
y = housing.target
print(f"\nCalifornia housing: {X.shape[0]:,} rows, {X.shape[1]} columns")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# ---------------------------------------------------------------- B1. forest vs boosting
# Settings:
#   Random Forest:     300 DEEP trees built independently (in parallel)
#   Gradient Boosting: 300 SMALL trees (max_depth=4) built one after another
#     learning_rate=0.1 -> each tree fixes 10% of the remaining error
#   XGBoost: same idea as GradientBoostingRegressor, but a much faster, optimised library.
#     n_jobs=-1 -> all CPU cores. (scikit-learn's GradientBoosting can only use 1 core,
#     because each tree must wait for the previous one; XGBoost speeds up INSIDE each tree.)
models = {
    "Random Forest (300 trees)": RandomForestRegressor(
        n_estimators=300, min_samples_leaf=2, random_state=42, n_jobs=-1),
    "Gradient Boosting (sklearn)": GradientBoostingRegressor(
        n_estimators=300, max_depth=4, learning_rate=0.1, random_state=42),
    "XGBoost": XGBRegressor(
        n_estimators=300, max_depth=4, learning_rate=0.1, random_state=42, n_jobs=-1),
}

print("\nAverage error on test houses (lower is better):")
for name, model in models.items():
    start = time.time()
    model.fit(X_train, y_train)
    seconds = time.time() - start
    mae = mean_absolute_error(y_test, model.predict(X_test))
    print(f"  {name:28} MAE ${mae * 100_000:>7,.0f}   (trained in {seconds:.1f} s)")
# Expect: both boosting models beat the forest, and XGBoost is the fastest.
# This is why boosting is the first choice for table data in industry.

# ---------------------------------------------------------------- B2. the danger: too many trees
# Random Forest: more trees NEVER overfit (they just vote more).
# Boosting: every new tree keeps fixing "mistakes" - after a while, the only mistakes
# left in the TRAINING data are noise/luck, and fixing those hurts on new data = overfitting.
#
# staged_predict -> gives the predictions after 1 tree, after 2 trees, ... after N trees.
# So we can see the test error at every step from ONE trained model.
plt.figure(figsize=(8, 4))
for lr in [1.0, 0.1]:
    gb = GradientBoostingRegressor(n_estimators=500, max_depth=4, learning_rate=lr,
                                   random_state=42)
    gb.fit(X_train, y_train)
    test_errors = [mean_absolute_error(y_test, p) for p in gb.staged_predict(X_test)]
    best = min(range(len(test_errors)), key=test_errors.__getitem__) + 1   # best tree count
    print(f"\nlearning_rate={lr}: best test MAE ${min(test_errors) * 100_000:,.0f} "
          f"at {best} trees, after 500 trees ${test_errors[-1] * 100_000:,.0f}")
    plt.plot(range(1, 501), test_errors, label=f"learning_rate={lr}")
# Expect: learning_rate=1.0 (big, greedy steps) is best very early, then gets WORSE.
# learning_rate=0.1 (small steps) learns slower, but ends up better and stays stable.
# Rule of thumb: small learning_rate + more trees + stop when test error stops improving
# ("early stopping" - we use it in part 2).

plt.title("Gradient Boosting: test error as trees are added")
plt.xlabel("Number of trees")
plt.ylabel("Test MAE (in $100,000s)")
plt.legend()
plt.tight_layout()
plt.show()

# ---------------------------------------------------------------- B3. feature importance
importance = pd.Series(models["XGBoost"].feature_importances_, index=X.columns).sort_values()
print("\nXGBoost feature importance:")
print(importance[::-1].round(3).to_string())
# Expect: income (MedInc) and location (Latitude/Longitude, AveOccup) matter most -
# "location, location, location" - just like in real life.









# Decision tree 

# Random forest = multiple decision trees vote

# Gradient Boosting = initial guess -> mistakes (actual value - guess value) -> next tree -> 





# 100 

# 1 -> mistake -> 2 -> mistake -> 3 ->  -> Diamond