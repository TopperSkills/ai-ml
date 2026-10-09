# Lesson: Polynomial Regression (Linear Regression that can BEND)
#
# Story: we invent 20 car trips. Mileage is best around 60 km/h and worse when
# driving too slow or too fast -> an arch-shaped curve. A straight line can't
# follow an arch, so we give Linear Regression extra "multiplied" columns
# (speed×speed, speed×speed×speed, ...) and compare different degrees:
#   degree 1  -> straight line     (too simple  -> underfit)
#   degree 2  -> 1 bend (arch)     (just right)
#   degree 3  -> up to 2 bends     (also fine)
#   degree 10 -> very wiggly       (too complex -> overfit, memorises noise)





# ---------------------------------------------------------------- imports
import numpy as np                 # numbers + random-number generator (to invent data)
import pandas as pd                # DataFrame = table with named columns
import matplotlib.pyplot as plt    # drawing charts

# train_test_split: divides rows into a training part (to learn from)
# and a testing part (hidden until the end, to check the model honestly)
from sklearn.model_selection import train_test_split

# PolynomialFeatures: creates the extra columns (speed×speed, speed×speed×speed, ...)
# StandardScaler: puts all columns on the same scale (average 0, similar spread).
#   Needed here because speed×speed×... creates HUGE numbers (120^3 = 1.7 million).
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
# speed = KMPH 
# Mileage = KMPL 

# make_pipeline: chains steps into one object -> extra columns -> scale -> regression
from sklearn.pipeline import make_pipeline

# The model itself is plain Linear Regression; the "bending" comes from the extra columns
from sklearn.linear_model import LinearRegression

# MAE: "on average, how far off were the guesses?" Lower = better
from sklearn.metrics import mean_absolute_error

# ---------------------------------------------------------------- 1. create fake data
# Random-number generator with fixed seed 0 -> same "random" data every run (reproducible)
rng = np.random.default_rng(0)

n = 20  # number of car trips (rows)

# 20 random speeds between 10 and 120 km/h, rounded to whole numbers
speed = rng.uniform(10, 120, n).round(0)

# Hidden "true" rule used to invent mileage (the model will NOT see this rule):
#   start at 20 km/l (the best mileage, reached at 60 km/h)
#   - 0.004 * (speed - 60) ** 2 -> the further from 60 km/h (slower OR faster),
#                                  the bigger the loss. This makes the arch shape.
#   + rng.normal(0, 1.5, n)     -> random "real life" noise (traffic, wind...),
#                                  usually between about -3 and +3 km/l
#   .round(1)                   -> 1 decimal place, e.g. 18.2 km/l
mileage = (20 - 0.004 * (speed - 60) ** 2 + rng.normal(0, 1.5, n)).round(1)

# Put both lists into one table with two columns
df = pd.DataFrame({"speed": speed, "mileage": mileage})
print(df.head())   # look at the first 5 trips

# ---------------------------------------------------------------- 2. features, target, split
# X = input. Double brackets [["speed"]] keep it as a TABLE (sklearn needs 2D input).
# y = answer to predict. Single brackets ["mileage"] give a single column (1D).
X = df[["speed"]]
y = df["mileage"]

# test_size=0.5 -> 10 trips to learn from, 10 hidden for testing
# random_state=42 -> same split every run
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

# Peek: what extra columns does PolynomialFeatures create? (degree 3, first 3 trips)
# Output columns: speed, speed×speed, speed×speed×speed
peek = PolynomialFeatures(degree=3, include_bias=False)
print("\nExtra columns for degree 3:")
print(pd.DataFrame(peek.fit_transform(X_train.head(3)),
                   columns=peek.get_feature_names_out()))   # names like "speed^2"
print()


# ---------------------------------------------------------------- 3. try different degrees
# Three new speeds to predict. The true mileage (from the hidden rule) is about
# 13.6 / 20.0 / 10.0 km/l, so we can judge how close each model gets.
new_speeds = pd.DataFrame({"speed": [20, 60, 110]})

# np.linspace(10, 120, 200) -> 200 evenly spaced speeds from 10 to 120.
# Used only to draw each model's curve smoothly on the chart.
smooth = pd.DataFrame({"speed": np.linspace(10, 120, 200)})

# Draw the real trips as dots: black = trips the model learned from,
# red X = hidden test trips (the "real exam")
plt.scatter(X_train["speed"], y_train, color="black", label="Training trips")
plt.scatter(X_test["speed"], y_test, color="red", marker="x", label="Test trips")

for degree in [1, 2, 3, 10]:
    # Build and train a 3-step pipeline:
    #   1) PolynomialFeatures -> adds speed², speed³ ... up to speed^degree
    #      include_bias=False -> don't add a column of 1s (LinearRegression has its own starting value)
    #   2) StandardScaler     -> brings those columns to the same scale
    #   3) LinearRegression   -> learns one importance per column, then adds them up
    # .fit() runs all 3 steps on the training data
    model = make_pipeline(
        PolynomialFeatures(degree=degree, include_bias=False),  # add speed², speed³ ...
        StandardScaler(),                                        # put columns on same scale
        LinearRegression(),                                      # normal linear regression
    ).fit(X_train, y_train)

    # Train MAE: error on trips it learned from. Test MAE: error on unseen trips (what matters).
    # If train is much smaller than test -> the model memorised (overfitting).
    train_mae = mean_absolute_error(y_train, model.predict(X_train))
    test_mae = mean_absolute_error(y_test, model.predict(X_test))

    # Predict mileage for 20, 60 and 110 km/h (pipeline adds extra columns + scales automatically)
    preds = model.predict(new_speeds).round(1)

    # {degree:2} -> pad to 2 characters so rows line up; {x:.2f} -> 2 decimal places
    print(f"degree {degree:2}  train MAE {train_mae:.2f}  test MAE {test_mae:.2f}  "
          f"predict 20/60/110 km/h -> {preds}")

    # Draw this model's curve: predict mileage for all 200 smooth speeds and join them as a line
    plt.plot(smooth["speed"], model.predict(smooth), label=f"degree {degree}")

# ---------------------------------------------------------------- 4. picture
plt.ylim(0, 30)   # fix the y-axis range, otherwise degree 10's wild swings squash the chart
plt.title("Mileage vs speed: which degree fits best?")
plt.xlabel("Speed (km/h)")
plt.ylabel("Mileage (km per litre)")
plt.legend()      # show the box explaining which colour is which
plt.show()        # open the chart window
