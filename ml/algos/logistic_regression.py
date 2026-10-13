import numpy as np                 # numbers + random-number generator (to invent data)
import pandas as pd                # DataFrame = table with named columns
import matplotlib.pyplot as plt    # drawing charts

# train_test_split: divides rows into a training part (to learn from)
# and a testing part (hidden until the end, to check the model honestly)
from sklearn.model_selection import train_test_split

# LogisticRegression: the classification algorithm. It makes a score like
# Linear Regression, then squashes it into a chance between 0 and 1 (S-curve).
from sklearn.linear_model import LogisticRegression

# accuracy_score: what fraction of predictions were correct (higher = better)
# confusion_matrix: 2x2 table showing WHERE the model was right and wrong
from sklearn.metrics import accuracy_score, confusion_matrix

# ---------------------------------------------------------------- 1. create fake data
# Fake data: 100 students. More study hours -> more likely to pass,
# but with some luck involved (a few hard workers fail, a few lazy ones pass).

# Random-number generator with fixed seed 0 -> same "random" data every run (reproducible)
rng = np.random.default_rng(0)

n = 100  # number of students (rows)

# 100 random study hours between 0 and 10, rounded to 1 decimal (e.g. 6.4)
hours = rng.uniform(0, 10, n).round(1)

# Hidden exam score (the model never sees this column):
#   8 * hours             -> each hour of study is worth about 8 marks
#   + rng.normal(0, 12, n) -> luck: usually between about -24 and +24 marks
score = 8 * hours + rng.normal(0, 12, n)        # hidden exam score with luck

# Pass mark is 40. (score >= 40) gives True/False for each student;
# .astype(int) converts True -> 1 (pass) and False -> 0 (fail).
# ML models need numbers, so categories are stored as 0 and 1.
passed = (score >= 40).astype(int)              # 1 = pass, 0 = fail

# Table with only what the model is allowed to see: hours and the pass/fail result
df = pd.DataFrame({"hours": hours, "passed": passed})
print(df.head(8))   # first 8 students

# .value_counts() -> how many 1s and how many 0s. Always check this in classification:
# if one class is very rare (e.g. 95 pass, 5 fail), accuracy becomes misleading.
print(df["passed"].value_counts())

# ---------------------------------------------------------------- 2. features, target, split
# X = input. Double brackets [["hours"]] keep it as a TABLE (sklearn needs 2D input).
# y = answer to predict (0 or 1). Single brackets give one column.
X = df[["hours"]]
y = df["passed"]

# test_size=0.3 -> 70 students to learn from, 30 hidden for testing
# random_state=42 -> same split every run
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# ---------------------------------------------------------------- 3. train
# .fit() -> TRAINING: the model adjusts its importance and base value so that
# students who really passed get a HIGH chance and those who failed get a LOW chance.
model = LogisticRegression().fit(X_train, y_train)

# ---------------------------------------------------------------- 4. predict for new students
# Predict for new students: the label AND the probability
new = pd.DataFrame({"hours": [1, 3, 5, 7, 9]})   # column name must match training ("hours")

# .predict() -> final answer for each row: 0 (fail) or 1 (pass).
# Internally: chance >= 50% -> 1, otherwise 0.
labels = model.predict(new)                  # 0 or 1
print(f"Labels: {labels}")
# .predict_proba() -> the chances, 2 columns per row: [chance of 0, chance of 1]
#   e.g. [0.46, 0.54] = 46% fail, 54% pass (each row adds up to 1)
# [:, 1] -> all rows, column 1 only = chance of passing
probs = model.predict_proba(new)[:, 1]       # chance of passing (column 1 = class "1")

# zip() walks through the three lists together: h = hours, l = label, p = chance
# {p:.0%} -> show as a percentage with no decimals (0.54 -> 54%)
# 'PASS' if l == 1 else 'FAIL' -> turn the 1/0 label into a readable word
for h, l, p in zip(new["hours"], labels, probs):
    print(f"{h} hrs -> chance of passing {p:.0%} -> predicts {'PASS' if l == 1 else 'FAIL'}")

# ---------------------------------------------------------------- 5. evaluate
# Evaluate on the hidden test students (the "real exam" for the model)
preds = model.predict(X_test)

# Accuracy = correct predictions / total predictions. 0.77 = 77% of test students
# were classified correctly. (Not 100% because luck cannot be predicted.)
print("Accuracy:", round(accuracy_score(y_test, preds), 2))

# Confusion matrix layout (rows = actual, columns = predicted):
#                  predicted FAIL   predicted PASS
#   actual FAIL    correct          false alarm
#   actual PASS    miss             correct
# The diagonal (top-left, bottom-right) holds the correct predictions.
print(confusion_matrix(y_test, preds))

# coef_      -> learned importance of "hours". Positive = more hours RAISE the chance of passing.
# intercept_ -> base value (the score when hours = 0). Negative = low chance with no study.
# (Trailing "_" in sklearn = something learned during .fit().)
print("Importance:", model.coef_, "Base:", model.intercept_)

# ---------------------------------------------------------------- 6. picture
# Picture: dots = real students (0 = fail, 1 = pass), curve = chance of passing

# 200 evenly spaced hours from 0 to 10, used only to draw the curve smoothly
smooth = pd.DataFrame({"hours": np.linspace(0, 10, 200)})

# Each student is a dot at height 0 (failed) or 1 (passed).
# alpha=0.5 -> half-transparent, so overlapping dots are still visible
plt.scatter(df["hours"], df["passed"], alpha=0.5, label="Students (1 = pass, 0 = fail)")

# The S-curve: the model's chance of passing for every value of hours
plt.plot(smooth["hours"], model.predict_proba(smooth)[:, 1], color="red", label="Chance of passing")

# Horizontal dashed line at 0.5: where the curve crosses it, the prediction
# switches from FAIL to PASS
plt.axhline(0.5, color="gray", linestyle="--", label="50% cut-off")

plt.xlabel("Hours studied")
plt.ylabel("Chance of passing")
plt.legend()   # box explaining what each colour/line means
plt.show()     # open the chart window
