# Lesson: Random Forest, part 2 - Breast cancer (where a forest clearly wins)
#
# In random_forest_titanic.py the forest only TIED a small tree, because Titanic has
# just 6 simple columns. Here we have 30 columns. With so many columns, every single
# tree finds a different, partly-lucky set of rules - and the forest's vote averages
# that luck out. This is where Random Forest shines.

# ---------------------------------------------------------------- imports
import pandas as pd
import matplotlib.pyplot as plt    # bar chart of feature importance

# load_breast_cancer: real medical dataset that comes INSIDE scikit-learn (no download)
from sklearn.datasets import load_breast_cancer

# train_test_split: one train/test split
# cross_val_score + StratifiedKFold: test a model on MANY different splits (explained below)
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold

# DecisionTreeClassifier: one tree (to compare against)
# RandomForestClassifier: many trees voting
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score, confusion_matrix

# ---------------------------------------------------------------- 1. load data
# 569 real tumour samples, each with 30 measurements from a microscope image
# (radius, texture, smoothness, ...).
# as_frame=True -> give the data as pandas tables.
# The result is a "Bunch" (like a dictionary): .data = inputs, .target = answers
cancer = load_breast_cancer(as_frame=True)
X = cancer.data      # 30 measurement columns
y = cancer.target    # 0 = malignant (cancer), 1 = benign (not cancer)

print(f"Rows: {X.shape[0]}, columns: {X.shape[1]}")
print(y.value_counts().rename({0: "malignant", 1: "benign"}), "\n")   # how many of each

# ---------------------------------------------------------------- 2. one tree vs a forest
# Fair comparison using 10 different splits (cross-validation).
# Why? With one split, a model can look better or worse just by LUCK of which
# rows landed in the test set. Cross-validation:
#   1) cuts the data into 10 equal parts ("folds")
#   2) trains 10 times; each time a different part is the test set, the other 9 train
#   3) gives 10 accuracy scores -> we report their average (and the worst/best)
#   StratifiedKFold -> every fold keeps the same malignant/benign ratio; shuffle -> mix rows first
folds = StratifiedKFold(n_splits=10, shuffle=True, random_state=0)

# n_estimators=200 -> 200 trees; random_state=42 -> reproducible; n_jobs=-1 -> all CPU cores
models = {
    "1 tree, no limit": DecisionTreeClassifier(random_state=42),
    "1 tree, max_depth=5": DecisionTreeClassifier(max_depth=5, random_state=42),
    "Random Forest, 200 trees": RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1),
}

print("Accuracy over 10 different splits:")
for name, model in models.items():
    scores = cross_val_score(model, X, y, cv=folds)   # 10 accuracy scores
    print(f"  {name:26} avg {scores.mean():.1%}   (worst {scores.min():.1%}, best {scores.max():.1%})")
# Expect: the forest wins clearly over both single trees.

# ---------------------------------------------------------------- 3. train one forest
# 75% train, 25% test; stratify=y keeps the same malignant/benign ratio in both parts
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

forest = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
forest.fit(X_train, y_train)
preds = forest.predict(X_test)
print(f"\nTest accuracy: {accuracy_score(y_test, preds):.1%}")

# Confusion matrix (rows = actual, columns = predicted):
#                     predicted MALIGNANT    predicted BENIGN
#   actual MALIGNANT  correct                MISSED CANCER  <- the dangerous mistake!
#   actual BENIGN     false alarm            correct
# In medicine a missed cancer is far worse than a false alarm (which just means more tests).
print("Confusion matrix:")
print(confusion_matrix(y_test, preds))

# ---------------------------------------------------------------- 4. top measurements
# Average, over all 200 trees, of how much each measurement helped make groups purer.
# .tail(10) -> the 10 largest (the Series is sorted smallest -> largest)
top10 = pd.Series(forest.feature_importances_, index=X.columns).sort_values().tail(10)
print("\nTop 5 most important measurements:")
print(top10[::-1].head(5).round(3).to_string())   # [::-1] -> largest first

# Horizontal bar chart of the 10 most useful measurements
top10.plot.barh(color="steelblue")
plt.title("Random Forest: top 10 measurements for detecting cancer")
plt.xlabel("Importance")
plt.tight_layout()   # avoid cut-off labels
plt.show()
