
# A single deep tree memorises the training data (overfits). A Random Forest
# fixes this by building MANY different trees and letting them VOTE:
#   - each tree learns from a random sample of rows (with repeats allowed)
#   - at each question, each tree may only look at a random few columns
# So every tree makes DIFFERENT mistakes, and the mistakes cancel out in the vote.
# Like asking 200 people instead of 1 expert: the crowd is usually wiser.
#
# Same Titanic data as decision_tree.py, so we can compare directly.
# Part 2 (random_forest_cancer.py) shows data where a forest clearly wins.

# ---------------------------------------------------------------- imports
import pandas as pd
import seaborn as sns              # also ships practice datasets (Titanic)
import matplotlib.pyplot as plt    # bar chart of feature importance

# train_test_split: one train/test split
# cross_val_score + StratifiedKFold: test a model on MANY different splits (explained below)
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold

# DecisionTreeClassifier: one tree (to compare against)
# RandomForestClassifier: many trees voting. "ensemble" = a group of models working together
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score

# ---------------------------------------------------------------- 1. load data
# Same preparation as decision_tree.py (see that file for column meanings)
df = sns.load_dataset("titanic")
df = df[["survived", "pclass", "sex", "age", "fare", "sibsp", "parch"]].dropna()
df["is_male"] = (df["sex"] == "male").astype(int)   # text -> number (1 = male, 0 = female)
df = df.drop(columns="sex")

X = df.drop(columns="survived")   # inputs
y = df["survived"]                # answer: 1 = survived, 0 = died

# ---------------------------------------------------------------- 2. one tree vs a forest
# Fair comparison using 10 different splits (cross-validation).
# Why? With one split, a model can look better or worse just by LUCK of which
# passengers landed in the test set. Cross-validation:
#   1) cuts the data into 10 equal parts ("folds")
#   2) trains 10 times; each time a different part is the test set, the other 9 train
#   3) gives 10 accuracy scores -> we report their average (and the worst/best)
#   StratifiedKFold -> every fold keeps the same survived/died ratio; shuffle -> mix rows first
folds = StratifiedKFold(n_splits=10, shuffle=True, random_state=0)

# Random Forest settings:
#   n_estimators=200   -> build 200 trees
#   min_samples_leaf=3 -> every leaf must hold at least 3 passengers (a gentle brake)
#   random_state=42    -> same random choices every run (reproducible)
#   n_jobs=-1          -> use all CPU cores, so trees are built in parallel (faster)
models = {
    "1 tree, no limit": DecisionTreeClassifier(random_state=42),
    "1 tree, max_depth=3": DecisionTreeClassifier(max_depth=3, random_state=42),
    "Random Forest, 200 trees": RandomForestClassifier(
        n_estimators=200, min_samples_leaf=3, random_state=42, n_jobs=-1),
}

print("Accuracy over 10 different splits:")
for name, model in models.items():
    scores = cross_val_score(model, X, y, cv=folds)   # 10 accuracy scores
    print(f"  {name:26} avg {scores.mean():.1%}   (worst {scores.min():.1%}, best {scores.max():.1%})")
# Expect: the forest clearly beats the unlimited tree, and roughly TIES the depth-3 tree.
# Lesson: with only 6 simple columns, a small tree already finds the main rules
# ("women and children first"), so there is little left for a forest to add.

# ---------------------------------------------------------------- 3. train one forest
# Same split as decision_tree.py, so we can look inside a trained forest
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

# oob_score=True -> also compute the "out-of-bag" score:
#   each tree learns from a random sample, so about 1/3 of passengers are NOT used by
#   that tree ("left out of the bag"). Each passenger is then tested only by the trees
#   that never saw them -> a free, honest accuracy estimate without a test set.
forest = RandomForestClassifier(n_estimators=200, min_samples_leaf=3, oob_score=True,
                                random_state=42, n_jobs=-1)
forest.fit(X_train, y_train)
print(f"\nTest accuracy: {accuracy_score(y_test, forest.predict(X_test)):.1%}")
print(f"OOB score (free estimate from training data only): {forest.oob_score_:.1%}")

# ---------------------------------------------------------------- 4. watch the trees vote
# Pick one test passenger: the first 3rd-class woman in the test set
passenger = X_test[(X_test["is_male"] == 0) & (X_test["pclass"] == 3)].head(1)
print("\nPassenger:")
print(passenger.to_string(index=False))

# forest.estimators_ -> the list of all 200 trained trees inside the forest.
# Ask each tree for its own answer (0 or 1). .values passes plain numbers, because the
# trees inside the forest were trained without column names.
votes = [int(tree.predict(passenger.values)[0]) for tree in forest.estimators_]
print(f"Trees voting 'survived': {sum(votes)} / {len(votes)}")

# predict_proba -> forest's chance of survival = AVERAGE of all trees' chances.
# Final answer: chance >= 50% -> survived.
chance = forest.predict_proba(passenger)[0, 1]
print(f"Forest's chance of survival: {chance:.0%} -> "
      f"{'SURVIVED' if chance >= 0.5 else 'DIED'}")
# y_test[passenger.index[0]] -> look up this passenger's real answer by their row label
print("Actual result:", "SURVIVED" if y_test[passenger.index[0]] == 1 else "DIED")

# ---------------------------------------------------------------- 5. feature importance
# Average, over all 200 trees, of how much each column helped make groups purer.
# Values add up to 1. Bigger = more useful for predicting survival.
importance = pd.Series(forest.feature_importances_, index=X.columns).sort_values()
print("\nFeature importance:")
print(importance[::-1].round(2).to_string())   # [::-1] -> largest first

# Horizontal bar chart: easier to compare than numbers
importance.plot.barh(color="steelblue")
plt.title("Random Forest (Titanic): which columns matter most?")
plt.xlabel("Importance (all bars add up to 1)")
plt.tight_layout()   # avoid cut-off labels
plt.show()
