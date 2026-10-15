
# Story: REAL data from the Titanic (1912). For each passenger we know their
# ticket class, gender, age, fare, family on board - and whether they SURVIVED.
# The tree learns survival rules by itself, e.g.
#   "female AND 1st/2nd class -> survived", "adult man in 3rd class -> died"
# At every step it picks the question that splits people into the PUREST groups
# (groups where almost everyone has the same answer), measured by "gini".

# ---------------------------------------------------------------- imports
import pandas as pd
import seaborn as sns              # also ships famous practice datasets (like Titanic)
import matplotlib.pyplot as plt    # drawing the tree

# train_test_split: divides rows into a training part (to learn from)
# and a testing part (hidden until the end, to check the model honestly)
from sklearn.model_selection import train_test_split

# DecisionTreeClassifier: the algorithm (Classifier = predicts a category: died/survived)
# export_text: prints the learned rules as text
# plot_tree:   draws the learned tree as a picture
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree

# accuracy_score: fraction of predictions that were correct (higher = better)
from sklearn.metrics import accuracy_score

# ---------------------------------------------------------------- 1. load real data
# Load REAL data: 891 Titanic passengers (seaborn downloads it once, then caches it)
df = sns.load_dataset("titanic")

# Keep only the columns we need:
#   survived -> TARGET: 1 = survived, 0 = died
#   pclass   -> ticket class: 1 = first (rich), 2 = second, 3 = third (poor)
#   sex      -> "male" / "female"
#   age      -> age in years
#   fare     -> ticket price paid
#   sibsp    -> number of siblings / husband / wife on board
#   parch    -> number of parents / children on board
# .dropna() -> remove rows with a missing value (mostly unknown ages): 891 -> 714 rows
df = df[["survived", "pclass", "sex", "age", "fare", "sibsp", "parch"]].dropna()

# Models need numbers, not text. (df["sex"] == "male") gives True/False,
# .astype(int) turns it into 1/0.
df["is_male"] = (df["sex"] == "male").astype(int)   # text -> number (1 = male, 0 = female)
df = df.drop(columns="sex")                          # the text column is no longer needed
print(df.head())                                     # first 5 passengers

# ---------------------------------------------------------------- 2. features, target, split
# X = inputs (all columns except survived), y = answer to predict
X = df.drop(columns="survived")
y = df["survived"]

# test_size=0.25  -> 75% to learn from (535 people), 25% hidden for testing (179 people)
# random_state=42 -> same split every run
# stratify=y      -> keep the same survived/died ratio in train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

# Dumb baseline: a "model" that always says "died" would be right this often.
# (y_test == 0).mean() -> share of 0s, because True counts as 1 and False as 0.
# Our tree must beat this number to be useful. {...:.1%} -> show as percent, 1 decimal.
print(f"Dumb baseline (everyone died): {(y_test == 0).mean():.1%}\n")

# ---------------------------------------------------------------- 3. choose the depth
# How deep should the tree grow? depth = how many questions in a row it may ask.
#   too shallow -> too simple (underfit)
#   too deep    -> memorises every passenger (overfit): great train score, bad test score
#   None        -> NO limit: keeps splitting until every leaf is pure
for depth in [1, 2, 3, 4, 6, None]:
    # max_depth = the "brake" for trees; random_state=42 -> same tree every run
    tree = DecisionTreeClassifier(max_depth=depth, random_state=42).fit(X_train, y_train)

    # Accuracy on passengers it learned from vs on unseen passengers (what matters)
    train_acc = accuracy_score(y_train, tree.predict(X_train))
    test_acc = accuracy_score(y_test, tree.predict(X_test))

    # get_n_leaves() -> number of final boxes (= number of rules)
    # str(depth):4   -> turn None into text and pad to 4 characters so rows line up
    # :3             -> pad the leaf count to 3 characters
    print(f"max_depth={str(depth):4}  leaves={tree.get_n_leaves():3}  "
          f"train acc {train_acc:.1%}  test acc {test_acc:.1%}")

# ---------------------------------------------------------------- 4. final model + its rules
# Final model: depth 3 had the best test accuracy above, so we use it
model = DecisionTreeClassifier(max_depth=3, random_state=42).fit(X_train, y_train)
print("\nTest accuracy:", f"{accuracy_score(y_test, model.predict(X_test)):.1%}\n")

# Print the learned rules as an indented flowchart.
#   feature_names     -> show column names instead of "feature_0, feature_1..."
#   show_weights=True -> show [died, survived] counts of training people in each leaf
# How to read a line like "is_male <= 0.50": TRUE means female (0), FALSE means male (1)
print(export_text(model, feature_names=list(X.columns), show_weights=True))

# ---------------------------------------------------------------- 5. feature importance
# Which questions mattered most? feature_importances_ = how much each column helped
# make the groups purer (all values add up to 1). 0 = never used by the tree.
# pd.Series(..., index=X.columns) pairs each value with its column name for easy reading.
print(pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False).round(2))

# ---------------------------------------------------------------- 6. predict new passengers
# Each list position is one passenger (column names must match the training columns):
#   0: rich woman, 1st class      1: poor young man, 3rd class
#   2: 6-year-old boy with family 3: 40-year-old man, 2nd class
new = pd.DataFrame({
    "pclass":  [1, 3, 3, 2],
    "age":     [30, 25, 6, 40],
    "fare":    [80, 8, 20, 15],
    "sibsp":   [0, 0, 1, 1],
    "parch":   [0, 0, 2, 0],
    "is_male": [0, 1, 1, 1],
})

# predict_proba() -> [chance of died, chance of survived] for each passenger.
# [:, 1] -> keep only the chance of SURVIVING.
# For a tree, this chance = share of survivors in the leaf the passenger lands in
# (e.g. leaf [5, 114] -> 114 / 119 = 0.96).
new["chance_survive"] = model.predict_proba(new)[:, 1].round(2)
print("\n", new)

# ---------------------------------------------------------------- 7. draw the tree
plt.figure(figsize=(16, 8))   # big canvas: 16 x 8 inches, so the boxes are readable

# Each box shows: question, gini (0 = pure, 0.5 = mixed), samples (people in box),
# value ([died, survived] counts), class (majority answer = the prediction).
# Left branch = question is TRUE, right branch = FALSE.
#   class_names -> label classes 0/1 as "died"/"survived"
#   filled=True -> colour boxes: orange = died, blue = survived, darker = purer
#   rounded=True -> rounded box corners; fontsize=9 -> text size
plot_tree(model, feature_names=list(X.columns), class_names=["died", "survived"],
          filled=True, rounded=True, fontsize=9)
plt.show()   # open the chart window





