# Lesson: Gradient Boosting, part 2 - XGBoost on credit card fraud (large, real-time)
#
# Same 284,807 card transactions as random_forest_fraud.py, same split, so we can
# compare XGBoost and Random Forest head to head.
#
# New ideas in this lesson:
#   - a VALIDATION set + EARLY STOPPING: let XGBoost decide how many trees it needs
#   - scale_pos_weight: telling boosting that the rare class (fraud) matters more
#   - saving an XGBoost model in its own portable format (.json)

# ---------------------------------------------------------------- imports
import time
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, confusion_matrix
from xgboost import XGBClassifier

# ---------------------------------------------------------------- 1. load data
# See random_forest_fraud.py for what the columns mean (V1..V28 hidden, Amount, Class)
cards = fetch_openml(data_id=1597, as_frame=True, data_home="ml/data", parser="auto")
X = cards.data
y = cards.target.astype(int)   # 1 = fraud, 0 = genuine

# ---------------------------------------------------------------- 2. three-way split
# Same test set as random_forest_fraud.py (same test_size, random_state, stratify).
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y)

# Boosting needs a VALIDATION set: a piece of the training data that XGBoost does NOT
# learn from, but checks after every new tree ("am I still improving on unseen data?").
# We never use the TEST set for this - the test set must stay untouched until the end,
# otherwise our final score would be too optimistic.
#   train (60%) -> learn     validation (15%) -> decide when to stop     test (25%) -> final exam
X_fit, X_val, y_fit, y_val = train_test_split(
    X_train, y_train, test_size=0.2, random_state=42, stratify=y_train)
print(f"Learn: {len(X_fit):,}   Validation: {len(X_val):,}   Test: {len(X_test):,} rows")

# ---------------------------------------------------------------- 3. Random Forest (baseline)
# Exactly the forest from random_forest_fraud.py
forest = RandomForestClassifier(n_estimators=100, min_samples_leaf=2, max_samples=0.5,
                                random_state=42, n_jobs=-1)
start = time.time()
forest.fit(X_train, y_train)
forest_seconds = time.time() - start

# ---------------------------------------------------------------- 4. XGBoost + early stopping
# Settings:
#   n_estimators=2000      -> an UPPER LIMIT; early stopping will usually stop far sooner
#   learning_rate=0.05     -> small, careful steps (see the chart in gradient_boosting.py)
#   max_depth=5            -> small trees; boosting gets its power from MANY small trees
#   subsample=0.8          -> each tree sees a random 80% of rows (like Random Forest's
#   colsample_bytree=0.8      randomness) and 80% of columns -> less overfitting
#   eval_metric="aucpr"    -> how XGBoost scores itself on the validation set.
#                             "aucpr" = area under the precision-recall curve: a single
#                             number for "how well does it rank frauds above genuine ones".
#                             Made for rare events; plain accuracy would be useless here.
#   early_stopping_rounds=100 -> if the validation score hasn't improved for 100 trees in a
#                             row, STOP, and keep the best number of trees.
#   tree_method="hist"     -> groups values into bins before splitting. Much faster on big data.
booster = XGBClassifier(
    n_estimators=2000, learning_rate=0.05, max_depth=5,
    subsample=0.8, colsample_bytree=0.8,
    eval_metric="aucpr", early_stopping_rounds=100,
    tree_method="hist", random_state=42, n_jobs=-1)

start = time.time()
# eval_set -> the data to check after each tree. verbose=False -> don't print every tree.
booster.fit(X_fit, y_fit, eval_set=[(X_val, y_val)], verbose=False)
booster_seconds = time.time() - start
print(f"\nXGBoost stopped early: best was {booster.best_iteration + 1} trees "
      f"(out of max 2000)")

# ---------------------------------------------------------------- 5. head to head
def report(name, model, seconds, threshold=0.5):
    chance = model.predict_proba(X_test)[:, 1]   # fraud chance for every test transaction
    pred = (chance >= threshold).astype(int)
    (tn, fp), (fn, tp) = confusion_matrix(y_test, pred)
    print(f"  {name:22} recall {recall_score(y_test, pred):5.1%}  "
          f"precision {precision_score(y_test, pred):5.1%}  "
          f"caught {tp:>3}  missed {fn:>3}  honest blocked {fp:>3}  ({seconds:.1f} s)")

print(f"\nTest set: {y_test.sum()} frauds among {len(y_test):,} transactions (threshold 50%)")
report("Random Forest", forest, forest_seconds)
report("XGBoost", booster, booster_seconds)
# Honest note: on THIS dataset, with these settings, the two are about TIED (the forest
# even catches a few more). Random Forest is already very strong here. Boosting usually
# needs a bit of tuning to win (see step 6), and its lead is bigger on messier real-world
# data with many column types (text categories, missing values...).
# Lesson: always compare against a simple baseline - "XGBoost is best" is not automatic.

# ---------------------------------------------------------------- 6. scale_pos_weight
# Only 0.17% of rows are fraud, so the trees mostly "see" genuine transactions.
# scale_pos_weight tells XGBoost: "each fraud row counts as N rows" while learning.
# A common starting value: (number of genuine) / (number of fraud) ~ 577 here.
# Too big -> it gets paranoid and blocks many honest customers. So we try a few values.
print("\nEffect of scale_pos_weight (threshold 50%):")
for weight in [1, 5, 25, round((y_fit == 0).sum() / (y_fit == 1).sum())]:
    model = XGBClassifier(
        n_estimators=2000, learning_rate=0.05, max_depth=5,
        subsample=0.8, colsample_bytree=0.8, scale_pos_weight=weight,
        eval_metric="aucpr", early_stopping_rounds=100,
        tree_method="hist", random_state=42, n_jobs=-1)
    start = time.time()
    model.fit(X_fit, y_fit, eval_set=[(X_val, y_val)], verbose=False)
    report(f"scale_pos_weight={weight}", model, time.time() - start)
# Expect: a small weight (5-25) catches MORE fraud than the forest, for about the same
# number of blocked customers. The "textbook" weight (~578) overdoes it: more honest
# customers blocked, and not more fraud caught. It's the same trade-off as lowering the
# threshold in random_forest_fraud.py.
# In practice teams pick ONE of the two knobs (weight or threshold) and tune it with the
# business: "how many angry customers are we willing to accept per fraud caught?"

# ---------------------------------------------------------------- 7. save + real-time scoring
# XGBoost's own .json format: it loads in Python, Java, C++, ... so a bank's payment
# server (often written in Java) can use a model trained in Python.
model_path = "ml/models/fraud_xgboost.json"
booster.save_model(model_path)
print(f"\nModel saved to {model_path}")

live_model = XGBClassifier()
live_model.load_model(model_path)
live_model.set_params(n_jobs=1)   # one transaction at a time -> 1 core is enough

txn = X_test[y_test == 1].head(1)   # one real fraud arriving at the payment server
start = time.perf_counter()
chance = live_model.predict_proba(txn)[0, 1]
ms = (time.perf_counter() - start) * 1000
print(f"Incoming transaction of {txn['Amount'].iloc[0]:.2f}: fraud chance {chance:.0%} "
      f"-> {'BLOCK' if chance >= 0.5 else 'approve'}  (scored in {ms:.1f} ms)")

# ---------------------------------------------------------------- 8. feature importance
importance = pd.Series(booster.feature_importances_, index=X.columns).sort_values()
print("\nTop 10 columns for XGBoost:")
print(importance[::-1].head(10).round(3).to_string())
# Compare with random_forest_fraud.py: the same few columns (V10, V11, V12, V14, V17)
# are at the top for both, but XGBoost leans much harder on just 2-3 of them.
# Boosting is "greedy": once a column works well, later trees keep reusing it.
