# Lesson: Random Forest, part 3 - Credit card fraud (a LARGE, real-time use case)
#
#  Here we use REAL card
# transactions from European cardholders: 284,807 rows, of which only 492 are fraud.
#
# This is how banks actually use Random Forest:
#   1) OFFLINE: train the forest once on millions of past transactions (slow, done at night)
#   2) ONLINE:  every time a card is swiped, the saved model scores it in milliseconds
#               -> "approve" or "block / send OTP"
#
# New ideas in this lesson:
#   - imbalanced data: why ACCURACY lies, and PRECISION / RECALL instead
#   - choosing a THRESHOLD (business decision: block more fraud vs annoy fewer customers)
#   - saving the model to a file and using it for real-time predictions

# ---------------------------------------------------------------- imports
import time                        # measure how long training / predicting takes
import pandas as pd
import matplotlib.pyplot as plt    # bar chart of feature importance
import joblib                      # save / load a trained model to a file

# fetch_openml: downloads a dataset from openml.org (only the FIRST time, ~80 MB),
# then reuses the saved copy in ml/data/openml
from sklearn.datasets import fetch_openml

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score

# ---------------------------------------------------------------- 1. load data
# Columns:
#   V1 ... V28 -> details of the transaction (location, merchant, device, time patterns...).
#                 The bank hid the real meaning for privacy (they were transformed with PCA,
#                 an algorithm we'll meet later), so they are just numbers to us.
#   Amount     -> transaction amount
#   Class      -> answer: 1 = fraud, 0 = genuine
print("Loading data (first run downloads ~80 MB)...")
cards = fetch_openml(data_id=1597, as_frame=True, data_home="ml/data", parser="auto")
X = cards.data                     # 29 input columns
y = cards.target.astype(int)       # comes as text "0"/"1" -> make it numbers

print(f"Rows: {X.shape[0]:,}, columns: {X.shape[1]}")
print(f"Genuine: {(y == 0).sum():,}   Fraud: {(y == 1).sum():,}   "
      f"-> only {y.mean():.2%} are fraud\n")

# ---------------------------------------------------------------- 2. why accuracy lies here
# A "lazy model" that ALWAYS says "genuine" never catches a single fraud...
# ...but it is still right 99.83% of the time!
print(f"Lazy model (always 'genuine') accuracy: {(y == 0).mean():.2%}  <- useless, yet looks great")
# So for rare events (fraud, disease, machine failure) we use two better scores:
#   RECALL    = of all REAL frauds, how many did we catch?     (missed fraud = bank loses money)
#   PRECISION = of all transactions we BLOCKED, how many were really fraud?
#               (low precision = many honest customers get their card blocked)

# ---------------------------------------------------------------- 3. split
# stratify=y -> train and test both keep the same 0.17% fraud ratio.
# Without it, the test set could end up with very few frauds just by luck.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y)

print(f"\nTrain: {len(X_train):,} rows ({y_train.sum()} fraud)   "
      f"Test: {len(X_test):,} rows ({y_test.sum()} fraud)")

# ---------------------------------------------------------------- 4. train the forest
# Settings:
#   n_estimators=100   -> 100 trees (enough here; more = slower, only slightly better)
#   min_samples_leaf=2 -> gentle brake: every leaf holds at least 2 transactions
#   max_samples=0.5    -> each tree learns from a random HALF of the rows.
#                         On big data this makes training ~2x faster and the trees
#                         even more different from each other (good for the vote).
#   n_jobs=-1          -> use all CPU cores. On large data this really matters.
forest = RandomForestClassifier(n_estimators=100, min_samples_leaf=2, max_samples=0.5,
                                random_state=42, n_jobs=-1)

print("\nTraining 100 trees on 213k transactions...")
start = time.time()
forest.fit(X_train, y_train)
print(f"Training took {time.time() - start:.1f} seconds")

# ---------------------------------------------------------------- 5. evaluate
pred = forest.predict(X_test)
print(f"\nAccuracy:  {accuracy_score(y_test, pred):.2%}  (remember: lazy model gets 99.83%)")
print(f"Recall:    {recall_score(y_test, pred):.1%}  of real frauds were caught")
print(f"Precision: {precision_score(y_test, pred):.1%}  of blocked transactions were really fraud")

# Confusion matrix: rows = truth, columns = prediction
#                  predicted genuine   predicted fraud
#   real genuine   [ honest, approved   honest, BLOCKED ]  <- annoyed customers
#   real fraud     [ fraud, MISSED      fraud, caught    ]  <- missed = money lost
(tn, fp), (fn, tp) = confusion_matrix(y_test, pred)
print(f"\nOut of {len(y_test):,} test transactions:")
print(f"  Fraud caught:              {tp}")
print(f"  Fraud missed:              {fn}")
print(f"  Honest customers blocked:  {fp}")
print(f"  Honest customers approved: {tn:,}")

# ---------------------------------------------------------------- 6. choose a threshold
# .predict() blocks a transaction when the forest's fraud chance >= 50%.
# But 50% is just a default. The BANK decides:
#   lower threshold  -> catch more fraud, but block more honest customers
#   higher threshold -> fewer complaints, but more fraud slips through
# predict_proba(...)[:, 1] -> fraud chance for every test transaction (average of the 100 trees)
fraud_chance = forest.predict_proba(X_test)[:, 1]

print("\nThreshold  fraud caught  fraud missed  honest blocked")
for threshold in [0.7, 0.5, 0.3, 0.1]:
    blocked = fraud_chance >= threshold
    caught = (blocked & (y_test == 1)).sum()
    missed = (~blocked & (y_test == 1)).sum()
    annoyed = (blocked & (y_test == 0)).sum()
    print(f"  {threshold:>4.0%}     {caught:>8}      {missed:>8}      {annoyed:>8}")
# Expect: going from 50% to 30% catches several more frauds while blocking only a few
# more honest customers. Many banks use a middle zone: low chance -> approve,
# medium -> send an OTP / call the customer, high -> block.

# ---------------------------------------------------------------- 7. save the model
# Training happens once (offline). The trained forest is saved to a file, and the
# live payment server just LOADS it - no retraining for each transaction.
model_path = "ml/models/fraud_forest.joblib"
joblib.dump(forest, model_path, compress=3)   # compress=3 -> smaller file
print(f"\nModel saved to {model_path}")

# ---------------------------------------------------------------- 8. real-time scoring
# Pretend we are the payment server: load the model and score transactions one by one,
# as they arrive. We "replay" some test transactions, including a few real frauds.
live_model = joblib.load(model_path)
live_model.set_params(n_jobs=1)   # one transaction at a time: 1 core is faster (no setup cost)

THRESHOLD = 0.3   # chosen in step 6
incoming = pd.concat([X_test[y_test == 0].head(5), X_test[y_test == 1].head(3)]).sample(
    frac=1, random_state=1)   # mix them up, like a real stream

print(f"\nLive stream (block if fraud chance >= {THRESHOLD:.0%}):")
for row_id, txn in incoming.iterrows():
    start = time.perf_counter()
    chance = live_model.predict_proba(txn.to_frame().T)[0, 1]   # .to_frame().T -> 1 row table
    ms = (time.perf_counter() - start) * 1000
    decision = "BLOCK  " if chance >= THRESHOLD else "approve"
    truth = "fraud" if y_test[row_id] == 1 else "genuine"
    print(f"  amount {txn['Amount']:>8.2f}  fraud chance {chance:>4.0%}  -> {decision}"
          f"  (actually {truth}, scored in {ms:.0f} ms)")
# Each decision takes only a few milliseconds - fast enough to run while the card is swiped.

# ---------------------------------------------------------------- 9. feature importance
# Which (hidden) columns does the forest rely on most to spot fraud?
importance = pd.Series(forest.feature_importances_, index=X.columns).sort_values()
print("\nTop 10 most important columns:")
print(importance[::-1].head(10).round(3).to_string())

importance.tail(15).plot.barh(color="steelblue")   # 15 most important
plt.title("Random Forest (card fraud): which columns matter most?")
plt.xlabel("Importance (all 29 bars add up to 1)")
plt.tight_layout()
plt.show()
