# Real project: SMS Spam Classifier with Logistic Regression
#
# Data: "SMS Spam Collection" - 5,572 REAL text messages, each labelled by humans as
#   ham  = normal message ("Are we meeting at 6?")
#   spam = junk / scam    ("WINNER!! Claim your FREE prize now, call 0900...")
#
# Problem: a model can only work with NUMBERS, but our data is TEXT.
# Solution: turn each message into numbers based on which words it contains
# (TfidfVectorizer), then Logistic Regression learns which words signal spam.

# ---------------------------------------------------------------- imports
from pathlib import Path   # handy way to work with file and folder paths

import pandas as pd

from sklearn.model_selection import train_test_split

# TfidfVectorizer: converts text into numbers (explained in step 4)
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

# classification_report: precision / recall / f1 for each class (explained in step 6)
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# ---------------------------------------------------------------- 1. load the data
URL = "https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv"

# Save a copy next to this project (ml/data/sms_spam.tsv) so we download only once.
# Path(__file__) = this file; .parent = ml/algos; .parent.parent = ml
DATA_FILE = Path(__file__).parent.parent / "data" / "sms_spam.tsv"

if DATA_FILE.exists():
    # Already downloaded before -> read the local copy (fast, works offline)
    df = pd.read_csv(DATA_FILE, sep="\t")
else:
    # First run -> download from the internet.
    # sep="\t"   -> columns are separated by TAB (a .tsv file, not commas)
    # header=None -> the file has no header row, so we give column names ourselves
    df = pd.read_csv(URL, sep="\t", header=None, names=["label", "message"])
    DATA_FILE.parent.mkdir(exist_ok=True)       # create ml/data folder if missing
    df.to_csv(DATA_FILE, sep="\t", index=False)  # save it (index=False -> no row-number column)

# ---------------------------------------------------------------- 2. explore the data
print("Rows, columns:", df.shape)
print(df.head(), "\n")

# How many of each class? Spam is only ~13% -> the data is IMBALANCED (important later!)
print(df["label"].value_counts(), "\n")

# Read a few real spam messages to get a feel for them
print("Sample spam messages:")
for msg in df[df["label"] == "spam"]["message"].head(3):
    print("  -", msg)
print()

# ---------------------------------------------------------------- 3. features, target, split
# Models need numbers: map the text labels to 0 / 1
#   .map({...}) replaces each value using the dictionary: "ham" -> 0, "spam" -> 1
X = df["message"]                                # input: the raw text (one column)
y = df["label"].map({"ham": 0, "spam": 1})       # answer: 0 = ham, 1 = spam

# 80% train, 20% test.
# stratify=y -> keep the SAME spam/ham ratio (about 13% spam) in both train and test.
#   Without it, by bad luck the test set might get too few spam messages.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Training messages: {len(X_train)}, Test messages: {len(X_test)}\n")

# ---------------------------------------------------------------- 4. build & train the model
# Step A - TfidfVectorizer turns every message into a long row of numbers:
#   - It first builds a VOCABULARY: every word seen in the training messages (~7,000 words).
#     Each word becomes one column.
#   - For each message, a word's column gets a number:
#       0                -> word not in this message
#       higher number    -> word is in the message AND is rare/meaningful overall
#       smaller number   -> word is very common everywhere (less useful)
#     (TF-IDF = "Term Frequency - Inverse Document Frequency": in plain words,
#      "words that are frequent in THIS message but rare in OTHER messages matter most")
#   - lowercase=True      -> "FREE", "Free" and "free" count as the same word
#   - stop_words="english" -> ignore filler words like "the", "is", "and"
#
# Step B - LogisticRegression learns one importance per word (per column):
#   positive importance -> word pushes towards SPAM ("free", "prize", "claim")
#   negative importance -> word pushes towards HAM  ("ok", "home", "later")
#   max_iter=1000 -> allow more learning steps (needed with thousands of columns)
model = make_pipeline(
    TfidfVectorizer(lowercase=True, stop_words="english"),
    LogisticRegression(max_iter=1000),
)

# .fit() -> learn the vocabulary from training text, convert it to numbers,
# then learn each word's importance. All 3 things happen in this one line.
model.fit(X_train, y_train)

vocab_size = len(model[0].vocabulary_)   # model[0] = the TfidfVectorizer step
print(f"Vocabulary learned: {vocab_size} words (= {vocab_size} feature columns!)\n")

# ---------------------------------------------------------------- 5. evaluate on the test set
preds = model.predict(X_test)

# Baseline: a "dumb" model that always says HAM would already be right ~87% of the time
# (because 87% of messages are ham). Our model must beat this to be useful!
baseline = (y_test == 0).mean()
print(f"Dumb baseline (always say 'ham'): {baseline:.1%}")
print(f"Our model accuracy:               {accuracy_score(y_test, preds):.1%}\n")

# Confusion matrix (rows = actual, columns = predicted):
#                 predicted HAM          predicted SPAM
#   actual HAM    correct                FALSE ALARM (real message sent to spam!)
#   actual SPAM   MISSED spam            correct
print("Confusion matrix:")
print(confusion_matrix(y_test, preds), "\n")

# ---------------------------------------------------------------- 6. precision & recall
# Accuracy alone is misleading when one class is rare, so we look at:
#   precision (spam) -> "When the model says SPAM, how often is it right?"
#                       High = few false alarms (good messages rarely lost)
#   recall (spam)    -> "Of all REAL spam, how much did the model catch?"
#                       High = few missed spam
#   f1-score         -> one number that balances precision and recall
#   support          -> how many test messages of that class exist
print(classification_report(y_test, preds, target_names=["ham", "spam"]))

# ---------------------------------------------------------------- 7. which words mean spam?
# Pair every word with its learned importance, then sort.
#   get_feature_names_out() -> list of all vocabulary words (column names)
#   model[-1].coef_[0]      -> importance of each word (same order)
words = model[0].get_feature_names_out()
importance = pd.Series(model[-1].coef_[0], index=words).sort_values()

print("Top 15 SPAM words (biggest positive importance):")
print(importance.tail(15)[::-1].round(2).to_string(), "\n")   # [::-1] = reverse order
print("Top 15 HAM words (biggest negative importance):")
print(importance.head(15).round(2).to_string(), "\n")

# ---------------------------------------------------------------- 8. look at the mistakes
# Learning from mistakes is how you improve a model
test_results = pd.DataFrame({"message": X_test, "actual": y_test, "predicted": preds})

missed = test_results[(test_results["actual"] == 1) & (test_results["predicted"] == 0)]
print(f"Missed spam ({len(missed)}) - first 3:")
for msg in missed["message"].head(3):
    print("  -", msg)

false_alarms = test_results[(test_results["actual"] == 0) & (test_results["predicted"] == 1)]
print(f"\nFalse alarms ({len(false_alarms)}) - good messages marked as spam:")
for msg in false_alarms["message"].head(3):
    print("  -", msg)
print()

# ---------------------------------------------------------------- 9. try your own messages
my_messages = [
    "Hey, are we still meeting for lunch tomorrow?",
    "Congratulations! You have WON a free iPhone. Call now to claim your prize!",
    "URGENT: your account has been suspended. Click the link to verify.",
    "Can you pick up milk on your way home?",
    "Get 50% off all items today only. Text STOP to unsubscribe.",
]
# predict_proba(...)[:, 1] -> chance of being spam (column 1 = class "spam")
spam_chances = model.predict_proba(my_messages)[:, 1]
for msg, chance in zip(my_messages, spam_chances):
    verdict = "SPAM" if chance >= 0.5 else "ham "
    print(f"{verdict} ({chance:.0%})  {msg}")



