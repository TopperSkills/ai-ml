
Traditional: Rules + Data ──► Answers
ML: Data + Answers ──► Rules (the "model")
Say you want to detect diabetes. In normal programming you'd write if glucose > 140 and bmi > 30: ... yourself. In ML, you give the computer thousands of patient records along with the known outcomes, and it learns the rules on its own.


Features (X) -> The inputs/columns -> used to predict Glucose, BMI, Age…

Target / Label (y) -> What we want to predict -> Outcome (0/1)

Algorithm -> The learning method (a recipe) -> RandomForestClassifier()

Model -> The algorithm after it has learned from data -> model = algorithm.fit(...)

Training -> The learning step -> .fit()

Prediction / Inference -> Using the model on new data -> .predict()


Algorithm vs Model: 
- an algorithm is like an empty brain with a learning method. 
- A model is that brain after it has studied. 
- The same algorithm trained on different data gives a different model.

The 3 types of ML
1. Supervised Learning: the data has answers (labels).
1.1. Regression predicts a number (house price, temperature)
1.2. Classification predicts a category (diabetic or not, spam or not)

2. Unsupervised Learning: no labels, so it finds hidden structure (e.g., grouping customers into segments)
3. Reinforcement Learning: an agent learns by trial and reward (games, robotics). 

patient file is Supervised → Classification.

The standard ML workflow

1. Get data → 2. Explore (EDA) → 3. Clean/Prepare → 4. Split train/test
   → 5. Choose algorithm → 6. Train (.fit) → 7. Predict → 8. Evaluate → 9. 


Why split into train and test?
Think of a student who memorizes last year's exam answers. They'll score 100% on that exact exam but fail a new one. A model can "memorize" the same way, which is called overfitting. So we hide about 20% of the data (the test set) and only check against it at the end. That gives an honest score.

Why fit the scaler on train only?

X_train_scaled = scaler.fit_transform(X_train) # learn mean/std from TRAIN
X_test_scaled = scaler.transform(X_test) # only APPLY to test
If the scaler learned from the test data, information about the test set would "leak" into training. This mistake is called data leakage

 
 Exercise:- 

Create ml/algos/lesson1_regression.py and try this. It's a regression problem, so you'll see the other kind of supervised learning:

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

data = fetch_california_housing(as_frame=True)
df = data.frame
print(df.head())

X = df.drop(columns=["MedHouseVal"])
y = df["MedHouseVal"] # house value in $100,000s

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression().fit(X_train, y_train)
preds = model.predict(X_test)

print("MAE:", mean_absolute_error(y_test, preds))
# ai-ml
