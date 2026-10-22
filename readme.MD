Machine Learning

Traditional: Rules + Data ──► Answers

ML: Data + Answers ──► Rules (the "model")

ML Algorithms + training = AI/ML model

Boy + teach driving 1 = driver  
Boy + teach driving 2 = driver

ML Algorithm

Medicine + Doctor = cure the patient

ML algo + AI/ML developer = solve real life problem

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

Most important ML algorithms
1️⃣ Regression (predict a number)
Algorithm -> Simple idea -> Real-life example -> scikit-learn

Linear Regression -> Finds a steady pattern and combines factors by importance -> House price, salary from experience -> LinearRegression

₹10 ₹20 ₹30 ₹40
2 cho 4 cho 6 ch 8

Student
features (hours_studied, sleep, marks, mobile_last_4, roll, age, bike_no,...)

junk features

Brake / Regulations

Teacher
Essay -> Reasons -> my dog was flying /

20 features(column)

LR - 10 students
training = check features of each student

MAE: Mean Absolute Error
In one line
MAE = on average, how far off were the model's guesses?

It's a score for regression models. Lower is better, and 0 would be perfect.

Ridge / Lasso -> Linear Regression with "brakes", so it doesn't over-trust any single feature -> Same uses, when there are many features Ridge, Lasso

Ridge:- multiples features, decrease importance of all features so importance of junk features becomes almost 0

Lasso:-

Polynomial Regression -> Linear Regression that can follow curves -> Car resale value dropping fast, then slowly -> PolynomialFeatures + LinearRegression

degree = 1,2,3,4,10

column column*column column*column\*column
20 400 8000
40 1600 . . .
60 3600
80 6400
100 10000

2️⃣ Classification (predict a category / yes-no)

Logistic Regression -> Gives a probability ("70% chance of yes") Will this customer buy? Is this loan risky? LogisticRegression
K-Nearest Neighbours (KNN) "Tell me who your neighbours are, and I'll tell you who you are": looks at the most similar past examples Recommending movies similar to ones you liked KNeighborsClassifier
Naive Bayes Counts which words or clues usually appear with each category Spam email filtering MultinomialNB, GaussianNB
Support Vector Machine (SVM) Draws the widest possible gap between two groups Face detection, handwriting recognition SVC
3️⃣ Tree-based (do both regression and classification) 🔥 most used in industry
Algorithm Simple idea Real-life example scikit-learn
Decision Tree ⭐ A flowchart of yes/no questions, like the game "20 Questions" "Income > 50k? → Has a job? → Approve loan" DecisionTreeClassifier / Regressor
Random Forest ⭐ Many decision trees vote together, like asking 100 experts instead of 1 Your patient_classify.py, fraud detection RandomForestClassifier / Regressor
Gradient Boosting (XGBoost, LightGBM, CatBoost) ⭐⭐ Trees built one after another, each fixing the previous one's mistakes Banking, e-commerce, Kaggle winners GradientBoostingClassifier, xgboost library
💡 For normal table data (Excel or CSV style), Random Forest and XGBoost are usually the best performers. They're what companies use most.

4️⃣ Unsupervised (no answers given, so it finds hidden groups)
Algorithm Simple idea Real-life example scikit-learn
K-Means Clustering ⭐ Groups similar items together, like sorting a mixed bag of fruits into piles Grouping customers: "budget", "premium", "occasional" KMeans
Hierarchical Clustering Builds a family tree of groups Grouping similar documents or species AgglomerativeClustering
PCA Compresses many columns into a few important ones, like summarising a long book into key points Reducing 100 features to 10, visualising data PCA
5️⃣ Deep Learning (Neural Networks), later in the course
Used for images, audio, text, and video, which regular algorithms struggle with.

CNN → images (face unlock, X-ray scans)
RNN / LSTM → sequences (older speech and text models)
Transformers → ChatGPT, Claude, Google Translate
These use different libraries (PyTorch, TensorFlow), and we'll get there after the basics.





🌲🌲🌲 Random Forest, point by point
What it is: many Decision Trees that vote together. "Forest" = many trees. "Random" = each tree is built a bit differently.

The problem it fixes: one deep tree memorises the training data (overfits). You saw this with Titanic: 99% train accuracy, only 70% test.

The core idea: the wisdom of the crowd. Ask 1 person to guess the number of sweets in a jar and they may be far off. Ask 200 people and average their guesses, and you get very close. Each person's mistakes cancel out.

Randomness #1: each tree gets a random sample of rows. It picks rows at random with repeats allowed, so a passenger may appear twice and others not at all. This is called bagging (bootstrap sampling).

Randomness #2: each question looks at only a few random columns. If every tree could always use is_male, they'd all look the same. Forcing variety makes the trees different from each other.

Why different trees help: they make different mistakes, and in the vote those mistakes cancel out. Trees that were all the same would just repeat each other's errors.

How it predicts:

Classification: each tree votes, and the majority wins. predict_proba = the average of the trees' chances.
Regression (RandomForestRegressor): it takes the average of the trees' numbers.
Main settings:

Setting	Meaning	Typical value
n_estimators	Number of trees	100–500
max_depth, min_samples_leaf	The brakes for each tree	Tune them
max_features	How many columns each question can look at	The default is fine
n_jobs=-1	Use all CPU cores (faster)	Always set it
More trees never overfit. They only make training slower. Accuracy improves and then levels off.

There's a free test score: the OOB score. Each tree never sees about 1/3 of the rows, so those rows can test it. This gives an honest accuracy without a separate test set.

No scaling needed. It's built from trees, which only ask questions like age ≤ 9.5.

Strengths: high accuracy, hard to overfit, works well on table data with little tuning, and gives feature_importances_.

Weaknesses: you can't read it as one flowchart (200 trees is too many), it's slower and bigger than one tree, and it's not ideal for images or text.

Use cases: fraud detection, loan default risk, medical diagnosis, customer churn, product recommendations, predicting stock or sales demand, and land-use mapping from satellite data.
