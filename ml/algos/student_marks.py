import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
# LinearRegression is algorithm
# scikit-learn 
# 1. Past data: hours studied and marks scored by 10 students
df = pd.DataFrame({
    "hours": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    "marks": [35, 42, 50, 55, 63, 68, 76, 80, 88, 92],
})

X = df[["hours"]]   # feature (double brackets -> keeps it as a table)
y = df["marks"]     # target we want to predict

# 2. Train: the model looks at the table and learns the pattern
model = LinearRegression().fit(X, y)

# 3. What did it learn? 10.2
print("Each extra hour adds about", round(model.coef_[0], 1), "marks")
print("Base marks with 0 hours  :", round(model.intercept_, 1))

# 4. Predict for new students
new_students = pd.DataFrame({"hours": [5.5, 12]})
for h, m in zip(new_students["hours"], model.predict(new_students)):
    print(f"Studies {h} hrs -> predicted marks: {m:.0f}")

# 5. Picture: dots = real students, line = what the model learned
plt.scatter(df["hours"], df["marks"], label="Real students")
plt.plot(df["hours"], model.predict(X), color="red", label="Model's pattern")
plt.xlabel("Hours studied")
plt.ylabel("Marks")
plt.legend()
plt.show()
