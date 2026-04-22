# Simple Machine Learning Example
#
import pandas as pd
from sklearn.linear_model import LinearRegression

# Step 1: Create a simple dataset
data = {
    "YearsExperience": [1, 2, 3, 4, 5],
    "Salary": [30000, 40000, 50000, 60000, 70000]
}

df = pd.DataFrame(data)

X = df[["YearsExperience"]]
y = df["Salary"]

model = LinearRegression()

model.fit(X, y)

experience = [[6]]
predicted_salary = model.predict(experience)

print("Predicted salary for 6 years experience:", predicted_salary[0])