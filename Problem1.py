import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# Data set
data = {
    'Experience': [2, 5, 1, 8, 4, 10, 3, 6, 7, 2],
    'Training_Hours': [40, 60, 20, 80, 50, 90, 30, 70, 75, 25],
    'Working_Hours': [38, 42, 35, 45, 40, 48, 37, 44, 46, 36],
    'Projects': [3, 6, 2, 8, 5, 9, 4, 7, 7, 3],
    'Productivity_Score': [62, 78, 55, 88, 72, 92, 65, 82, 85, 60]
}

df = pd.DataFrame(data)

print("\n Dataset:")
print(df)

X = df[['Experience', 'Training_Hours', 'Working_Hours', 'Projects']]
y = df['Productivity_Score']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)
y_pred_all = model.predict(X)

# coefficients
print(f"\nIntercept: {model.intercept_:.4f}")
print("\nCoefficients:")
for feature, coef in zip(X.columns, model.coef_):
    print(f"  {feature}: {coef:.4f}")

# 1. Which factor most strongly impacts productivity?
coefficients_abs = pd.DataFrame({
    'Feature': X.columns,
    'Coefficient': model.coef_,
    'Absolute_Impact': np.abs(model.coef_)
}).sort_values('Absolute_Impact', ascending=False)

print("\n1. Which factor most strongly impacts productivity?")
print(f"   - {coefficients_abs.iloc[0]['Feature']} (coefficient: {coefficients_abs.iloc[0]['Coefficient']:.4f})")
print("\n   Impact Ranking:")
for idx, row in coefficients_abs.iterrows():
    print(f"   {idx+1}. {row['Feature']}: {row['Coefficient']:.4f}")

# 2. How does training affect productivity?
training_coef = model.coef_[list(X.columns).index('Training_Hours')]
print(f"\n2. How does training affect productivity?")
print(f"   - For each additional training hour, productivity increases by {training_coef:.4f} points")
print(f"   - Training has a {'positive' if training_coef > 0 else 'negative'} impact on productivity")

# 3. Should the company increase training hours or working hours?
training_coef = model.coef_[list(X.columns).index('Training_Hours')]
working_coef = model.coef_[list(X.columns).index('Working_Hours')]
print(f"\n3. Should the company increase training hours or working hours?")
if abs(training_coef) > abs(working_coef):
    print(f"   - Increase TRAINING HOURS (impact: {training_coef:.4f} vs {working_coef:.4f})")
else:
    print(f"   - Increase WORKING HOURS (impact: {working_coef:.4f} vs {training_coef:.4f})")

# 4. What happens if Working Hours increase beyond optimal limits?
print(f"\n4. What happens if Working Hours increase beyond optimal limits?")
print(f"   - Current model shows linear relationship: {working_coef:.4f} per hour")
print(f"   - Note: Linear models don't capture diminishing returns automatically")
print(f"   - In reality, excessive working hours likely decrease productivity (burnout)")

# 5. Can productivity ever decrease with more experience?
exp_coef = model.coef_[list(X.columns).index('Experience')]
print(f"\n5. Can productivity ever decrease with more experience?")
print(f"   - According to this model: {'NO' if exp_coef > 0 else 'YES'}")
print(f"   - Experience coefficient: {exp_coef:.4f}")
print(f"   - Each year of experience changes productivity by {exp_coef:.4f} points")

# 6. How to detect overfitting?
print(f"\n6. How would you detect overfitting in this model?")
print(f"   - Compare train vs test R² scores:")
print(f"     - Train R²: {r2_score(y_train, y_pred_train):.4f}")
print(f"     - Test R²: {r2_score(y_test, y_pred_test):.4f}")
print(f"     - Difference: {abs(r2_score(y_train, y_pred_train) - r2_score(y_test, y_pred_test)):.4f}")
if abs(r2_score(y_train, y_pred_train) - r2_score(y_test, y_pred_test)) > 0.1:
    print(f"   -  Possible overfitting detected (large difference)")
else:
    print(f"   - No significant overfitting detected")

# 7. Suggest one new feature
print(f"\n7. Suggest one new feature to improve prediction accuracy:")
print(f"   - Team size")



# sample prediction
print("\n *******Example*******")
sample_employee = pd.DataFrame({
    'Experience': [6],
    'Training_Hours': [65],
    'Working_Hours': [42],
    'Projects': [5]
})
predicted_score = model.predict(sample_employee)[0]
print(f"\nFor an employee with:")
print(f"  - Experience: 6 years")
print(f"  - Training Hours: 65")
print(f"  - Working Hours: 42")
print(f"  - Projects: 5")
print(f"\n- Predicted Productivity Score: {predicted_score:.2f}")
