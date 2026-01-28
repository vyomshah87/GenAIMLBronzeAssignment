import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.tree import DecisionTreeClassifier


data = {
    'Age': [28, 45, 35, 50, 30, 42, 26, 48, 38, 55],
    'AnnualIncome': [6.5, 12, 8, 15, 7, 10, 5.5, 14, 9, 16],
    'CreditScore': [720, 680, 750, 640, 710, 660, 730, 650, 700, 620],
    'LoanAmount': [5.5, 10, 6.7, 12, 5.5, 9, 4.4, 11, 7.8, 13],
    'LoanTerm': [5, 10, 7, 15, 5, 10, 4, 12, 8, 15],
    'EmploymentType': ['Salaried', 'Self-Employed', 'Salaried', 'Self-Employed', 
                       'Salaried', 'Salaried', 'Salaried', 'Self-Employed', 
                       'Salaried', 'Self-Employed'],
    'Loan': [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
}

df = pd.DataFrame(data)

print("\n Dataset")
print(df)


labelencode = LabelEncoder()
df['EmploymentType_Encoded'] = labelencode.fit_transform(df['EmploymentType'])

X = df[['Age', 'AnnualIncome', 'CreditScore', 'LoanAmount', 'LoanTerm', 'EmploymentType_Encoded']]
y = df['Loan']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
X_scaled = scaler.fit_transform(X)

k_range = range(1, 11)
k_scores = []

for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, X_train_scaled, y_train, cv=3, scoring='accuracy')
    k_scores.append(scores.mean())

optimal_k = k_range[np.argmax(k_scores)]

knn_model = KNeighborsClassifier(n_neighbors=optimal_k)
knn_model.fit(X_train_scaled, y_train)

y_pred_knn = knn_model.predict(X_test_scaled)
y_pred_knn_train = knn_model.predict(X_train_scaled)
y_pred_knn_all = knn_model.predict(X_scaled)
y_pred_proba_knn = knn_model.predict_proba(X_test_scaled)[:, 1]


print(f"\n Classification Report:")
print(classification_report(y_test, y_pred_knn, target_names=['No Default', 'Default'], zero_division=0))

print(f"\nConfusion Matrix:")
cm_knn = confusion_matrix(y_test, y_pred_knn)
print(cm_knn)

dt_model = DecisionTreeClassifier(random_state=42, max_depth=3)
dt_model.fit(X_train_scaled, y_train)
y_pred_dt = dt_model.predict(X_test_scaled)
y_pred_dt_train = dt_model.predict(X_train_scaled)

# 1. Identify high-risk customers
print("\n1. Identify high-risk customers:")
df['Default_Probability'] = knn_model.predict_proba(X_scaled)[:, 1]
df['Risk_Category'] = pd.cut(df['Default_Probability'], 
                              bins=[0, 0.3, 0.7, 1.0], 
                              labels=['Low Risk', 'Medium Risk', 'High Risk'])
high_risk = df[df['Risk_Category'] == 'High Risk']
print(f"\n   High-Risk Customers ({len(high_risk)}):")
if len(high_risk) > 0:
    print(high_risk[['Age', 'AnnualIncome', 'CreditScore', 'LoanAmount', 'Default_Probability', 'Risk_Category']])
else:
    print("   No high-risk customers identified in current data")

# 2. Patterns leading to default
print("\n2. What patterns lead to loan default?")
defaulters = df[df['Loan'] == 1]
non_defaulters = df[df['Loan'] == 0]
print(f"\n   Defaulters Profile:")
print(f"   - Average Credit Score: {defaulters['CreditScore'].mean():.1f}")
print(f"   - Average Annual Income: {defaulters['AnnualIncome'].mean():.1f} lakhs")
print(f"   - Average Loan Amount: {defaulters['LoanAmount'].mean():.1f} lakhs")
print(f"   - Average Age: {defaulters['Age'].mean():.1f} years")
print(f"\n   Non-Defaulters Profile:")
print(f"   - Average Credit Score: {non_defaulters['CreditScore'].mean():.1f}")
print(f"   - Average Annual Income: {non_defaulters['AnnualIncome'].mean():.1f} lakhs")
print(f"   - Average Loan Amount: {non_defaulters['LoanAmount'].mean():.1f} lakhs")
print(f"   - Average Age: {non_defaulters['Age'].mean():.1f} years")

# 3. Credit score and income influence
print("\n3. How do credit score and income influence predictions?")
print(f"   Credit Score Impact:")
print(f"   - Defaulters avg: {defaulters['CreditScore'].mean():.1f}")
print(f"   - Non-defaulters avg: {non_defaulters['CreditScore'].mean():.1f}")
print(f"   - Difference: {non_defaulters['CreditScore'].mean() - defaulters['CreditScore'].mean():.1f} points")
print(f"\n   Annual Income Impact:")
print(f"   - Defaulters avg: {defaulters['AnnualIncome'].mean():.1f} lakhs")
print(f"   - Non-defaulters avg: {non_defaulters['AnnualIncome'].mean():.1f} lakhs")
print(f"   - Difference: {non_defaulters['AnnualIncome'].mean() - defaulters['AnnualIncome'].mean():.1f} lakhs")

# 4. Banking policies
print("\n4. Suggest banking policies based on model output:")
print(f"   - Reject loans for customers with credit score < 650")
print(f"   - Require higher down payment for loan amounts > 10 lakhs")
print(f"   - Offer lower interest rates to customers with credit score > 700")
print(f"   - Implement stricter verification for self-employed applicants")
print(f"   - Consider debt-to-income ratio in approval process")

# 5. KNN vs Decision Tree comparison
print("\n5. Compare KNN with Decision Trees for this problem:")
print(f"   KNN Performance:")
print(f"   - Test Accuracy: {accuracy_score(y_test, y_pred_knn):.4f}")
print(f"   - F1-Score: {f1_score(y_test, y_pred_knn, zero_division=0):.4f}")
print(f"   - Pros: Captures complex patterns, no assumptions about data")
print(f"   - Cons: Sensitive to feature scaling, computationally expensive")
print(f"\n   Decision Tree Performance:")
print(f"   - Test Accuracy: {accuracy_score(y_test, y_pred_dt):.4f}")
print(f"   - F1-Score: {f1_score(y_test, y_pred_dt, zero_division=0):.4f}")
print(f"   - Pros: Interpretable, handles mixed features well")
print(f"   - Cons: Prone to overfitting, sensitive to small variations")

if accuracy_score(y_test, y_pred_knn) > accuracy_score(y_test, y_pred_dt):
    print(f"\n   - Recommendation: KNN performs better for this dataset")
else:
    print(f"\n   - Recommendation: Decision Tree performs better for this dataset")

# 6. LoanAmount dominating distance
print("\n6. What happens if LoanAmount dominates distance calculation?")
print(f"   - Without scaling: LoanAmount (range ~4-16) would dominate")
print(f"   - Credit scores (600-750) would have different scale")
print(f"   - Solution: StandardScaler normalizes all features to same scale")
print(f"   - This ensures fair contribution from all features")

# 7. Real-time loan approval
print("\n7. Should KNN be used in real-time loan approval systems?")
print(f"   Considerations:")
print(f"   Yes, Prediction time for KNN: O(n) - depends on training set size")
print(f"   Yes, Current dataset size: {len(X_train)} samples")
print(f"   Yes, For small datasets (<10,000): KNN is feasible")
print(f"   No, For large datasets (>100,000): Consider faster alternatives")
print(f"\n   Recommendation: Use KNN with optimizations (KD-trees, Ball trees)")
print(f"   Or consider ensemble methods for better real-time performance")

# Sample prediction

print("\n *******Example********")

sample_customer = pd.DataFrame({
    'Age': [40],
    'AnnualIncome': [9],
    'CreditScore': [675],
    'LoanAmount': [8],
    'LoanTerm': [7],
    'EmploymentType_Encoded': [0]
})
sample_scaled = scaler.transform(sample_customer)
prediction = knn_model.predict(sample_scaled)[0]
probability = knn_model.predict_proba(sample_scaled)[0]

print(f"\nSample Customer Profile:")
print(f"  - Age: 40 years")
print(f"  - Annual Income: 9 lakhs")
print(f"  - Credit Score: 675")
print(f"  - Loan Amount: 8 lakhs")
print(f"  - Loan Term: 7 years")
print(f"  - Employment: Salaried")
print(f"\n Prediction: {'DEFAULT' if prediction == 1 else 'NO DEFAULT'}")
print(f" Probability of Default: {probability[1]:.2%}")
print(f" Probability of No Default: {probability[0]:.2%}")

if probability[1] > 0.7:
    print(f" Risk Level: HIGH RISK")
elif probability[1] > 0.3:
    print(f" Risk Level: MEDIUM RISK")
else:
    print(f" Risk Level: LOW RISK")
