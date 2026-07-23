# Imbalance dataset with class weights

# ==========================================
# 1. Import Libraries
# ==========================================

import random
import pandas as pd
import matplotlib.pyplot as plt

# Make the random numbers reproducible
random.seed(42)

# ==========================================
# 2. Generate Dataset
# ==========================================

data = []

for i in range(200):

    # ----------------------------
    # Genuine Transaction (90%)
    # ----------------------------
    if i < 180:

        fraud = 0

        amount = random.randint(500, 50000)

        age = random.randint(20, 65)

        transaction_hour = random.randint(6, 22)

        card_type = random.choice(
            ["Debit", "Credit"]
        )

        international = random.choice(
            ["No", "No", "No", "Yes"]
        )

        previous_fraud = random.choice(
            [0,0,0,0,1]
        )

        device_trusted = random.choice(
            ["Yes","Yes","Yes","No"]
        )

        location_match = random.choice(
            ["Yes","Yes","Yes","No"]
        )

        merchant = random.choice(
            [
                "Shopping",
                "Restaurant",
                "Fuel",
                "Medical",
                "Grocery"
            ]
        )

    # ----------------------------
    # Fraud Transaction (10%)
    # ----------------------------
    else:

        fraud = 1

        amount = random.randint(60000, 200000)

        age = random.randint(18, 70)

        transaction_hour = random.choice(
            [0,1,2,3,4,23]
        )

        card_type = random.choice(
            ["Credit","Credit","Debit"]
        )

        international = random.choice(
            ["Yes","Yes","Yes","No"]
        )

        previous_fraud = random.choice(
            [1,1,2,3]
        )

        device_trusted = random.choice(
            ["No","No","Yes"]
        )

        location_match = random.choice(
            ["No","No","Yes"]
        )

        merchant = random.choice(
            [
                "Electronics",
                "Luxury",
                "Jewelry",
                "Online"
            ]
        )

    data.append({

        "TransactionAmount": amount,

        "CustomerAge": age,

        "TransactionHour": transaction_hour,

        "CardType": card_type,

        "International": international,

        "PreviousFrauds": previous_fraud,

        "DeviceTrusted": device_trusted,

        "LocationMatch": location_match,

        "MerchantCategory": merchant,

        "Fraud": fraud

    })

# ==========================================
# 3. Create DataFrame
# ==========================================

df = pd.DataFrame(data)

print(df.head())

print("\nDataset Shape")

print(df.shape)

# ==========================================
# 4. Check Class Distribution
# ==========================================

print("\nFraud Count")

print(df["Fraud"].value_counts())

print("\nFraud Percentage")

print(df["Fraud"].value_counts(normalize=True) * 100)

# ==========================================
# 5. Visualize Class Distribution
# ==========================================

df["Fraud"].value_counts().plot(
    kind="bar"
)

plt.title("Fraud Class Distribution")

plt.xlabel("Fraud")

plt.ylabel("Count")

plt.show()

# ==========================================
# 6. Import Machine Learning Libraries
# ==========================================

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

# ==========================================
# 7. Encode Categorical Variables
# ==========================================

encoder = LabelEncoder()

categorical_columns = [

    "CardType",

    "International",

    "DeviceTrusted",

    "LocationMatch",

    "MerchantCategory"

]

for column in categorical_columns:

    df[column] = encoder.fit_transform(df[column])

print("\nEncoded Dataset")

print(df.head())

# ==========================================
# 8. Separate Features and Target
# ==========================================

X = df.drop("Fraud", axis=1)

y = df["Fraud"]

# ==========================================
# 9. Train-Test Split
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(

    X,

    y,

    test_size=0.2,

    random_state=42,

    stratify=y

)

print("\nTraining Shape :", X_train.shape)

print("Testing Shape  :", X_test.shape)

# ==========================================
# 10. Create Random Forest Model
# ==========================================

model = RandomForestClassifier(

    n_estimators=100,

    random_state=42

)

# ==========================================
# 11. Train Model
# ==========================================

model.fit(X_train, y_train)

print("\nModel Training Completed Successfully!")

# ==========================================
# 12. Import Evaluation Metrics
# ==========================================

from sklearn.metrics import (

    accuracy_score,

    precision_score,

    recall_score,

    f1_score,

    confusion_matrix,

    ConfusionMatrixDisplay,

    classification_report

)

# ==========================================
# 13. Predict Test Data
# ==========================================

y_pred = model.predict(X_test)

print("\nActual Values")

print(y_test.values)

print("\nPredicted Values")

print(y_pred)

# ==========================================
# 14. Model Evaluation
# ==========================================

accuracy = accuracy_score(y_test, y_pred)

precision = precision_score(y_test, y_pred)

recall = recall_score(y_test, y_pred)

f1 = f1_score(y_test, y_pred)

print("\nModel Evaluation")

print("----------------------------")

print("Accuracy :", accuracy)

print("Precision:", precision)

print("Recall   :", recall)

print("F1 Score :", f1)

# ==========================================
# 15. Classification Report
# ==========================================

print("\nClassification Report")

print(classification_report(y_test, y_pred))

# ==========================================
# 16. Confusion Matrix
# ==========================================

cm = confusion_matrix(y_test, y_pred)

display = ConfusionMatrixDisplay(

    confusion_matrix=cm,

    display_labels=["Genuine", "Fraud"]

)

display.plot()

plt.title("Confusion Matrix")

plt.show()

# ==========================================
# 17. Train Balanced Random Forest
# ==========================================

balanced_model = RandomForestClassifier(

    n_estimators=100,

    class_weight="balanced",

    random_state=42

)

balanced_model.fit(X_train, y_train)

# ==========================================
# 18. Predict Again
# ==========================================

balanced_pred = balanced_model.predict(X_test)

# ==========================================
# 19. Evaluate Balanced Model
# ==========================================

print("\nBalanced Random Forest Results")

print("--------------------------------")

print("Accuracy :", accuracy_score(y_test, balanced_pred))

print("Precision:", precision_score(y_test, balanced_pred))

print("Recall   :", recall_score(y_test, balanced_pred))

print("F1 Score :", f1_score(y_test, balanced_pred))

print("\nClassification Report")

print(classification_report(y_test, balanced_pred))

# ==========================================
# 20. Confusion Matrix
# ==========================================

cm = confusion_matrix(y_test, balanced_pred)

display = ConfusionMatrixDisplay(

    confusion_matrix=cm,

    display_labels=["Genuine","Fraud"]

)

display.plot()

plt.title("Balanced Random Forest")

plt.show()

# ==========================================
# 21. Feature Importance
# ==========================================

importance = pd.DataFrame({

    "Feature": X.columns,

    "Importance": balanced_model.feature_importances_

})

importance = importance.sort_values(

    by="Importance",

    ascending=False

)

print("\nFeature Importance")

print(importance)

# ==========================================
# 22. Predict New Transaction
# ==========================================

new_transaction = pd.DataFrame({

    "TransactionAmount":[120000],

    "CustomerAge":[24],

    "TransactionHour":[2],

    "CardType":[0],

    "International":[1],

    "PreviousFrauds":[2],

    "DeviceTrusted":[0],

    "LocationMatch":[0],

    "MerchantCategory":[1]

})

prediction = balanced_model.predict(new_transaction)

if prediction[0] == 1:

    print("\nPrediction : FRAUD TRANSACTION")

else:

    print("\nPrediction : GENUINE TRANSACTION")