# ==========================================
# 1. Import Libraries
# ==========================================

import random
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

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
# NEW IMPORT
# ==========================================

from imblearn.over_sampling import SMOTE

# ==========================================
# 2. Generate Dataset
# ==========================================

random.seed(42)

data = []

for i in range(200):

    # ----------------------------
    # Genuine Transaction (90%)
    # ----------------------------

    if i < 180:

        fraud = 0

        amount = random.randint(500, 50000)

        age = random.randint(20,65)

        transaction_hour = random.randint(6,22)

        card_type = random.choice(["Debit","Credit"])

        international = random.choice(
            ["No","No","No","Yes"]
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

        amount = random.randint(60000,200000)

        age = random.randint(18,70)

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

        "TransactionAmount":amount,

        "CustomerAge":age,

        "TransactionHour":transaction_hour,

        "CardType":card_type,

        "International":international,

        "PreviousFrauds":previous_fraud,

        "DeviceTrusted":device_trusted,

        "LocationMatch":location_match,

        "MerchantCategory":merchant,

        "Fraud":fraud

    })

df = pd.DataFrame(data)

print(df.head())

print("\nDataset Shape")

print(df.shape)

# ==========================================
# 3. Check Class Distribution
# ==========================================

print("\nBefore SMOTE")

print(df["Fraud"].value_counts())

print("\nPercentage")

print(df["Fraud"].value_counts(normalize=True)*100)

df["Fraud"].value_counts().plot(

    kind="bar"

)

plt.title("Class Distribution Before SMOTE")

plt.xlabel("Fraud")

plt.ylabel("Count")

plt.show()

# ==========================================
# 4. Encode Categorical Variables
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

# ==========================================
# 5. Separate Features and Target
# ==========================================

X = df.drop("Fraud", axis=1)

y = df["Fraud"]

# ==========================================
# 6. Train-Test Split
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(

    X,

    y,

    test_size=0.2,

    random_state=42,

    stratify=y

)

# ==========================================
# 7. Check Training Data Before SMOTE
# ==========================================

print("\nTraining Data Before SMOTE")

print(y_train.value_counts())

# ==========================================
# 8. Apply SMOTE
# ==========================================

smote = SMOTE(

    random_state=42

)

X_train_smote, y_train_smote = smote.fit_resample(

    X_train,

    y_train

)

# ==========================================
# 9. Check Training Data After SMOTE
# ==========================================

print("\nTraining Data After SMOTE")

print(y_train_smote.value_counts())

# ==========================================
# 10. Create Random Forest Model
# ==========================================

model = RandomForestClassifier(

    n_estimators=100,

    random_state=42

)

# ==========================================
# 11. Train Model using SMOTE Data
# ==========================================

model.fit(

    X_train_smote,

    y_train_smote

)

print("\nRandom Forest trained successfully using SMOTE data!")

# ==========================================
# 12. Predict Test Data
# ==========================================

y_pred = model.predict(X_test)

print("\nActual Values")

print(y_test.values)

print("\nPredicted Values")

print(y_pred)

# ==========================================
# 13. Model Evaluation
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
# 14. Classification Report
# ==========================================

print("\nClassification Report")

print(classification_report(y_test, y_pred))

# ==========================================
# 15. Confusion Matrix
# ==========================================

cm = confusion_matrix(y_test, y_pred)

display = ConfusionMatrixDisplay(

    confusion_matrix=cm,

    display_labels=["Genuine","Fraud"]

)

display.plot()

plt.title("Confusion Matrix (SMOTE + Random Forest)")

plt.show()

# ==========================================
# 16. Feature Importance
# ==========================================

importance = pd.DataFrame({

    "Feature": X.columns,

    "Importance": model.feature_importances_

})

importance = importance.sort_values(

    by="Importance",

    ascending=False

)

print("\nFeature Importance")

print(importance)

# ==========================================
# 17. Predict New Transaction
# ==========================================

new_transaction = pd.DataFrame({

    "TransactionAmount":[150000],

    "CustomerAge":[25],

    "TransactionHour":[2],

    "CardType":[0],          # Credit/Debit (encoded)

    "International":[1],     # Yes

    "PreviousFrauds":[2],

    "DeviceTrusted":[0],     # No

    "LocationMatch":[0],     # No

    "MerchantCategory":[1]   # Encoded value

})

prediction = model.predict(new_transaction)

if prediction[0] == 1:

    print("\nPrediction : FRAUD TRANSACTION")

else:

    print("\nPrediction : GENUINE TRANSACTION")