# SVM (Support Vector Machine)

# ==========================================
# 1. Import Libraries
# ==========================================

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay
)

# ==========================================
# 2. Create Dataset
# ==========================================

df = pd.DataFrame({

    "TransactionAmount":[
        200,5000,150,12000,300,8000,250,9500,400,7000,
        350,10000,220,11000,450,7500,280,9000,320,8500,
        180,6500,270,13000,500,7200,210,9800,390,8100,
        260,10500,310,11500,420,6900,240,9200,360,8700,
        190,7600,230,12500,410,8300,290,9700,340,8900
    ],

    "TransactionTime":[
        10,2,14,1,11,3,15,2,9,4,
        13,1,16,3,12,2,17,4,10,5,
        11,2,15,1,9,3,14,2,13,4,
        16,1,10,2,12,5,17,3,11,4,
        15,2,13,1,9,3,16,2,12,4
    ],

    "LocationMatch":[
        "Yes","No","Yes","No","Yes","No","Yes","No","Yes","No",
        "Yes","No","Yes","No","Yes","No","Yes","No","Yes","No",
        "Yes","No","Yes","No","Yes","No","Yes","No","Yes","No",
        "Yes","No","Yes","No","Yes","No","Yes","No","Yes","No",
        "Yes","No","Yes","No","Yes","No","Yes","No","Yes","No"
    ],

    "DeviceTrusted":[
        "Yes","No","Yes","No","Yes","No","Yes","No","Yes","No",
        "Yes","No","Yes","No","Yes","No","Yes","No","Yes","No",
        "Yes","No","Yes","No","Yes","No","Yes","No","Yes","No",
        "Yes","No","Yes","No","Yes","No","Yes","No","Yes","No",
        "Yes","No","Yes","No","Yes","No","Yes","No","Yes","No"
    ],

    "PreviousFraud":[
        "No","Yes","No","Yes","No","Yes","No","Yes","No","Yes",
        "No","Yes","No","Yes","No","Yes","No","Yes","No","Yes",
        "No","Yes","No","Yes","No","Yes","No","Yes","No","Yes",
        "No","Yes","No","Yes","No","Yes","No","Yes","No","Yes",
        "No","Yes","No","Yes","No","Yes","No","Yes","No","Yes"
    ],

    "AccountAge":[
        5,1,8,1,6,2,7,1,9,2,
        8,1,10,2,7,1,9,2,6,3,
        8,2,7,1,9,2,6,1,8,2,
        10,1,7,2,8,3,9,2,6,1,
        7,2,8,1,9,2,10,1,6,3
    ],

    "NumTransactionsToday":[
        2,12,1,15,3,10,2,13,1,9,
        2,14,1,16,3,11,2,12,1,10,
        2,9,1,17,3,8,2,15,1,11,
        2,13,1,16,3,9,2,14,1,10,
        2,12,1,15,3,11,2,13,1,9
    ],

    "Fraud":[
        0,1,0,1,0,1,0,1,0,1,
        0,1,0,1,0,1,0,1,0,1,
        0,1,0,1,0,1,0,1,0,1,
        0,1,0,1,0,1,0,1,0,1,
        0,1,0,1,0,1,0,1,0,1
    ]

})

print(df.head())

# ==========================================
# 3. Encode Categorical Variables
# ==========================================

encoder = LabelEncoder()

df["LocationMatch"] = encoder.fit_transform(df["LocationMatch"])

df["DeviceTrusted"] = encoder.fit_transform(df["DeviceTrusted"])

df["PreviousFraud"] = encoder.fit_transform(df["PreviousFraud"])

# ==========================================
# 4. Separate Features and Target
# ==========================================

X = df.drop("Fraud", axis=1)

y = df["Fraud"]

# ==========================================
# 5. Train-Test Split
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(

    X,
    y,
    test_size=0.2,
    random_state=42

)

print("Training Shape :", X_train.shape)

print("Testing Shape :", X_test.shape)

# ==========================================
# 6. Feature Scaling
# ==========================================

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)

X_test = scaler.transform(X_test)

# ==========================================
# 7. Create SVM Model
# ==========================================

model = SVC(

    kernel="rbf",

    C=1,

    gamma="scale",

    random_state=42

)

# model = SVC(
#     kernel="linear",
#     C=1,
#     random_state=42
# )

# model = SVC(
#     kernel="poly"
#     degree=3,
#     C=1,
#     random_state=42
# )


# ==========================================
# 8. Train Model
# ==========================================

model.fit(X_train, y_train)

# ==========================================
# 9. Predict Test Data
# ==========================================

y_pred = model.predict(X_test)

print("\nActual Values")
print(y_test.values)

print("\nPredicted Values")
print(y_pred)

# ==========================================
# 10. Evaluate Model
# ==========================================

accuracy = accuracy_score(y_test, y_pred)

precision = precision_score(y_test, y_pred)

recall = recall_score(y_test, y_pred)

f1 = f1_score(y_test, y_pred)

print("\nModel Evaluation")
print("---------------------------")

print("Accuracy :", accuracy)

print("Precision:", precision)

print("Recall   :", recall)

print("F1 Score :", f1)

# ==========================================
# 11. Confusion Matrix
# ==========================================

cm = confusion_matrix(y_test, y_pred)

display = ConfusionMatrixDisplay(

    confusion_matrix=cm,

    display_labels=["Normal", "Fraud"]

)

display.plot()

plt.title("Confusion Matrix")

plt.show()

# ==========================================
# 12. Predict New Transaction
# ==========================================

new_transaction = pd.DataFrame({

    "TransactionAmount":[9500],

    "TransactionTime":[2],

    "LocationMatch":[0],

    "DeviceTrusted":[0],

    "PreviousFraud":[1],

    "AccountAge":[1],

    "NumTransactionsToday":[14]

})