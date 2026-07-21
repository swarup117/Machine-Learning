# ==========================================
# 1. Import Libraries
# ==========================================

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neighbors import KNeighborsClassifier

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

    "Age":[
        25,32,45,29,38,50,41,27,36,48,
        30,44,52,33,40,28,46,35,39,31,
        43,26,37,49,34,42,29,47,38,30,
        51,36,45,27,41,33,39,28,44,35,
        50,32,40,29,46,37,31,48,34,43
    ],

    "Income":[
        30000,50000,80000,45000,70000,90000,75000,35000,65000,85000,
        48000,78000,95000,52000,72000,40000,82000,60000,68000,50000,
        76000,32000,67000,88000,55000,79000,43000,84000,71000,49000,
        92000,64000,81000,37000,74000,53000,69000,41000,80000,61000,
        93000,51000,73000,44000,83000,66000,47000,87000,56000,77000
    ],

    "CreditScore":[
        620,700,780,690,760,810,740,640,730,790,
        680,770,820,710,750,650,800,720,740,700,
        780,630,735,805,705,775,670,790,755,690,
        815,725,785,645,760,715,745,660,780,730,
        820,700,750,675,795,740,685,810,710,770
    ],

    "LoanAmount":[
        100000,150000,200000,120000,180000,250000,190000,110000,170000,220000,
        130000,210000,260000,140000,185000,105000,230000,160000,175000,145000,
        205000,115000,172000,240000,150000,215000,125000,225000,180000,135000,
        255000,168000,208000,108000,195000,148000,178000,112000,218000,165000,
        248000,142000,188000,118000,235000,174000,132000,228000,155000,212000
    ],

    "Employment":[
        "Yes","Yes","Yes","No","Yes","Yes","Yes","No","Yes","Yes",
        "No","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","No",
        "Yes","No","Yes","Yes","Yes","Yes","No","Yes","Yes","No",
        "Yes","Yes","Yes","No","Yes","Yes","Yes","No","Yes","Yes",
        "Yes","No","Yes","No","Yes","Yes","No","Yes","Yes","Yes"
    ],

    "OwnHouse":[
        "No","Yes","Yes","No","Yes","Yes","Yes","No","Yes","Yes",
        "No","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","No",
        "Yes","No","Yes","Yes","Yes","Yes","No","Yes","Yes","No",
        "Yes","Yes","Yes","No","Yes","Yes","Yes","No","Yes","Yes",
        "Yes","No","Yes","No","Yes","Yes","No","Yes","Yes","Yes"
    ],

    "LoanApproved":[
        0,1,1,0,1,1,1,0,1,1,
        0,1,1,1,1,0,1,1,1,0,
        1,0,1,1,1,1,0,1,1,0,
        1,1,1,0,1,1,1,0,1,1,
        1,0,1,0,1,1,0,1,1,1
    ]

})

print(df.head())

# ==========================================
# 3. Encode Categorical Variables
# ==========================================

encoder = LabelEncoder()

df["Employment"] = encoder.fit_transform(df["Employment"])

df["OwnHouse"] = encoder.fit_transform(df["OwnHouse"])

# ==========================================
# 4. Separate Features and Target
# ==========================================

X = df.drop("LoanApproved", axis=1)

y = df["LoanApproved"]

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
# 7. Create KNN Model
# ==========================================

model = KNeighborsClassifier(

    n_neighbors=5

)

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

    display_labels=["Rejected", "Approved"]

)

display.plot()

plt.title("Confusion Matrix")

plt.show()

# ==========================================
# 12. Predict New Customer
# ==========================================

new_customer = pd.DataFrame({

    "Age":[35],

    "Income":[70000],

    "CreditScore":[750],

    "LoanAmount":[180000],

    "Employment":[1],

    "OwnHouse":[1]

})

new_customer = scaler.transform(new_customer)
prediction = model.predict(new_customer)

if prediction[0] == 1:

    print("\nPrediction : LOAN APPROVED")

else:

    print("\nPrediction : LOAN REJECTED")
    