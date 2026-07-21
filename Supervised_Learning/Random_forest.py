# Random Forest Implementation

# ==========================================
# 1. Import Libraries
# ==========================================

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
    ConfusionMatrixDisplay
)

# ==========================================
# 2. Create Dataset
# ==========================================

df = pd.DataFrame({

    "Age":[
        25,30,45,35,28,40,50,29,33,42,
        31,27,46,38,36,52,41,26,34,48,
        37,39,43,32,29,44,51,30,35,47,
        28,36,49,33,40,45,27,38,42,31,
        29,50,34,41,37,46,32,39,43,35
    ],

    "AnnualIncome":[
        30000,45000,80000,60000,35000,90000,120000,40000,55000,85000,
        50000,32000,95000,70000,65000,130000,88000,31000,52000,110000,
        68000,75000,98000,48000,39000,92000,125000,46000,61000,105000,
        37000,69000,115000,54000,87000,96000,34000,72000,89000,51000,
        42000,118000,56000,91000,76000,99000,47000,74000,97000,62000
    ],

    "CreditScore":[
        580,620,780,700,590,810,850,610,690,790,
        670,600,820,740,710,840,800,570,680,830,
        720,760,810,650,605,790,845,630,705,825,
        595,715,835,660,805,815,585,730,785,675,
        615,840,695,800,745,820,640,755,810,700
    ],

    "LoanAmount":[
        150000,200000,250000,220000,180000,300000,350000,170000,210000,280000,
        230000,160000,320000,260000,240000,380000,310000,140000,205000,340000,
        255000,270000,330000,195000,175000,290000,360000,185000,225000,345000,
        165000,245000,370000,215000,305000,315000,155000,265000,295000,235000,
        190000,355000,225000,300000,275000,325000,200000,285000,310000,240000
    ],

    "EmploymentYears":[
        2,4,10,7,3,12,18,5,6,11,
        8,2,13,9,7,20,14,1,5,16,
        8,10,15,4,3,11,19,5,6,17,
        2,9,18,5,13,14,2,8,12,6,
        4,16,7,11,9,15,5,10,13,8
    ],

    "ExistingLoans":[
        2,1,0,1,2,0,0,2,1,0,
        1,2,0,1,1,0,0,3,1,0,
        1,0,0,2,2,0,0,2,1,0,
        2,1,0,2,0,0,3,1,0,1,
        2,0,1,0,1,0,2,1,0,1
    ],

    "PropertyOwner":[
        "No","No","Yes","Yes","No","Yes","Yes","No","Yes","Yes",
        "Yes","No","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes",
        "Yes","Yes","Yes","No","No","Yes","Yes","No","Yes","Yes",
        "No","Yes","Yes","No","Yes","Yes","No","Yes","Yes","Yes",
        "No","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes"
    ],

    "Married":[
        "No","Yes","Yes","Yes","No","Yes","Yes","No","Yes","Yes",
        "Yes","No","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes",
        "Yes","Yes","Yes","No","No","Yes","Yes","No","Yes","Yes",
        "No","Yes","Yes","No","Yes","Yes","No","Yes","Yes","Yes",
        "No","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes"
    ],

    "LoanApproved":[
        0,0,1,1,0,1,1,0,1,1,
        1,0,1,1,1,1,1,0,1,1,
        1,1,1,0,0,1,1,0,1,1,
        0,1,1,0,1,1,0,1,1,1,
        0,1,1,1,1,1,0,1,1,1
    ]

})

print(df.head())

# ==========================================
# 3. Encode Categorical Variables
# ==========================================

encoder = LabelEncoder()

df["PropertyOwner"] = encoder.fit_transform(df["PropertyOwner"])

df["Married"] = encoder.fit_transform(df["Married"])

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

print("\nTraining Shape:", X_train.shape)
print("Testing Shape:", X_test.shape)

# ==========================================
# 6. Create Random Forest Model
# ==========================================

model = RandomForestClassifier(
    n_estimators=100,
    criterion="gini",
    max_depth=5,
    random_state=42
)

#we need this when we want to change the parameters of the model. 
# For example, if we want to increase the number of trees in the forest,
#  we can change n_estimators to 200. If we want to change the maximum depth of the trees, 
# we can change max_depth to 10. If we want to change the criterion for splitting, 
# we can change criterion to "entropy".


# model = RandomForestClassifier(

#     n_estimators=200,

#     criterion="gini",

#     max_depth=10,

#     min_samples_split=5,

#     min_samples_leaf=2,

#     max_features="sqrt",

#     bootstrap=True,

#     oob_score=True, (advance option to use out-of-bag samples to estimate the
#                            generalization accuracy of the model)

#     random_state=42

# )

# ==========================================
# 7. Train Model
# ==========================================

model.fit(X_train, y_train)

# ==========================================
# 8. Predict Test Data
# ==========================================

y_pred = model.predict(X_test)

print("\nActual Values")
print(y_test.values)

print("\nPredicted Values")
print(y_pred)

# ==========================================
# 9. Evaluate Model
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
# 10. Confusion Matrix
# ==========================================

cm = confusion_matrix(y_test, y_pred)

display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Fail", "Pass"]
)

display.plot()

plt.title("Confusion Matrix")

plt.show()

# ==========================================
# 11. Feature Importance
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
# 12. Predict New Student
# ==========================================

new_customer = pd.DataFrame({

    "Age":[35],
    "AnnualIncome":[85000],
    "CreditScore":[760],
    "LoanAmount":[250000],
    "EmploymentYears":[8],
    "ExistingLoans":[1],
    "PropertyOwner":[1],
    "Married":[1]

})

prediction = model.predict(new_customer)

if prediction[0] == 1:
    print("\nPrediction : LOAN APPROVED")
else:
    print("\nPrediction : LOAN REJECTED")