# ==========================================
# 1. Import Libraries
# ==========================================

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression

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

    "StudyHours":[
        2,3,5,6,1,4,7,8,2,5,
        6,3,4,5,7,8,2,1,6,5,
        4,3,7,8,2,5,6,4,3,7,
        5,6,1,2,8,7,4,5,6,3,
        2,5,7,8,1,4,6,5,3,7
    ],

    "Attendance":[
        60,65,80,90,50,75,92,95,55,82,
        88,70,78,84,93,96,58,45,89,83,
        77,68,91,97,57,81,87,76,69,94,
        85,90,48,54,98,92,79,84,86,72,
        59,80,93,95,46,74,88,82,67,91
    ],

    "PreviousMarks":[
        45,50,70,80,35,65,88,90,40,72,
        78,55,67,74,89,92,42,30,81,76,
        68,54,90,95,43,71,79,66,53,91,
        75,82,34,39,94,87,69,73,80,56,
        44,70,89,93,32,64,77,74,52,88
    ],

    "Assignments":[
        2,3,8,9,1,6,10,10,2,8,
        9,4,6,8,10,10,2,1,9,8,
        6,4,10,10,2,7,9,6,4,10,
        8,9,1,2,10,10,6,8,9,4,
        2,8,10,10,1,6,9,8,4,10
    ],

    "Internet":[
        "No","Yes","Yes","Yes","No","Yes","Yes","Yes","No","Yes",
        "Yes","Yes","Yes","Yes","Yes","Yes","No","No","Yes","Yes",
        "Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes",
        "Yes","Yes","No","No","Yes","Yes","Yes","Yes","Yes","Yes",
        "No","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes"
    ],

    "ExtraClasses":[
        "No","No","Yes","Yes","No","Yes","Yes","Yes","No","Yes",
        "Yes","No","Yes","Yes","Yes","Yes","No","No","Yes","Yes",
        "Yes","No","Yes","Yes","No","Yes","Yes","Yes","No","Yes",
        "Yes","Yes","No","No","Yes","Yes","Yes","Yes","Yes","No",
        "No","Yes","Yes","Yes","No","Yes","Yes","Yes","No","Yes"
    ],

    "SleepHours":[
        5,6,7,8,5,6,8,8,5,7,
        7,6,6,7,8,8,5,4,7,7,
        6,6,8,8,5,7,7,6,6,8,
        7,7,5,5,8,8,6,7,7,6,
        5,7,8,8,4,6,7,7,6,8
    ],

    "Pass":[
        0,0,1,1,0,1,1,1,0,1,
        1,0,1,1,1,1,0,0,1,1,
        1,0,1,1,0,1,1,1,0,1,
        1,1,0,0,1,1,1,1,1,0,
        0,1,1,1,0,1,1,1,0,1
    ]

})

print(df.head())

# ==========================================
# 3. Encode Categorical Variables
# ==========================================

encoder = LabelEncoder()

df["Internet"] = encoder.fit_transform(df["Internet"])

df["ExtraClasses"] = encoder.fit_transform(df["ExtraClasses"])

# ==========================================
# 4. Separate Features and Target
# ==========================================

X = df.drop("Pass", axis=1)

y = df["Pass"]

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
# 6. Feature Scaling
# ==========================================

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)

X_test = scaler.transform(X_test)

# ==========================================
# 7. Create Model
# ==========================================

model = LogisticRegression()

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
print("----------------------------")
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
    display_labels=["Fail","Pass"]
)

display.plot()

plt.title("Confusion Matrix")

plt.show()

# ==========================================
# 12. Predict New Student
# ==========================================

new_student = pd.DataFrame({
    "StudyHours":[6],
    "Attendance":[90],
    "PreviousMarks":[82],
    "Assignments":[8],
    "Internet":[1],
    "ExtraClasses":[1],
    "SleepHours":[7]
})

new_student = scaler.transform(new_student)

prediction = model.predict(new_student)

if prediction[0] == 1:
    print("\nPrediction : PASS")
else:
    print("\nPrediction : FAIL")