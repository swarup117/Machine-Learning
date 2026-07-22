# Naive Bayes

# ==========================================
# 1. Import Libraries
# ==========================================

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import GaussianNB

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
        25,42,35,28,50,31,45,29,38,55,
        27,48,36,33,52,30,41,26,47,39,
        34,53,32,44,29,49,37,28,46,40,
        31,54,35,27,51,33,43,30,48,36,
        29,45,38,32,50,34,42,28,47,39
    ],

    "Fever":[
        "Yes","Yes","No","No","Yes","Yes","Yes","No","No","Yes",
        "No","Yes","No","No","Yes","No","Yes","No","Yes","No",
        "No","Yes","No","Yes","No","Yes","No","No","Yes","No",
        "No","Yes","No","No","Yes","No","Yes","No","Yes","No",
        "No","Yes","No","No","Yes","No","Yes","No","Yes","No"
    ],

    "Cough":[
        "Yes","Yes","No","No","Yes","Yes","Yes","No","No","Yes",
        "No","Yes","No","No","Yes","No","Yes","No","Yes","No",
        "No","Yes","No","Yes","No","Yes","No","No","Yes","No",
        "No","Yes","No","No","Yes","No","Yes","No","Yes","No",
        "No","Yes","No","No","Yes","No","Yes","No","Yes","No"
    ],

    "Headache":[
        "Yes","Yes","No","No","Yes","No","Yes","No","No","Yes",
        "No","Yes","No","No","Yes","No","Yes","No","Yes","No",
        "No","Yes","No","Yes","No","Yes","No","No","Yes","No",
        "No","Yes","No","No","Yes","No","Yes","No","Yes","No",
        "No","Yes","No","No","Yes","No","Yes","No","Yes","No"
    ],

    "Fatigue":[
        "Yes","Yes","No","No","Yes","Yes","Yes","No","No","Yes",
        "No","Yes","No","No","Yes","No","Yes","No","Yes","No",
        "No","Yes","No","Yes","No","Yes","No","No","Yes","No",
        "No","Yes","No","No","Yes","No","Yes","No","Yes","No",
        "No","Yes","No","No","Yes","No","Yes","No","Yes","No"
    ],

    "BodyPain":[
        "Yes","Yes","No","No","Yes","Yes","Yes","No","No","Yes",
        "No","Yes","No","No","Yes","No","Yes","No","Yes","No",
        "No","Yes","No","Yes","No","Yes","No","No","Yes","No",
        "No","Yes","No","No","Yes","No","Yes","No","Yes","No",
        "No","Yes","No","No","Yes","No","Yes","No","Yes","No"
    ],

    "Disease":[
        1,1,0,0,1,1,1,0,0,1,
        0,1,0,0,1,0,1,0,1,0,
        0,1,0,1,0,1,0,0,1,0,
        0,1,0,0,1,0,1,0,1,0,
        0,1,0,0,1,0,1,0,1,0
    ]

})

print(df.head())

# ==========================================
# 3. Encode Categorical Variables
# ==========================================

encoder = LabelEncoder()

df["Fever"] = encoder.fit_transform(df["Fever"])

df["Cough"] = encoder.fit_transform(df["Cough"])

df["Headache"] = encoder.fit_transform(df["Headache"])

df["Fatigue"] = encoder.fit_transform(df["Fatigue"])

df["BodyPain"] = encoder.fit_transform(df["BodyPain"])

# ==========================================
# 4. Separate Features and Target
# ==========================================

X = df.drop("Disease", axis=1)

y = df["Disease"]

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
# 6. Create Model
# ==========================================

model = GaussianNB()

# from sklearn.naive_bayes import MultinomialNB

# model = MultinomialNB()


# from sklearn.naive_bayes import BernoulliNB

# model = BernoulliNB()


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

    display_labels=["Healthy", "Disease"]

)

display.plot()

plt.title("Confusion Matrix")

plt.show()

# ==========================================
# 11. Predict New Patient
# ==========================================

new_patient = pd.DataFrame({

    "Age":[40],

    "Fever":[1],

    "Cough":[1],

    "Headache":[1],

    "Fatigue":[1],

    "BodyPain":[1]

})

prediction = model.predict(new_patient)


if prediction[0] == 1:

    print("\nPrediction : DISEASE DETECTED")

else:

    print("\nPrediction : HEALTHY")

probability = model.predict_proba(new_patient)

print(probability)    