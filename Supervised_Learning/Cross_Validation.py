# Cross Validation is a technique used to evaluate the performance of a machine learning model
# by partitioning the data into subsets, training the model on some subsets, and validating it 
# on the remaining subsets. This helps in assessing how well the model generalizes to unseen data.

#cross validation on XGBoost model

#XGBoost 
# ==========================================
# 1. Import Libraries
# ==========================================

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import (train_test_split, cross_val_score)
from sklearn.preprocessing import LabelEncoder

from xgboost import XGBClassifier


# ==========================================
# dataset
# ==========================================


df = pd.DataFrame({

    "EmailLength":[
        320,180,450,210,380,150,500,275,340,160,
        420,195,365,480,230,310,140,390,250,430,
        170,355,465,205,300,155,410,280,335,190,
        460,225,375,145,490,260,325,175,440,240,
        360,200,470,215,305,165,415,285,345,185
    ],

    "NumLinks":[
        5,1,7,0,6,0,8,3,4,1,
        7,2,5,8,1,4,0,6,2,7,
        1,5,8,0,3,0,6,2,4,1,
        8,1,5,0,9,3,4,1,7,2,
        5,1,8,0,3,1,6,2,5,0
    ],

    "NumAttachments":[
        0,1,0,2,0,1,0,1,0,2,
        0,1,0,0,2,1,2,0,1,0,
        2,0,0,1,1,2,0,1,0,2,
        0,2,0,1,0,1,0,2,0,1,
        0,2,0,1,1,2,0,1,0,2
    ],

    "ContainsOffer":[
        "Yes","No","Yes","No","Yes","No","Yes","No","Yes","No",
        "Yes","No","Yes","Yes","No","Yes","No","Yes","No","Yes",
        "No","Yes","Yes","No","Yes","No","Yes","No","Yes","No",
        "Yes","No","Yes","No","Yes","No","Yes","No","Yes","No",
        "Yes","No","Yes","No","Yes","No","Yes","No","Yes","No"
    ],

    "ContainsMoney":[
        "Yes","No","Yes","No","Yes","No","Yes","No","Yes","No",
        "Yes","No","Yes","Yes","No","Yes","No","Yes","No","Yes",
        "No","Yes","Yes","No","Yes","No","Yes","No","Yes","No",
        "Yes","No","Yes","No","Yes","No","Yes","No","Yes","No",
        "Yes","No","Yes","No","Yes","No","Yes","No","Yes","No"
    ],

    "ContainsUrgent":[
        "Yes","No","Yes","No","Yes","No","Yes","No","No","No",
        "Yes","No","Yes","Yes","No","Yes","No","Yes","No","Yes",
        "No","Yes","Yes","No","Yes","No","Yes","No","Yes","No",
        "Yes","No","Yes","No","Yes","No","Yes","No","Yes","No",
        "Yes","No","Yes","No","Yes","No","Yes","No","Yes","No"
    ],

    "NumCapitalWords":[
        18,3,25,2,20,1,30,8,16,2,
        24,5,18,28,4,14,1,22,6,26,
        2,19,27,3,12,1,21,7,15,2,
        29,4,18,1,32,9,14,2,25,5,
        17,3,28,2,13,2,23,8,18,3
    ],

    "Spam":[
        1,0,1,0,1,0,1,0,1,0,
        1,0,1,1,0,1,0,1,0,1,
        0,1,1,0,1,0,1,0,1,0,
        1,0,1,0,1,0,1,0,1,0,
        1,0,1,0,1,0,1,0,1,0
    ]

})

print(df.head())

# ==========================================
# encode categorical variables
# ==========================================

encoder = LabelEncoder()

df["ContainsOffer"] = encoder.fit_transform(df["ContainsOffer"])

df["ContainsMoney"] = encoder.fit_transform(df["ContainsMoney"])

df["ContainsUrgent"] = encoder.fit_transform(df["ContainsUrgent"])


# ==========================================
# prepare features and target
# ==========================================
X = df.drop("Spam", axis=1)

y = df["Spam"]

print(X.head())

print(y.head())

# ==========================================
# train-test split
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(

    X,

    y,

    test_size=0.2,

    random_state=42

)

# ==========================================
# create XGBoost model
# ==========================================

model = XGBClassifier(

    n_estimators=100,

    learning_rate=0.1,

    max_depth=5,

    random_state=42,

    eval_metric="logloss"

)

# ==========================================
# Cross Validation
# ==========================================

scores = cross_val_score(

    model,

    X_train,

    y_train,

    cv=5

)

print("Cross Validation Scores")
print(scores)

print("\nAverage Accuracy")
print(scores.mean())

print("\nStandard Deviation")
print(scores.std())

# ==========================================
# train model
# ==========================================
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("\nActual Values")
print(y_test.values)

print("\nPredicted Values")
print(y_pred)


# ==========================================
# evaluate model performance
# ==========================================

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay
)

accuracy = accuracy_score(y_test, y_pred)

precision = precision_score(y_test, y_pred)

recall = recall_score(y_test, y_pred)

f1 = f1_score(y_test, y_pred)

print("\nModel Evaluation")
print("-------------------------")

print("Accuracy :", accuracy)

print("Precision:", precision)

print("Recall   :", recall)

print("F1 Score :", f1)


# ==========================================
# confusion matrix
# ==========================================

cm = confusion_matrix(y_test, y_pred)

display = ConfusionMatrixDisplay(

    confusion_matrix=cm,

    display_labels=["Not Spam", "Spam"]

)

display.plot()

plt.title("Confusion Matrix")

plt.show()

# ==========================================
#feature importance
# ==========================================

importance = model.feature_importances_

feature_importance = pd.DataFrame({

    "Feature": X.columns,

    "Importance": importance

})

feature_importance = feature_importance.sort_values(

    by="Importance",

    ascending=False

)

print(feature_importance)

plt.figure(figsize=(8,5))

plt.bar(

    feature_importance["Feature"],

    feature_importance["Importance"]

)

plt.xticks(rotation=45)

plt.title("Feature Importance")

plt.xlabel("Features")

plt.ylabel("Importance")

plt.show()


# ==========================================
#prediction
# ==========================================
new_email = pd.DataFrame({

    "EmailLength":[420],

    "NumLinks":[6],

    "NumAttachments":[0],

    "ContainsOffer":[1],

    "ContainsMoney":[1],

    "ContainsUrgent":[1],

    "NumCapitalWords":[25]

})
prediction = model.predict(new_email)

if prediction[0] == 1:

    print("\nPrediction : SPAM EMAIL")

else:

    print("\nPrediction : NOT SPAM")


print("\n xgboost model completed successfully")    