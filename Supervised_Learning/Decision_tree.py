# practice project of decision tree algorithm

# =====================================================
# 1. IMPORT LIBRARIES
# =====================================================

import random
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
# =====================================================
# 8. IMPORT LIBRARIES
# =====================================================

from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

# Make the random values reproducible
random.seed(42)

# =====================================================
# 2. GENERATE DATASET
# =====================================================

students = []

for i in range(100):

    # -------------------------
    # Numeric Features
    # -------------------------

    study_hours = random.randint(1,10)

    attendance = random.randint(50,100)

    previous_marks = random.randint(35,100)

    assignments = random.randint(0,10)

    sleep_hours = random.randint(4,9)

    family_income = random.randint(20000,150000)

    # -------------------------
    # Categorical Features
    # -------------------------

    internet = random.choice(["Yes","No"])

    extra_class = random.choice(["Yes","No"])

    parent_education = random.choice([
        "School",
        "Bachelor",
        "Master"
    ])

    sports = random.choice(["Yes","No"])

    # -------------------------
    # Create Target Variable
    # -------------------------

    score = 0

    if study_hours >= 6:
        score += 2

    if attendance >= 75:
        score += 2

    if previous_marks >= 60:
        score += 2

    if assignments >= 5:
        score += 1

    if internet == "Yes":
        score += 1

    if extra_class == "Yes":
        score += 1

    if sleep_hours >= 6:
        score += 1

    # Small randomness
    score += random.choice([-1,0,1])

    if score >= 6:
        result = "Pass"
    else:
        result = "Fail"

    students.append([
        study_hours,
        attendance,
        previous_marks,
        assignments,
        sleep_hours,
        family_income,
        internet,
        extra_class,
        parent_education,
        sports,
        result
    ])

# =====================================================
# 3. CREATE DATAFRAME
# =====================================================

df = pd.DataFrame(

    students,

    columns=[
        "StudyHours",
        "Attendance",
        "PreviousMarks",
        "Assignments",
        "SleepHours",
        "FamilyIncome",
        "Internet",
        "ExtraClasses",
        "ParentEducation",
        "Sports",
        "Result"
    ]

)
#------------------------------------------------

print(df.head())



# =====================================================
# 4. EDA
# =====================================================


print("="*60)
print("FIRST 5 ROWS")
print("="*60)

print(df.head())

# --------------------------------------------

print("="*60)
print("LAST 5 ROWS")
print("="*60)

print(df.tail())

# --------------------------------------------

print("="*60)
print("DATASET SHAPE")
print("="*60)

print(df.shape)

# --------------------------------------------

print("="*60)
print("DATA INFORMATION")
print("="*60)

print(df.info())

# --------------------------------------------

print("="*60)
print("STATISTICAL SUMMARY")
print("="*60)

print(df.describe())

# --------------------------------------------

print("="*60)
print("MISSING VALUES")
print("="*60)

print(df.isnull().sum())

# --------------------------------------------

print("="*60)
print("DUPLICATE VALUES")
print("="*60)

print(df.duplicated().sum())

# --------------------------------------------

print("="*60)
print("PASS / FAIL DISTRIBUTION")
print("="*60)

print(df["Result"].value_counts())

# =====================================================
# 5. VISUALIZATION
# =====================================================

#histogram of numeric features
df.hist(
    figsize=(15,10),
    bins=10
)

plt.tight_layout()

plt.show()

#count plot
plt.figure(figsize=(6,4))

sns.countplot(
    x="Result",
    data=df
)

plt.title("Pass vs Fail")

plt.show()

#corelation heatmap
plt.figure(figsize=(8,6))

numeric_df = df.select_dtypes(include=np.number)

sns.heatmap(
    numeric_df.corr(),
    annot=True,
    cmap="coolwarm"
)

plt.show()

#boxplot

numeric_columns = numeric_df.columns

plt.figure(figsize=(18,10))

for i, col in enumerate(numeric_columns):

    plt.subplot(2,3,i+1)

    sns.boxplot(
        y=df[col]
    )

    plt.title(col)

plt.tight_layout()

plt.show()


# =====================================================
#feature engineering
# =====================================================
print(df.isnull().sum())

print(df.duplicated().sum())

df.drop_duplicates(inplace=True)

# --------------------------------------------
df["FamilySupport"] = (
    (df["Internet"] == "Yes").astype(int)
    +
    (df["ExtraClasses"] == "Yes").astype(int)
)

df["StudyEfficiency"] = (
    df["StudyHours"] *
    df["Attendance"]
)

df["AssignmentRatio"] = (
    df["Assignments"] /
    df["StudyHours"]
)

# --------------------------------------------
encoder = LabelEncoder()

df["Internet"] = encoder.fit_transform(df["Internet"])

df["ExtraClasses"] = encoder.fit_transform(df["ExtraClasses"])

df["ParentEducation"] = encoder.fit_transform(df["ParentEducation"])

df["Sports"] = encoder.fit_transform(df["Sports"])

df["Result"] = encoder.fit_transform(df["Result"])

# --------------------------------------------
print(df.head())


# =====================================================
# model training
# =====================================================
X = df.drop("Result", axis=1)

y = df["Result"]

#-------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(

    X,
    y,

    test_size=0.2,

    random_state=42

)

#=====================================================
#crating the decision tree model

model = DecisionTreeClassifier(

    criterion="gini",

    max_depth=4,

    random_state=42

)

model.fit(

    X_train,

    y_train

)

y_pred = model.predict(

    X_test

)
#=====================================================
#model evaluation

comparison = df_compare = X_test.copy()

comparison["Actual"] = y_test.values

comparison["Predicted"] = y_pred

print(comparison.head(10))

accuracy = accuracy_score(

    y_test,

    y_pred

)

precision = precision_score(

    y_test,

    y_pred

)

recall = recall_score(

    y_test,

    y_pred

)

f1 = f1_score(

    y_test,

    y_pred
)

print("="*50)

print("MODEL EVALUATION")

print("="*50)

print("Accuracy :", accuracy)

print("Precision :", precision)

print("Recall :", recall)

print("F1 Score :", f1)

cm = confusion_matrix(

    y_test,

    y_pred

)

print(cm)

cm = confusion_matrix(

    y_test,

    y_pred

)

print(cm)

plt.figure(figsize=(6,5))

sns.heatmap(

    cm,

    annot=True,

    fmt="d",

    cmap="Blues"

)

plt.xlabel("Predicted")

plt.ylabel("Actual")

plt.title("Confusion Matrix")

plt.show()

print(classification_report(

    y_test,

    y_pred

))

#=====================================================
#predicting the result of a new student using the trained model
#=====================================================
# New Student Data
new_student = pd.DataFrame({

    "StudyHours":[8],
    "Attendance":[90],
    "PreviousMarks":[75],
    "Assignments":[9],
    "SleepHours":[7],
    "FamilyIncome":[80000],
    "Internet":[1],
    "ExtraClasses":[1],
    "ParentEducation":[0],
    "Sports":[1],

    # Engineered Features
    "FamilySupport":[2],
    "StudyEfficiency":[720],
    "AssignmentRatio":[1.125]

})

prediction = model.predict(new_student)

if prediction[0] == 1:
    print("Prediction : PASS")
else:
    print("Prediction : FAIL")

probability = model.predict_proba(new_student)

print("Prediction Probability")

print(probability)

from sklearn.tree import plot_tree

plt.figure(figsize=(20,10))

plot_tree(

    model,

    feature_names=X.columns,

    class_names=["Fail","Pass"],

    filled=True,

    rounded=True,

    fontsize=10

)

plt.title("Decision Tree")

plt.show()

importance = pd.DataFrame({

    "Feature":X.columns,

    "Importance":model.feature_importances_

})

importance = importance.sort_values(

    by="Importance",

    ascending=False

)

print(importance)

plt.figure(figsize=(10,6))

sns.barplot(

    data=importance,

    x="Importance",

    y="Feature"

)

plt.title("Feature Importance")

plt.show()

print("="*50)

print("Decision Tree Summary")

print("="*50)

print("Tree Depth :", model.get_depth())

print("Number of Leaves :", model.get_n_leaves())

print("Training Accuracy :", model.score(X_train,y_train))

print("Testing Accuracy :", model.score(X_test,y_test))