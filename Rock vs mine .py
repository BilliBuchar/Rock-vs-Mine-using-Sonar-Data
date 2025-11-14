import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, auc

# Load dataset
data = pd.read_csv(r"C:\Users\MANGALAM TRIPATHI\Desktop\python\Sonar data.csv", header=None)

# Split features and label
X = data.drop(columns=60, axis=1)
Y = data[60]

# Train-test split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.10, stratify=Y, random_state=1)

# Build SVM model
model = SVC(kernel='rbf', probability=True)
model.fit(X_train, Y_train)

# Predictions
train_pred = model.predict(X_train)
test_pred = model.predict(X_test)

# Accuracy scores
print("\nTraining Accuracy :", accuracy_score(Y_train, train_pred))
print("Testing Accuracy  :", accuracy_score(Y_test, test_pred))

# Confusion matrix
print("\nConfusion Matrix:\n")
cm = confusion_matrix(Y_test, test_pred)
print(cm)

# Classification report
print("\nClassification Report:\n")
print(classification_report(Y_test, test_pred))

# ROC curve
y_scores = model.predict_proba(X_test)[:, 1]
fpr, tpr, thresh = roc_curve((Y_test == 'M').astype(int), y_scores)
roc_auc = auc(fpr, tpr)

# Plot ROC
plt.figure(figsize=(7,5))
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
plt.plot([0,1], [0,1], linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.grid(True)
plt.show()

# Input function
def get_user_input():
    print("\nEnter 60 comma-separated values:")
    while True:
        try:
            raw = input("Values: ").strip()
            vals = [float(x) for x in raw.split(",")]
            if len(vals) != 60:
                print("Error: Exactly 60 values required.")
                continue
            return np.asarray(vals).reshape(1, -1)
        except:
            print("Error: Only numeric values allowed.")

# Prediction loop
while True:
    ask = input("\nDo you want to test new input? (y/n): ").lower()
    if ask != "y":
        print("\nExiting program.")
        break

    user_values = get_user_input()
    prediction = model.predict(user_values)[0]

    print("\nPrediction:", prediction)
    if prediction == "R":
        print("Classified as: ROCK")
    else:
        print("Classified as: MINE")
