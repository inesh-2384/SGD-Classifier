# SGD-Classifier
## AIM:
To write a program to predict the type of species of the Iris flower using the SGD Classifier.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Start the program and import required libraries (sklearn, pandas, numpy).
2.Load the Iris dataset from scikit-learn.
3.Split dataset into training and testing sets.
4.Train the model using SGDClassifier.
5.Predict species on the test data.
6.Evaluate accuracy and display results.
7.End the program.  

## Program:
```
/*
Program to implement the prediction of iris species using SGD Classifier.
Developed by:  Inesh N
RegisterNumber:  2122232220036 
*/
```

```py 
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load dataset
iris = datasets.load_iris()
X = iris.data
y = iris.target

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train SGD Classifier
sgd_clf = SGDClassifier(max_iter=1000, tol=1e-3, random_state=42)
sgd_clf.fit(X_train, y_train)

# Prediction
y_pred = sgd_clf.predict(X_test)

# Accuracy
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=iris.target_names))

```

## Output:

<img width="544" height="292" alt="image" src="https://github.com/user-attachments/assets/d7f3ab92-6fa5-4386-8066-d2efdafe9d9a" />


## Result:
Thus, the program to implement the prediction of the Iris species using SGD Classifier is written and verified using Python programming.
