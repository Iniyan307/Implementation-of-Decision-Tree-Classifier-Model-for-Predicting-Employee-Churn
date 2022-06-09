# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Moodle-Code Runner

## Algorithm
1. Import the standard Libraries.
2. Import LabelEncoder and fit_transform "Salary".
3. Assign values for x and y.
4. Import test_train_split from sklearn and assign values.
5. Import DecisionTreeClassifier and predict x_test
6. Find the accuracy and predict the given values.

## Program:
```
/*
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by:  Iniyan S
RegisterNumber:  212220040053
*/

import pandas as pd
data = pd.read_csv("/content/sample_data/Employee.csv")

data.head()

data.info()

data.isnull().sum()

data["left"].value_counts()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data["salary"] = le.fit_transform(data["salary"])
data.head()

x = data[["satisfaction_level","last_evaluation","number_project","average_montly_hours","time_spend_company","Work_accident","promotion_last_5years","salary"]]
x.head()

y = data["left"]

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=100)

from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier(criterion="entropy")
dt.fit(x_train,y_train)
y_pred = dt.predict(x_test)

from sklearn import metrics
accuracy = metrics.accuracy_score(y_test,y_pred)
accuracy

dt.predict([[0.5,0.8,9,260,6,0,1,2]])

```

## Output:
### data.head():

![OP1](/OP1.png)

### data.info():

![OP2](/OP2.png)

### data.isnull().sum():

![OP3](/OP3.png)

### data["left"].value_counts():

![OP4](/OP4.png)

### Label Encoded Salary:

![OP5](/OP5.png)

### x.head():

![OP6](/OP6.png)

### Accuracy:

![OP7](/OP7.png)

### dt.predict():

![OP8](/OP8.png)


## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
