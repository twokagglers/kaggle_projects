import numpy as np
import pandas as pd 

train_csv = pd.read_csv("/kaggle/input/titanic/train.csv")
test_csv = pd.read_csv("/kaggle/input/titanic/test.csv")

def extract_features(data):
    #remove PassengerId and Name
    #Pclass: Passenger Class (1 = 1st; 2 = 2nd; 3 = 3rd)
    #SibSp: Number of Siblings/Spouses Aboard
    #Parch: Number of Parents/Children Aboard
    #Embarked: Port of Embarkation (C = Cherbourg; Q = Queenstown; S = Southampton)

    data = data.drop(["Name",'Cabin'], axis = 1)
    #drop these two col because there are too many null values
    #fill missing age with median value
    #data["Age"].fillna(data.groupby("Title")["Age"].transform("median"), inplace=True)
    #fill missing Fare with median value
    data["Fare"].fillna(data.groupby("Pclass")["Fare"].transform("median"), inplace=True)

    sex_mapping = {"male": 0, "female": 1}
    data['Sex'] = data['Sex'].map(sex_mapping)

    data["Age"].fillna(-1, inplace = True)
    data.loc[ data['Age'] <= 16, 'Age'] = [0],
    data.loc[(data['Age'] > 16) & (data['Age'] <= 26), 'Age'] = [1],
    data.loc[(data['Age'] > 26) & (data['Age'] <= 36), 'Age'] = [2],
    data.loc[(data['Age'] > 36) & (data['Age'] <= 62), 'Age'] = [3],
    data.loc[ data['Age'] > 62, 'Age'] = 4

    data['Embarked'] = data['Embarked'].fillna('S')
    embarked_mapping = {"S": 0, "C": 1, "Q": 2}
    data['Embarked'] = data['Embarked'].map(embarked_mapping)

    data.loc[ data['Fare'] <= 17, 'Fare'] = [0],
    data.loc[(data['Fare'] > 17) & (data['Fare'] <= 30), 'Fare'] = [1],
    data.loc[(data['Fare'] > 30) & (data['Fare'] <= 100), 'Fare'] = [2],
    data.loc[data['Fare'] > 100, 'Fare'] = 3

    features_drop = ['Ticket', 'SibSp', 'Parch']
    data = data.drop(features_drop, axis=1)

    return data

train = extract_features(train_csv)
test = extract_features(test_csv)

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold

k_fold = KFold(n_splits=10, shuffle=True, random_state=0)
train_data = train.drop(["PassengerId", 'Survived'], axis=1)
test_data = test.drop(["PassengerId"], axis=1)
target = train['Survived']
clf = RandomForestClassifier()
scoring = 'accuracy'
score = cross_val_score(clf, train_data, target, cv=k_fold, n_jobs=1, scoring=scoring)

clf.fit(train_data, target)

prediction = clf.predict(test_data)
submission = pd.DataFrame({
        "PassengerId": test["PassengerId"],
        "Survived": prediction
    })

submission.to_csv('submission.csv', index=False)
print("finish")
