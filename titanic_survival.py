import numpy as np
import pandas as pd

#read in inputs from CSVs
data=pd.read_csv("train.csv",sep=",")
test=pd.read_csv('test.csv',sep=',')

data.info()
''' To identify the columns with missing values
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 891 entries, 0 to 890
Data columns (total 12 columns):
PassengerId    891 non-null int64
Survived       891 non-null int64
Pclass         891 non-null int64
Name           891 non-null object
Sex            891 non-null object
Age            714 non-null float64
SibSp          891 non-null int64
Parch          891 non-null int64
Ticket         891 non-null object
Fare           891 non-null float64
Cabin          204 non-null object
Embarked       889 non-null object
dtypes: float64(2), int64(5), object(5)
memory usage: 83.6+ KB
'''

#Converting string data to numeric data
data.loc[data["Sex"]=="female","Sex"] = 0
data.loc[data["Sex"]=="male","Sex"] = 1
test.loc[test["Sex"]=="male","Sex"] = 1
test.loc[test["Sex"]=="female","Sex"] = 0

#Exploring the data
data.describe(include=['O'])
'''
                             Name  Sex    Ticket        Cabin Embarked
count                         891  891       891          204      889
unique                        891    2       681          147        3
top     Graham, Mr. George Edward    1  CA. 2343  C23 C25 C27        S
freq                            1  577         7            4      644
'''

#Cleaning Names to extract the salutation
y=data['Name'].str.split(',',expand=True)[1]
y=y.str.split(expand=True)[0]
data['Name']=y
z=test['Name'].str.split(',',expand=True)[1]
z=z.str.split(expand=True)[0]
test['Name']=z

#Filling missing values in 'Age' in test data
test.loc[(test['Name']=="Mr.") & (test['SibSp']>1) & (test['Parch']==0)& (test['Age'].isnull()),"Age"].median()
test.loc[(test['Name']=="Mr.") & (test['SibSp']>1) & (test['Parch']==0)& (test['Age'].isnull()),"Age"] = 22
test.loc[(test['Name']=="Miss.") & (test['SibSp']>1) & (test['Parch']==0),"Age"].median()
test.loc[(test['Name']=="Miss.") & (test['SibSp']>1) & (test['Parch']==0) & (test['Age'].isnull()),"Age"] = 22
test.loc[(test['Name']=="Miss.") & (test['SibSp']==0) & (test['Parch']==0) & (test['Age'].isnull()),"Age"]=24
test.loc[(test['Name']=="Miss.") & (test['SibSp']>0) & (test['Parch']==0) & (test['Age'].isnull()),"Age"]=22
test.loc[(test['Name']=="Mr.") & (test['SibSp']>0) & (test['Parch']==0)& (test['Age'].isnull()) ,"Age"] = 30
test.loc[(test['Name']=="Mr.") & (test['SibSp']>0) & (test['Parch']>0) & (test['Age'].isnull()),"Age"] = 41
test.loc[(test['Name']=="Mr.") & (test['SibSp']==0) & (test['Parch']==0) & (test['Pclass']==1) & (test['Age'].isnull()),"Age"] = 42
test.loc[(test['Name']=="Mr.") & (test['SibSp']==0) & (test['Parch']==0) & (test['Pclass']==2) & (test['Age'].isnull()),"Age"] = 27
test.loc[(test['Name']=="Mr.") & (test['SibSp']==0) & (test['Parch']==0) & (test['Pclass']==3) & (test['Age'].isnull()),"Age"] = 25
test.loc[(test['Name']=="Mrs.") & (test['SibSp']==0) & (test['Parch']==0) & (test['Age'].isnull()),"Age"] = 38
test.loc[(test['Name']=="Mrs.") & (test['SibSp']==0) & (test['Parch']>0) & (test['Age'].isnull()),"Age"] = 45
test.loc[(test['Name']=="Mrs.") & (test['SibSp']>0) & (test['Parch']==0) & (test['Age'].isnull()),"Age"]=36
test.loc[(test['Name']=="Mrs.") & (test['SibSp']>0) & (test['Parch']>0)& (test['Age'].isnull()) ,"Age"]=30
test.loc[(test['Name']=="Master.") & (test['Age'].isnull()) ,"Age"] =7
test['Age'].median()
test.loc[ (test['Age'].isnull()) ,"Age"] = 26
test.info()
test_cleaned=test

#Filling in missing values in 'Age' in the training data
data.loc[(data['Name']=="Mr.") & (data['SibSp']>1) & (data['Parch']==0)].median()
data.loc[(data['Name']=="Mr.") & (data['SibSp']>1) & (data['Parch']==0)& (data['Age'].isnull()),"Age"]=26
data.loc[(data['Name']=="Miss.") & (data['SibSp']>0) & (data['Parch']>0)& (data['Age'].isnull()),"Age"]=8
data.loc[(data['Name']=="Miss.") & (data['SibSp']==0) & (data['Parch']==0)& (data['Age'].isnull()),"Age"]=26
data.loc[(data['Name']=="Miss.") & (data['SibSp']>0) & (data['Parch']==0) & (data['Age'].isnull()),"Age"] = 23
data.loc[(data['Name']=="Miss.") & (data['SibSp']==0) & (data['Parch']>0) & (data['Age'].isnull()),"Age"]=15.5
data.loc[(data['Name']=="Mr.") & (data['SibSp']>0) & (data['Parch']==0) & (data['Age'].isnull()) ,"Age"]= 29
data.loc[(data['Name']=="Mr.") & (data['SibSp']>0) & (data['Parch']>0)& (data['Age'].isnull()) ,"Age"] = 31
data.loc[(data['Name']=="Mr.") & (data['SibSp']==0) & (data['Parch']==0) & (data['Pclass']==1) & (data['Age'].isnull()),"Age"] = 45
data.loc[(data['Name']=="Mr.") & (data['SibSp']==0) & (data['Parch']==0) & (data['Pclass']==2) & (data['Age'].isnull()),"Age"] = 30
data.loc[(data['Name']=="Mr.") & (data['SibSp']==0) & (data['Parch']==0) & (data['Pclass']==3) & (data['Age'].isnull()),"Age"]=27
data.loc[(data['Name']=="Mrs.") & (data['SibSp']==0) & (data['Parch']==0) & (data['Age'].isnull()),"Age"] = 41
data.loc[(data['Name']=="Mrs.") & (data['SibSp']==0) & (data['Parch']>0) & (data['Age'].isnull()) ,"Age"] = 36.5
data.loc[(data['Name']=="Mrs.") & (data['SibSp']>0) & (data['Parch']==0) & (data['Age'].isnull()),"Age"] = 33
data.loc[(data['Name']=="Master.")  & (data['Age'].isnull()),"Age"] = 3.5
data.loc[(data['Age'].isnull()),"Age"] = 27

#Drop Cabin
data.drop('Cabin',axis=1,inplace=True)
test_cleaned.drop('Cabin',axis=1,inplace=True)

#Fill in few missing 'Embarked' values with median
data.loc[(data['Embarked'].isnull()),"Embarked"]='S'
data_cleaned=data

data.info()
'''
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 891 entries, 0 to 890
Data columns (total 11 columns):
PassengerId    891 non-null int64
Survived       891 non-null int64
Pclass         891 non-null int64
Name           891 non-null object
Sex            891 non-null object
Age            891 non-null float64
SibSp          891 non-null int64
Parch          891 non-null int64
Ticket         891 non-null object
Fare           891 non-null float64
Embarked       891 non-null object
dtypes: float64(2), int64(5), object(4)
memory usage: 76.6+ KB
'''
from sklearn import tree

Y=data_cleaned['Survived']
X=data_cleaned[['PassengerId','Pclass','Name','Sex','Age','SibSp','Parch','Fare','Ticket','Embarked']]
pd.get_dummies(X['Name'])
names=pd.get_dummies(X['Name'])
X=X.join(names)
X.drop('Name',axis=1,inplace=True)

names_test=pd.get_dummies(test_cleaned['Name'])
test_cleaned.info()
test_cleaned=test_cleaned.join(names_test)
test_cleaned.drop('Name',inplace=True,axis=1)
test_cleaned.loc[test_cleaned['Fare'].isnull(),'Fare'] = 3.17

#Categorize Age Data
X.loc[(X['Age']<=16),'Age']=0
X.loc[(X['Age']>16) & (X['Age']<=32),'Age']
X.loc[(X['Age']>16) & (X['Age']<=32),'Age']=1
X.loc[(X['Age']>32) & (X['Age']<=48),'Age']=2
X.loc[(X['Age']>48) & (X['Age']<=64),'Age']=3
X.loc[(X['Age']>64) ,'Age']=4

test_cleaned.loc[(test_cleaned['Age']<=16),'Age']=0
test_cleaned.loc[(test_cleaned['Age']>16) & (test_cleaned['Age']<=32),'Age']=1
test_cleaned.loc[(test_cleaned['Age']>32) & (test_cleaned['Age']<=48),'Age']=2
test_cleaned.loc[(test_cleaned['Age']>48) & (test_cleaned['Age']<=64),'Age']=3
test_cleaned.loc[(test_cleaned['Age']>64),'Age']=4

#Drop a few additional columns of salutation, not present in test data
X=X.drop('Capt.',axis=1)
X=X.drop('Don.',axis=1)
X=X.drop('Jonkheer.',axis=1)
X=X.drop('Lady.',axis=1)
X=X.drop('Major.',axis=1)
X=X.drop('Mlle.',axis=1)
X=X.drop('Mme.',axis=1)
X=X.drop('the',axis=1)
X=X.drop('Sir.',axis=1)
em=pd.get_dummies(X['Embarked'])
X=X.join(em)
X=X.drop('Embarked',axis=1)

emt=pd.get_dummies(test_cleaned['Embarked'])
test_cleaned=test_cleaned.join(emt)
test_cleaned=test_cleaned.drop('Embarked',axis=1)
test_cleaned=test_cleaned.drop('Dona.',axis=1)
X=X.drop('Ticket',axis=1)
test_cleaned=test_cleaned.drop('Ticket',axis=1)

#first model attempted : Decision Tree Classifier
clf = tree.DecisionTreeClassifier()
clf = clf.fit(X,Y)
output=model.predict(test_cleaned)

#Additional feature to determine if person is with family onboard
X['withFamily']=np.where((X['SibSp']>0) | (X['Parch']>0), 1, 0)
test_cleaned['withFamily']=np.where((test_cleaned['SibSp']>0) | (test_cleaned['Parch']>0), 1, 0)

#Random Forest Classifier
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(max_depth=15, random_state=0)
rf.fit(X,Y)
out_values=rf.predict(test_cleaned)
