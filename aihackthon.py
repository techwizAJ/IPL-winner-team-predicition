# -*- coding: utf-8 -*-
"""
Created on Sun Apr 22 10:16:21 2018

@author: techwiz
"""

import pandas as pd

""" Data Preprocessing """
train_matches = pd.read_csv("Trainmatches.csv")
test_matches = pd.read_csv("Testmatches.csv")

#train_matches.isnull().values.any()
#train_matches.isnull().sum().sum()
""" Only one row with na so dropping it"""
train_matches.dropna(inplace=True)
#train_matches.isnull().values.any()
#train_matches.isnull().sum().sum()

# Combining Train and test data for further encoding
combined = pd.concat([train_matches,test_matches],join='inner')
#combined.drop('season',axis=1,inplace=True)
combined.drop('result',axis=1,inplace=True)
combined.drop('dl_applied',axis=1,inplace=True)
#combined = pd.concat([combined,train_matches["winner"]],axis=1)
combined['city'].value_counts()
combined['venue'].value_counts()
combined['season'].value_counts()
train_matches['winner'].value_counts()

""" Encoding values manaually """
# Manual Encoding for teams , city , stadiums and toss decisions
cleanUpEncoding = { "season" : {2017: 10,2016:9,2015:8,2014:7,2013:6,2012:5,2011:4,
                                2010:3,2009:2 ,2008:1},
        "toss_decision": {"bat": 0 , "field": 1} ,
                     "team1": {"Team1":1, "Team2":2, "Team3":3, "Team4":4,
                             "Team5":5, "Team6":6,"Team7":7,"Team8":8,"Team9":9,"Team10":10,"Team11":11},
                             "team2": {"Team1":1, "Team2":2, "Team3":3, "Team4":4,
                             "Team5":5, "Team6":6,"Team7":7,"Team8":8,"Team9":9,"Team10":10,"Team11":11},
                                "toss_winner": {"Team1":1, "Team2":2, "Team3":3, "Team4":4,
                             "Team5":5, "Team6":6,"Team7":7,"Team8":8,"Team9":9,"Team10":10,"Team11":11} ,
                                "city": { "City1":1,"City2":2,"City3":3,"City4":4,"City5":5,"City6":6,"City7":7,"City8":8,"City9":9,"City10":10,"City11":11,"City12":12,"City13":13,"City14":14,"City15":15,"City16":16,
                                         "City17":17,"City18":18,"City19":19,"City20":20,"City21":21,"City22":22,"City23":23,"City24":24,"City25":25,"City26":26,"City27":27,"City28":28,"City29":29,"City30":30,
                                         "City31":1,"City32":32},
                                "venue": { "Stadium1":1,"Stadium2":2,"Stadium3":3,"Stadium4":4,"Stadium5":5,"Stadium6":6,"Stadium7":7,"Stadium8":8,"Stadium9":9,"Stadium10":10,"Stadium11":11,"Stadium12":12,"Stadium13":13,"Stadium14":14 ,"Stadium15":15,"Stadium16":16,"Stadium17":17,"Stadium18":18,"Stadium19":1,"Stadium20":20,"Stadium21":21,"Stadium22":22,"Stadium23":23,"Stadium24":24,"Stadium25":25,"Stadium26":26,"Stadium27":27,
                                          "Stadium28":28,"Stadium29":29,"Stadium30":30,"Stadium31":31,"Stadium32":32,"Stadium33":33,"Stadium34":34,"stadium35":35 } }
combined.replace(cleanUpEncoding, inplace=True)
dependedVariableEncoding = { "winner": {"Team1":1, "Team2":2, "Team3":3, "Team4":4,
                             "Team5":5, "Team6":6,"Team7":7,"Team8":8,"Team9":9,"Team10":10,"Team11":11}}
checkEncoding = { "team1": {"Team1":1, "Team2":2, "Team3":3, "Team4":4,
                             "Team5":5, "Team6":6,"Team7":7,"Team8":8,"Team9":9,"Team10":10,"Team11":11},
                             "team2": {"Team1":1, "Team2":2, "Team3":3, "Team4":4,
                             "Team5":5, "Team6":6,"Team7":7,"Team8":8,"Team9":9,"Team10":10,"Team11":11}
        }
train_matches.replace(dependedVariableEncoding, inplace=True)
test_matches.replace(checkEncoding, inplace=True)

""" Feature Selection  """
X = combined.iloc[:,:].values
y = train_matches.iloc[:,9].values

X_train = X[0:499,:]
X_test = X[499:,:]

""" Feature Scaling """
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
#sc_Y = StandardScaler()
#y = sc_Y.fit_transform(y)

""" Now Appling Various ML Models For Classification """

from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier(n_estimators=1000,min_samples_split=30,min_samples_leaf=4,random_state=42,warm_start=True)
clf.fit(X_train,y)
y_pred = clf.predict(X_test)

"""
#Testing accuracy using cross_validiation techniques

from sklearn.cross_validation import train_test_split
x_train , x_test , y_train , y_test = train_test_split(X_train,y,test_size=0.2,random_state=42)
clf_new  = RandomForestClassifier(n_estimators=1000,min_samples_split=30,min_samples_leaf=4,random_state=42,warm_start=True)
clf_new.fit(x_train,y_train)
y_pred_new = clf_new.predict(x_test)
from sklearn.metrics import confusion_matrix
cnf = confusion_matrix(y_test,y_pred_new)
from sklearn.metrics import accuracy_score
acu = accuracy_score(y_test,y_pred_new)
"""
team = test_matches.iloc[:,3:5].values
newsub = []
sub = pd.read_csv("submission.csv")
count =0
for i in range(0,136):
    if(team[i][0] == y_pred[i]):
        #sub[i]["team_1_win_flag"] = 1
        newsub.append(1)
        
    else:
        newsub.append(0)

sub["team_1_win_flag"] = newsub
sub.to_csv('submissionsNEW1.csv',index=False)
final = pd.read_csv('submissionsNEW1.csv')