from email_preprocess import preprocess
from sklearn import tree
from sklearn.metrics import accuracy_score


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()
print (len(features_train[0]))



#########################################################
### your code goes here ###
###clf1 = tree.DecisionTreeClassifier(min_samples_split=2)
clf2 = tree.DecisionTreeClassifier(min_samples_split=40)

###clf1.fit(features_train, labels_train)
clf2.fit(features_train, labels_train)

###pred1 = clf1.predict(features_test)
pred2 = clf2.predict(features_test)


###acc_min_samples_split_2 = accuracy_score(pred1, labels_test)
acc = accuracy_score(pred2, labels_test)
print acc

#########################################################




