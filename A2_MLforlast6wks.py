
# Import all libraries
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold
import os



# Ignore future warnings
from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)

os.chdir(r'C:\Users\selin\5318_project2\Assignment2Data')

X = np.load('X_train.npy')
y = np.load('y_train.npy')
X_test = np.load('X_test.npy')
y_test = np.load('y_test.npy')

X = X / 255.
X_test = X_test / 255.

X = X.reshape(X.shape[0], -1)

## Setting the 10 fold stratified cross-validation
cvKFold=StratifiedKFold(n_splits=14, shuffle=True, random_state=0)


# Logistic Regression
def logregClassifier(X, y):
    model = LogisticRegression(random_state=0)
    scores = cross_val_score(model, X, y, cv=cvKFold)
    return scores.mean()



#Naïve Bayes
def nbClassifier(X, y):
    model = GaussianNB()
    scores = cross_val_score(model, X, y, cv=cvKFold)
    return scores.mean()



# Decision Tree
def dtClassifier(X, y):
    model = DecisionTreeClassifier(criterion='entropy', random_state=0)
    scores = cross_val_score(model, X, y, cv=cvKFold)
    return scores.mean()



# Ensembles: Bagging, Ada Boost and Gradient Boosting
def bagDTClassifier(X, y, n_estimators, max_samples, max_depth):
    base_estimator = DecisionTreeClassifier(criterion='entropy', max_depth=max_depth, random_state=0)
    model = BaggingClassifier(estimator=base_estimator, n_estimators=n_estimators, 
                             max_samples=max_samples, random_state=0)
    scores = cross_val_score(model, X, y, cv=cvKFold)
    return scores.mean()

def adaDTClassifier(X, y, n_estimators, learning_rate, max_depth):
    base_estimator = DecisionTreeClassifier(criterion='entropy', max_depth=max_depth, random_state=0)
    model = AdaBoostClassifier(estimator=base_estimator, n_estimators=n_estimators, 
                              learning_rate=learning_rate, random_state=0)
    scores = cross_val_score(model, X, y, cv=cvKFold)
    return scores.mean()

def gbClassifier(X, y, n_estimators, learning_rate):
    model = GradientBoostingClassifier(n_estimators=n_estimators, learning_rate=learning_rate, random_state=0)
    scores = cross_val_score(model, X, y, cv=cvKFold)
    return scores.mean()


# ### Part 1 Results


# Parameters for Part 1:

#Bagging
bag_n_estimators = 50
bag_max_samples = 100
bag_max_depth = 5

#AdaBoost
ada_n_estimators = 50
ada_learning_rate = 0.5
ada_bag_max_depth = 5

#GB
gb_n_estimators = 50
gb_learning_rate = 0.5

# Print results for each classifier in part 1 to 4 decimal places here:
print("LogR average cross-validation accuracy:",format(logregClassifier(X, y),'.4f'))
print("NB average cross-validation accuracy:",format(nbClassifier(X, y),'.4f'))
print("DT average cross-validation accuracy:",format(dtClassifier(X, y),'.4f'))
print("Bagging average cross-validation accuracy:",format(bagDTClassifier(X, y, bag_n_estimators, bag_max_samples, bag_max_depth),'.4f'))
print("AdaBoost average cross-validation accuracy:",format(adaDTClassifier(X, y, ada_n_estimators, ada_learning_rate, ada_bag_max_depth),'.4f'))
print("GB average cross-validation accuracy:",format(gbClassifier(X, y, gb_n_estimators, gb_learning_rate),'.4f'))



# KNN
k = [1, 3, 5, 7]
p = [1, 2]


def bestKNNClassifier(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, 
                                                      random_state=0, stratify=y)
    
    param_grid = {'n_neighbors': k, 'p': p}
    knn = KNeighborsClassifier()
    
    grid_search = GridSearchCV(knn, param_grid, cv=cvKFold)
    grid_search.fit(X_train, y_train)
    
    best_k = grid_search.best_params_['n_neighbors']
    best_p = grid_search.best_params_['p']
    best_score = grid_search.best_score_
    
    best_model = grid_search.best_estimator_
    test_score = best_model.score(X_test, y_test)
    
    return [best_k, best_p, best_score, test_score]



# SVM
# You should use SVC from sklearn.svm with kernel set to 'rbf'
C = [0.01, 0.1, 1, 5] 
gamma = [0.01, 0.1, 1, 10]

def bestSVMClassifier(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, 
                                                      random_state=0, stratify=y)
    
    param_grid = {'C': C, 'gamma': gamma}
    svm = SVC(kernel='rbf', random_state=0)
    
    grid_search = GridSearchCV(svm, param_grid, cv=cvKFold)
    grid_search.fit(X_train, y_train)
    
    best_C = grid_search.best_params_['C']
    best_gamma = grid_search.best_params_['gamma']
    best_score = grid_search.best_score_
    
    best_model = grid_search.best_estimator_
    test_score = best_model.score(X_test, y_test)
    
    return [best_C, best_gamma, best_score, test_score]



# Random Forest
# You should use RandomForestClassifier from sklearn.ensemble with information gain and max_features set to ‘sqrt’.
n_estimators = [10, 30, 60, 100]
max_leaf_nodes = [6, 12]

def bestRFClassifier(X, y):
    # Split data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, 
                                                      random_state=0, stratify=y)
    
    param_grid = {'n_estimators': n_estimators, 'max_leaf_nodes': max_leaf_nodes}
    rf = RandomForestClassifier(criterion='entropy', max_features='sqrt', random_state=0)

    grid_search = GridSearchCV(rf, param_grid, cv=cvKFold)
    grid_search.fit(X_train, y_train)

    best_n_estimators = grid_search.best_params_['n_estimators']
    best_max_leaf_nodes = grid_search.best_params_['max_leaf_nodes']
    best_score = grid_search.best_score_

    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)
    test_score = accuracy_score(y_test, y_pred)

    macro_f1 = f1_score(y_test, y_pred, average='macro')
    weighted_f1 = f1_score(y_test, y_pred, average='weighted')
    
    return [best_n_estimators, best_max_leaf_nodes, best_score, test_score, macro_f1, weighted_f1]


# ### Part 2: Results

# Perform Grid Search with 10-fold stratified cross-validation (GridSearchCV in sklearn). 
# The stratified folds from cvKFold should be provided to GridSearchV
# Run best parameter classifiers
knn_best_k, knn_best_p, knn_cv_score, knn_test_score = bestKNNClassifier(X, y)
svm_best_C, svm_best_gamma, svm_cv_score, svm_test_score = bestSVMClassifier(X, y)
rf_best_n_estimators, rf_best_max_leaf_nodes, rf_cv_score, rf_test_score, rf_macro_f1, rf_weighted_f1 = bestRFClassifier(X, y)

# This should include using train_test_split from sklearn.model_selection with stratification and random_state=0
# Print results for each classifier here. All results should be printed to 4 decimal places except for
# "k", "p", n_estimators" and "max_leaf_nodes" which should be printed as integers.

print("KNN best k:",knn_best_k)
print("KNN best p:",knn_best_p)
print("KNN cross-validation accuracy:",format(knn_cv_score,'.4f'))
print("KNN test set accuracy:",format(knn_test_score,'.4f'))

print()

print("SVM best C:=",svm_best_C)
print("SVM best gamma:=",svm_best_gamma)
print("SVM cross-validation accuracy:",format(svm_cv_score,'.4f'))
print("SVM test set accuracy:",format(svm_test_score,'.4f'))

print()

print("RF best n_estimators:",rf_best_n_estimators)
print("RF best max_leaf_nodes:",rf_best_max_leaf_nodes)
print("RF cross-validation accuracy:",format(rf_cv_score,'.4f'))
print("RF test set accuracy:",format(rf_test_score,'.4f'))
print("RF test set macro average F1:",format(rf_macro_f1,'.4f'))
print("RF test set weighted average F1:",format(rf_weighted_f1,'.4f'))




