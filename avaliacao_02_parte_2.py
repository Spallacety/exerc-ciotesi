import pandas as pd
import numpy as np
from collections import Counter
from sklearn.naive_bayes import MultinomialNB, BernoulliNB, GaussianNB
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier
from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix

def fit_and_predict(name, model, training_data, training_markers, test_data, test_markers):
  model.fit(training_data, training_markers)
  result = model.predict(test_data)

  correct = 0
  size = len(test_markers)
  for i in range(size):
    if test_markers[i] == result[i]:
      correct += 1

  print('Confusion Matrix for %s' %(name))
  print(confusion_matrix(result, test_markers))
  print('%s: %.2f%% correctly predict\n' %(name, (correct*100/size)))

def k_fold(name, model, training_data, training_markers, k):
  scores = cross_val_score(model, training_data, training_markers, cv = k)
  print("%s %d-fold: %.2f%% correctly predict" %(name, k, (np.mean(scores)*100.0)))

def find_null():
  for field in train_df.columns:
    print(field, 'NaN:', train_df[field].isnull().sum())
    
train_df = pd.read_csv('random.csv')

#find_null()

train_df = train_df.drop(['country_off_res', 'age_desc', 'used_app_before', 'relation'], 1)

#train_df.loc[train_df['relation'].isnull(), 'relation'] = 'Unknown'
train_df.loc[train_df['ethnicity'].isnull(), 'ethnicity'] = 'Unknown'
train_df['gender'] = train_df['gender'].map({'m' : 0, 'f' : 1}).astype(int)
train_df['jaundice'] = train_df['jaundice'].map({'no' : 0, 'yes' : 1}).astype(int)
train_df['pdd'] = train_df['pdd'].map({'no' : 0, 'yes' : 1}).astype(int)
train_df['ethnicity'] = train_df['ethnicity'].map({'White-European' : 0 ,'Middle-Eastern' : 1 ,'Hispanic' : 2 ,'Asian' : 3 ,'Black' : 4 ,'South-Asian' : 5 ,'Latino' : 6 ,'Pasifika' : 7 ,'Turkish' : 8, 'Others' : 9, 'Unknown': 10}).astype(int)
#train_df['used_app_before'] = train_df['used_app_before'].map({'no' : 0, 'yes' : 1}).astype(int)
#train_df['relation'] = train_df['relation'].map({'Self' : 0, 'Parent' : 1, 'Relative' : 2, 'Health care professional' : 3, 'Others' : 4, 'Unknown' : 5}).astype(int)
train_df['classification'] = train_df['classification'].map({'NO' : 0, 'YES' : 1}).astype(int)

train_df = train_df.dropna(how='any',axis=0)

#print(len(train_df[(train_df['ethnicity']==0) & (train_df['classification']==0)]))
#print(len(train_df[(train_df['ethnicity']==1) & (train_df['classification']==0)]))
#print(len(train_df[(train_df['ethnicity']==2) & (train_df['classification']==0)]))
#print(len(train_df[(train_df['ethnicity']==3) & (train_df['classification']==0)]))
#print(len(train_df[(train_df['ethnicity']==4) & (train_df['classification']==0)]))
#print(len(train_df[(train_df['ethnicity']==5) & (train_df['classification']==0)]))
#print(len(train_df[(train_df['ethnicity']==6) & (train_df['classification']==0)]))
#print(len(train_df[(train_df['ethnicity']==7) & (train_df['classification']==0)]))
#print(len(train_df[(train_df['ethnicity']==8) & (train_df['classification']==0)]))
#print(len(train_df[(train_df['ethnicity']==9) & (train_df['classification']==0)]))
#print(len(train_df[(train_df['ethnicity']==10) & (train_df['classification']==0)]))

X_df = train_df [['q1', 'q2', 'q3', 'q4', 'q5', 'q6', 'q7', 'q8', 'q9', 'q10', 'age', 'ethnicity', 'gender', 'jaundice', 'pdd', 'result']]
Y_df = train_df ['classification']

Xdummies_df = pd.get_dummies(X_df)

X = Xdummies_df.values
Y = Y_df.values

training_percentage = 0.7
test_percentage = 0.3

training_size = int(training_percentage * len(Y))
test_size = int(test_percentage * len(Y))

training_data = X[:training_size]
training_markers = Y[:training_size]

training_end = training_size + test_size

test_data = X[training_size:training_end]
test_markers = Y[training_size:training_end]

modelMultinomial = MultinomialNB()
modelGaussian = GaussianNB()
modelBernoulli = BernoulliNB()
modelAdaBoost = AdaBoostClassifier()
modelRandomForest = RandomForestClassifier()
modelOneVsRest = OneVsRestClassifier(LinearSVC(random_state=0))
modelOneVsOne = OneVsOneClassifier(LinearSVC(random_state=0))

fit_and_predict("MultinomialNB", modelMultinomial, training_data, training_markers, test_data, test_markers)
fit_and_predict("GaussianNB", modelGaussian, training_data, training_markers, test_data, test_markers)
fit_and_predict("BernoulliNB", modelBernoulli, training_data, training_markers, test_data, test_markers)
fit_and_predict("AdaBoostClassifier", modelAdaBoost, training_data, training_markers, test_data, test_markers)
fit_and_predict("RandomForestClassifier", modelRandomForest, training_data, training_markers, test_data, test_markers)
fit_and_predict("OneVsRestClassifier", modelOneVsRest, training_data, training_markers, test_data, test_markers)
fit_and_predict("OneVsOneClassifier", modelOneVsOne, training_data, training_markers, test_data, test_markers)

k = 5

k_fold("MultinomialNB", modelMultinomial, training_data, training_markers, k)
k_fold("GaussianNB", modelGaussian, training_data, training_markers, k)
k_fold("BernoulliNB", modelBernoulli, training_data, training_markers, k)
k_fold("AdaBoostClassifier", modelAdaBoost, training_data, training_markers, k)
k_fold("RandomForestClassifier", modelRandomForest, training_data, training_markers, k)
k_fold("OneVsRestClassifier", modelOneVsRest, training_data, training_markers, k)
k_fold("OneVsOneClassifier", modelOneVsOne, training_data, training_markers, k)