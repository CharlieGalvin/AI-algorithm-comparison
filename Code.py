import pandas as pd
import numpy as np

from sklearn.feature_selection import SelectKBest

# use chi squared, f-test and mutual information
from sklearn.feature_selection import chi2, RFE, f_classif, mutual_info_classif
from sklearn.preprocessing import MinMaxScaler

# we can then use logistic regression or RFE
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder
from sklearn.linear_model import  LogisticRegression
import matplotlib.pyplot as plt

df = pd.read_csv('latestdata.csv')
del df['ID']
del df['additional_information']
del df['source']
del df['sequence_available']
del df['notes_for_discussion']
del df['chronic_disease']
del df['latitude']
del df['longitude']

del df['sex']
del df['geo_resolution']
del df["date_onset_symptoms"]
del df["date_admission_hospital"]
del df['symptoms']
del df["travel_history_dates"]
del df["travel_history_location"]
del df['reported_market_exposure']
del df["date_death_or_discharge"]
del df["location"]
del df['admin3']
del df['admin2']
del df['admin1']
del df['country_new']
del df['data_moderator_initials']



binary_false = ["Death", "death", "deceased", "Deceased", "Died", "died", "Dead", "dead", "Unstable"]

for death_word in binary_false:
    df.loc[df["outcome"] == death_word,"outcome"] = 0

binary_true = ["Alive", "alive", "Discharge", "discharge", "Discharged", "discharged",
       "Discharged from hospital", "discharged from hospital",
       "not hospitalized", "Recovered", "recovered", "recovering at home 03.03.2020",
        "Stable", "stable", "stable condition", "Stable condition"]

for living_word in binary_true:
    df.loc[df["outcome"] == living_word,"outcome"] = 1



df[(df['outcome'] != 0) & (df['outcome'] != 1)] = np.NaN
df = df[df.outcome.notnull()]


# we assume those that did not put yet to "lives in Wuhan", do not live in Wuhan.
df.loc[df["lives_in_Wuhan"] != "yes", "lives_in_Wuhan"] = "no"

print(df['outcome'].value_counts())

df['outcome'].value_counts().plot(kind="bar")
plt.show()


# clening not null values
df = df[df.age.notnull()]
df = df[df.city.notnull()]
df = df[df.province.notnull()]
df = df[df.country.notnull()]
df = df[df.chronic_disease_binary.notnull()]
df = df[df.date_confirmation.notnull()]
df = df[df.travel_history_binary.notnull()]

print(df.isnull().sum())
print(len(df.index))

label = df['outcome'].astype("float64")

# remove labels from features
df1 = df.pop('outcome')
#df['outcome'] = df1
# putting target value at the end

# fixing age to continuous

df["age"].replace({"90-99" :95}, inplace= True)
df["age"].replace({"80-89" :85}, inplace= True)
df["age"].replace({"70-79" :75}, inplace= True)
df["age"].replace({"60-69" :65}, inplace= True)
df["age"].replace({"50-59" :55}, inplace= True)
df["age"].replace({"40-49" :45}, inplace= True)
df["age"].replace({"20-29" :25}, inplace= True)
df["age"].replace({"20-29" :25}, inplace= True)
df["age"].replace({"15-88" :85}, inplace= True)
df["age"].replace({"38-68" :53}, inplace= True)
df["age"].replace({"20-57" :34}, inplace= True)
df["age"].replace({"21-72" :0}, inplace= True)
df["age"].replace({"22-80" :0}, inplace= True)
df["age"].replace({"19-77" :0}, inplace= True)
df["age"].replace({"80-" :5}, inplace= True)
df["age"].replace({"0.5" :5}, inplace= True)
df["age"].replace({"0.75" :5}, inplace= True)
df["age"].replace({"0.25" :5}, inplace= True)

df.to_csv("processed_data_for_FS2.csv", header= True)

categorical_features = ["province", "country","city","date_confirmation",
                        "travel_history_binary", "chronic_disease_binary","admin_id","lives_in_Wuhan"]

# we now have a full database.

df[categorical_features] = df[categorical_features].astype("category")
print(df['age'].value_counts())

df['age'] = df['age'].astype('float64')
print(df.dtypes)

age_tag = True

if not age_tag:
    oe = OrdinalEncoder()
    oe.fit(df[categorical_features])
    df[categorical_features] = oe.transform(df[categorical_features])


if age_tag:
    # modelling age as categoric
    df["age_bins"] = pd.cut(df.age, bins= [-1,20,30,40,50,60,70,80,90,200], labels= ["<20","20-29","30-39","40-49","50-59","60-69","70-79","80-89","90+"])
    categorical_features.append("age_bins")
    oe = OrdinalEncoder()
    oe.fit(df[categorical_features])
    df[categorical_features] = oe.transform(df[categorical_features])
    df_norm = df.copy()
    del df_norm["age"]

else:
    continuous_features = ["age"]
    scaler = MinMaxScaler()
    df_norm = df.copy()
    df_norm[continuous_features] = scaler.fit_transform(df[continuous_features])

#print(df["age_bins"].value_counts())
print(df.columns)
print(categorical_features)

# we can maybe use different forms of coding if we want here.


oe = OrdinalEncoder()
results = df_norm.copy()

print(results)
print(results.dtypes)

if not age_tag:
    results['age'] = results['age'].astype('float64')


results['outcome'] = df1
print(results.columns)
print(results['outcome'].value_counts())

results['outcome'].value_counts().plot(kind="bar")
plt.show()
results.to_csv("ML_dataset.csv")


#df = pd.read_csv('PROCESSED.csv')
#OrdinalEncoder()
#label = df['outcome']

# remove labels from features
#df.drop('outcome', axis= 1, inplace= True)

#print(label.value_counts())
#label.value_counts().plot(kind="bar")
#plt.show()

# need to train the model several times to see if this improves the results with addition of K-cross validation.

#########################################################################################################################
# this is the second way it can be run to adjust the sampling for comaprisons

import pandas as pd
import numpy as np

from sklearn.feature_selection import SelectKBest

# use chi squared, f-test and mutual information
from sklearn.feature_selection import chi2, RFE, f_classif, mutual_info_classif
from sklearn.preprocessing import MinMaxScaler

# we can then use logistic regression or RFE
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder
from sklearn.linear_model import  LogisticRegression
import matplotlib.pyplot as plt

df = pd.read_csv('latestdata.csv')
del df['ID']
del df['additional_information']
del df['source']
del df['sequence_available']
del df['notes_for_discussion']
del df['chronic_disease']
del df['latitude']
del df['longitude']

del df['sex']
del df['geo_resolution']
del df["date_onset_symptoms"]
del df["date_admission_hospital"]
del df['symptoms']
del df['lives_in_Wuhan']
del df["travel_history_dates"]
del df["travel_history_location"]
del df['reported_market_exposure']
del df["date_death_or_discharge"]
del df["location"]
del df['admin3']
del df['admin2']
del df['admin1']
del df['country_new']
del df['data_moderator_initials']



binary_false = ["Death", "death", "deceased", "Deceased", "Died", "died", "Dead", "dead", "Unstable"]

for death_word in binary_false:
    df.loc[df["outcome"] == death_word,"outcome"] = 0

binary_true = ["Alive", "alive", "Discharge", "discharge", "Discharged", "discharged",
       "Discharged from hospital", "discharged from hospital",
       "not hospitalized", "Recovered", "recovered", "recovering at home 03.03.2020",
        "Stable", "stable", "stable condition", "Stable condition"]

for living_word in binary_true:
    df.loc[df["outcome"] == living_word,"outcome"] = 1



df[(df['outcome'] != 0) & (df['outcome'] != 1)] = np.NaN
df = df[df.outcome.notnull()]

print(df['outcome'].value_counts())

df['outcome'].value_counts().plot(kind="bar")


# clening not null values
df = df[df.age.notnull()]
df = df[df.city.notnull()]
df = df[df.province.notnull()]
df = df[df.country.notnull()]
df = df[df.chronic_disease_binary.notnull()]
df = df[df.date_confirmation.notnull()]
df = df[df.travel_history_binary.notnull()]

print(df.isnull().sum())
print(len(df.index))

label = df['outcome'].astype("float64")

# remove labels from features
df1 = df.pop('outcome')
#df['outcome'] = df1
# putting target value at the end

# fixing age to continuous

df["age"].replace({"90-99" :95}, inplace= True)
df["age"].replace({"80-89" :85}, inplace= True)
df["age"].replace({"70-79" :75}, inplace= True)
df["age"].replace({"60-69" :65}, inplace= True)
df["age"].replace({"50-59" :55}, inplace= True)
df["age"].replace({"40-49" :45}, inplace= True)
df["age"].replace({"20-29" :25}, inplace= True)
df["age"].replace({"20-29" :25}, inplace= True)
df["age"].replace({"15-88" :85}, inplace= True)
df["age"].replace({"38-68" :53}, inplace= True)
df["age"].replace({"20-57" :34}, inplace= True)
df["age"].replace({"21-72" :0}, inplace= True)
df["age"].replace({"22-80" :0}, inplace= True)
df["age"].replace({"19-77" :0}, inplace= True)
df["age"].replace({"80-" :5}, inplace= True)
df["age"].replace({"0.5" :5}, inplace= True)
df["age"].replace({"0.75" :5}, inplace= True)
df["age"].replace({"0.25" :5}, inplace= True)

df.to_csv("processed_data_for_FS2.csv", header= True)

categorical_features = ["province", "country","city","date_confirmation",
                        "travel_history_binary", "chronic_disease_binary","admin_id"]

# we now have a full database.

df[categorical_features] = df[categorical_features].astype("category")
print(df['age'].value_counts())

df['age'] = df['age'].astype('float64')
print(df.dtypes)

age_tag = True

if not age_tag:
    oe = OrdinalEncoder()
    oe.fit(df[categorical_features])
    df[categorical_features] = oe.transform(df[categorical_features])


if age_tag:
    # modelling age as categoric
    df["age_bins"] = pd.cut(df.age, bins= [-1,20,30,40,50,60,70,80,90,200], labels= ["<20","20-29","30-39","40-49","50-59","60-69","70-79","80-89","90+"])
    categorical_features.append("age_bins")
    oe = OrdinalEncoder()
    oe.fit(df[categorical_features])
    df[categorical_features] = oe.transform(df[categorical_features])
    df_norm = df.copy()
    del df_norm["age"]

else:
    continuous_features = ["age"]
    scaler = MinMaxScaler()
    df_norm = df.copy()
    df_norm[continuous_features] = scaler.fit_transform(df[continuous_features])

#print(df["age_bins"].value_counts())
print(df.columns)
print(categorical_features)

# we can maybe use different forms of coding if we want here.


oe = OrdinalEncoder()
results = df_norm.copy()

print(results)
print(results.dtypes)

if not age_tag:
    results['age'] = results['age'].astype('float64')





results['outcome'] = df1
print(results.columns)
print(results['outcome'].value_counts())

results['outcome'].value_counts().plot(kind="bar")
plt.show()

outcomes = [0, 1]
count = 0


for outcome_option in outcomes:
    data2 = results.copy()
    data2.loc[data2["outcome"] != outcome_option, "outcome"] = np.NaN
    data2 = data2[data2.outcome.notnull()]

    if count == 0:
        data3 = data2.sample(n=298, replace=True, random_state=1)
        count += 1

    else:
        data8 = data2.sample(n=4860, replace=True, random_state=1)
        data3 = data3.append(data8)

results = data3

results['outcome'].value_counts().plot(kind="bar")
plt.show()


results.to_csv("ML_dataset.csv")




#df = pd.read_csv('PROCESSED.csv')
#OrdinalEncoder()
#label = df['outcome']

# remove labels from features
#df.drop('outcome', axis= 1, inplace= True)

#print(label.value_counts())
#label.value_counts().plot(kind="bar")
#plt.show()

# need to train the model several times to see if this improves the results with addition of K-cross validation.


####################################################################################################
# this is where all of the graphs are outputted. I used code from online to output a nicer heat map than my own but
# I cannot find the website for reference anywhere. I will email the lecturer to confirm that this is okay as I don't
# want to be convicted of plagiarism.

# Corona Virus is spreading.
# You need to think about what problems you want to answer, build results from data and carefully think about your methods

# Need to look into accuracy of each model as well as how well it performs for one justification
# Need to look at several justifications and use a good model question

# Need to think about the following algorithms:
# SVM, Decision trees and Neural Networks (maybe one other one as well)
# Can look at using sklearn to do all of these things and work on looking at accuracy

# Also need to look at different methodologies for the dataset we plan to use.

# He is looking for something exceptional and I really need to give it to him otherwise I am screwed for the entire module.
# I have to average 60% or higher in both of them.

# Think it definitely makes sense to either use a classification question or a regression question.

# Potential ideas include: How long will a patient be in hospital, as well as whether a patient is going to die or not.

import pandas as pd
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression


col_names = ['age','province','country','date_confirmation','country_new','travel_history_binary','outcome']

# load dataset
data = pd.read_csv("ML_dataset.csv")

data.head()
feature_cols = ['admin_id', 'chronic_disease_binary', 'province', 'country', 'date_confirmation', 'city', 'travel_history_binary','age_bins']

X = data[feature_cols] # Features
y = data.outcome # Target variable

# need to look into how this properly works
from sklearn.model_selection import train_test_split
# this represents the proportion of the dataset that should be included
# random_state controls the shuffling of the data before applying the split.
# could investigate this further.

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=0)

# hyper parameter tuning:
#model = LogisticRegression(max_iter= 1000)
#solvers = ['newton-cg', 'lbfgs', 'liblinear']
#penalty = ['l1','l2','elasticnet','none']
#c_values = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
#grid = dict(solver=solvers,penalty=penalty,C=c_values)
#cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
#grid_search = GridSearchCV(estimator=model, param_grid=grid, n_jobs=-1, cv=cv, scoring='accuracy',error_score=0)
#grid_result = grid_search.fit(X_train, y_train)
# summarize results
#print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
#means = grid_result.cv_results_['mean_test_score']
##stds = grid_result.cv_results_['std_test_score']
#params = grid_result.cv_results_['params']
#for mean, stdev, param in zip(means, stds, params):
#    print("%f (%f) with: %r" % (mean, stdev, param))

from sklearn.ensemble import RandomForestClassifier
import numpy as np
rf = RandomForestClassifier()

# I used this code from: https://towardsdatascience.com/hyperparameter-tuning-the-random-forest-in-python-using-scikit-learn-28d2aa77dd74

# Number of trees in random forest
#n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
# Number of features to consider at every split
#max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
#max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
#max_depth.append(None)
# Minimum number of samples required to split a node
#min_samples_split = [2, 5, 10]
# Minimum number of samples required at each leaf node
#min_samples_leaf = [1, 2, 4]
# Method of selecting samples for training each tree
#bootstrap = [True, False]
# Create the random grid
#random_grid = {'n_estimators': n_estimators,
#               'max_features': max_features,
#               'max_depth': max_depth,
#               'min_samples_split': min_samples_split,
#               'min_samples_leaf': min_samples_leaf,
#               'bootstrap': bootstrap}


#rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 100, cv = 3, verbose=2, random_state=42, n_jobs = -1)
#rf_random.fit(X_train, y_train)
#print("Best: %f using %s" % (rf_random.best_score_, rf_random.best_params_))

import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

lr = LogisticRegression(max_iter=1000, solver= 'newton-cg', C=0.1, penalty= 'l2')
lr.fit(X_train, y_train)
y_pred = lr.predict(X_test)

# the code I used to output the heatmap I NEED TO FIND ONLINE AND REFERENCE

from sklearn import metrics
cnf_matrix = metrics.confusion_matrix(y_test, y_pred)
print(cnf_matrix)


import seaborn as sns

class_names = [0,1]
fix, ax = plt.subplots()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)

# create heatmap
sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu" ,fmt='g')
ax.xaxis.set_label_position("top")
plt.tight_layout()
plt.title('Confusion matrix', y=1.1)
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
plt.show()


# precision means when a model makes a prediction, how often it is correct.
# when my model predicts patients are going to suffer from diabetes, it was correct
# 76% of the time.

y_prob = lr.predict_proba(X_test)[::,1]
fpr, tpr, _ = metrics.roc_curve(y_test, y_prob)

auc = metrics.roc_auc_score(y_test, y_prob)
plt.plot(fpr, tpr, label="data 1, auc" + str(auc))
plt.legend(loc=4)
plt.show()
print(classification_report(y_test, y_pred))


rfr = RandomForestClassifier(n_estimators= 1800, min_samples_split=2, min_samples_leaf=1, max_features= 'auto', max_depth= 20, bootstrap= False,  random_state= 42)
rfr.fit(X_train, y_train)
y_pred2 = rfr.predict(X_test)
cnf_matrix2 = metrics.confusion_matrix(y_test, y_pred2)
print(cnf_matrix2)

import seaborn as sns

class_names = [0,1]
fix, ax = plt.subplots()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)

# create heatmap
sns.heatmap(pd.DataFrame(cnf_matrix2), annot=True, cmap="YlGnBu" ,fmt='g')
ax.xaxis.set_label_position("top")
plt.tight_layout()
plt.title('Confusion matrix', y=1.1)
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
plt.show()


# precision means when a model makes a prediction, how often it is correct.
# when my model predicts patients are going to suffer from diabetes, it was correct
# 76% of the time.

y_prob = rfr.predict_proba(X_test)[::,1]
fpr, tpr, _ = metrics.roc_curve(y_test, y_prob)

auc = metrics.roc_auc_score(y_test, y_prob)
plt.plot(fpr, tpr, label="data 1, auc" + str(auc))
plt.legend(loc=4)
plt.show()
print(classification_report(y_test, y_pred2))

from sklearn import svm

x = len(data.index)
data2 = data.sample(n= int(x), replace=False, random_state=1)

X = data2[feature_cols] # Features
y = data2.outcome # Target variable
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=0)

import matplotlib.pyplot as plt
svmd = svm.SVC(kernel='rbf', gamma= 0.0001, C= 10,probability= True)
svmd.fit(X_train, y_train)
y_pred3 = svmd.predict(X_test)
cnf_matrix3 = metrics.confusion_matrix(y_test, y_pred3)
print(cnf_matrix3)
import seaborn as sns

class_names = [0,1]
fix, ax = plt.subplots()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)

# create heatmap
sns.heatmap(pd.DataFrame(cnf_matrix3), annot=True, cmap="YlGnBu" ,fmt='g')
ax.xaxis.set_label_position("top")
plt.tight_layout()
plt.title('Confusion matrix', y=1.1)
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
plt.show()


# precision means when a model makes a prediction, how often it is correct.
# when my model predicts patients are going to suffer from diabetes, it was correct
# 76% of the time.

y_prob = svmd.predict_proba(X_test)[::,1]
fpr, tpr, _ = metrics.roc_curve(y_test, y_prob)

auc = metrics.roc_auc_score(y_test, y_prob)
plt.plot(fpr, tpr, label="data 1, auc" + str(auc))
plt.legend(loc=4)
plt.show()
print(classification_report(y_test, y_pred3))


#################################################################################################
# voting classifier and k-cross validation

# Corona Virus is spreading.
# You need to think about what problems you want to answer, build results from data and carefully think about your methods

# Need to look into accuracy of each model as well as how well it performs for one justification
# Need to look at several justifications and use a good model question

# Need to think about the following algorithms:
# SVM, Decision trees and Neural Networks (maybe one other one as well)
# Can look at using sklearn to do all of these things and work on looking at accuracy

# Also need to look at different methodologies for the dataset we plan to use.

# He is looking for something exceptional and I really need to give it to him otherwise I am screwed for the entire module.
# I have to average 60% or higher in both of them.

# Think it definitely makes sense to either use a classification question or a regression question.

# Potential ideas include: How long will a patient be in hospital, as well as whether a patient is going to die or not.
import matplotlib.pyplot as plt
import numpy as np

import pandas as pd
from sklearn.metrics import classification_report
from sklearn.ensemble import VotingClassifier
col_names = ['age','province','country','date_confirmation','country_new','travel_history_binary','outcome']

# load dataset
data = pd.read_csv("ML_dataset.csv")

data.head()
feature_cols = ['admin_id', 'chronic_disease_binary', 'province', 'country', 'date_confirmation', 'city', 'travel_history_binary','age_bins']



X = data[feature_cols] # Features
y = data.outcome # Target variable


# need to look into how this properly works
from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=0)



from sklearn.model_selection import StratifiedKFold
# this represents the proportion of the dataset that should be included
# random_state controls the shuffling of the data before applying the split.
# could investigate this further.

folds = StratifiedKFold(n_splits= 10)
folds.get_n_splits(X,y)

# for train_index, test_index in folds.split(X,y):
#X_train, X_test = X.iloc[train_index], X.iloc[test_index]
#y_train, y_test = y.iloc[train_index], y.iloc[test_index]

from sklearn.linear_model import LogisticRegression

lr = LogisticRegression(max_iter=1000)

lr.fit(X_train, y_train)
y_pred = lr.predict(X_test)

from sklearn import metrics
cnf_matrix = metrics.confusion_matrix(y_test, y_pred)
print(cnf_matrix)
import seaborn as sns

class_names = [0, 1]
fix, ax = plt.subplots()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)

# create heatmap
sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu", fmt='g')
ax.xaxis.set_label_position("top")
plt.tight_layout()
plt.title('Confusion matrix', y=1.1)
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
plt.show()

# precision means when a model makes a prediction, how often it is correct.
# when my model predicts patients are going to suffer from diabetes, it was correct
# 76% of the time.

y_prob = lr.predict_proba(X_test)[::, 1]
fpr, tpr, _ = metrics.roc_curve(y_test, y_prob)

auc = metrics.roc_auc_score(y_test, y_prob)
plt.plot(fpr, tpr, label="data 1, auc" + str(auc))
plt.legend(loc=4)
plt.show()

print(classification_report(y_test, y_pred))

from sklearn.ensemble import RandomForestClassifier

rfr = RandomForestClassifier(n_estimators= 100, random_state= 42)
rfr.fit(X_train, y_train)
y_pred2 = rfr.predict(X_test)
cnf_matrix2 = metrics.confusion_matrix(y_test, y_pred2)
print(cnf_matrix2)
import seaborn as sns

class_names = [0, 1]
fix, ax = plt.subplots()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)

# create heatmap
sns.heatmap(pd.DataFrame(cnf_matrix2), annot=True, cmap="YlGnBu", fmt='g')
ax.xaxis.set_label_position("top")
plt.tight_layout()
plt.title('Confusion matrix', y=1.1)
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
plt.show()

# precision means when a model makes a prediction, how often it is correct.
# when my model predicts patients are going to suffer from diabetes, it was correct
# 76% of the time.

y_prob = rfr.predict_proba(X_test)[::, 1]
fpr, tpr, _ = metrics.roc_curve(y_test, y_prob)

auc = metrics.roc_auc_score(y_test, y_prob)
plt.plot(fpr, tpr, label="data 1, auc" + str(auc))
plt.legend(loc=4)
plt.show()
print(classification_report(y_test, y_pred2))

from sklearn import svm

x = len(data.index)
data2 = data.sample(n= int(x * 0.4), replace=False, random_state=1)

X = data2[feature_cols] # Features
y = data2.outcome # Target variable
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=0)

svmd = svm.SVC(kernel='linear',probability= True)
svmd.fit(X_train, y_train)
y_pred3 = svmd.predict(X_test)
cnf_matrix3 = metrics.confusion_matrix(y_test, y_pred3)
print(cnf_matrix3)
import seaborn as sns

class_names = [0, 1]
fix, ax = plt.subplots()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)

# create heatmap
sns.heatmap(pd.DataFrame(cnf_matrix3), annot=True, cmap="YlGnBu", fmt='g')
ax.xaxis.set_label_position("top")
plt.tight_layout()
plt.title('Confusion matrix', y=1.1)
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
plt.show()

# precision means when a model makes a prediction, how often it is correct.
# when my model predicts patients are going to suffer from diabetes, it was correct
# 76% of the time.

y_prob = svmd.predict_proba(X_test)[::, 1]
fpr, tpr, _ = metrics.roc_curve(y_test, y_prob)

auc = metrics.roc_auc_score(y_test, y_prob)
plt.plot(fpr, tpr, label="data 1, auc" + str(auc))
plt.legend(loc=4)
plt.show()
print(classification_report(y_test, y_pred3))


voting = VotingClassifier(estimators= [("svmd",svmd),("rfr",rfr),("lr",lr)])

voting.fit(X_train, y_train)
y_pred4 = voting.predict(X_test)
cnf_matrix4 = metrics.confusion_matrix(y_test, y_pred4)
print(cnf_matrix4)
print(classification_report(y_test, y_pred4))

import seaborn as sns

class_names = [0,1]
fix, ax = plt.subplots()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)

# create heatmap
sns.heatmap(pd.DataFrame(cnf_matrix4), annot=True, cmap="YlGnBu" ,fmt='g')
ax.xaxis.set_label_position("top")
plt.tight_layout()
plt.title('Confusion matrix', y=1.1)
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
plt.show()


# precision means when a model makes a prediction, how often it is correct.
# when my model predicts patients are going to suffer from diabetes, it was correct
# 76% of the time.

y_prob = voting.predict_proba(X_test)[::,1]
fpr, tpr, _ = metrics.roc_curve(y_test, y_prob)

auc = metrics.roc_auc_score(y_test, y_prob)
plt.plot(fpr, tpr, label="data 1, auc" + str(auc))
plt.legend(loc=4)
plt.show()