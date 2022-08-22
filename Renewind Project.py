#!/usr/bin/env python
# coding: utf-8

# ### Business Context
# 
# Renewable energy sources play an increasingly important role in the global energy mix, as the effort to reduce the environmental impact of energy production increases.
# 
# Out of all the renewable energy alternatives, wind energy is one of the most developed technologies worldwide. The U.S Department of Energy has put together a guide to achieving operational efficiency using predictive maintenance practices.
# 
# Predictive maintenance uses sensor information and analysis methods to measure and predict degradation and future component capability. The idea behind predictive maintenance is that failure patterns are predictable and if component failure can be predicted accurately and the component is replaced before it fails, the costs of operation and maintenance will be much lower.
# 
# The sensors fitted across different machines involved in the process of energy generation collect data related to various environmental factors (temperature, humidity, wind speed, etc.) and additional features related to various parts of the wind turbine (gearbox, tower, blades, break, etc.). 
# 
# 
# 
# 

# ## Objective
# “ReneWind” is a company working on improving the machinery/processes involved in the production of wind energy using machine learning and has collected data of generator failure of wind turbines using sensors. They have shared a ciphered version of the data, as the data collected through sensors is confidential (the type of data collected varies with companies). 
# 
# The objective is to build various classification models, tune them, and find the best one that will help identify failures so that the generators could be repaired before failing/breaking to reduce the overall maintenance cost. 
# The nature of predictions made by the classification model will translate as follows:
# 
# It is given that the cost of repairing a generator is much less than the cost of replacing it, and the cost of inspection is less than the cost of repair.
# 
# “1” in the target variables should be considered as “failure” and “0” represents “No failure”.

# ## Data Description
# - The data provided is a transformed version of original data which was collected using sensors.
# - Train.csv - To be used for training and tuning of models. 
# - Test.csv - To be used only for testing the performance of the final best model.
# - Both the datasets consist of 40 predictor variables and 1 target variable

# In[1]:


# To help with reading and manipulating data
import pandas as pd
import numpy as np


# To help with data visualization
import matplotlib.pyplot as plt
import seaborn as sns

# To get different metric scores, and split data
from sklearn.metrics import (
    f1_score,
    accuracy_score,
    recall_score,
    precision_score,
    confusion_matrix,
    roc_auc_score,
    plot_confusion_matrix,
)
from sklearn import metrics
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

# To define maximum number of columns to be displayed in a dataframe
pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", None)

# To supress scientific notations for a dataframe
pd.set_option("display.float_format", lambda x: "%.3f" % x)

# To help with model building
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    AdaBoostClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
    BaggingClassifier,
)
from xgboost import XGBClassifier

pd.set_option("display.float_format", lambda x: "%.3f" % x)

# To supress warnings
import warnings


warnings.filterwarnings("ignore")

get_ipython().run_line_magic('load_ext', 'nb_black')


# In[2]:


rene = pd.read_csv("Downloads/Train.csv.csv")  
rene_Test = pd.read_csv("Downloads/Test.csv.csv")


# Loadng the training set and the testing set

# In[3]:


data = rene
data2 = rene_Test


# ## Data Overview

# In[4]:


data.shape


# In[5]:


data2.shape


# The trainig set has 20,000 rows and 41 columns.
# The testing test has 5,000 rows and 41 columns.

# ### Displaying the first few rows of the dataset

# In[6]:


data.head()


# In[7]:


data.tail()


# There are negative values in the data sets; data collected from the sensors.

# ### Statistical summary of the dataset

# In[8]:


data.describe().T


# In[9]:


data2.describe().T


# The datasets seems to be evenly distributed, will confirm distribution on EDA.

# ### Checking the data types of the columns for the dataset

# In[10]:


data.info()


# ### Checking for missing values

# In[11]:


data.isnull().sum()


# In[12]:


data2.isnull().sum()


# V1 and V2 from the training(18) and testing(5 and 6 respectively) datasets have missing values.

# In[13]:


data.isnull().values.any()


# In[14]:


bool_series = pd.isnull(data["V1"])
bool_series3 = pd.isnull(data["V2"])


# In[15]:


data[bool_series]


# In[16]:


data[bool_series3]


# ### Values in target variable

# In[17]:


data.dtypes.value_counts()


# In[18]:


data["Target"].value_counts()


# In[19]:


1110 / 18890 * 100


# In[20]:


data2["Target"].value_counts()


# In[21]:


282 / 4718 * 100


# Only 6% from both the traing and testing test is the reading as “failure” 

# In[ ]:





# ## Exploratory Data Analysis

# ### Univariate analysis

# In[22]:


def histogram_boxplot(data, feature, figsize=(12, 7), kde=False, bins=None):
    """
    Boxplot and histogram combined

    data: dataframe
    feature: dataframe column
    figsize: size of figure (default (12,7))
    kde: whether to the show density curve (default False)
    bins: number of bins for histogram (default None)
    """
    f2, (ax_box2, ax_hist2) = plt.subplots(
        nrows=2,
        sharex=True,  # x-axis will be shared among all subplots
        gridspec_kw={"height_ratios": (0.25, 0.75)},
        figsize=figsize,
    )
    sns.boxplot(data=data, x=feature, ax=ax_box2, showmeans=True, color="violet")
    sns.histplot(
        data=data, x=feature, kde=kde, ax=ax_hist2, bins=bins, palette="winter"
    ) if bins else sns.histplot(data=data, x=feature, kde=kde, ax=ax_hist2)
    ax_hist2.axvline(
        data[feature].mean(), color="green", linestyle="--"
    )  # Add mean to the histogram
    ax_hist2.axvline(data[feature].median(), color="black", linestyle="-")


# In[23]:


sns.set_style("darkgrid")
data.hist(figsize=(15, 10))
plt.show()


# #### Plotting histograms and boxplots for all the variables

# In[24]:


for feature in data.columns:
    histogram_boxplot(data, feature, figsize=(12, 7), kde=False, bins=None)


#     All the variables are evenly disrtibuted.

# ## Data Pre-Processing

# **Since the have been separated to training and testing set, diving the training set to training and validation set.**
# 

# In[25]:


X = data.drop(["Target"], axis=1)
Y = data["Target"]

X_train, X_val, y_train, y_val = train_test_split(
    X, Y, test_size=0.2, random_state=1, stratify=Y
)

print("Shape of Training set : ", X_train.shape)
print("Shape of validation test : ", X_val.shape)
print("Percentage of classes in training set:")
print(y_train.value_counts(normalize=True))
print("Percentage of classes in validation set:")
print(y_val.value_counts(normalize=True))


# In[26]:


X_test = data2.drop(["Target"], axis=1)
Y_test = data2["Target"]


# In[27]:


print(X_test.shape)


# ## Missing value imputation
# 
# 
# 

# In[28]:


imputer = SimpleImputer(strategy="median")
# Fit and transform the train data
X_train = pd.DataFrame(imputer.fit_transform(X_train), columns=X_train.columns)

# Transform the validation data
X_val = pd.DataFrame(imputer.transform(X_val), columns=X_train.columns)

# Transform the test data
X_test = pd.DataFrame(imputer.transform(X_test), columns=X_train.columns)


# In[29]:


print(X_train.isna().sum())
print("-" * 30)
print(X_val.isna().sum())
print("-" * 30)
print(X_test.isna().sum())


# ## Model Building

# The nature of predictions made by the classification model will translate as follows:
# 
# - True positives (TP) are failures correctly predicted by the model.
# - False negatives (FN) are real failures in a generator where there is no detection by model. 
# - False positives (FP) are failure detections in a generator where there is no failure.
# 
# **Which metric to optimize?**
# 
# * We need to choose the metric which will ensure that the maximum number of generator failures are predicted correctly by the model.
# * We would want Recall to be maximized as greater the Recall, the higher the chances of minimizing false negatives.
# * We want to minimize false negatives because if a model predicts that a machine will have no failure when there will be a failure, it will increase the maintenance cost.

# In[30]:


# defining a function to compute different metrics to check performance of a classification model built using sklearn
def model_performance_classification_sklearn(model, predictors, target):
    """
    Function to compute different metrics to check classification model performance

    model: classifier
    predictors: independent variables
    target: dependent variable
    """

    pred = model.predict(predictors)

    acc = accuracy_score(target, pred)  # to compute Accuracy
    recall = recall_score(target, pred)  # to compute Recall
    precision = precision_score(target, pred)  # to compute Precision
    f1 = f1_score(target, pred)  # to compute F1-score

    df_perf = pd.DataFrame(
        {"Accuracy": acc, "Recall": recall, "Precision": precision, "F1": f1},
        index=[0],
    )

    return df_perf


# In[31]:


def confusion_matrix_sklearn(model, predictors, target):
    """
    To plot the confusion_matrix with percentages

    model: classifier
    predictors: independent variables
    target: dependent variable
    """
    y_pred = model.predict(predictors)
    cm = confusion_matrix(target, y_pred)
    labels = np.asarray(
        [
            ["{0:0.0f}".format(item) + "\n{0:.2%}".format(item / cm.flatten().sum())]
            for item in cm.flatten()
        ]
    ).reshape(2, 2)

    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=labels, fmt="")
    plt.ylabel("True label")
    plt.xlabel("Predicted label")


# ### Defining scorer to be used for cross-validation and hyperparameter tuning

# In[32]:


scorer = metrics.make_scorer(metrics.recall_score)


# ### Model Building on original data

# In[33]:


get_ipython().run_cell_magic('time', '', '\nmodels = [] \n\n# Appending models into the list\nmodels.append(("Logistic regression", LogisticRegression(random_state=1)))\nmodels.append(("Bagging", BaggingClassifier(random_state=1)))\nmodels.append(("Random forest", RandomForestClassifier(random_state=1)))\nmodels.append(("GBM", GradientBoostingClassifier(random_state=1)))\nmodels.append(("Adaboost", AdaBoostClassifier(random_state=1)))\nmodels.append(("Xgboost", XGBClassifier(random_state=1, eval_metric="logloss")))\nmodels.append(("dtree", DecisionTreeClassifier(random_state=1)))\n\nresults1 = []  \nnames = []  \n\n\nprint("\\n" "Cross-Validation performance on training dataset:" "\\n")\n\nfor name, model in models:\n    kfold = StratifiedKFold(\n        n_splits=5, shuffle=True, random_state=1\n    )  \n    cv_result = cross_val_score(\n        estimator=model, X=X_train, y=y_train, scoring = scorer,cv=kfold\n    )\n    results1.append(cv_result)\n    names.append(name)\n    print("{}: {}".format(name, cv_result.mean()))\n\nprint("\\n" "Validation Performance:" "\\n")\n\nfor name, model in models:\n    model.fit(X_train, y_train)\n    scores = recall_score(y_val, model.predict(X_val))\n    print("{}: {}".format(name, scores))')


# ### Plotting boxplots for CV scores of all models defined above

# In[34]:


fig = plt.figure(figsize=(10, 7))

fig.suptitle("Algorithm Comparison")
ax = fig.add_subplot(111)

plt.boxplot(results1)
ax.set_xticklabels(names)

plt.show()


# - We can see that Xgboost is giving the highest cross-validated recall followed by Dtree, Randomforest, and GBM.
# 
# - We will tune the Xgboost, GMB, and Random-forest models and see if the performance improves.
# 
# - Recall is very low, we can try oversampling (increase training data) to see if the model performance can be improved

# ### Model Building with oversampled data

# In[35]:


print("Before OverSampling, counts of label '1': {}".format(sum(y_train == 1)))
print("Before OverSampling, counts of label '0': {} \n".format(sum(y_train == 0)))

# Synthetic Minority Over Sampling Technique
sm = SMOTE(sampling_strategy=1, k_neighbors=5, random_state=1)
X_train_over, y_train_over = sm.fit_resample(X_train, y_train)


print("After OverSampling, counts of label '1': {}".format(sum(y_train_over == 1)))
print("After OverSampling, counts of label '0': {} \n".format(sum(y_train_over == 0)))


print("After OverSampling, the shape of train_X: {}".format(X_train_over.shape))
print("After OverSampling, the shape of train_y: {} \n".format(y_train_over.shape))


# In[36]:


get_ipython().run_cell_magic('time', '', '\nmodels = []  \n\n# Appending models into the list\nmodels.append(("Logistic regression", LogisticRegression(random_state=1)))\nmodels.append(("Bagging", BaggingClassifier(random_state=1)))\nmodels.append(("Random forest", RandomForestClassifier(random_state=1)))\nmodels.append(("GBM", GradientBoostingClassifier(random_state=1)))\nmodels.append(("Adaboost", AdaBoostClassifier(random_state=1)))\nmodels.append(("Xgboost", XGBClassifier(random_state=1, eval_metric="logloss")))\nmodels.append(("dtree", DecisionTreeClassifier(random_state=1)))\n\nresults1 = []  \nnames = []  \n\nprint("\\n" "Cross-Validation performance on training dataset:" "\\n")\n\nfor name, model in models:\n    kfold = StratifiedKFold(\n        n_splits=5, shuffle=True, random_state=1\n    )  # Setting number of splits equal to 5\n    cv_result = cross_val_score(\n        estimator=model, X=X_train, y=y_train, scoring = scorer,cv=kfold\n    )\n    results1.append(cv_result)\n    names.append(name)\n    print("{}: {}".format(name, cv_result.mean()))\n\nprint("\\n" "Validation Performance:" "\\n")\n\nfor name, model in models:\n    model.fit(X_train, y_train)\n    scores = recall_score(y_val, model.predict(X_val))\n    print("{}: {}".format(name, scores))')


# ### Plotting boxplots for CV scores of all over sampled data

# In[37]:


fig = plt.figure(figsize=(10, 7))

fig.suptitle("Algorithm Comparison")
ax = fig.add_subplot(111)

plt.boxplot(results1)
ax.set_xticklabels(names)

plt.show()


# ### Model Building with undersampled data

# In[38]:


rus = RandomUnderSampler(random_state=1, sampling_strategy=1)
X_train_under, y_train_under = rus.fit_resample(X_train, y_train)


print("Before UnderSampling, counts of label '1': {}".format(sum(y_train == 1)))
print("Before UnderSampling, counts of label '0': {} \n".format(sum(y_train == 0)))


print("After UnderSampling, counts of label '1': {}".format(sum(y_train_under == 1)))
print("After UnderSampling, counts of label '0': {} \n".format(sum(y_train_under == 0)))


print("After UnderSampling, the shape of train_X: {}".format(X_train_under.shape))
print("After UnderSampling, the shape of train_y: {} \n".format(y_train_under.shape))


# In[39]:


get_ipython().run_cell_magic('time', '', '\nmodels = []  \n\n# Appending models into the list\nmodels.append(("Logistic regression", LogisticRegression(random_state=1)))\nmodels.append(("Bagging", BaggingClassifier(random_state=1)))\nmodels.append(("Random forest", RandomForestClassifier(random_state=1)))\nmodels.append(("GBM", GradientBoostingClassifier(random_state=1)))\nmodels.append(("Adaboost", AdaBoostClassifier(random_state=1)))\nmodels.append(("Xgboost", XGBClassifier(random_state=1, eval_metric="logloss")))\nmodels.append(("dtree", DecisionTreeClassifier(random_state=1)))\n\nresults1 = []  # Empty list to store all model\'s CV scores\nnames = []  # Empty list to store name of the models\n\n# loop through all models to get the mean cross validated score\nprint("\\n" "Cross-Validation performance on training dataset:" "\\n")\n\nfor name, model in models:\n    kfold = StratifiedKFold(\n        n_splits=5, shuffle=True, random_state=1\n    )  # Setting number of splits equal to 5\n    cv_result = cross_val_score(\n        estimator=model, X=X_train, y=y_train, scoring = scorer,cv=kfold\n    )\n    results1.append(cv_result)\n    names.append(name)\n    print("{}: {}".format(name, cv_result.mean()))\n\nprint("\\n" "Validation Performance:" "\\n")\n\nfor name, model in models:\n    model.fit(X_train, y_train)\n    scores = recall_score(y_val, model.predict(X_val))\n    print("{}: {}".format(name, scores))')


# ### Plotting boxplots for CV scores of all undersampled data

# In[40]:


fig = plt.figure(figsize=(10, 7))

fig.suptitle("Algorithm Comparison")
ax = fig.add_subplot(111)

plt.boxplot(results1)
ax.set_xticklabels(names)

plt.show()


# **After looking at performance of all the models, let's decide which models can further improve with hyperparameter tuning.**

# ## Hyperparameter Tuning

# ## Decision-tree

# ### Tuning decision-tree using oversampled data

# In[41]:


get_ipython().run_cell_magic('time', '', '\n# defining model\nmodel = DecisionTreeClassifier(random_state=1)\n\n\n# Parameter grid to pass in RandomSearchCV\nparam_grid = {\n    "max_depth": np.arange(2, 6),\n    "min_samples_leaf": [1,2,3, 4, 7],\n    "max_leaf_nodes": [10, 15,20],\n    "min_impurity_decrease": [0.0001, 0.001],\n}\n\n\n\n#Calling RandomizedSearchCV\nrandomized_cv = RandomizedSearchCV(estimator=model, param_distributions=param_grid, n_jobs = -1, n_iter=50, scoring=scorer, cv=5, random_state=1)\n\n#Fitting parameters in RandomizedSearchCV\nrandomized_cv.fit(X_train_over,y_train_over)\n\nprint("Best parameters are {} with CV score={}:" .format(randomized_cv.best_params_,randomized_cv.best_score_))')


# In[42]:


# building model with best parameters
dt_over = DecisionTreeClassifier(
    max_depth=5, min_samples_leaf=4, max_leaf_nodes=15, min_impurity_decrease=0.0001,
)

# Fit the model on training data
dt_over.fit(X_train_over, y_train_over)


# In[43]:


## To check the performance on training set
dt_train_performance_over = model_performance_classification_sklearn(
    dt_over, X_train_over, y_train_over
)

print("Training performance on oversample decision-tree model:")
dt_train_performance_over


# In[44]:


print("confusion matrix training performance on oversampled decision-tree model:")
confusion_matrix_sklearn(dt_over, X_train_over, y_train_over)


# In[45]:


## To check the performance on validation set
dt_val_performance_over = model_performance_classification_sklearn(
    dt_over, X_val, y_val
)

print("validation performance on oversample decision-tree model:")
dt_val_performance_over


# In[46]:


confusion_matrix_sklearn(dt_over, X_val, y_val)


# ### Decision-tree model Building with undersampled data

# In[47]:


get_ipython().run_cell_magic('time', '', '\n# defining model\nmodel = DecisionTreeClassifier(random_state=1)\n\n\n# Parameter grid to pass in RandomSearchCV\nparam_grid = {\n    "max_depth": np.arange(2, 6),\n    "min_samples_leaf": [1,2,3, 4, 7],\n    "max_leaf_nodes": [10, 15,20],\n    "min_impurity_decrease": [0.0001, 0.001],\n}\n\n\n\n#Calling RandomizedSearchCV\nrandomized_cv = RandomizedSearchCV(estimator=model, param_distributions=param_grid, n_jobs = -1, n_iter=50, scoring=scorer, cv=5, random_state=1)\n\n#Fitting parameters in RandomizedSearchCV\nrandomized_cv.fit(X_train_under,y_train_under)\n\nprint("Best parameters are {} with CV score={}:" .format(randomized_cv.best_params_,randomized_cv.best_score_))')


# In[48]:


# building model with best parameters
dt_under = DecisionTreeClassifier(
    max_depth=5, min_samples_leaf=4, max_leaf_nodes=15, min_impurity_decrease=0.0001,
)

# Fit the model on training data
dt_under.fit(X_train_under, y_train_under)


# In[49]:


## To check the performance on training set
dt_train_performance_under = model_performance_classification_sklearn(
    dt_under, X_train_under, y_train_under
)

print("Training performance on undersample decision-tree model:")
dt_train_performance_under


# In[50]:


print("confusion matrix training performance on oversampled decision-tree model:")
confusion_matrix_sklearn(dt_under, X_train_under, y_train_under)


# In[51]:


## To check the performance on validation set
dt_val_performance_under = model_performance_classification_sklearn(
    dt_under, X_val, y_val
)

print("validation performance on underrsample decision-tree model:")
dt_val_performance_under


# - The tuned dtree model is  overfitting the training data
# - The validation precision is still less than 50% on bothe the validation set of the undersample and over sample data i.e. the model is not good at identifying potential default.

# ## Adaboost model

# ### Tuning AdaBoost using oversampled data

# In[52]:


get_ipython().run_cell_magic('time', '', '\nmodel = AdaBoostClassifier(random_state=1)\n\n\nparam_grid = {\n    "n_estimators": np.arange(10, 110, 10),\n    "learning_rate": [0.1, 0.01, 0.2, 0.05, 1],\n    "base_estimator": [\n        DecisionTreeClassifier(max_depth=1, random_state=1),\n        DecisionTreeClassifier(max_depth=2, random_state=1),\n        DecisionTreeClassifier(max_depth=3, random_state=1),\n    ],\n}\n\nscorer = metrics.make_scorer(metrics.recall_score)\n\nrandomized_cv = RandomizedSearchCV(estimator=model, param_distributions=param_grid, n_jobs = -1, n_iter=50, scoring=scorer, cv=5, random_state=1)\n\n#Fitting parameters in RandomizedSearchCV\nrandomized_cv.fit(X_train_over,y_train_over)\n\nprint("Best parameters are {} with CV score={}:" .format(randomized_cv.best_params_,randomized_cv.best_score_))')


# In[53]:


adb_tuned_over = AdaBoostClassifier(
    n_estimators=70,
    learning_rate=1,
    random_state=1,
    base_estimator=DecisionTreeClassifier(max_depth=3, random_state=1),
)

# Fit the model on training data
adb_tuned_over.fit(X_train_over, y_train_over)


# In[54]:


## To check the performance on training set
adb_train_performance_over = model_performance_classification_sklearn(
    adb_tuned_over, X_train_over, y_train_over
)

print("Training performance on oversample adaboost model:")
adb_train_performance_over


# In[55]:


print("confusion matrix training performance on oversampled adaboost model:")
confusion_matrix_sklearn(adb_tuned_over, X_train_over, y_train_over)


# In[56]:


## To check the performance on validation set
adb_val_performance_over = model_performance_classification_sklearn(
    adb_tuned_over, X_val, y_val
)

print("validation performance on oversample adaboost model:")
adb_val_performance_over


# In[57]:


print("confusion matrix validation performance on oversampled adaboost model:")
confusion_matrix_sklearn(adb_tuned_over, X_val, y_val)


# ### Adaboost model Building with undersampled data

# In[58]:


get_ipython().run_cell_magic('time', '', '\nmodel = AdaBoostClassifier(random_state=1)\n\n\nparam_grid = {\n    "n_estimators": np.arange(10, 110, 10),\n    "learning_rate": [0.1, 0.01, 0.2, 0.05, 1],\n    "base_estimator": [\n        DecisionTreeClassifier(max_depth=1, random_state=1),\n        DecisionTreeClassifier(max_depth=2, random_state=1),\n        DecisionTreeClassifier(max_depth=3, random_state=1),\n    ],\n}\n\nscorer = metrics.make_scorer(metrics.recall_score)\n\nrandomized_cv = RandomizedSearchCV(estimator=model, param_distributions=param_grid, n_jobs = -1, n_iter=50, scoring=scorer, cv=5, random_state=1)\n\n#Fitting parameters in RandomizedSearchCV\nrandomized_cv.fit(X_train_over,y_train_over)\n\nprint("Best parameters are {} with CV score={}:" .format(randomized_cv.best_params_,randomized_cv.best_score_))')


# In[59]:


adb_tuned_under = AdaBoostClassifier(
    n_estimators=70,
    learning_rate=1,
    random_state=1,
    base_estimator=DecisionTreeClassifier(max_depth=3, random_state=1),
)

# Fit the model on training data
adb_tuned_under.fit(X_train_over, y_train_over)


# In[60]:


## To check the performance on training set
adb_train_performance_under = model_performance_classification_sklearn(
    adb_tuned_under, X_train_under, y_train_under
)

print("Training performance on undersample adaboost model:")
adb_train_performance_under


# In[61]:


print("confusion matrix training performance on undersampled adaboost model:")
confusion_matrix_sklearn(adb_tuned_under, X_train_under, y_train_under)


# In[62]:


## To check the performance on validation set
adb_val_performance_under = model_performance_classification_sklearn(
    adb_tuned_under, X_val, y_val
)

print("Validation performance on undersample adaboost model:")
adb_val_performance_under


# In[63]:


print("confusion matrix validation performance on undersampled adaboost model:")
confusion_matrix_sklearn(adb_tuned_under, X_val, y_val)


# - The tuned adaboost model is overfitting the training data. However, it is a better model than decision tree.

# ## Gradient-boost model

# ### Tuning Gradient Boosting using oversampled data

# In[64]:


get_ipython().run_cell_magic('time', '', '\n# defining model\nModel = GradientBoostingClassifier(random_state=1)\n\n#Parameter grid to pass in RandomSearchCV\nparam_grid={"n_estimators": np.arange(50,150,25), "learning_rate": [0.2,0.01, 0.05, 1], "subsample":[0.3,0.4,0.5,0.7,0.6], "max_features":[0.3,0.4,0.5,0.6,0.7]}\n\n#Calling RandomizedSearchCV\nrandomized_cv = RandomizedSearchCV(estimator=Model, param_distributions=param_grid, scoring=scorer, n_iter=50, n_jobs = -1, cv=5, random_state=1)\n\n#Fitting parameters in RandomizedSearchCV\nrandomized_cv.fit(X_train, y_train)\n\nprint("Best parameters are {} with CV score={}:" .format(randomized_cv.best_params_,randomized_cv.best_score_))')


# In[65]:


gbm_tuned_over = GradientBoostingClassifier(
    max_features=0.7, random_state=1, learning_rate=1, n_estimators=75, subsample=0.3
)

gbm_tuned_over.fit(X_train, y_train)


# In[66]:


## To check the performance on training set
gbm_train_performance_over = model_performance_classification_sklearn(
    gbm_tuned_over, X_train_over, y_train_over
)

print("Training performance on oversample gradient-boost model:")
gbm_train_performance_over


# In[67]:


print("confusion matrix validation performance on undersampled gradient-boost model:")
confusion_matrix_sklearn(gbm_tuned_over, X_train_over, y_train_over)


# In[68]:


# To check the performance on validation set
gbm_train_performance_over = model_performance_classification_sklearn(
    gbm_tuned_over, X_val, y_val
)

print("Training performance on validation set:")
gbm_train_performance_over


# In[69]:


print("confusion matrix validation performance on oversampled gbm model:")
confusion_matrix_sklearn(gbm_tuned_over, X_val, y_val)


# ### Tuning Gradient Boosting using undersampled data

# In[70]:


get_ipython().run_cell_magic('time', '', '\n# defining model\nModel = GradientBoostingClassifier(random_state=1)\n\n#Parameter grid to pass in RandomSearchCV\nparam_grid={"n_estimators": np.arange(50,150,25), "learning_rate": [0.2,0.01, 0.05, 1], "subsample":[0.3,0.4,0.5,0.7,0.6], "max_features":[0.3,0.4,0.5,0.6,0.7]}\n\n#Calling RandomizedSearchCV\nrandomized_cv = RandomizedSearchCV(estimator=Model, param_distributions=param_grid, scoring=scorer, n_iter=50, n_jobs = -1, cv=5, random_state=1)\n\n#Fitting parameters in RandomizedSearchCV\nrandomized_cv.fit(X_train, y_train)\n\nprint("Best parameters are {} with CV score={}:" .format(randomized_cv.best_params_,randomized_cv.best_score_))')


# In[71]:


gbm_tuned_under = GradientBoostingClassifier(
    max_features=0.7, random_state=1, learning_rate=1, n_estimators=75, subsample=0.3
)

gbm_tuned_under.fit(X_train, y_train)


# In[72]:


## To check the performance on training set
gbm_train_performance_under = model_performance_classification_sklearn(
    gbm_tuned_under, X_train_under, y_train_under
)

print("Training performance on undersample gradient-boost model:")
gbm_train_performance_under


# In[73]:


print("confusion matrix on training performance on undersampled gbm-model:")
confusion_matrix_sklearn(gbm_tuned_under, X_train_under, y_train_under)


# In[74]:


# To check the performance on validation set
gbm_val_performance_under = model_performance_classification_sklearn(
    gbm_tuned_under, X_val, y_val
)

print("Training performance on validation set:")
gbm_val_performance_under


# In[75]:


print("confusion matrix validation performance on undersampled gradient-boost model:")
confusion_matrix_sklearn(adb_tuned_under, X_val, y_val)


# ## XGBoost

# ### Tuning XGBoost using oversampled data

# In[76]:


get_ipython().run_cell_magic('time', '', '\n# defining model\nModel = XGBClassifier(random_state=1,eval_metric=\'logloss\')\n\n#Parameter grid to pass in RandomSearchCV\nparam_grid={\'n_estimators\':[150,200,250],\'scale_pos_weight\':[5,10], \'learning_rate\':[0.1,0.2], \'gamma\':[0,3,5], \'subsample\':[0.8,0.9]}\n\n#Calling RandomizedSearchCV\nrandomized_cv = RandomizedSearchCV(estimator=Model, param_distributions=param_grid, n_iter=50, n_jobs = -1, scoring=scorer, cv=5, random_state=1)\n\n#Fitting parameters in RandomizedSearchCV\nrandomized_cv.fit(X_train, y_train) ## Complete the code to fit the model on over sampled data\n\nprint("Best parameters are {} with CV score={}:" .format(randomized_cv.best_params_,randomized_cv.best_score_))')


# In[77]:


xgb_tuned_over = XGBClassifier(
    max_features=0.7, random_state=1, learning_rate=1, n_estimators=75, subsample=0.3
)

xgb_tuned_over.fit(X_train, y_train)


# In[78]:


## To check the performance on training set
xgb_train_performance_over = model_performance_classification_sklearn(
    xgb_tuned_over, X_train_over, y_train_over
)

print("Training performance on oversample xgboost model:")
xgb_train_performance_over


# In[79]:


print("confusion matrix validation performance on oversampled xgb-model:")
confusion_matrix_sklearn(xgb_tuned_over, X_train_over, y_train_over)


# In[80]:


# To check ther performance on validation set
xgb_val_performance_over = model_performance_classification_sklearn(
    xgb_tuned_over, X_val, y_val
)

print("training performance on validation set:")
xgb_val_performance_over


# In[81]:


print("confusion matrix validation performance on overersampled xgb-model:")
confusion_matrix_sklearn(xgb_tuned_over, X_val, y_val)


# ### Tuning XGBoost using undersampled data

# In[82]:


xgb_tuned_under = XGBClassifier(
    max_features=0.7, random_state=1, learning_rate=1, n_estimators=75, subsample=0.3
)

xgb_tuned_under.fit(X_train, y_train)


# In[83]:


## To check the performance on training set
xgb_train_performance_under = model_performance_classification_sklearn(
    xgb_tuned_under, X_train_under, y_train_under
)

print("Training performance on oversample xgboost model:")
xgb_train_performance_under


# In[84]:


print("confusion matrix on training performance on undersampled xgboost:")
confusion_matrix_sklearn(gbm_tuned_under, X_train_under, y_train_under)


# In[85]:


# To check the performance on validation set
xgb_val_performance_under = model_performance_classification_sklearn(
    xgb_tuned_under, X_val, y_val
)

print("Training performance on validation set:")
xgb_val_performance_under


# In[86]:


print("confusion matrix validation performance on undersampled xgboost model:")
confusion_matrix_sklearn(xgb_tuned_under, X_val, y_val)


# ## Random-forest

# ### Tuning Random-forest using oversampled data

# In[87]:


get_ipython().run_cell_magic('time', '', '\n# defining model\nModel =RandomForestClassifier(random_state=1)\n\n# Parameter grid to pass in RandomSearchCV\nparam_grid = {\n    "n_estimators": [200,250,300],\n    "min_samples_leaf": np.arange(1, 4),\n    "max_features": [np.arange(0.3, 0.6, 0.1),\'sqrt\'],\n    "max_samples": np.arange(0.4, 0.7, 0.1)}\n\n\n#Calling RandomizedSearchCV\nrandomized_cv = RandomizedSearchCV(estimator=Model, param_distributions=param_grid, n_iter=50, n_jobs = -1, scoring=scorer, cv=5, random_state=1)\n\n#Fitting parameters in RandomizedSearchCV\nrandomized_cv.fit(X_train, y_train) ## Complete the code to fit the model on under sampled data\n\nprint("Best parameters are {} with CV score={}:" .format(randomized_cv.best_params_,randomized_cv.best_score_))')


# In[88]:


rf_tuned_over = RandomForestClassifier(
    random_state=1, max_samples=0.6, n_estimators=300, min_samples_leaf=1,
)

rf_tuned_over.fit(X_train, y_train)


# In[89]:


## To check the performance on training set
rf_train_performance_over = model_performance_classification_sklearn(
    rf_tuned_over, X_train_over, y_train_over
)

print("Training performance on oversample random-forest model:")
rf_train_performance_over


# In[90]:


print("confusion matrix validation performance on oversampled random-forest:")
confusion_matrix_sklearn(rf_tuned_over, X_train_over, y_train_over)


# In[91]:


# To check ther performance on validation set
rf_val_performance_over = model_performance_classification_sklearn(
    rf_tuned_over, X_val, y_val
)

print("training performance on validation set:")
rf_val_performance_over


# In[92]:


print("confusion matrix validation performance on overersampled random-forest:")
confusion_matrix_sklearn(rf_tuned_over, X_val, y_val)


# ### Tuning random-forest using undersampled data

# In[93]:


rf_tuned_under = RandomForestClassifier(
    random_state=1, max_samples=0.6, n_estimators=300, min_samples_leaf=1,
)

rf_tuned_under.fit(X_train, y_train)


# In[94]:


## To check the performance on training set
rf_train_performance_under = model_performance_classification_sklearn(
    rf_tuned_under, X_train_under, y_train_under
)

print("Training performance on undersample random-forest model:")
rf_train_performance_under


# In[95]:


print("confusion matrix on training performance on undersampled random-forest:")
confusion_matrix_sklearn(rf_tuned_under, X_train_under, y_train_under)


# In[96]:


# To check the performance on validation set
rf_val_performance_under = model_performance_classification_sklearn(
    rf_tuned_under, X_val, y_val
)

print("Training performance on validation set:")
rf_val_performance_under


# In[97]:


print("confusion matrix validation performance on undersampled random-forest:")
confusion_matrix_sklearn(rf_tuned_under, X_val, y_val)


# **After tuning all the models, next step is to compare the performance of all tuned models and see which one is the best.**

# In[98]:


# training performance comparison

models_train_comp_df = pd.concat(
    [
        dt_train_performance_under.T,
        adb_train_performance_over.T,
        gbm_train_performance_under.T,
        xgb_train_performance_under.T,
        rf_train_performance_under.T,
    ],
    axis=1,
)
models_train_comp_df.columns = [
    "Decision tree tuned with undersampled data",
    "AdaBoost classifier tuned with oversampled data",
    "Gradientboost with undersampled data",
    "XGBoost tuned with oversampled data",
    "Random forest tuned with undersampled data",
]
print("Training performance comparison:")
models_train_comp_df


# In[99]:


# training performance comparison

models_val_comp_df = pd.concat(
    [
        dt_val_performance_under.T,
        adb_val_performance_over.T,
        gbm_val_performance_under.T,
        xgb_val_performance_under.T,
        rf_val_performance_under.T,
    ],
    axis=1,
)
models_val_comp_df.columns = [
    "Decision tree tuned with undersampled data",
    "AdaBoost classifier tuned with oversampled data",
    "Gradientboost with undersampled data",
    "XGBoost tuned with undersampled data",
    "Random forest tuned with undersampled data",
]
print("Validation performance comparison:")
models_val_comp_df


# - The XGBoost is giving overall accuracy on both training and validation tuned sets..
# - Let's check the model's performance on test set and then see the feature importance from the tuned XGB model

# **Now we have our final model, so let's find out how our final model is performing on unseen test data.**

# In[100]:


pipeline_model = model_performance_classification_sklearn(
    xgb_tuned_over, X_test, Y_test
)
pipeline_model


# In[101]:


feature_names = X.columns
importances = xgb_tuned_over.feature_importances_
indices = np.argsort(importances)

plt.figure(figsize=(12, 12))
plt.title("Feature Importances")
plt.barh(range(len(indices)), importances[indices], color="violet", align="center")
plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
plt.xlabel("Relative Importance")
plt.show()


# ## Pipelines

# ### Building pipeline

# In[102]:


# creating a transformer for numerical variables, which will apply simple imputer on the numerical variables
numeric_transformer = Pipeline(steps=[("imputer", SimpleImputer(strategy="median"))])



# creating a transformer for categorical variables, which will first apply simple imputer and 
#then do one hot encoding for categorical variables
categorical_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ]
)
# handle_unknown = "ignore", allows model to handle any unknown category in the test data

# combining categorical transformer and numerical transformer using a column transformer

preprocessor = ColumnTransformer(
    transformers=[
       
        
    ],
    remainder="passthrough",
)
# remainder = "passthrough" has been used, it will allow variables that are present in original data 
# but not in "numerical_columns" and "categorical_columns" to pass through the column transformer without any changes


# In[103]:


# Creating new pipeline with best parameters
pipeline_model = Pipeline(
    steps=[
        ("pre", preprocessor),
        (
            "XGB",
            XGBClassifier(
                max_features=0.7,
                random_state=1,
                learning_rate=1,
                n_estimators=50,
                subsample=0.2,
            ),
        ),
    ]
)
# Fit the model on training data
pipeline_model.fit(X_train, y_train)


# In[104]:


pipeline_model.score(X_train, y_train)


# In[105]:


pipeline_model.score(X_val, y_val)


# In[106]:


X1 = data2.drop(columns="Target")
Y1 = data2["Target"]


# In[107]:


# Let's check the performance on test set
model_test = model_performance_classification_sklearn(pipeline_model, X_test, Y_test)
model_test


# In[ ]:





# In[ ]:





# ## Business Insights and Recommendations

# ### Insights

# ## Data Description
# - The data provided is a transformed version of original data which was collected using sensors.
# - Train.csv - To be used for training and tuning of models. 
# - Test.csv - To be used only for testing the performance of the final best model.
# - Both the datasets consist of 40 predictor variables and 1 target variable

# In[126]:


sns.set_style("darkgrid")
data.hist(figsize=(15, 10))
plt.show()


# In[127]:


plt.figure(figsize=(18, 10))
sns.heatmap(data.corr(), annot=True, vmin=-1, vmax=1, fmt=".2f", cmap="Spectral")
plt.show()


# - All the variables are evenly distributed.
# - V1 and V2 from the training(18) and testing(5 and 6 respectively) datasets have missing values.
# - Only 6% from both the traing and testing test is the reading as “failure” 
# - V32 has the highest and lowest read from the is V32, with 26.38 and 17.2 respectively
# - Correlation seems to be randomly distruted in the datasets.

# ## Model Building

# The nature of predictions made by the classification model will translate as follows:
# 
# - True positives (TP) are failures correctly predicted by the model.
# - False negatives (FN) are real failures in a generator where there is no detection by model. 
# - False positives (FP) are failure detections in a generator where there is no failure.
# 
# **Which metric to optimize?**
# 
# * We need to choose the metric which will ensure that the maximum number of generator failures are predicted correctly by the model.
# * We would want Recall to be maximized as greater the Recall, the higher the chances of minimizing false negatives.
# * We want to minimize false negatives because if a model predicts that a machine will have no failure when there will be a failure, it will increase the maintenance cost.

# In[128]:


print("Training performance comparison:")
models_train_comp_df


# In[129]:


print("Validation performance comparison:")
models_val_comp_df


# - We can see that Xgboost is giving the highest cross-validated recall followed by Dtree, Randomforest, and GBM.
# 
# - We will tune the Xgboost, GMB, and Random-forest models and see if the performance improves.
# 
# - Recall is very low, we can try oversampling (increase training data) to see if the model performance can be improved
# 
# - All the models are giving a generalized performance on training and test set.
# 
# - The highest recall is .992% on the training set.
# 
# - However using a model that generally fir on validation set is sthe best practise, hence XGBoostClassifier.
# 
# - XGBoostClassifier  model minimizes real failures (False negatives (FN)) in a generator where there is no detection by model, therefore increasing the inspection cost, and decreasing maintenance, and repair cost.

# In[130]:


feature_names = X.columns
importances = xgb_tuned_over.feature_importances_
indices = np.argsort(importances)

plt.figure(figsize=(12, 12))
plt.title("Feature Importances")
plt.barh(range(len(indices)), importances[indices], color="violet", align="center")
plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
plt.xlabel("Relative Importance")
plt.show()


# - V18 is the most important feature, followed by V21 and V21.
# 
# - V19 has the least important feature.

# ### Insights and Conclusion

# - XGBoosClassifier is recomended for the best model in detecting failures in the generators.
# 
# - Data from various environmental factors like the temperature, humidity, wind speed, etc. are needed for predictive maintenace.
# 
# - I recomend inspecting the V32 sensor as it is gving both the highest and lowest readings.
