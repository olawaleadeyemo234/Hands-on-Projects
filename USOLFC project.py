#!/usr/bin/env python
# coding: utf-8

# ## EasyVisa Project

# ## Problem Statement

# ### Context:
# 
# Business communities in the United States are facing high demand for human resources, but one of the constant challenges is identifying and attracting the right talent, which is perhaps the most important element in remaining competitive. Companies in the United States look for hard-working, talented, and qualified individuals both locally as well as abroad.
# 
# The Immigration and Nationality Act (INA) of the US permits foreign workers to come to the United States to work on either a temporary or permanent basis. The act also protects US workers against adverse impacts on their wages or working conditions by ensuring US employers' compliance with statutory requirements when they hire foreign workers to fill workforce shortages. The immigration programs are administered by the Office of Foreign Labor Certification (OFLC).
# 
# OFLC processes job certification applications for employers seeking to bring foreign workers into the United States and grants certifications in those cases where employers can demonstrate that there are not sufficient US workers available to perform the work at wages that meet or exceed the wage paid for the occupation in the area of intended employment.
# 
# ### Objective:
# 
# In FY 2016, the OFLC processed 775,979 employer applications for 1,699,957 positions for temporary and permanent labor certifications. This was a nine percent increase in the overall number of processed applications from the previous year. The process of reviewing every case is becoming a tedious task as the number of applicants is increasing every year.
# 
# The increasing number of applicants every year calls for a Machine Learning based solution that can help in shortlisting the candidates having higher chances of VISA approval. OFLC has hired the firm EasyVisa for data-driven solutions. You as a data  scientist at EasyVisa have to analyze the data provided and, with the help of a classification model:
# 
# * Facilitate the process of visa approvals.
# * Recommend a suitable profile for the applicants for whom the visa should be certified or denied based on the drivers that significantly influence the case status. 
# 
# ### Data Description
# 
# The data contains the different attributes of employee and the employer. The detailed data dictionary is given below.
# 
# * case_id: ID of each visa application
# * continent: Information of continent the employee
# * education_of_employee: Information of education of the employee
# * has_job_experience: Does the employee has any job experience? Y= Yes; N = No
# * requires_job_training: Does the employee require any job training? Y = Yes; N = No 
# * no_of_employees: Number of employees in the employer's company
# * yr_of_estab: Year in which the employer's company was established
# * region_of_employment: Information of foreign worker's intended region of employment in the US.
# * prevailing_wage:  Average wage paid to similarly employed workers in a specific occupation in the area of intended employment. The purpose of the prevailing wage is to ensure that the foreign worker is not underpaid compared to other workers offering the same or similar service in the same area of employment. 
# * unit_of_wage: Unit of prevailing wage. Values include Hourly, Weekly, Monthly, and Yearly.
# * full_time_position: Is the position of work full-time? Y = Full Time Position; N = Part Time Position
# * case_status:  Flag indicating if the Visa was certified or denied

# ## Importing necessary libraries

# In[1]:


import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
import seaborn as sns

pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", 100)


from sklearn.ensemble import (
    BaggingClassifier,
    RandomForestClassifier,
    AdaBoostClassifier,
    GradientBoostingClassifier,
    StackingClassifier,
)

from xgboost import XGBClassifier
from sklearn.tree import DecisionTreeClassifier

from sklearn import metrics
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

from sklearn.model_selection import GridSearchCV


# ## Importing Dataset

# In[2]:


visa = pd.read_csv("Downloads/Easyvisa.csv")  ##  Fill the blank to read the data


# In[3]:


data = visa.copy()


# ## Overview of the Dataset

# ### View the first and last 5 rows of the dataset

# In[4]:


data.head(5)  


# In[5]:


data.tail(5) 


# ### Understand the shape of the dataset

# In[6]:


data.shape 


# * The data set has 25480 rows and columns

# ### Check the data types of the columns for the dataset

# In[7]:


data.info()


# * no of employees, year of establishment, and prevailing wage are nummeric features while rest are objects.
# * There is no null-values in the data set

# In[8]:


data.duplicated().sum()


# In[9]:


data.isna().sum()  


# * There are no duplicate values in the data

# ## Exploratory Data Analysis

# #### Let's check the statistical summary of the data

# In[10]:


data.describe().T 


# * The number of emplyee have a negative value; likely an anomaly. 
# * Year of employee is betweeen 1800 - 2016
# * The median prevailing wage is 70,308. There is a huge differnfe between maximum value and the 75th percentile; likely an outlier in this column

# #### Fixing the negative values in number of employees columns

# In[11]:


data.loc[data["no_of_employees"] < 0].shape


# In[12]:


data["no_of_employees"] = abs(data["no_of_employees"])


# * Changing the 33 negative observations as data entry errors and taking the absolute values for this column.

# #### Let's check the count of each unique category in each of the categorical variables

# In[13]:


cat_col = list(data.select_dtypes("object").columns)
for column in cat_col:
    print(data[column].value_counts())
    print("-" * 50)


# * I cahnged the list of all categorical variables to object to get details information from the data sets.
# * Asians have the highest number of job applications.
# * Most of the application have a bachelors degree.
# * Most applicants have job experience and do not recquire training.
# * The Northe-East region has the highest number of location for apllicant.
# * Most of the visa application are for full time job postions
# * The target column "case status" is imbalance with many applicantnhaving a certifes visa

# In[14]:


data["case_id"].nunique() 


# In[15]:


data.drop(
    ["case_id"], axis=1, inplace=True) 


# * Dropping the "case-id" column because they are unigue and are not significant in the data set

# ### Univariate Analysis

# In[16]:


def histogram_boxplot(data, feature, figsize=(15, 10), kde=False, bins=None):
    """
    Boxplot and histogram combined

    data: dataframe
    feature: dataframe column
    figsize: size of figure (default (15,10))
    kde: whether to show the density curve (default False)
    bins: number of bins for histogram (default None)
    """
    f2, (ax_box2, ax_hist2) = plt.subplots(
        nrows=2,  
        sharex=True,  
        gridspec_kw={"height_ratios": (0.25, 0.75)},
        figsize=figsize,
    )  
    sns.boxplot(
        data=data, x=feature, ax=ax_box2, showmeans=True, color="violet"
    )  
    sns.histplot(
        data=data, x=feature, kde=kde, ax=ax_hist2, bins=bins
    ) if bins else sns.histplot(
        data=data, x=feature, kde=kde, ax=ax_hist2
    )  
    ax_hist2.axvline(
        data[feature].mean(), color="green", linestyle="--"
    )  
    ax_hist2.axvline(
        data[feature].median(), color="black", linestyle="-"
    )  


# #### Observations on number of employees

# In[17]:


histogram_boxplot(data, "no_of_employees")


# * The number of employees distribution is right skewed.
# * Some comoaniees have more than 500, 000 number of employees, probably international companies.

# #### Observations on prevailing wage

# In[18]:


histogram_boxplot(data, "prevailing_wage")  


# In[19]:


data.loc[data["prevailing_wage"] < 100, "unit_of_wage"].shape 


# In[20]:


data.loc[data["prevailing_wage"] < 100]


# * The prevailing wage is right skewed
# * 176 applicant get paid 0$, probaly intern positions, futehr investigation is needed.

# ### Categorical Varibale EDA

# In[21]:


def labeled_barplot(data, feature, perc=False, n=None):
    """
    Barplot with percentage at the top

    data: dataframe
    feature: dataframe column
    perc: whether to display percentages instead of count (default is False)
    n: displays the top n category levels (default is None, i.e., display all levels)
    """

    total = len(data[feature])  
    count = data[feature].nunique()
    if n is None:
        plt.figure(figsize=(count + 2, 6))
    else:
        plt.figure(figsize=(n + 2, 6))

    plt.xticks(rotation=90, fontsize=15)
    ax = sns.countplot(
        data=data,
        x=feature,
        palette="Paired",
        order=data[feature].value_counts().index[:n],
    )

    for p in ax.patches:
        if perc == True:
            label = "{:.1f}%".format(
                100 * p.get_height() / total
            )
        else:
            label = p.get_height() 

        x = p.get_x() + p.get_width() / 2 
        y = p.get_height()  

        ax.annotate(
            label,
            (x, y),
            ha="center",
            va="center",
            size=12,
            xytext=(0, 5),
            textcoords="offset points",
        )  

    plt.show()  


# #### Observations on continent

# In[22]:


labeled_barplot(data, "continent", perc=True)


# * The highest applicant are from Asia, lowest from Oceania

# #### Observations on education of employee

# In[23]:


labeled_barplot(data, "education_of_employee", perc=True)


# * Highest applicant have a masters degree lowest have Doctorate degree

# #### Observations on job experience

# In[24]:


labeled_barplot(data, "has_job_experience", perc=True)


# * Most applicant have job experience prior to visa application

# #### Observations on job training

# In[25]:


labeled_barplot(data, "requires_job_training", perc=True)


# * 88.4% of the applicant do not require training.

# #### Observations on region of employment

# In[26]:


labeled_barplot(data, "region_of_employment", perc=True)


# * North-East, South and West rrgion have highest nimber of applicants
# * The Island with the lowest

# #### Observations on unit of wage

# In[27]:


labeled_barplot(data, "unit_of_wage", perc=True)


# * 90.1% of the applicant have a yearly unit of the wag, folllw by Hour, Week, and Month.

# #### Observations on case status

# In[28]:


labeled_barplot(data, "case_status", perc=True)


# * 66.8% of the visas were certified.

# In[29]:


from pandas_profiling import ProfileReport

prof = ProfileReport(data)
prof


# ### Bivariate Analysis

# In[30]:


cols_list = data.select_dtypes(include=np.number).columns.tolist()

plt.figure(figsize=(10, 5))
sns.heatmap(data[cols_list].corr(), annot=True, vmin=-1, vmax=1, fmt=".2f", cmap="Spectral")
plt.show()


# * There's little to no correlation within the independent features of the data.

# **Further Analysis.**

# In[31]:


def distribution_plot_wrt_target(data, predictor, target):

    fig, axs = plt.subplots(2, 2, figsize=(12, 10))

    target_uniq = data[target].unique()

    axs[0, 0].set_title("Distribution of target for target=" + str(target_uniq[0]))
    sns.histplot(
        data=data[data[target] == target_uniq[0]],
        x=predictor,
        kde=True,
        ax=axs[0, 0],
        color="teal",
        stat="density",
    )

    axs[0, 1].set_title("Distribution of target for target=" + str(target_uniq[1]))
    sns.histplot(
        data=data[data[target] == target_uniq[1]],
        x=predictor,
        kde=True,
        ax=axs[0, 1],
        color="orange",
        stat="density",
    )

    axs[1, 0].set_title("Boxplot w.r.t target")
    sns.boxplot(data=data, x=target, y=predictor, ax=axs[1, 0], palette="gist_rainbow")

    axs[1, 1].set_title("Boxplot (without outliers) w.r.t target")
    sns.boxplot(
        data=data,
        x=target,
        y=predictor,
        ax=axs[1, 1],
        showfliers=False,
        palette="gist_rainbow",
    )

    plt.tight_layout()
    plt.show()


# In[32]:


def stacked_barplot(data, predictor, target):
    """
    Print the category counts and plot a stacked bar chart

    data: dataframe
    predictor: independent variable
    target: target variable
    """
    count = data[predictor].nunique()
    sorter = data[target].value_counts().index[-1]
    tab1 = pd.crosstab(data[predictor], data[target], margins=True).sort_values(
        by=sorter, ascending=False
    )
    print(tab1)
    print("-" * 120)
    tab = pd.crosstab(data[predictor], data[target], normalize="index").sort_values(
        by=sorter, ascending=False
    )
    tab.plot(kind="bar", stacked=True, figsize=(count + 5, 5))
    plt.legend(
        loc="lower left", frameon=False,
    )
    plt.legend(loc="upper left", bbox_to_anchor=(1, 1))
    plt.show()


# * Do level of education have any impact on visa certification?

# In[33]:


stacked_barplot(data, "education_of_employee", "case_status")


# * Education seems to have a positive relationship with the certification of visa that is higher the education higher are the chances of visa getting certified.
# * Around 85% of the visa applications got certified for the applicants with Doctorate degree. While 80% of the visa applications got certified for the applicants with Master's degree.
# * Around 60% of the visa applications got certified for applicants with Bachelor's degrees.
# * Applicants who do not have a degree and have graduated from high school are more likely to have their applications denied.

# #### Are there regional requirements for talent in diverse educational backgrounds?

# In[34]:


plt.figure(figsize=(10, 5))
sns.heatmap(pd.crosstab(data.education_of_employee, data.region_of_employment),
    annot=True,
    fmt="g",
    cmap="viridis") 

plt.ylabel("Education")
plt.xlabel("Region")
plt.show()


# * The requirement for the applicants who have passed high school is most in the South region, followed by Northeast region.
# * The requirement for Bachelor's is mostly in South region, followed by West region.
# * The requirement for Master's is most in Northeast region, followed by South region.
# * The requirement for Doctorate's is mostly in West region, followed by Northeast region.

# ####  Percentage of visa certifications across each region

# In[35]:


stacked_barplot(data, "region_of_employment", "case_status")  


# * Midwest region sees the highest number of visa certifications - around 75%, followed by the south region that sees around 70% of the visa applications getting certified.
# * Island, West, and Northeast region has an almost equal percentage of visa certifications.

# #### Visa status vary across different continents.

# In[36]:


stacked_barplot(data, "continent", "case_status")


# * Applications from Europe and Africa have a higher chance of getting certified.
# * Around 80% of the applications from Europe are certified.
# * Asia has the third-highest percentage (Around 60%) of visa certification and has the highest number of applications.

# #### Experienced professionals across visa appplication and case status. 

# In[37]:


stacked_barplot(data, "has_job_experience", "case_status")


# * Having job experience seems to be a key differentiator between visa applications getting certified or denied.
# * Around 80% of the applications were certified for the applicants who have some job experience as compared to the applicants who do not have any job experience. 
# * Applicants without job experiences saw only 60% of the visa applications getting certified.

# #### Do the employees who have prior work experience require any job training?

# In[38]:


stacked_barplot(data, "has_job_experience", "requires_job_training")


# * Less percentage of applicants require job training if they have prior work experience.

# #### "Prevailing wage" across "case status"

# In[39]:


distribution_plot_wrt_target(data, "prevailing_wage", "case_status")


# In[40]:


sns.displot(data, x="prevailing_wage", hue="case_status", kind="kde")


# * The median prevailing wage for the certified applications is slightly higher as compared to denied applications.

# #### Checking if the prevailing wage is similar across all the regions of the US

# In[41]:


plt.figure(figsize=(10, 5))
sns.boxplot(
    data=data, x="region_of_employment", y="prevailing_wage"
)  ## Complete the code to create boxplot for region of employment and prevailing wage
plt.show()


# * Midwest and Island regions have slightly higher prevailing wages as compared to other regions. 
# * The distribution of prevailing wage is similar across West, Northeast, and South regions.

# #### The prevailing wage has different units (Hourly, Weekly, etc). Let's find out if it has any impact on visa applications getting certified.

# In[42]:


stacked_barplot(data, "unit_of_wage", "case_status")


# * Unit of prevailing wage is an important factor for differentiating between a certified and a denied visa application.
# * If the unit of prevailing wage is Yearly, there's a high chance of the application getting certified.
# * Around 75% of the applications were certified for the applicants who have a yearly unit of wage. While only 35% of the applications were certified for applicants who have an hourly unit of wage.
# * Monthly and Weekly units of prevailing wage have the same percentage of visa applications getting certified.

# ## Data Preprocessing

# ### Outlier Check
# 
# - Let's check for outliers in the data.

# In[43]:


numeric_columns = data.select_dtypes(include=np.number).columns.tolist()

plt.figure(figsize=(15, 12))

for i, variable in enumerate(numeric_columns):
    plt.subplot(4, 4, i + 1)
    plt.boxplot(data[variable], whis=1.5)
    plt.tight_layout()
    plt.title(variable)

plt.show()


# * There are quite a few outliers in the data.

# ### Data Preparation for modeling - predicting which visa will be certified.

# In[44]:


data["case_status"] = data["case_status"].apply(lambda x: 1 if x == "Certified" else 0)

X = data.drop(["case_status"], axis=1)
Y = data["case_status"]

X = pd.get_dummies(X, drop_first=True) 

X_train, X_test, y_train, y_test = train_test_split(
    X, Y, test_size=0.3, random_state=1)  


# In[45]:


print("Shape of Training set : ", X_train.shape)
print("Shape of test set : ", X_test.shape)
print("Percentage of classes in training set:")
print(y_train.value_counts(normalize=True))
print("Percentage of classes in test set:")
print(y_test.value_counts(normalize=True))


# ## Model evaluation criterion

# ### Model can make wrong predictions as:
# 
# 1. Model predicts that the visa application will get certified but in reality, the visa application should get denied.
# 2. Model predicts that the visa application will not get certified but in reality, the visa application should get certified. 
# 
# ### Which case is more important? 
# * Both the cases are important as:
# 
# * If a visa is certified when it had to be denied a wrong employee will get the job position while US citizens will miss the opportunity to work on that position.
# 
# * If a visa is denied when it had to be certified the U.S. will lose a suitable human resource that can contribute to the economy. 
# 
# 
# 
# ### How to reduce the losses?
# 
# * `F1 Score` can be used a the metric for evaluation of the model, greater the F1  score higher are the chances of minimizing False Negatives and False Positives. 
# * We will use balanced class weights so that model focuses equally on both classes.

# **First, let's create functions to calculate different metrics and confusion matrix so that we don't have to use the same code repeatedly for each model.**
# * The model_performance_classification_sklearn function will be used to check the model performance of models. 
# * The confusion_matrix_sklearn function will be used to plot the confusion matrix.

# In[46]:


def model_performance_classification_sklearn(model, predictors, target):
    """
    Function to compute different metrics to check classification model performance

    model: classifier
    predictors: independent variables
    target: dependent variable
    """

    pred = model.predict(predictors)

    acc = accuracy_score(target, pred)  
    recall = recall_score(target, pred)  
    precision = precision_score(target, pred) 
    f1 = f1_score(target, pred)  

    
    df_perf = pd.DataFrame(
        {"Accuracy": acc, "Recall": recall, "Precision": precision, "F1": f1,},
        index=[0],)

    return df_perf


# In[47]:


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


# ## Decision Tree - Model Building and Hyperparameter Tuning

# ### Decision Tree Model

# In[48]:


dtree = DecisionTreeClassifier(random_state=1)
dtree.fit(X_train, y_train)  ## Complete the code to fit decision tree on train data


# #### Checking model performance on training set

# In[49]:


confusion_matrix_sklearn(dtree, X_train, y_train)


# In[50]:


dtree.score(X_train, y_train)


# decision_tree_perf_train = model_performance_classification_sklearn(
#     dtree, X_train, y_train)
# decision_tree_perf_train

# * 0 errors on the training set, each sample has been classified correctly.
# * Model has performed very well on the training set.

# #### Checking model performance on test set

# In[51]:


confusion_matrix_sklearn(dtree, X_test, y_test)


# In[52]:


decision_tree_perf_test = model_performance_classification_sklearn(
    dtree, X_test, y_test)
decision_tree_perf_test


# * The decision tree model is overfitting the data as expected and not able to generalize well on the test set.
# Pruning is recomended.

# ### Hyperparameter Tuning - Decision Tree

# In[53]:


dtree_estimator = DecisionTreeClassifier(
    class_weight={0: 0.35, 1: 0.65}, random_state=1
)


parameters = {
    "max_depth": np.arange(2, 10),
    "min_samples_leaf": [5, 7, 10, 15],
    "max_leaf_nodes": [2, 3, 5, 10, 15],
    "min_impurity_decrease": [0.0001, 0.001, 0.01, 0.1],
}


scorer = metrics.make_scorer(metrics.recall_score)


grid_obj = GridSearchCV(dtree_estimator, parameters, scoring=scorer, n_jobs=-1)
grid_obj = grid_obj.fit(X_train, y_train)


dtree_estimator = grid_obj.best_estimator_


dtree_estimator.fit(X_train, y_train)


# In[54]:


confusion_matrix_sklearn(dtree_estimator, X_train, y_train)


# In[55]:


dtree_estimator_model_train_perf = model_performance_classification_sklearn(
    dtree_estimator, X_train, y_train
)
dtree_estimator_model_train_perf


# In[56]:


confusion_matrix_sklearn(dtree_estimator, X_test, y_test)


# In[57]:


dtree_estimator_model_test_perf = model_performance_classification_sklearn(
    dtree_estimator, X_test, y_test)
dtree_estimator_model_test_perf


# * The decision tree model has a very high recall but, the precision is quite less.
# * The performance of the model after hyperparameter tuning has become generalized.
# * Getting an F1 score of 0.79 and 0.80 on the training and test set, respectively.
# * Building some ensemble models and see if the metrics improve.

# ## Bagging - Model Building and Hyperparameter Tuning

# ### Bagging Classifier

# In[58]:


bagging = BaggingClassifier(random_state=1)
bagging.fit(X_train, y_train)


# #### Checking model performance on training set

# In[59]:


confusion_matrix_sklearn(bagging, X_train, y_train)


# In[60]:


bagging_model_train_perf = model_performance_classification_sklearn(
    bagging, X_train, y_train
)
bagging_model_train_perf


# #### Checking model performance on test set

# In[61]:


confusion_matrix_sklearn(bagging, X_test, y_test)


# In[62]:


bagging_model_test_perf = model_performance_classification_sklearn(
    bagging, X_test, y_test
)
bagging_model_train_perf


# * The bagging classifier is overfitting on the training set like the decision tree model.
# * Reducing overfitting and improve the performance by hyperparameter tuning.

# ### Hyperparameter Tuning - Bagging Classifier

# In[63]:


bagging_estimator_tuned = BaggingClassifier(random_state=1)


parameters = {
    "max_samples": [0.7, 0.9],
    "max_features": [0.7, 0.9],
    "n_estimators": np.arange(90, 120, 10),}


acc_scorer = metrics.make_scorer(metrics.f1_score)


grid_obj = GridSearchCV(bagging_estimator_tuned, parameters, scoring=acc_scorer, cv=5)  
grid_obj = grid_obj.fit(X_train, y_train) 


bagging_estimator_tuned = grid_obj.best_estimator_


bagging_estimator_tuned.fit(X_train, y_train)


# #### Checking model performance on training set

# In[64]:


confusion_matrix_sklearn(bagging_estimator_tuned, X_train, y_train)


# In[65]:


bagging_estimator_tuned_model_train_perf = model_performance_classification_sklearn(bagging_estimator_tuned, X_train, y_train)
bagging_estimator_tuned_model_train_perf


# #### Checking model performance on test set

# In[66]:


confusion_matrix_sklearn(bagging_estimator_tuned, X_test, y_test)


# In[67]:


bagging_estimator_tuned_model_test_perf = model_performance_classification_sklearn(bagging_estimator_tuned, X_test, y_test)
bagging_estimator_tuned_model_test_perf


# * After tuning the hyperparameters the bagging classifier is still overfitting.
# * However, there's a big difference in the training and the test recall.

# ### Random Forest

# In[68]:


rf_estimator = RandomForestClassifier(random_state=1, class_weight="balanced")
rf_estimator.fit(X_train, y_train)


# #### Checking model performance on training set

# In[69]:


confusion_matrix_sklearn(rf_estimator, X_train, y_train)


# In[70]:


rf_estimator_model_train_perf = model_performance_classification_sklearn(
    rf_estimator, X_train, y_train
)
rf_estimator_model_train_perf


# #### Checking model performance on test set

# In[71]:


confusion_matrix_sklearn(rf_estimator, X_test, y_test)


# In[72]:


rf_estimator_model_test_perf = model_performance_classification_sklearn(
    rf_estimator, X_test, y_test
)
rf_estimator_model_test_perf


# - With default parameters, random forest is overfitting the training data.
# - Reducing overfitting and improving recall by hyperparameter tuning.

# ### Hyperparameter Tuning - Random Forest

# In[73]:


rf_tuned = RandomForestClassifier(random_state=1, oob_score=True, bootstrap=True)

parameters = {
    "max_depth": list(np.arange(5, 15, 5)),
    "max_features": ["sqrt", "log2"],
    "min_samples_split": [3, 5, 7],
    "n_estimators": np.arange(10, 40, 10),
}


acc_scorer = metrics.make_scorer(metrics.f1_score)


grid_obj = GridSearchCV(rf_tuned, parameters, scoring=acc_scorer, cv=5, n_jobs=-1)
grid_obj = grid_obj.fit(X_train, y_train)


rf_tuned = grid_obj.best_estimator_


rf_tuned.fit(X_train, y_train)


# #### Checking model performance on training set

# In[74]:


confusion_matrix_sklearn(rf_tuned, X_train, y_train)


# In[75]:


rf_tuned_model_train_perf = model_performance_classification_sklearn(
    rf_tuned, X_train, y_train
)
rf_tuned_model_train_perf


# #### Checking model performance on test set

# In[76]:


confusion_matrix_sklearn(rf_tuned, X_test, y_test)


# In[77]:


rf_tuned_model_test_perf = model_performance_classification_sklearn(
    rf_tuned, X_test, y_test)

rf_tuned_model_test_perf


# After hyperparameter tuning the model performance has generalized.
# F1 score of 0.83 and 0.82 on the training and test data, respectively.
# The model has a high recall and a good precision.

# ## Boosting - Model Building and Hyperparameter Tuning

# ### AdaBoost Classifier

# In[78]:


ab_classifier = AdaBoostClassifier(random_state=1)
ab_classifier.fit(X_train, y_train)


# #### Checking model performance on training set

# In[79]:


confusion_matrix_sklearn(ab_classifier, X_train, y_train)


# In[80]:


ab_classifier_model_train_perf = model_performance_classification_sklearn(
    ab_classifier, X_train, y_train
)
ab_classifier_model_train_perf


# #### Checking model performance on test set

# In[81]:


confusion_matrix_sklearn(ab_classifier, X_test, y_test)


# In[82]:


ab_classifier_model_test_perf = model_performance_classification_sklearn(
    ab_classifier, X_test, y_test
)
ab_classifier_model_test_perf


# The model is giving a generalized performance.
# We have received a good F1 score of 0.81 and 0.82 on both the training and test set.

# ### Hyperparameter Tuning - AdaBoost Classifier

# In[83]:


abc_tuned = AdaBoostClassifier(random_state=1)

parameters = {
   
    "base_estimator": [
        DecisionTreeClassifier(max_depth=1, class_weight="balanced", random_state=1),
        DecisionTreeClassifier(max_depth=2, class_weight="balanced", random_state=1),
    ],
    "n_estimators": np.arange(80, 101, 10),
    "learning_rate": np.arange(0.1, 0.4, 0.1),
}


acc_scorer = metrics.make_scorer(metrics.f1_score)


grid_obj = GridSearchCV(abc_tuned, parameters, scoring=acc_scorer, cv=5) ## Complete the code to run grid search with cv = 5
grid_obj = grid_obj.fit(X_train, y_train) ## Complete the code to fit the grid_obj on train data

abc_tuned = grid_obj.best_estimator_

abc_tuned.fit(X_train, y_train)


# #### Checking model performance on training set

# In[84]:


confusion_matrix_sklearn(abc_tuned, X_train, y_train)


# In[85]:


abc_tuned_model_train_perf = model_performance_classification_sklearn(
    abc_tuned, X_train, y_train)

abc_tuned_model_train_perf


# #### Checking model performance on test set

# In[86]:


confusion_matrix_sklearn(abc_tuned, X_test, y_test)## Complete the code to create confusion matrix for test data on tuned estimator


# In[87]:


abc_tuned_model_test_perf = model_performance_classification_sklearn(
    abc_tuned, X_test, y_test)

abc_tuned_model_test_perf


# * After tuning the F1 score has reduced.
# * The recall of the model has reduced but the precision has improved.

# ### Gradient Boosting Classifier

# In[88]:


gb_classifier = GradientBoostingClassifier(random_state=1)
gb_classifier.fit(X_train, y_train)


# #### Checking model performance on training set

# In[89]:


confusion_matrix_sklearn(gb_classifier, X_train, y_train) 


# In[90]:


gb_classifier_model_train_perf = model_performance_classification_sklearn(
    gb_classifier, X_train, y_train)

gb_classifier_model_train_perf


# #### Checking model performance on test set

# In[91]:


confusion_matrix_sklearn(gb_classifier, X_test, y_test)


# In[92]:


gb_classifier_model_test_perf = model_performance_classification_sklearn(
    gb_classifier, X_test, y_test)

gb_classifier_model_test_perf


# * The model is giving a good and generalized performance.
# * We are getting the F1 score of 0.82 and 0.82 on the training and test set, respectively.
# * Let's see if the performance can be improved further by hyperparameter tuning.

# ### Hyperparameter Tuning - Gradient Boosting Classifier

# In[114]:


gbc_tuned = GradientBoostingClassifier(
    init=AdaBoostClassifier(random_state=1), random_state=1
)

parameters = {
    "n_estimators": [200, 250],
    "subsample": [0.9, 1],
    "max_features": [0.8, 0.9],
    "learning_rate": np.arange(0.1, 0.21, 0.1),
}

acc_scorer = metrics.make_scorer(metrics.f1_score)

grid_obj = GridSearchCV(gbc_tuned, parameters, scoring=acc_scorer, cv=5) 
grid_obj = grid_obj.fit(X_train, y_train)

gbc_tuned = grid_obj.best_estimator_

gbc_tuned.fit(X_train, y_train)


# #### Checking model performance on training set

# In[115]:


confusion_matrix_sklearn(gbc_tuned, X_train, y_train)


# In[116]:


gbc_tuned_model_train_perf = model_performance_classification_sklearn(
    gbc_tuned, X_train, y_train
)
gbc_tuned_model_train_perf 


# #### Checking model performance on test set

# In[117]:


confusion_matrix_sklearn(gbc_tuned, X_test, y_test) ## Complete the code to create confusion matrix for test data on tuned estimator


# In[118]:


gbc_tuned_model_test_perf = model_performance_classification_sklearn(
    gbc_tuned, X_test, y_test
)
gbc_tuned_model_test_perf


# * After tuning there is not much change in the model performance as compared to the model with default values of hyperparameters.

# ### XGBoost Classifier

# In[119]:


xgb_classifier = XGBClassifier(random_state=1, eval_metric="logloss")
xgb_classifier.fit(X_train, y_train)


# #### Checking model performance on training set

# In[120]:


confusion_matrix_sklearn(xgb_classifier, X_train, y_train) ## Complete the code to create confusion matrix for train data


# In[121]:


xgb_classifier_model_train_perf = model_performance_classification_sklearn(
    xgb_classifier, X_train, y_train)## Complete the code to check performance on train data
xgb_classifier_model_train_perf


# #### Checking model performance on test set

# In[122]:


confusion_matrix_sklearn(xgb_classifier, X_test, y_test) ## Complete the code to create confusion matrix for test data


# In[123]:


xgb_classifier_model_test_perf = model_performance_classification_sklearn(
    xgb_classifier, X_test, y_test
)## Complete the code to check performance for test data
xgb_classifier_model_test_perf


# ### Hyperparameter Tuning - XGBoost Classifier

# In[124]:


# Choose the type of classifier.
xgb_tuned = XGBClassifier(random_state=1, eval_metric="logloss")

# Grid of parameters to choose from
parameters = {
    "n_estimators": np.arange(150, 250, 50),
    "scale_pos_weight": [1, 2],
    "subsample": [0.9, 1],
    "learning_rate": np.arange(0.1, 0.21, 0.1),
    "gamma": [3, 5],
    "colsample_bytree": [0.8, 0.9],
    "colsample_bylevel": [ 0.9, 1],
}

# Type of scoring used to compare parameter combinations
acc_scorer = metrics.make_scorer(metrics.f1_score)

# Run the grid search
grid_obj = GridSearchCV(xgb_tuned, parameters, scoring=acc_scorer, cv=5) ## Complete the code to run grid search with cv = 5
grid_obj = grid_obj.fit(X_train, y_train)## Complete the code to fit the grid_obj on train data

# Set the clf to the best combination of parameters
xgb_tuned = grid_obj.best_estimator_

# Fit the best algorithm to the data.
xgb_tuned.fit(X_train, y_train)


# #### Checking model performance on training set

# In[125]:


confusion_matrix_sklearn(xgb_tuned, X_train, y_train) ## Complete the code to create confusion matrix for train data on tuned estimator


# In[126]:


xgb_tuned_model_train_perf = model_performance_classification_sklearn(
    xgb_tuned, X_train, y_train
) ## Complete the code to check performance for train data on tuned estimator
xgb_tuned_model_train_perf


# #### Checking model performance on test set

# In[127]:


confusion_matrix_sklearn(xgb_tuned, X_test, y_test) ## Complete the code to create confusion matrix for test data on tuned estimator


# In[128]:


xgb_tuned_model_test_perf = model_performance_classification_sklearn(
    xgb_tuned, X_test, y_test
) ## Complete the code to check performance for test data on tuned estimator
xgb_tuned_model_test_perf


# ## Stacking Classifier

# In[129]:


estimators = [
    ("AdaBoost", ab_classifier),
    ("Gradient Boosting", gbc_tuned),
    ("Random Forest", rf_tuned),
]

final_estimator = xgb_tuned

stacking_classifier = StackingClassifier(
    estimators=estimators, final_estimator=final_estimator
) ## Complete the code to define Stacking Classifier

stacking_classifier.fit(X_train, y_train)## Complete the code to fit Stacking Classifier on the train data


# ### Checking model performance on training set

# In[130]:


confusion_matrix_sklearn(stacking_classifier, X_train, y_train)## Complete the code to create confusion matrix for train data


# In[131]:


stacking_classifier_model_train_perf = model_performance_classification_sklearn(
    stacking_classifier, X_train, y_train
) ## Complete the code to check performance on train data
stacking_classifier_model_train_perf


# ### Checking model performance on test set

# In[132]:


confusion_matrix_sklearn(stacking_classifier, X_train, y_train)## Complete the code to create confusion matrix for test dataconfusion_matrix_sklearn(stacking_classifier, X_test, y_test)


# In[133]:


stacking_classifier_model_test_perf = model_performance_classification_sklearn(
    stacking_classifier, X_test, y_test
) ## Complete the code to check performance for test data
stacking_classifier_model_test_perf


# ## Model Performance Comparison and Final Model Selection

# In[136]:


# training performance comparison

models_train_comp_df = pd.concat(
    [
        dtree_estimator_model_train_perf.T,
        ab_classifier_model_train_perf.T,
        abc_tuned_model_train_perf.T,
        gb_classifier_model_train_perf.T,
        gbc_tuned_model_train_perf.T,
        xgb_classifier_model_train_perf.T,
        xgb_tuned_model_train_perf.T,
        stacking_classifier_model_train_perf.T,
    ],
    axis=1,
)
models_train_comp_df.columns = [
    "Decision Tree",
    "Adaboost Classifier",
    "Tuned Adaboost Classifier",
    "Gradient Boost Classifier",
    "Tuned Gradient Boost Classifier",
    "XGBoost Classifier",
    "XGBoost Classifier Tuned",
    "Stacking Classifier",
]
print("Training performance comparison:")
models_train_comp_df


# In[137]:


# testing performance comparison

# testing performance comparison

models_test_comp_df = pd.concat(
    [
        dtree_estimator_model_test_perf.T,
        dtree_estimator_model_test_perf.T,
        rf_estimator_model_test_perf.T,
        rf_tuned_model_test_perf.T,
        ab_classifier_model_test_perf.T,
        abc_tuned_model_test_perf.T,
        gb_classifier_model_test_perf.T,
        gbc_tuned_model_test_perf.T,
        xgb_classifier_model_test_perf.T,
        xgb_tuned_model_test_perf.T,
        stacking_classifier_model_test_perf.T,
    ],
    axis=1,
)
models_test_comp_df.columns = [
    "Decision Tree",
    "Tuned Decision Tree",
    "Random Forest",
    "Tuned Random Forest",
    "Adaboost Classifier",
    "Tuned Adaboost Classifier",
    "Gradient Boost Classifier",
    "Tuned Gradient Boost Classifier",
    "XGBoost Classifier",
    "XGBoost Classifier Tuned",
    "Stacking Classifier",
]
print("Testing performance comparison:")
models_test_comp_df ## Complete the code to check performance for test data


# ### Important features of the final model

# In[138]:


feature_names = X_train.columns
importances = gb_classifier.feature_importances_
indices = np.argsort(importances)

plt.figure(figsize=(12, 12))
plt.title("Feature Importances")
plt.barh(range(len(indices)), importances[indices], color="violet", align="center")
plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
plt.xlabel("Relative Importance")
plt.show()


# ## Business Insights and Recommendations

# #### The profile of the applicants for whom the visa status can be approved:
# 
# Primary information to look at:
# 
# Education level - At least has a Bachelor's degree - Master's and doctorate are preferred.
# Job Experience - Should have some job experience.
# Prevailing wage - The median prevailing wage of the employees for whom the visa got certified is around 72k.
# Secondary information to look at:
# 
# Unit of Wage - Applicants having a yearly unit of wage.
# Continent - Ideally the nationality and ethnicity of an applicant shouldn't matter to work in a country but previously it has been observed that applicants from Europe, Africa, and Asia have higher chances of visa certification.
# Region of employment - Our analysis suggests that the applications to work in the Mid-West region have more chances of visa approval. The approvals can also be made based on requirement of talent, from our analysis we see that:
# The requirement for the applicants who have passed high school is most in the South region, followed by Northeast region.
# The requirement for Bachelor's is mostly in South region, followed by West region.
# The requirement for Master's is most in Northeast region, followed by South region.
# The requirement for Doctorate's is mostly in West region, followed by Northeast region.
# 
# #### The profile of the applicants for whom the visa status can be denied:
# 
# Primary information to look at:
# 
# Education level - Doesn't have any degree and has completed high school.
# Job Experience - Doesn't have any job experience.
# Prevailing wage - The median prevailing wage of the employees for whom the visa got certified is around 65k.
# Secondary information to look at:
# 
# Unit of Wage - Applicants having an hourly unit of wage.
# Continent - Ideally the nationality and ethnicity of an applicant shouldn't matter to work in a country but previously it has been observed that applicants from South America, North America, and Oceania have higher chances of visa applications getting denied.
# 
# #### Additional information of employers and employees can be collected to gain better insights. Information such as:
# Employers: Information about the wage they are offering to the applicant, Sector in which company operates in, etc
# Employee's: Specialization in their educational degree, Number of years of experience, etc
# 

# In[ ]:




