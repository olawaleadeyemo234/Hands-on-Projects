#!/usr/bin/env python
# coding: utf-8

# # ExtraaLearn Project
# ***Marks: 60***
# 
# ## Context
# 
# The EdTech industry has been surging in the past decade immensely, and according to a forecast, the Online Education market would be worth $286.62bn by 2023 with a compound annual growth rate (CAGR) of 10.26% from 2018 to 2023. The modern era of online education has enforced a lot in its growth and expansion beyond any limit. Due to having many dominant features like ease of information sharing, personalized learning experience, transparency of assessment, etc, it is now preferable to traditional education. 
# 
# In the present scenario due to the Covid-19, the online education sector has witnessed rapid growth and is attracting a lot of new customers. Due to this rapid growth, many new companies have emerged in this industry. With the availability and ease of use of digital marketing resources, companies can reach out to a wider audience with their offerings. The customers who show interest in these offerings are termed as leads. There are various sources of obtaining leads for Edtech companies, like
# 
# * The customer interacts with the marketing front on social media or other online platforms. 
# * The customer browses the website/app and downloads the brochure
# * The customer connects through emails for more information.
# 
# The company then nurtures these leads and tries to convert them to paid customers. For this, the representative from the organization connects with the lead on call or through email to share further details.
# 
# ## Objective
# 
# ExtraaLearn is an initial stage startup that offers programs on cutting-edge technologies to students and professionals to help them upskill/reskill. With a large number of leads being generated regularly, one of the issues faced by ExtraaLearn is to identify which of the leads are more likely to convert so that they can allocate resources accordingly. You, as a data scientist at ExtraaLearn, have been provided the leads data to:
# * Analyze and build an ML model to help identify which leads are more likely to convert to paid customers, 
# * Find the factors driving the lead conversion process
# * Create a profile of the leads which are likely to convert
# 
# 
# ## Data Description
# 
# The data contains the different attributes of leads and their interaction details with ExtraaLearn. The detailed data dictionary is given below.
# 
# 
# **Data Dictionary**
# * ID: ID of the lead
# * age: Age of the lead
# * current_occupation: Current occupation of the lead. Values include 'Professional','Unemployed',and 'Student'
# * first_interaction: How did the lead first interact with ExtraaLearn. Values include 'Website', 'Mobile App'
# * profile_completed: What percentage of the profile has been filled by the lead on the website/mobile app. Values include Low - (0-50%), Medium - (50-75%), High (75-100%)
# * website_visits: How many times has a lead visited the website
# * time_spent_on_website: Total time spent on the website
# * page_views_per_visit: Average number of pages on the website viewed during the visits.
# * last_activity: Last interaction between the lead and ExtraaLearn. 
#     * Email Activity: Seeking for details about the program through email, Representative shared information with a lead like a brochure of program, etc 
#     * Phone Activity: Had a Phone Conversation with a representative, Had conversation over SMS with a representative, etc
#     * Website Activity: Interacted on live chat with a representative, Updated profile on the website, etc
# 
# * print_media_type1: Flag indicating whether the lead had seen the ad of ExtraaLearn in the Newspaper.
# * print_media_type2: Flag indicating whether the lead had seen the ad of ExtraaLearn in the Magazine.
# * digital_media: Flag indicating whether the lead had seen the ad of ExtraaLearn on the digital platforms.
# * educational_channels: Flag indicating whether the lead had heard about ExtraaLearn in the education channels like online forums, discussion threads, educational websites, etc.
# * referral: Flag indicating whether the lead had heard about ExtraaLearn through reference.
# * status: Flag indicating whether the lead was converted to a paid customer or not.

# ### **Please read the instructions carefully before starting the project.** 
# This is a commented Jupyter IPython Notebook file in which all the instructions and tasks to be performed are mentioned. 
# * Blanks '_______' are provided in the notebook that 
# needs to be filled with an appropriate code to get the correct result. With every '_______' blank, there is a comment that briefly describes what needs to be filled in the blank space. 
# * Identify the task to be performed correctly, and only then proceed to write the required code.
# * Fill the code wherever asked by the commented lines like "# write your code here" or "# complete the code". Running incomplete code may throw error.
# * Please run the codes in a sequential manner from the beginning to avoid any unnecessary errors.
# * Add the results/observations (wherever mentioned) derived from the analysis in the presentation and submit the same.

# In[1]:


# this will help in making the Python code more structured automatically (good coding practice)


get_ipython().run_line_magic('load_ext', 'nb_black')

import warnings

warnings.filterwarnings("ignore")
from statsmodels.tools.sm_exceptions import ConvergenceWarning

warnings.simplefilter("ignore", ConvergenceWarning)

# Libraries to help with reading and manipulating data

import pandas as pd
import numpy as np

# Library to split data
from sklearn.model_selection import train_test_split

# libaries to help with data visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Removes the limit for the number of displayed columns
pd.set_option("display.max_columns", None)
# Sets the limit for the number of displayed rows
pd.set_option("display.max_rows", 200)
# setting the precision of floating numbers to 5 decimal points
pd.set_option("display.float_format", lambda x: "%.5f" % x)

# To build model for prediction
import statsmodels.stats.api as sms
from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels.api as sm
from statsmodels.tools.tools import add_constant
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
from sklearn import tree

# To tune different models
from sklearn.model_selection import GridSearchCV


# To get diferent metric scores
from sklearn.metrics import (
    f1_score,
    accuracy_score,
    recall_score,
    precision_score,
    confusion_matrix,
    roc_auc_score,
    plot_confusion_matrix,
    precision_recall_curve,
    roc_curve,
    make_scorer,
)


# ## Import Dataset

# In[2]:


learn = pd.read_csv("Extraaalearn.csv")  ##  Complete the code to read the data


# In[3]:


# copying data to another variable to avoid any changes to original data
data = learn


# ### View the first and last 5 rows of the dataset

# In[4]:


data.head(5)  ##  Complete the code to view top 5 rows of the data


# In[5]:


data.tail(5)  ##  Complete the code to view last 5 rows of the data


# ### Understand the shape of the dataset

# In[6]:


data.shape  ## Complete the code to get the shape of data


# ### Check the data types of the columns for the dataset

# In[7]:


data.info()


# In[8]:


data.dtypes.value_counts()


# In[9]:


data.isnull().sum()


# ## Exploratory Data Analysis

# **Let's check the statistical summary of the data.**

# In[10]:


data.describe().T


# In[11]:


data.describe(exclude="number").T


# In[12]:


# Making a list of all catrgorical variables
cat_col = list(data.select_dtypes("object").columns)

# Printing number of count of each unique value in each column
for column in cat_col:
    print(data[column].value_counts())
    print("-" * 50)


# In[13]:


# checking the number of unique values
data["ID"].unique()  # Complete the code to check the number of unique values


# In[14]:


data.drop(
    ["ID"], axis=1, inplace=True
)  # Complete the code to drop "ID" column from data


# In[15]:


data.info()


# In[16]:


data.head(10)


# In[17]:


data.info


# ### Univariate Analysis

# In[18]:


# function to plot a boxplot and a histogram along the same scale.


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
        nrows=2,  # Number of rows of the subplot grid= 2
        sharex=True,  # x-axis will be shared among all subplots
        gridspec_kw={"height_ratios": (0.25, 0.75)},
        figsize=figsize,
    )  # creating the 2 subplots
    sns.boxplot(
        data=data, x=feature, ax=ax_box2, showmeans=True, color="violet"
    )  # boxplot will be created and a star will indicate the mean value of the column
    sns.histplot(
        data=data, x=feature, kde=kde, ax=ax_hist2, bins=bins, palette="winter"
    ) if bins else sns.histplot(
        data=data, x=feature, kde=kde, ax=ax_hist2
    )  # For histogram
    ax_hist2.axvline(
        data[feature].mean(), color="green", linestyle="--"
    )  # Add mean to the histogram
    ax_hist2.axvline(
        data[feature].median(), color="black", linestyle="-"
    )  # Add median to the histogram


# ### Observations on age

# In[19]:


histogram_boxplot(data, "age")


# ### Observations on website_visits

# In[20]:


histogram_boxplot(data, "website_visits") # Complete the code to plot a histogram_boxplot for website_visits


# In[21]:


# To check how many leads have not visited web-site
data[data["website_visits"] == 0].shape


# ### Observations on number of time_spent_on_website

# In[22]:


histogram_boxplot(data, "time_spent_on_website")
 # Complete the code to plot a histogram_boxplot for time_spent_on_website


# ### Observations on number of page_views_per_visit

# In[23]:


histogram_boxplot(data, "page_views_per_visit")


# In[24]:


# function to create labeled barplots


def labeled_barplot(data, feature, perc=False, n=None):
    """
    Barplot with percentage at the top

    data: dataframe
    feature: dataframe column
    perc: whether to display percentages instead of count (default is False)
    n: displays the top n category levels (default is None, i.e., display all levels)
    """

    total = len(data[feature])  # length of the column
    count = data[feature].nunique()
    if n is None:
        plt.figure(figsize=(count + 1, 5))
    else:
        plt.figure(figsize=(n + 1, 5))

    plt.xticks(rotation=90, fontsize=15)
    ax = sns.countplot(
        data=data,
        x=feature,
        palette="Paired",
        order=data[feature].value_counts().index[:n].sort_values(),
    )

    for p in ax.patches:
        if perc == True:
            label = "{:.1f}%".format(
                100 * p.get_height() / total
            )  # percentage of each class of the category
        else:
            label = p.get_height()  # count of each level of the category

        x = p.get_x() + p.get_width() / 2  # width of the plot
        y = p.get_height()  # height of the plot

        ax.annotate(
            label,
            (x, y),
            ha="center",
            va="center",
            size=12,
            xytext=(0, 5),
            textcoords="offset points",
        )  # annotate the percentage

    plt.show()  # show the plot


# ### Observations on current_occupation

# In[25]:


labeled_barplot(data, "current_occupation", perc=True)


# ### Observations on number of first_interaction

# In[26]:


labeled_barplot(data, "current_occupation")


# ### Observations on profile_completed

# In[27]:


labeled_barplot(data, "first_interaction")


# ### Observations on last_activity

# In[28]:


labeled_barplot(
    data, "last_activity"
)  # Complete the code to plot labeled_barplot for last_activity


# ### Observations on print_media_type1

# In[29]:


labeled_barplot(data, "print_media_type1")


# ### Observations on print_media_type2

# In[30]:


labeled_barplot(
    data, "print_media_type2"
)  # Complete the code to plot labeled_barplot for print_media_type2


# ### Observations on digital_media

# In[31]:


labeled_barplot(data, "digital_media")


# ### Observations on educational_channels

# In[32]:


labeled_barplot(
    data, "educational_channels"
)  # Complete the code to plot labeled_barplot for educational_channels


# ### Observations on referral

# In[33]:


labeled_barplot(
    data, "referral"
)  # Complete the code to plot labeled_barplot for referral


# ### Observations on status

# In[34]:


labeled_barplot(data, "status")  # Complete the code to plot labeled_barplot for status


# ### Bivariate Analysis

# In[35]:


cols_list = data.select_dtypes(include=np.number).columns.tolist()

plt.figure(figsize=(12, 7))
sns.heatmap(
    data[cols_list].corr(), annot=True, vmin=-1, vmax=1, fmt=".2f", cmap="Spectral"
)
plt.show()


# In[36]:


sns.pairplot(data)
plt.show()


# **Creating functions that will help us with further analysis.**

# In[37]:


### function to plot distributions wrt target


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


# In[38]:


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


# **Leads will have different expectations from the outcome of the course and the current occupation may play a key role for them to take the program. Let's analyze it**

# In[39]:


stacked_barplot(data, "current_occupation", "status")


# **Age can be a good factor to differentiate between such leads**

# In[40]:


plt.figure(figsize=(10, 5))
sns.boxplot(data["current_occupation"], data["age"])
plt.show()


# In[41]:


data.groupby(["current_occupation"])["age"].describe()


# **The company's first interaction with leads should be compelling and persuasive. Let's see if the channels of the first interaction have an impact on the conversion of leads**

# In[42]:


stacked_barplot(data, "first_interaction", "status")


# In[43]:


distribution_plot_wrt_target(data, "time_spent_on_website", "status")


# In[44]:


# checking the median value
data.groupby(["status"])["time_spent_on_website"].median()


# **Let's do a similar analysis for time spent on website and page views per visit.**

# **People browsing the website or the mobile app are generally required to create a profile by sharing their personal details before they can access more information. Let's see if the profile completion level has an impact on lead status**

# In[45]:


stacked_barplot(
    data, "profile_completed", "status"
)  # Complete the code to plot stacked_barplot for profile_completed and status


# **After a lead shares their information by creating a profile, there may be interactions between the lead and the company to proceed with the process of enrollment. Let's see how the last activity impacts lead conversion status**

# In[46]:


stacked_barplot(
    data, "last_activity", "status"
)  # Complete the code to plot stacked_barplot for last_activity and status


# **Let's see how advertisement and referrals impact the lead status**

# In[47]:


stacked_barplot(
    data, "print_media_type1", "status"
)  # Complete the code to plot stacked_barplot for print_media_type1 and status


# In[48]:


stacked_barplot(
    data, "print_media_type2", "status"
)  # Complete the code to plot stacked_barplot for print_media_type2 and status


# In[49]:


stacked_barplot(data, "digital_media", "status")


# In[50]:


stacked_barplot(
    data, "educational_channels", "status"
)  # Complete the code to plot stacked_barplot for educational_channels and status


# In[51]:


stacked_barplot(
    data, "referral", "status"
)  # Complete the code to plot stacked_barplot for referral and status


# In[52]:


plt.figure(figsize=(15, 7))
sns.heatmap(data.corr(), annot=True, vmin=-1, vmax=1, fmt=".2f", cmap="Spectral")
plt.show()


# ### Outlier Check
# 
# - Let's check for outliers in the data.

# In[53]:


# outlier detection using boxplot
numeric_columns = data.select_dtypes(include=np.number).columns.tolist()
# dropping release_year as it is a temporal variable
numeric_columns.remove("status")

plt.figure(figsize=(15, 12))

for i, variable in enumerate(numeric_columns):
    plt.subplot(4, 4, i + 1)
    plt.boxplot(data[variable], whis=1.5)
    plt.tight_layout()
    plt.title(variable)

plt.show()


# ### Data Preparation for modeling
# 
# - We want to predict which lead is more likely to be converted.
# - Before we proceed to build a model, we'll have to encode categorical features.
# - We'll split the data into train and test to be able to evaluate the model that we build on the train data.

# In[54]:


X = data.drop(["status"], axis=1)
Y = data["status"] # Complete the code to define the dependent (target) variable

X = pd.get_dummies(X, drop_first = True) # Complete the code to get dummies for X

# Splitting the data in 70:30 ratio for train to test data
X_train, X_test, y_train, y_test = train_test_split(
    X, Y, test_size=0.30, random_state=1
)



# In[55]:


print("Shape of Training set : ", X_train.shape)
print("Shape of test set : ", X_test.shape)
print("Percentage of classes in training set:")
print(y_train.value_counts(normalize=True))
print("Percentage of classes in test set:")
print(y_test.value_counts(normalize=True))


# ### Model evaluation criterion
# 
# ### Model can make wrong predictions as:
# 
# 1. Predicting a lead will not be converted to a paid customer in reality, the lead would have converted to a paid customer.
# 2. Predicting a lead will be converted to a paid customer in reality, the lead would not have converted to a paid customer. 
# 
# ### Which case is more important? 
# 
# * If we predict that a lead will not get converted and the lead would have converted then the company will lose a potential customer. 
# 
# * If we predict that a lead will get converted and the lead doesn't get converted the company might lose resources by nurturing false-positive cases.
# 
# Losing a potential customer is a greater loss.
# 
# ### How to reduce the losses?
# 
# * Company would want `Recall` to be maximized, greater the Recall score higher are the chances of minimizing False Negatives. 

# #### First, let's create functions to calculate different metrics and confusion matrix so that we don't have to use the same code repeatedly for each model.
# * The model_performance_classification_statsmodels function will be used to check the model performance of models. 
# * The confusion_matrix_statsmodels function will be used to plot the confusion matrix.

# In[56]:


# defining a function to compute different metrics to check performance of a classification model built using statsmodels
def model_performance_classification_statsmodels(
    model, predictors, target, threshold=0.5
):
    """
    Function to compute different metrics to check classification model performance

    model: classifier
    predictors: independent variables
    target: dependent variable
    threshold: threshold for classifying the observation as class 1
    """

    # checking which probabilities are greater than threshold
    pred_temp = model.predict(predictors) > threshold
    # rounding off the above values to get classes
    pred = np.round(pred_temp)

    acc = accuracy_score(target, pred)  # to compute Accuracy
    recall = recall_score(target, pred)  # to compute Recall
    precision = precision_score(target, pred)  # to compute Precision
    f1 = f1_score(target, pred)  # to compute F1-score

    # creating a dataframe of metrics
    df_perf = pd.DataFrame(
        {"Accuracy": acc, "Recall": recall, "Precision": precision, "F1": f1,},
        index=[0],
    )

    return df_perf


# In[57]:


# defining a function to plot the confusion_matrix of a classification model


def confusion_matrix_statsmodels(model, predictors, target, threshold=0.5):
    """
    To plot the confusion_matrix with percentages

    model: classifier
    predictors: independent variables
    target: dependent variable
    threshold: threshold for classifying the observation as class 1
    """
    y_pred = model.predict(predictors) > threshold
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


# ### Logistic Regression (with statsmodels library)

# In[58]:


X = data.drop(["status"], axis=1)
Y = data["status"]


# adding constant
X = sm.add_constant(X)  ## Complete the code to add constant to X
X = pd.get_dummies(X, drop_first=True)  # Complete the code to get dummies for X

# Splitting data in train and test sets
(X_train, X_test, y_train, y_test,) = train_test_split(X, Y, test_size=0.3, random_state=1)


# In[59]:


# fitting logistic regression model
logit = sm.Logit(y_train, X_train.astype(float))
lg = logit.fit(disp=False)

print(lg.summary())  # Complete the code to get model summary


# In[60]:


print("Training performance:")
model_performance_classification_statsmodels(lg, X_train, y_train)


# ### Multicollinearity

# In[61]:


# we will define a function to check VIF
def checking_vif(predictors):
    vif = pd.DataFrame()
    vif["feature"] = predictors.columns

    # calculating VIF for each feature
    vif["VIF"] = [
        variance_inflation_factor(predictors.values, i)
        for i in range(len(predictors.columns))
    ]
    return vif


# In[62]:


checking_vif(X_train)


# ### Dropping high p-value variables
# 
# - We will drop the predictor variables having a p-value greater than 0.05 as they do not significantly impact the target variable.
# - But sometimes p-values change after dropping a variable. So, we'll not drop all variables at once.
# - Instead, we will do the following:
#     - Build a model, check the p-values of the variables, and drop the column with the highest p-value.
#     - Create a new model without the dropped feature, check the p-values of the variables, and drop the column with the highest p-value.
#     - Repeat the above two steps till there are no columns with p-value > 0.05.
# 
# The above process can also be done manually by picking one variable at a time that has a high p-value, dropping it, and building a model again. But that might be a little tedious and using a loop will be more efficient.

# In[63]:


# initial list of columns
cols = X_train.columns.tolist()

# setting an initial max p-value
max_p_value = 1

while len(cols) > 0:
    # defining the train set
    x_train_aux = X_train[cols]

    # fitting the model
    model = sm.Logit(y_train, x_train_aux).fit(disp=False)

    # getting the p-values and the maximum p-value
    p_values = model.pvalues
    max_p_value = max(p_values)

    # name of the variable with maximum p-value
    feature_with_p_max = p_values.idxmax()

    if max_p_value > 0.05:
        cols.remove(feature_with_p_max)
    else:
        break

selected_features = cols
print(selected_features)


# In[64]:


X_train1 = X_train[selected_features]
X_test1 = X_test[selected_features]


# In[65]:


logit1 = sm.Logit(
    y_train, X_train1.astype(float)
)  ## Complete the code to train logistic regression on X_train1 and y_train
lg1 = logit1.fit(disp=False)  ## Complete the code to fit logistic regression
print(lg1.summary())  ## Complete the code to print summary of the model


logit1 = sm.Logit(y_train, X_train1.astype(float))
lg1 = logit1.fit(disp=False)
print(lg1.summary())


# In[66]:


print("Training performance:")
model_performance_classification_statsmodels(
    lg1, X_train1, y_train
)  ## Complete the code to check performance on X_train1 and y_train


# ###  Converting coefficients to odds
# * The coefficients of the logistic regression model are in terms of log(odd), to find the odds we have to take the exponential of the coefficients. 
# * Therefore, **odds =  exp(b)**
# * The percentage change in odds is given as **odds = (exp(b) - 1) * 100**

# In[67]:


# converting coefficients to odds
odds = np.exp(lg1.params)

# finding the percentage change
perc_change_odds = (np.exp(lg1.params) - 1) * 100

# removing limit from number of columns to display
pd.set_option("display.max_columns", None)

# adding the odds to a dataframe
pd.DataFrame({"Odds": odds, "Change_odd%": perc_change_odds}, index=X_train1.columns).T


# #### Checking model performance on the training set

# In[68]:


# creating confusion matrix
confusion_matrix_statsmodels(lg1, X_train1, y_train)


# In[69]:


print("Training performance:")
log_reg_model_train_perf = model_performance_classification_statsmodels(
    lg1, X_train1, y_train
)  ## Complete the code to check performance on X_train1 and y_train
log_reg_model_train_perf

print("Training performance:")
model_performance_classification_statsmodels(lg, X_train, y_train)


# #### ROC-AUC
# * ROC-AUC on training set

# In[70]:


logit_roc_auc_train = roc_auc_score(y_train, lg1.predict(X_train1))
fpr, tpr, thresholds = roc_curve(y_train, lg1.predict(X_train1))
plt.figure(figsize=(7, 5))
plt.plot(fpr, tpr, label="Logistic Regression (area = %0.2f)" % logit_roc_auc_train)
plt.plot([0, 1], [0, 1], "r--")
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.01])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Receiver operating characteristic")
plt.legend(loc="lower right")
plt.show()


# ### Model Performance Improvement

# * Let's see if the recall score can be improved further, by changing the model threshold using AUC-ROC Curve.

# ### Optimal threshold using AUC-ROC curve

# In[71]:


# Optimal threshold as per AUC-ROC curve
# The optimal cut off would be where tpr is high and fpr is low
fpr, tpr, thresholds = roc_curve(y_train, lg1.predict(X_train1))

optimal_idx = np.argmax(tpr - fpr)
optimal_threshold_auc_roc = thresholds[optimal_idx]
print(optimal_threshold_auc_roc)


# In[72]:


# creating confusion matrix
confusion_matrix_statsmodels(lg1, X_train1, y_train)
## Complete the code to create the confusion matrix for X_train1 and y_train with optimal_threshold_auc_roc as threshold


# In[73]:


# checking model performance for this model
log_reg_model_train_perf_threshold_auc_roc = model_performance_classification_statsmodels(
    lg1, X_train1, y_train, threshold=optimal_threshold_auc_roc
)
print("Training performance:")
log_reg_model_train_perf_threshold_auc_roc


# #### Let's use Precision-Recall curve and see if we can find a better threshold

# In[74]:


y_scores = lg1.predict(X_train1)
prec, rec, tre = precision_recall_curve(y_train, y_scores,)


def plot_prec_recall_vs_tresh(precisions, recalls, thresholds):
    plt.plot(thresholds, precisions[:-1], "b--", label="precision")
    plt.plot(thresholds, recalls[:-1], "g--", label="recall")
    plt.xlabel("Threshold")
    plt.legend(loc="upper left")
    plt.ylim([0, 1])


plt.figure(figsize=(10, 7))
plot_prec_recall_vs_tresh(prec, rec, tre)
plt.show()


# In[75]:


# setting the threshold
optimal_threshold_curve = 0.38


# #### Checking model performance on training set

# In[76]:


# creating confusion matrix
confusion_matrix_statsmodels(
    lg1, X_train1, y_train, threshold=optimal_threshold_auc_roc
)
## Complete the code to create the confusion matrix for X_train1 and y_train with optimal_threshold_curve as threshold


# In[77]:


log_reg_model_train_perf_threshold_curve = model_performance_classification_statsmodels(
    lg1, X_train1, y_train, threshold=optimal_threshold_curve
)
print("Training performance:")
log_reg_model_train_perf_threshold_curve


# ### Let's check the performance on the test set

# **Using model with default threshold**

# In[78]:


# creating confusion matrix
confusion_matrix_statsmodels(
    lg1, X_test1, y_test
)  ## Complete the code to create confusion matrix for X_test1 and y_test


# In[79]:


log_reg_model_test_perf = model_performance_classification_statsmodels(lg1, X_train1, y_train)
   ## Complete the code to check performance on X_test1 and y_test

print("Test performance:")
log_reg_model_test_perf



# * ROC curve on test set

# In[80]:


logit_roc_auc_train = roc_auc_score(y_test, lg1.predict(X_test1))
fpr, tpr, thresholds = roc_curve(y_test, lg1.predict(X_test1))
plt.figure(figsize=(7, 5))
plt.plot(fpr, tpr, label="Logistic Regression (area = %0.2f)" % logit_roc_auc_train)
plt.plot([0, 1], [0, 1], "r--")
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.01])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Receiver operating characteristic")
plt.legend(loc="lower right")
plt.show()


# **Using model with threshold=0.26** 

# In[81]:


# creating confusion matrix
confusion_matrix_statsmodels(lg1, X_test1, y_test)
## Complete the code to create confusion matrix for X_test1 and y_test using optimal_threshold_auc_roc as threshold


# In[82]:


# checking model performance for this model
log_reg_model_test_perf_threshold_auc_roc = model_performance_classification_statsmodels(
    lg1, X_test1, y_test, threshold=optimal_threshold_auc_roc
)
print("Test performance:")
log_reg_model_test_perf_threshold_auc_roc


# **Using model with threshold = 0.38**

# In[83]:


# creating confusion matrix
confusion_matrix_statsmodels(
    lg1, X_test1, y_test
)  ## Complete the code to create confusion matrix for X_test1 and y_test using optimal_threshold_curve as threshold


# In[84]:


log_reg_model_test_perf_threshold_curve = model_performance_classification_statsmodels(
    lg1, X_test1, y_test, threshold=optimal_threshold_curve
)
print("Test performance:")
log_reg_model_test_perf_threshold_curve


# ### Model performance summary

# In[85]:


# training performance comparison

models_train_comp_df = pd.concat(
    [
        log_reg_model_train_perf.T,
        log_reg_model_train_perf_threshold_auc_roc.T,
        log_reg_model_train_perf_threshold_curve.T,
    ],
    axis=1,
)
models_train_comp_df.columns = [
    "Logistic Regression-default Threshold",
    "Logistic Regression-0.37 Threshold",
    "Logistic Regression-0.42 Threshold",
]

print("Training performance comparison:")
models_train_comp_df


# In[86]:


# test performance comparison

# testing performance comparison

models_test_comp_df = pd.concat(
    [
        log_reg_model_test_perf.T,
        log_reg_model_test_perf_threshold_auc_roc.T,
        log_reg_model_test_perf_threshold_curve.T,
    ],
    axis=1,
)
models_test_comp_df.columns = [
    "Logistic Regression statsmodel",
    "Logistic Regression-0.26 Threshold",
    "Logistic Regression-0.38 Threshold",
]

print("Test set performance comparison:")
models_test_comp_df

## Complete the code to compare test performance


# ## Decision Tree

# In[87]:


X = data.drop(["status"], axis=1)
Y = data["status"]


X = pd.get_dummies(X, drop_first=True)  ## Complete the code to create dummies for X

# Splitting data in train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, Y, test_size=0.3, random_state=1
)  ## Complete the code to split the data into train test in the ratio 70:30 with random_state = 1


# In[88]:


y_train.value_counts(normalize=True)


# #### First, let's create functions to calculate different metrics and confusion matrix so that we don't have to use the same code repeatedly for each model.
# * The model_performance_classification_sklearn function will be used to check the model performance of models. 
# * The confusion_matrix_sklearnfunction will be used to plot the confusion matrix.

# In[89]:


# defining a function to compute different metrics to check performance of a classification model built using sklearn
def model_performance_classification_sklearn(model, predictors, target):
    """
    Function to compute different metrics to check classification model performance

    model: classifier
    predictors: independent variables
    target: dependent variable
    """

    # predicting using the independent variables
    pred = model.predict(predictors)

    acc = accuracy_score(target, pred)  # to compute Accuracy
    recall = recall_score(target, pred)  # to compute Recall
    precision = precision_score(target, pred)  # to compute Precision
    f1 = f1_score(target, pred)  # to compute F1-score

    # creating a dataframe of metrics
    df_perf = pd.DataFrame(
        {"Accuracy": acc, "Recall": recall, "Precision": precision, "F1": f1,},
        index=[0],
    )

    return df_perf


# In[90]:


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


# ### Building Decision Tree Model

# In[91]:


model = DecisionTreeClassifier(criterion="gini", random_state=1)
model.fit(X_train, y_train)  ## Complete the code to fit decision tree on train data


# # Checking model performance on training set

# In[92]:


model.score(X_train, y_train)


# # Checking model performance on testing set

# In[93]:


model.score(X_test, y_test)


# In[94]:


confusion_matrix_sklearn(model, X_train, y_train)


# In[95]:


decision_tree_perf_train = model_performance_classification_sklearn(
    model, X_train, y_train
)
decision_tree_perf_train


# #### Checking model performance on test set

# In[96]:


confusion_matrix_sklearn(model, X_test, y_test)


# In[97]:


decision_tree_perf_test = model_performance_classification_sklearn(
    model, X_test, y_test
)
decision_tree_perf_test

  ## Complete the code to check performance on test set


# In[98]:


model.score(X_test, y_test)


# **Before pruning the tree let's check the important features.**

# In[99]:


feature_names = list(X_train.columns)
importances = model.feature_importances_
indices = np.argsort(importances)
print(feature_names)

plt.figure(figsize=(8, 8))
plt.title("Feature Importances")
plt.barh(range(len(indices)), importances[indices], color="violet", align="center")
plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
plt.xlabel("Relative Importance")
plt.show()


# In[100]:


plt.figure(figsize=(20, 30))
tree.plot_tree(
    model,
    feature_names=feature_names,
    filled=True,
    fontsize=9,
    node_ids=True,
    class_names=True,
)
plt.show()


# In[101]:


# Text report showing the rules of a decision tree -

print(tree.export_text(model, feature_names=feature_names, show_weights=True))


# ### Pruning the tree

# **Pre-Pruning**

# In[102]:


# Choose the type of classifier.
estimator = DecisionTreeClassifier(random_state=1, class_weight={0: 0.3, 1: 0.7})

# Grid of parameters to choose from
parameters = {
    "max_depth": np.arange(5, 13, 2),
    "max_leaf_nodes": [10, 20, 40, 50, 75, 100],
    "min_samples_split": [2, 5, 7, 10, 20, 30],
}

# Type of scoring used to compare parameter combinations
acc_scorer = make_scorer(recall_score)

# Run the grid search
grid_obj = GridSearchCV(estimator, parameters, scoring=acc_scorer, cv=5)
grid_obj = grid_obj.fit(X_train, y_train)

# Set the clf to the best combination of parameters
estimator = grid_obj.best_estimator_

# Fit the best algorithm to the data.
estimator.fit(X_train, y_train)


# #### Checking performance on training set

# In[103]:


confusion_matrix_sklearn(model, X_train, y_train)


## Complete the code to create confusion matrix for train data


# In[104]:


decision_tree_tune_perf_train = model_performance_classification_sklearn(
    model, X_train, y_train
)
## Complete the code to check performance on train set
decision_tree_tune_perf_train


# #### Checking performance on test set

# In[105]:


# importance of features in the tree building ( The importance of a feature is computed as the
# (normalized) total reduction of the criterion brought by that feature. It is also known as the Gini importance )

print(
    pd.DataFrame(
        model.feature_importances_, columns=["Imp"], index=X_train.columns
    ).sort_values(by="Imp", ascending=False)
)


# In[106]:


confusion_matrix_sklearn(
    model, X_test, y_test
)  ## Complete the code to create confusion matrix for test data


# In[107]:


decision_tree_tune_perf_test = model_performance_classification_sklearn(
    model, X_test, y_test
)  ## Complete the code to check performance on test set
decision_tree_tune_perf_test


# ### Visualizing the Decision Tree

# In[108]:


plt.figure(figsize=(20, 10))
out = tree.plot_tree(
    estimator,
    feature_names=feature_names,
    filled=True,
    fontsize=9,
    node_ids=False,
    class_names=None,
)
# below code will add arrows to the decision tree split if they are missing
for o in out:
    arrow = o.arrow_patch
    if arrow is not None:
        arrow.set_edgecolor("black")
        arrow.set_linewidth(1)
plt.show()


# In[109]:


# Text report showing the rules of a decision tree -
print(tree.export_text(estimator, feature_names=feature_names, show_weights=True))


# In[110]:


# importance of features in the tree building

importances = estimator.feature_importances_
indices = np.argsort(importances)

plt.figure(figsize=(8, 8))
plt.title("Feature Importances")
plt.barh(range(len(indices)), importances[indices], color="violet", align="center")
plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
plt.xlabel("Relative Importance")
plt.show()


# **Cost Complexity Pruning**

# In[111]:


clf = DecisionTreeClassifier(random_state=1)
path = clf.cost_complexity_pruning_path(X_train, y_train)
ccp_alphas, impurities = abs(path.ccp_alphas), path.impurities


# In[112]:


pd.DataFrame(path)


# In[113]:


fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(ccp_alphas[:-1], impurities[:-1], marker="o", drawstyle="steps-post")
ax.set_xlabel("effective alpha")
ax.set_ylabel("total impurity of leaves")
ax.set_title("Total Impurity vs effective alpha for training set")
plt.show()


# Next, we train a decision tree using effective alphas. The last value
# in ``ccp_alphas`` is the alpha value that prunes the whole tree,
# leaving the tree, ``clfs[-1]``, with one node.

# In[114]:


clfs = []
for ccp_alpha in ccp_alphas:
    clf = DecisionTreeClassifier(random_state=1, ccp_alpha=ccp_alpha)
    clf.fit(
        X_train, y_train
    )  ## Complete the code to fit decision tree on training data
    clfs.append(clf)
print(
    "Number of nodes in the last tree is: {} with ccp_alpha: {}".format(
        clfs[-1].tree_.node_count, ccp_alphas[-1]
    )
)


# In[115]:


clfs = clfs[:-1]
ccp_alphas = ccp_alphas[:-1]

node_counts = [clf.tree_.node_count for clf in clfs]
depth = [clf.tree_.max_depth for clf in clfs]
fig, ax = plt.subplots(2, 1, figsize=(10, 7))
ax[0].plot(ccp_alphas, node_counts, marker="o", drawstyle="steps-post")
ax[0].set_xlabel("alpha")
ax[0].set_ylabel("number of nodes")
ax[0].set_title("Number of nodes vs alpha")
ax[1].plot(ccp_alphas, depth, marker="o", drawstyle="steps-post")
ax[1].set_xlabel("alpha")
ax[1].set_ylabel("depth of tree")
ax[1].set_title("Depth vs alpha")
fig.tight_layout()


# ### Recall Score vs alpha for training and testing sets

# In[116]:


recall_train = []
for clf in clfs:
    pred_train = clf.predict(X_train)
    values_train = recall_score(y_train, pred_train)
    recall_train.append(values_train)

recall_test = []
for clf in clfs:
    pred_test = clf.predict(X_test)
    values_test = recall_score(y_test, pred_test)
    recall_test.append(values_test)


# In[117]:


fig, ax = plt.subplots(figsize=(15, 5))
ax.set_xlabel("alpha")
ax.set_ylabel("F1 Score")
ax.set_title("F1 Score vs alpha for training and testing sets")
ax.plot(ccp_alphas, recall_train, marker="o", label="train", drawstyle="steps-post")
ax.plot(ccp_alphas, recall_test, marker="o", label="test", drawstyle="steps-post")
ax.legend()
plt.show()


# In[118]:


index_best_model = np.argmax(recall_test)
best_model = clfs[index_best_model]
print(best_model)


# #### Checking performance on training set

# In[119]:


confusion_matrix_sklearn(best_model, X_train, y_train)


# In[120]:


decision_tree_post_perf_train = model_performance_classification_sklearn(
    best_model, X_train, y_train
)
decision_tree_post_perf_train


# In[ ]:





# #### Checking performance on test set

# In[121]:


confusion_matrix_sklearn(best_model, X_test, y_test) ## Complete the code to create confusion matrix for test data on best model


# In[132]:


decision_tree_post_perf_test = model_performance_classification_sklearn (best_model, X_test, y_test) ## Complete the code to check performance of test set on best model
decision_tree_post_test


# In[133]:


plt.figure(figsize=(20, 10))

out = tree.plot_tree(
    best_model,
    feature_names=feature_names,
    filled=True,
    fontsize=9,
    node_ids=False,
    class_names=None,
)
for o in out:
    arrow = o.arrow_patch
    if arrow is not None:
        arrow.set_edgecolor("black")
        arrow.set_linewidth(1)
plt.show()


# In[134]:


# Text report showing the rules of a decision tree -

print(tree.export_text(best_model, feature_names=feature_names, show_weights=True))


# In[135]:


importances = best_model.feature_importances_
indices = np.argsort(importances)

plt.figure(figsize=(8, 8))
plt.title("Feature Importances")
plt.barh(range(len(indices)), importances[indices], color="violet", align="center")
plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
plt.xlabel("Relative Importance")
plt.show()


# ### Comparing Decision Tree models

# In[136]:


# training performance comparison

models_train_comp_df = pd.concat(
    [
        decision_tree_perf_train.T,
        decision_tree_tune_perf_train.T,
        decision_tree_post_perf_train.T,
    ],
    axis=1,
)
models_train_comp_df.columns = [
    "Decision Tree sklearn",
    "Decision Tree (Pre-Pruning)",
    "Decision Tree (Post-Pruning)",
]
print("Training performance comparison:")
models_train_comp_df


# In[137]:


# test performance comparison

models_train_comp_df = pd.concat(
    [
        decision_tree_perf_test.T,
        decision_tree_tune_perf_test.T,
        decision_tree_post_perf_test.T,
    ],
    axis=1,
)
models_train_comp_df.columns = [
    "Decision Tree sklearn",
    "Decision Tree (Pre-Pruning)",
    "Decision Tree (Post-Pruning)",
]
print("Test set performance comparison:")
models_train_comp_df


# ### Business Recommendations
# - Time spent on website and first interaction on the website are the two highest factor driving conversion rate. - - The company should focus more on these two feature.
# - Students biographic is most likely to convert.
# - The company should allocate maximum resources on students spending less than 415.5 minutes on the website.
# - Features like website visits and page view per visit are not relevant in the data set.
# - The data-sets final decison tree have four depth.

# In[ ]:




