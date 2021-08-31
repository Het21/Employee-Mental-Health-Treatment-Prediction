# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from IPython import get_ipython

# %% [markdown]
# # <h1><center>Data Importing & Loading</center></h1>

# %%
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import warnings
warnings.filterwarnings("ignore")

# PreProcessing
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, IterativeImputer
from sklearn.preprocessing import MinMaxScaler
from category_encoders import BinaryEncoder

# Splitting Data
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score

# Modeling
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, recall_score, plot_roc_curve
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from xgboost.sklearn import XGBClassifier

# Tuning
from sklearn.model_selection import GridSearchCV


# %%
mh = pd.read_csv('Mental_Health_Tech_Survey.csv')
mh

# %% [markdown]
# # <h1><center>Phase 1 - Data Cleaning</center></h1>

# %%
mh.info()

# %% [markdown]
# ***Missing Values***

# %%
mh.isna().sum()/len(mh.index)*100

# %% [markdown]
# ***Drop irrelevant Columns***

# %%
mh.drop(columns=['Timestamp', 'state', 'comments'], inplace = True)
# Timestamp: Timestamp of when the respondents answered the survey is not relevant for predicition
# State: It only includes whether a respondent lives in a US state or not, rest all are missing values which account for 40% of total state data

# %% [markdown]
# ***Renaming Columns***

# %%
mh.rename({'self_employed' : 'Self_Employed', 'family_history' : 'Family_History', 
           'treatment' : 'Treatment', 'work_interfere' : 'Work_Interfere', 
           'no_employees': 'Employee_Numbers', 'remote_work': 'Remote_Work', 'tech_company': 'Tech_Company', 
           'benefits': 'Benefits', 'care_options': 'Care_Options', 'wellness_program': 'Wellness_Program', 
           'seek_help': 'Seek_Help', 'anonymity': 'Anonymity', 'leave': 'Medical_Leave', 
           'mental_health_consequence': 'Mental_Health_Consequence', 
           'phys_health_consequence': 'Physical_Health_Consequence', 'coworkers': 'Coworkers', 
           'supervisor': 'Supervisor', 'mental_health_interview': 'Mental_Health_Interview', 
           'phys_health_interview': 'Physical_Health_Interview', 'mental_vs_physical': 'Mental_VS_Physical', 
           'obs_consequence': 'Observed_Consequence'} , inplace = True , axis = 1)

# %% [markdown]
# ***Feature's Value Validity***
# %% [markdown]
# *Age* - Setting age limit to 15 as per guidelines from  ILO (International Labor Organization)

# %%
mh['Age'].unique()


# %%
mh['Age'].replace([mh['Age'][mh['Age'] < 15]], np.nan, inplace = True)
mh['Age'].replace([mh['Age'][mh['Age'] > 100]], np.nan, inplace = True)

mh['Age'].unique()

# %% [markdown]
# *Gender*- Genders with same meaning but different text have been combined into one meaningful category<br>
# Final Categories after cleaning:<br>
# Male or cis Male - born as male and decide to be male.<br>
# Female or cis Female - born as female and decide to be female.<br>
# Queer is a word that describes sexual and gender identities other than straight and cisgender. It may also include Lesbian, gay, bisexual, and transgender people.<br>

# %%
mh['Gender'].unique()


# %%
mh['Gender'].replace(['Male ', 'male', 'M', 'm', 'Male', 'Cis Male',
                     'Man', 'cis male', 'Mail', 'Male-ish', 'Male (CIS)',
                      'Cis Man', 'msle', 'Malr', 'Mal', 'maile', 'Make',], 'Male', inplace = True)

mh['Gender'].replace(['Female ', 'female', 'F', 'f', 'Woman', 'Female',
                     'femail', 'Cis Female', 'cis-female/femme', 'Femake', 'Female (cis)',
                     'woman',], 'Female', inplace = True)

mh["Gender"].replace(['Female (trans)', 'queer/she/they', 'non-binary',
                     'fluid', 'queer', 'Androgyne', 'Trans-female', 'male leaning androgynous',
                      'Agender', 'A little about you', 'Nah', 'All',
                      'ostensibly male, unsure what that really means',
                      'Genderqueer', 'Enby', 'p', 'Neuter', 'something kinda male?',
                      'Guy (-ish) ^_^', 'Trans woman',], 'Queer', inplace = True)


# %%
mh['Gender'].value_counts()

# %% [markdown]
# # <h1><center>Phase 2 - Data Mining and Analysis</center></h1>

# %%
mh_eda = mh.copy()

# %% [markdown]
# ### Target of Data

# %%
sns.set_style("whitegrid")
plt.figure(figsize = (7,5))
eda_percentage = mh_eda['Treatment'].value_counts(normalize = True).rename_axis('Treatment').reset_index(name = 'Percentage')
sns.barplot(x = 'Treatment', y = 'Percentage', data = eda_percentage.head(10))
plt.title('Get Treatment of Survey Respondents')
plt.show()

# %% [markdown]
# * This is the respondents result of question, **'Have you got treatment for a mental health condition?'**.
# * The percentage of respondents who want to get treatment is 50%. Workplaces that promote mental health and support people with mental disorders are more likely to reduce absenteeism, increase productivity and benefit from associated economic gains.
# 
# * We opted to separate them into 3 aspects to see what what factors motivated employees to get a treatment:
#     - Employee's profiling
#     - Employee's work environtment
#     - Employee's mental health facilities
# %% [markdown]
# ### Profile of Respondents

# %%
plt.figure(figsize = (18,5))
plt.subplot(1,2,1)
sns.distplot(mh_eda['Age'], label = 'Skewness : %.2f'%(mh_eda['Age'].skew()))
plt.legend(loc = 0, fontsize = 10)
plt.title('Distribution for Age of Survey Respondents')
plt.subplot(1,2,2)
sns.boxplot(x = "Age", y = "Treatment", data = mh_eda)
plt.title('Boxplot for Age of Survey Respondents')
age = str(mh_eda['Age'].describe().round(2))
plt.text(56, 0.85, age)
plt.show()

# %% [markdown]
# #### Skewness
# - Based on the plot, **the skewness score is 1.01, which means the data is highly skewed and with Positive skewness** where the mode is smaller than mean or median.
# - It's indicated that most of the employees that fill the survey around the end 20s to early 40s. It can be assumed that they are on between mid to senior-level positions. **The distribution of ages is right-skewed which is expected as the tech industry tend to have younger employees**.
# 
# #### Boxplot
# - From the boxplot, there is no statistically significant difference of ages between respondents that get treatment and no treatment.

# %%
plt.figure(figsize = (20,5))
plt.subplot(1,2,1)
eda_percentage = mh_eda['Gender'].value_counts(normalize = True).rename_axis('Gender').reset_index(name = 'Percentage')
sns.barplot(x = 'Gender', y = 'Percentage', data = eda_percentage.head(10))
plt.title('Gender of Survey Respondents')
plt.subplot(1,2,2)
sns.countplot(mh_eda['Gender'], hue = mh_eda['Treatment'])
plt.title('Gender of Survey Respondents')
plt.show()

# %% [markdown]
# * This is the respondents result of question, **'What is your gender identities?'**.
# * **Almost 79% of respondents are male**, not surprisingly, especially in the tech field. The very large gap between men and women causes higher competitive pressure for women than men. Based on the plot, females that want to get treatment is high around 70%.
# 
# * There is a Queer entry of less than 2%. Although the percentage of queer is very low, some insights could still be extracted. For example, such a small proportion can show a significant difference in the count of who wants the treatments, indicating that for the queer, mental health problems are serious too.

# %%
plt.figure(figsize = (20,5))
plt.subplot(1,2,1)
eda_percentage = mh_eda['Family_History'].value_counts(normalize = True).rename_axis('Family_History').reset_index(name = 'Percentage')
sns.barplot(x = 'Family_History', y = 'Percentage', data = eda_percentage)
plt.title('Family History of Survey Respondents')
plt.subplot(1,2,2)
sns.countplot(mh_eda['Family_History'], hue = mh_eda['Treatment'])
plt.title('Family History of Survey Respondents')
plt.show()

# %% [markdown]
# * This is the respondents result of question, **'Do you have a family history of mental illness?'**.
# * From 40% of respondents who say that they have a family history of mental illness, the plot shows that they significantly want to get treatment rather than without a family history. This is acceptable, normally, people with a family history pay more attention to mental illness. Family history is a significant risk factor for many mental health disorders. The apple does not fall far from the tree, as it is relatively common for families with mental illness symptoms to have one or more relatives with histories of similar difficulties.
# %% [markdown]
# ### Work Environment of Respondents

# %%
plt.figure(figsize = (20,5))
plt.subplot(1,2,1)
eda_percentage = mh_eda['Work_Interfere'].value_counts(normalize = True).rename_axis('Work_Interfere').reset_index(name = 'Percentage')
sns.barplot(x = 'Work_Interfere', y = 'Percentage', data = eda_percentage)
plt.title('Work Interfere of Survey Respondents')
plt.subplot(1,2,2)
sns.countplot(mh_eda['Work_Interfere'], hue = mh_eda['Treatment'])
plt.title('Work Interfere of Survey Respondents')
plt.show()

# %% [markdown]
# * This is the respondents result of question, **'If you have a mental health condition, do you feel that it interferes with your work?'**.
# * About 78% of respondents have experienced interference at work with a ratio of rarely, sometimes, and frequently.
# * Mental health conditions sometimes become an interference while working about 45%. The plots prove that almost 80% want to get treatment. But **it's surprising to know even mental health never has interfered at work, there is a small group of people that still wanted to get treatment before it became a job stress**.

# %%
plt.figure(figsize = (20,5))
plt.subplot(1,2,1)
eda_percentage = mh_eda['Remote_Work'].value_counts(normalize = True).rename_axis('Remote_Work').reset_index(name = 'Percentage')
sns.barplot(x = 'Remote_Work', y = 'Percentage', data = eda_percentage)
plt.title('Working Style of Survey Respondents')
plt.subplot(1,2,2)
sns.countplot(mh_eda['Remote_Work'], hue = mh_eda['Treatment'])
plt.title('Working Style of Survey Respondents')
plt.show()

# %% [markdown]
# * This is the respondents result of question, **'Do you work remotely (outside of an office) at least 50% of the time?'**.
# * Around 70% of respondents didn't work remotely, which means the biggest factor of mental health disorder came up triggered on the workplace. On the other side, it is slightly different between an employee that wants to get treatment and that who didn't want to get a treatment. Although, it is interesting to note that when we see a respondent who works 50% of the workday remotely, their demand of wanting treatment is a little bit higher than the other groups.

# %%
plt.figure(figsize = (20,5))
plt.subplot(1,2,1)
eda_percentage = mh_eda['Tech_Company'].value_counts(normalize = True).rename_axis('Tech_Company').reset_index(name = 'Percentage')
sns.barplot(x = 'Tech_Company', y = 'Percentage', data = eda_percentage)
plt.title('Company Type of Survey Respondents')
plt.subplot(1,2,2)
sns.countplot(mh_eda['Tech_Company'], hue = mh_eda['Treatment'])
plt.title('Company Type of Survey Respondents')
plt.show()

# %% [markdown]
# * This is the respondents result of question, **'Would you be willing to discuss a mental health issue with your coworkers?'**.
# * From 18% of respondents who say yes to discuss it with coworkers, 60% of them wanted to get treatment. 
# * About 60% of respondents decide to discuss some of them with coworkers. Employees who do that and want to get treatment are half of them.
# * This is the respondents result of question, **'Would you be willing to discuss a mental health issue with your direct supervisor(s)?'**.
# * From 40% of respondents who say yes to discuss with supervisor, only 55% of them wanted to get treatment. A very good assumption is that maybe talking to someone in a higher position could help the relief. It's the opposite while employees discuss with coworkers.

# %%
plt.figure(figsize = (20,5))
plt.subplot(1,2,1)
eda_percentage = mh_eda['Observed_Consequence'].value_counts(normalize = True).rename_axis('Observed_Consequence').reset_index(name = 'Percentage')
sns.barplot(x = 'Observed_Consequence', y = 'Percentage', data = eda_percentage)
plt.title('Observed Consequence Survey Respondents')
plt.subplot(1,2,2)
sns.countplot(mh_eda['Observed_Consequence'], hue = mh_eda['Treatment'])
plt.title('Observed Consequence of Survey Respondents')
plt.show()

# %% [markdown]
# * This is the respondents result of question, **'Have you heard of or observed negative consequences for coworkers with mental health conditions in your workplace?'**.
# * From 15% of respondents who say yes about knowing the negative consequences for coworkers with mental heatlh condition, almost 70% of them that want to get treatment. After the employee knows about the negative consequences, it becomes a good trigger for someone to get treatment to prevent mental health conditions.
# %% [markdown]
# ### Mental Health Facilities of Respondents

# %%
plt.figure(figsize = (20,5))
plt.subplot(1,2,1)
eda_percentage = mh_eda['Benefits'].value_counts(normalize = True).rename_axis('Benefits').reset_index(name = 'Percentage')
sns.barplot(x = 'Benefits', y = 'Percentage', data = eda_percentage)
plt.title('Benefits of Survey Respondents')
plt.subplot(1,2,2)
sns.countplot(mh_eda['Benefits'], hue = mh_eda['Treatment'])
plt.title('Benefits of Survey Respondents')
plt.show()

# %% [markdown]
# * This is the respondents result of question, **'Does your employer provide mental health benefits?'**.
# * Only 35% of respondents know about mental health benefits that the company provides for them.
# * For employees who know the benefits, almost 60% of the employees want to get treatment.

# %%
plt.figure(figsize = (20,5))
plt.subplot(1,2,1)
eda_percentage = mh_eda['Wellness_Program'].value_counts(normalize = True).rename_axis('Wellness_Program').reset_index(name = 'Percentage')
sns.barplot(x = 'Wellness_Program', y = 'Percentage', data = eda_percentage)
plt.title('Wellness Program of Survey Respondents')
plt.subplot(1,2,2)
sns.countplot(mh_eda['Wellness_Program'], hue = mh_eda['Treatment'])
plt.title('Wellness Program of Survey Respondents')
plt.show()

# %% [markdown]
# * This is the respondents result of question, **'Has your employer ever discussed mental health as part of an employee wellness program?'**.
# * About 19% of the repondents say yes about become a part of employee wellness program and 60% of employee want to get treatment.
# * More than 65% of respondents say that there aren't any wellness programs that provide by their company. But half of the respondents want to get treatment, which means the company need to provide it soon.

# %%
plt.figure(figsize = (20,5))
plt.subplot(1,2,1)
eda_percentage = mh_eda['Anonymity'].value_counts(normalize = True).rename_axis('Anonymity').reset_index(name = 'Percentage')
sns.barplot(x = 'Anonymity', y = 'Percentage', data = eda_percentage)
plt.title('Anonymity of Survey Respondents')
plt.subplot(1,2,2)
sns.countplot(mh_eda['Anonymity'], hue = mh_eda['Treatment'])
plt.title('Anonymity of Survey Respondents')
plt.show()

# %% [markdown]
# * This is the respondents result of question, **'Is your anonymity protected if you choose to take advantage of mental health or substance abuse treatment resources?'**.
# * About 30% of respondents say yes if their anonymity is protected while taking advantage of mental health or substance abuse treatment resources and almost 65% of employees want to get treatment. The employee feels that the company protected their privacy and it's a good move for the company to build trust with their employees. Because of that, the employee wants to get treatment to be better.
# %% [markdown]
# ###  Conclusion
# 
# Nearly 86% of employees report improved work performance and lower rates of absenteeism after receiving treatment for depression, according to an April 2018 article in the Journal of Occupational and Environmental Medicine. This means big gains in retention and productivity for employers. By providing employees access to mental health benefits, the company can begin to create a culture of understanding and compassion at the tech company. And having employees who feel cared for and happy isn’t just good, it’s good business.
# 
# #### Based on profiling the respondents
# * Companies must know that gender and family history greatly influence the decision to get treatment for employees. So if the company wants to provide more support, the company must make an assessment of the employee's personality because different characters can determine different needs. Age can also be a trigger, considering that most of them are young so there is a high chance that they will be open-minded to get treatment.
# 
# #### Based on the work environment of respondents
# * Work interference is the most influential of employees who want to get treatment. This means the company should consider providing facilities to anticipate job stress on employees. Some of the companies decide to make a private room or silent room in case employees suddenly feel stress and need a private moment to relieve.
# 
# #### Based on the mental health facilities of respondents
# * The company needs to provide a good benefit for employees so they can maintain their mental health. If the company can don't have resources for it, there are so many third parties who can be hired to maintain a wellness program for the company. Building trust like keep private about whom employee that gets treatment also can also give a trigger for employee want to get treatment.
# %% [markdown]
# # <h1><center>Phase 3 - Prediction and Cross Validation</center></h1>
# %% [markdown]
# *PreProcessing*
# %% [markdown]
# ***Preprocessing Scheme***
# 
# - OneHotEncoding: Gender, Family History, Employee Numbers, Remote Work, Tech Company, Benefits, Care Options, Wellness Program, Seek Help, Anonymity, Medical Leave, Mental Health Consequence, Physical Health Consequence, Coworkers, Supervisor, Mental Health Interview, Physical Health Interview, Mental VS Physical, Observed Consequence
#     * Simple Imputer Most Frequent: Self Employed, Work Interfere
# - Iterative Impute: Age
# - Target: Treatment

# %%
mode_onehot_pipe = Pipeline([
    ('encoder', SimpleImputer(strategy = 'most_frequent')),
    ('one hot encoder', OneHotEncoder(handle_unknown = 'ignore'))])

transformer = ColumnTransformer([
    ('one hot', OneHotEncoder(handle_unknown = 'ignore'), ['Gender', 'Family_History', 'Employee_Numbers',
                                                           'Remote_Work', 'Tech_Company', 'Benefits', 'Care_Options',
                                                           'Wellness_Program', 'Seek_Help', 'Anonymity',
                                                           'Medical_Leave', 'Mental_Health_Consequence',
                                                           'Physical_Health_Consequence', 'Coworkers', 'Supervisor',
                                                           'Mental_Health_Interview', 'Physical_Health_Interview',
                                                           'Mental_VS_Physical', 'Observed_Consequence']),
    ('mode_onehot_pipe', mode_onehot_pipe, ['Self_Employed', 'Work_Interfere']),
    ('iterative', IterativeImputer(max_iter = 10, random_state = 0), ['Age'])])

# %% [markdown]
# ***Define Target Data***

# %%
mh['Treatment'].value_counts()/mh.shape[0]*100

# %% [markdown]
# * The data looks normal. It doesn't indicate imbalanced data

# %%
mh['Treatment'] = np.where(mh['Treatment'] == 'Yes', 1, 0)

# %% [markdown]
# * *0 = No Treatment*
# * *1 = Get Treatment*
# 
#         - TN: Employee's Mental Health predict with No Treatment and the actual is No Treatment
#         - TP: Employee's Mental Health predict with Get Treatment and the actual is Get Treatment
#         - FP: Employee's Mental Health predict with Get Treatment and the actual is No Treatment
#         - FN: Employee's Mental Health predict with No Treatment and the actual is Get Treatment
# 
# Actions:
# * FN: There is a feeling of excessive stress and anxiety at work. It's not detected in employees do their work performance decreases due to prediction errors.
# * FP: It would be nice if the company provides mental health treatment to them so that employees can maintain mental health at the workplace. 
# 
#     -> In terms of mental health, the company should know what factors for employee  deciding to get treatment. To anticipate error prediction, the metric used to determine the predicted score is the recall score (FN).

# %%
X = mh.drop('Treatment', axis = 1)
y = mh['Treatment']

# %% [markdown]
# ***Splitting Data***

# %%
X.shape


# %%
X_train, X_test, y_train, y_test = train_test_split(X,y,
                                                   stratify = y,
                                                    test_size = 0.3,
                                                   random_state = 2222)

# %% [markdown]
# * We use 0.3 as default score for test_size and X.shape for random_state so the data will be divided equally.
# %% [markdown]
# *Modeling*
# %% [markdown]
# ***Define Model***
# 
# - In supervised learning, algorithms learn from labeled data. In this case, we used Classification technique for determining which class is yes and no.
# - We used 3 basic models and 4 ensemble models to predict.
# - Basic models:
#     * Logistic Regression (logreg)
#     * Decision Tree Classifier (tree)
#     * K-Nearest Neighbor (knn)
# - Ensemble models:
#     * Random Forest Classifier (rf)
#     * Ada Boost Classifier (ada)
#     * Gradient Boosting Classifier (grad)
#     * XGB Classifier (xgboost)

# %%
logreg = LogisticRegression()
tree = DecisionTreeClassifier(random_state = 2222)
knn = KNeighborsClassifier()
rf = RandomForestClassifier(random_state = 2222)
ada = AdaBoostClassifier(random_state = 2222)
grad = GradientBoostingClassifier(random_state = 2222)
xgboost = XGBClassifier(random_state = 2222)


# %%
logreg_pipe = Pipeline([('transformer', transformer), ('logreg', logreg)])
tree_pipe = Pipeline([('transformer', transformer), ('tree', tree)])
knn_pipe = Pipeline([('transformer', transformer), ('knn', knn)])
rf_pipe = Pipeline([('transformer', transformer), ('rf', rf)])
ada_pipe = Pipeline([('transformer', transformer), ('ada', ada)])
grad_pipe = Pipeline([('transformer', transformer), ('grad', grad)])
xgb_pipe = Pipeline([('transformer', transformer), ('xgboost', xgboost)])

def model_evaluation(model, metric):
    model_cv = cross_val_score(model, X_train, y_train, cv = StratifiedKFold(n_splits = 5), scoring = metric)
    return model_cv

logreg_pipe_cv = model_evaluation(logreg_pipe, 'recall')
tree_pipe_cv = model_evaluation(tree_pipe, 'recall')
knn_pipe_cv = model_evaluation(knn_pipe, 'recall')
rf_pipe_cv = model_evaluation(rf_pipe, 'recall')
ada_pipe_cv = model_evaluation(ada_pipe, 'recall')
grad_pipe_cv = model_evaluation(grad_pipe, 'recall')
xgb_pipe_cv = model_evaluation(xgb_pipe, 'recall')

for model in [logreg_pipe, tree_pipe, knn_pipe, rf_pipe, ada_pipe, grad_pipe, xgb_pipe]:
    model.fit(X_train, y_train)

score_cv = [logreg_pipe_cv.round(5), tree_pipe_cv.round(5), knn_pipe_cv.round(5),
            rf_pipe_cv.round(5), ada_pipe_cv.round(5), grad_pipe_cv.round(5), xgb_pipe_cv.round(5)]
score_mean = [logreg_pipe_cv.mean(), tree_pipe_cv.mean(), knn_pipe_cv.mean(), rf_pipe_cv.mean(),
              ada_pipe_cv.mean(), grad_pipe_cv.mean(), xgb_pipe_cv.mean()]
score_std = [logreg_pipe_cv.std(), tree_pipe_cv.std(), knn_pipe_cv.std(), rf_pipe_cv.std(),
             ada_pipe_cv.std(), grad_pipe_cv.std(), xgb_pipe_cv.std()]
score_recall_score = [recall_score(y_test, logreg_pipe.predict(X_test)),
            recall_score(y_test, tree_pipe.predict(X_test)), 
            recall_score(y_test, knn_pipe.predict(X_test)), 
            recall_score(y_test, rf_pipe.predict(X_test)),
            recall_score(y_test, ada_pipe.predict(X_test)),
            recall_score(y_test, grad_pipe.predict(X_test)),
            recall_score(y_test, xgb_pipe.predict(X_test))]
method_name = ['Logistic Regression', 'Decision Tree Classifier', 'KNN Classifier', 'Random Forest Classifier',
               'Ada Boost Classifier', 'Gradient Boosting Classifier', 'XGB Classifier']
cv_summary = pd.DataFrame({
    'method': method_name,
    'cv score': score_cv,
    'mean score': score_mean,
    'std score': score_std,
    'recall score': score_recall_score
})
cv_summary

# %% [markdown]
# - From the cross validation process, there are 2 models that pop up with high precision scores. The first is Logistic Regression for the basic model and the second is Ada Boost Classifier for the ensemble model. But we decided to continue with Logistic Regression because Ada Boost Classifier is really heavy to process.
# - Let's tune using Logistic Regression model.
# %% [markdown]
# # <h1><center>Phase 4 - Feature Selection using Tuning Result</center></h1>
# %% [markdown]
# *HyperParam Tuning*

# %%
lr_estimator = Pipeline([
    ('transformer', transformer),
    ('model', logreg)])

hyperparam_space = {
    'model__C': [ 1, 0.5, 0.1, 0.05, 0.01],
    'model__solver': ['newton-cg', 'lbfgs', 'liblinear'],
    'model__class_weight': ['balanced', 'dict'],
    'model__max_iter': [100, 200, 300],
    'model__multi_class': ['auto', 'ovr', 'multinomial'],
    'model__random_state': [2222]
}

grid_lr = GridSearchCV(
                lr_estimator,
                param_grid = hyperparam_space,
                cv = StratifiedKFold(n_splits = 5),
                scoring = 'recall',
                n_jobs = -1)

grid_lr.fit(X_train, y_train)

print('best score', grid_lr.best_score_)
print('best param', grid_lr.best_params_)


# %%
logreg_pipe.fit(X_train, y_train)
recall_logreg = (recall_score(y_test, logreg_pipe.predict(X_test)))

grid_lr.best_estimator_.fit(X_train, y_train)
recall_grid = (recall_score(y_test, grid_lr.predict(X_test)))

score_list = [recall_logreg, recall_grid]
method_name = ['Logistic Regression Before Tuning', 'Logistic Regression After Tuning']
best_summary = pd.DataFrame({
    'method': method_name,
    'score': score_list
})
best_summary

# %% [markdown]
# - This is the comparison between before tuning score and after tuning score using Logistic Regression. We chose to use Logistic Regression after tuning score in this section.

# %%
features = list(transformer.transformers_[0][1].get_feature_names())+list(transformer.transformers_[1][1][1].get_feature_names())+['Age']
coef_table = pd.DataFrame({'coef': grid_lr.best_estimator_[1].coef_.flatten()}, index = features)
abs(coef_table).plot(kind = 'barh', figsize = (18,20))

# %% [markdown]
# * Based on selecting features based on coefficient score, we dropped 4 features manually who got a score under 0.05 for all answer choices for every feature. There are Age, x3(Remote_work), x7(Wellness_Program), x12(Physical_Health_Consequence).
# %% [markdown]
# *Re-run Using Feature Selection*
# %% [markdown]
# ***Preprocessing Scheme***
# 
# - OneHotEncoding: Gender, Family History, Employee Numbers, Tech Company, Benefits, Care Options, Seek Help, Anonymity, Medical Leave, Mental Health Consequence, Coworkers, Supervisor, Mental Health Interview, Physical Health Interview, Mental_VS_Physical, Observed_Consequence
#     * Mode: Self Employed, Work Interfere
# - Target: Treatment

# %%
mh_tuning = mh.copy()
mh_tuning.drop(columns = ['Age', 'Remote_Work', 'Wellness_Program', 'Physical_Health_Consequence'], inplace = True)
mh_tuning.head()


# %%
mode_onehot_pipe_second = Pipeline([
    ('encoder', SimpleImputer(strategy = 'most_frequent')),
    ('one hot encoder', OneHotEncoder(handle_unknown = 'ignore'))])

transformer_second = ColumnTransformer([
    ('one hot', OneHotEncoder(handle_unknown = 'ignore'), ['Gender', 'Family_History', 'Employee_Numbers',
                                                           'Tech_Company', 'Benefits', 'Care_Options', 
                                                           'Seek_Help', 'Anonymity', 'Medical_Leave',
                                                           'Mental_Health_Consequence', 'Coworkers',
                                                           'Supervisor','Mental_Health_Interview',
                                                           'Physical_Health_Interview', 'Mental_VS_Physical',
                                                           'Observed_Consequence',]),
    ('mode_onehot_pipe', mode_onehot_pipe_second, ['Self_Employed', 'Work_Interfere']),])


# %%
X_select = mh_tuning.drop('Treatment', axis = 1)
y_select = mh_tuning['Treatment']


# %%
X_select_train, X_select_test, y_select_train, y_select_test = train_test_split(X_select,y_select,
                                                   stratify = y_select,
                                                    test_size = 0.3,
                                                   random_state = 2222)


# %%
logreg_second = LogisticRegression(C = 0.5, class_weight = 'dict', max_iter = 100,
                                   multi_class = 'auto', random_state = 2222, solver = 'newton-cg')
logreg_second_pipe = Pipeline([('transformer', transformer_second), ('model', logreg_second)])
logreg_second_pipe.fit(X_select_train, y_select_train)
print('After Feature Selection Process, the score is ', recall_score(y_select_test, logreg_second_pipe.predict(X_select_test)))

# %% [markdown]
# - The score that used as a reference for predicting is 0.7277, which is similar to the tuning score before.

# %%
dfTestPredictions = logreg_second_pipe.predict(X_select_test)
results = pd.DataFrame({'Index': X_test.index, 'Treatment': dfTestPredictions})
results


