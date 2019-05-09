#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 19 11:45:37 2019

"""

# Importing libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Importing new libraries
from sklearn.model_selection import train_test_split # train/test split
from sklearn.neighbors import KNeighborsRegressor # KNN for Regression
import statsmodels.formula.api as smf # regression modeling
from sklearn.model_selection import cross_val_score # k-folds cross validation

from sklearn.linear_model import LogisticRegression
# Importing dataset
file = 'XXXXX path XXXXXX / ML/GOT_character_predictions.xlsx'
got = pd.read_excel(file)


########################
# Fundamental Dataset Exploration
########################
# Column names
got.columns

# Displaying the first rows of the DataFrame
print(got.head())


# Dimensions of the DataFrame
got.shape


# Information about each variable
got.info()

# Descriptive statistics
got.describe().round(2)


got.sort_values('isAlive', ascending = False)


###############################################################################
# Create flags for missing values

###############################################################################

for col in got:

    """ Create columns that are 0s if a value was not missing and 1 if
    a value is missing. """
    
    if got[col].isnull().any():
        got['m_'+col] = got[col].isnull().astype(int)

###############################################################################
# Imputing Missing Values
###############################################################################

print(
      got
      .isnull()
      .sum()
      )

# Creating unknown intead of NAN in categorical variable
got = pd.DataFrame(got)

fill= 'unknown'

# Using the replace command with the dictionary
for col in got:
    if got[col].dtype == 'O': 
        got[col] = got[col].fillna(fill)
    

###############################################################################
# Correlation Analysis
###############################################################################

got.head()


df_corr = got.corr().round(2)
print(df_corr)
df_corr.loc['isAlive'].sort_values(ascending = False)

df_corr_2 = got2.corr().round(2)
df_corr_2.loc['isAlive'].sort_values(ascending = False)
########################
# Correlation Heatmap
########################

# Using palplot to view a color scheme
sns.palplot(sns.color_palette('coolwarm', 12))

fig, ax = plt.subplots(figsize=(15,15))

df_corr2 = df_corr.iloc[1:19, 1:19]

sns.heatmap(df_corr2,
            cmap = 'coolwarm',
            square = True,
            annot = True,
            linecolor = 'black',
            linewidths = 0.5)


plt.savefig('GOT Correlation Heatmap.png')
plt.show()
        
        
        
        
###############################################################################
# Fundamental Dataset Exploration 

# Visual EDA steps:
# -  plotting each variable's distribution 
# -  scatter plots that illustrate relationship between the variables
# -  heatmap of correlations between the variables
        
# Key EDA Visualizations
plt.subplot(2, 2, 1)
plt.scatter(x="popularity", y="isAlive", alpha = 0.5,
            color = 'blue',data=got)
plt.title("Popularity")






###############################################################################
# Feature Engineering
###############################################################################
import pandas as pd
# got['house'].unique()
# 328 unique houses
# 1946 entries, 427 missing
        

# One-Hot Encoding Qualitative Variables
#street_dummies = pd.get_dummies(list(housing['Street']), drop_first = True)
#lot_config_dummies = pd.get_dummies(list(housing['Lot Config']), drop_first = True)
#neighborhod_dummies = pd.get_dummies(list(housing['Neighborhood']), drop_first = True)




# Concatenating One-Hot Encoded Values with the Larger DataFrame
#h ousing_2 = pd.concat(
#        [housing.loc[:,:],
#         street_dummies, lot_config_dummies, neighborhod_dummies],
#         axis = 1)




got['house_Targ']=0
def conditions(got):
    if (got['house'] == 'House Targaryen'):
        return 1
    else:
        return 0
got['house_Targ'] = got.apply(conditions, axis=1)




got['house_Frey']=0
def conditions(got):
    if (got['house'] == 'House Frey'):
        return 1
    else:
        return 0
got['house_Frey'] = got.apply(conditions, axis=1)



got['alive_heir']=0
def conditions(got):
    if got['isAliveHeir'] == 1:
        return 1
    else:
        return 0
got['alive_heir'] = got.apply(conditions, axis=1)

got['house_nights']=0
def conditions(got):
    if (got['house'] == "Night's Watch" and got['book5_A_Dance_with_Dragons']==0
        and got['book4_A_Feast_For_Crows']==0):
        return 1
    else:
        return 0
got['house_nights'] = got.apply(conditions, axis=1)



got['old']=0
def conditions(got):
    if (got['age'] >= 100):
        return 1
    else:
        return 0
got['old'] = got.apply(conditions, axis=1)



got['lord']=0
def conditions(got):
    if 'Lord' in got['title']:
        return 1
    else:
        return 0
got['lord'] = got.apply(conditions, axis=1)

got['queen']=0
def conditions(got):
    if 'Queen' in got['title']:
        return 1
    else:
        return 0
got['queen'] = got.apply(conditions, axis=1)


got['brotherhood']=0
def conditions(got):
    if 'Brotherhood' in got['house']:
        return 1
    else:
        return 0
got['brotherhood'] = got.apply(conditions, axis=1)



got['heir_targ']=0
def conditions(got):
    if 'Targaryen' in got['heir']:
        return 1
    else:
        return 0
got['heir_targ'] = got.apply(conditions, axis=1)



got['all_books']=0
def conditions(got):
    if (got['book1_A_Game_Of_Thrones']==1 and got['book2_A_Clash_Of_Kings']==1 and 
        got['book3_A_Storm_Of_Swords']==1 and  got['book4_A_Feast_For_Crows']==1 and 
        got['book5_A_Dance_with_Dragons']==1):
        return 1
    else:
        return 0
got['all_books'] = got.apply(conditions, axis=1)




got['no_books']=0
def conditions(got):
    if (got['book1_A_Game_Of_Thrones']==0 and got['book2_A_Clash_Of_Kings']==0 and 
        got['book3_A_Storm_Of_Swords']==0 and  got['book4_A_Feast_For_Crows']==0 and 
        got['book5_A_Dance_with_Dragons']==0):
        return 1
    else:
        return 0
got['no_books'] = got.apply(conditions, axis=1)



got['only_book'] = 0
got['one_book']= got['book1_A_Game_Of_Thrones']+got['book2_A_Clash_Of_Kings']
got['one_book']= got['one_book'] +got['book3_A_Storm_Of_Swords']
got['one_book']= got['one_book'] +got['book4_A_Feast_For_Crows']
got['one_book']= got['one_book'] +got['book5_A_Dance_with_Dragons']
def conditions(got):
    if got['one_book']==1:
        return 1
    else:
        return 0
got['only_book'] = got.apply(conditions, axis=1)



got['date']=0
def conditions(got):
    if got['dateOfBirth'] <= 210:
        return 1
    else:
        return 0
got['date'] = got.apply(conditions, axis=1)



got['pop']=0
def conditions(got):
    if got['popularity'] <= 0.86 and got['popularity']>=0.73:
        return 1
    else:
        return 0
got['pop'] = got.apply(conditions, axis=1)


got['unpop']=0
def conditions(got):
    if got['popularity'] <= 0.013 and got['popularity']>=0.003:
        return 1
    else:
        return 0
got['unpop'] = got.apply(conditions, axis=1)




#house = got['house'].value_counts().reset_index(name='count')
#got['isAlive'].value_counts() 

#got.groupby(['house'])[["isAlive"]].sum()

#alive=got.groupby(['house', 'isAlive']).size().reset_index(name='Time')
#alive['house'].value_counts() 
#alive.info()






#primitive ols model 

# One-Hot Encoding Qualitative Variables
#
#title_dummies = pd.get_dummies(list(got['title']), drop_first = True)
#lot_config_dummies = pd.get_dummies(list(housing['Lot Config']), drop_first = True)
#neighborhod_dummies = pd.get_dummies(list(housing['Neighborhood']), drop_first = True)




# Concatenating One-Hot Encoded Values with the Larger DataFrame
got3 = pd.concat(
       [got.loc[:,:],
       title_dummies],
         axis = 1)
  
       
import pandas as pd
import statsmodels.formula.api as smf # regression modeling
        
lm_prim = smf.ols(formula = """isAlive ~  got['m_dateOfBirth']
+got['m_age']
+got['m_mother']
+ got['male']
+got['house_Frey']
+got['house_Targ']
+ got['old']+got['all_books']+got['no_books']+got['date']+
         got['brotherhood']+got['book5_A_Dance_with_Dragons']+ got['book4_A_Feast_For_Crows']+
         got['lord']+got['isNoble']+got['unpop']+got['pop']+
         got['only_book']+got['heir_targ']+
         got['queen']+ got['isMarried']+got['alive_heir']
                                               """, data = got)



# Fitting Results
results = lm_prim.fit()


# Printing Summary Statistics
print(results.summary())

got['isAlive'].corr(got['book5_A_Dance_with_Dragons'])    













# Split first

got2 = pd.concat(
        [got['isAlive'], got['name'],got['m_dateOfBirth'],got['m_age'],
         got['m_mother'], got['male'], got['house_Frey'], got['house_Targ'], 
         got['old'], got['all_books'],got['no_books'],got['date'],
         got['brotherhood'],got['book5_A_Dance_with_Dragons'], got['book4_A_Feast_For_Crows'],got['lord'],
         got['unpop'], got['pop'],
         got['only_book'],got['heir_targ'],
         got['queen'], got['isMarried'], got['alive_heir']],
         axis = 1)

got_data = got2.drop(['name','isAlive'],
                                axis = 1)
got_target = got2.loc[:, 'isAlive']


X_train, X_test, y_train, y_test = train_test_split(
            got_data,
            got_target,
            test_size = 0.10,
            random_state = 508)


# Training set 
print(X_train.shape)
print(y_train.shape)

# Testing set
print(X_test.shape)
print(y_test.shape)


###############################################################################
# Building a Base Model with statsmodels
###############################################################################

# Original Value Counts
got_target.value_counts()
got_target.sum() / got_target.count()

# Training set value counts
y_train.value_counts()
y_train.sum() / y_train.count()

# Testing set value counts
y_test.value_counts()
y_test.sum() / y_test.count()

########################
# Logistic Regression Modeling 
########################
# Biserial point correlations
c = got.corr().round(3)

# Modeling based on the most correlated explanatory variable
logistic_small = smf.logit(formula = """isAlive ~ book5_A_Dance_with_Dragons""",
                  data = got_train)


results_logistic = logistic_small.fit()
results_logistic.summary()





# Significant LS model
logistic_sig= smf.logit(formula = """choice ~ cartime +
                                              carcost +
                                              traincost""",
                                              data = sydney_train)

results_logistic_sig = logistic_sig.fit()
results_logistic_sig.summary()







# We need to merge our X_train and y_train sets so that they can be
# used in statsmodels
got_train = pd.concat([X_train, y_train], axis = 1)

###############################################################################
# Developing a Classification Base with KNN
###############################################################################

########################
# Using KNN  On Our Optimal Model (same code as our previous script on KNN)
########################

# Exact loop as before
training_accuracy = []
test_accuracy = []
neighbors_settings = range(1, 51)
for n_neighbors in neighbors_settings:
    # build the model
    clf = KNeighborsRegressor(n_neighbors = n_neighbors)
    clf.fit(X_train, y_train)
    
    # record training set accuracy
    training_accuracy.append(clf.score(X_train, y_train))
    
    # record generalization accuracy
    test_accuracy.append(clf.score(X_test, y_test))

plt.plot(neighbors_settings, training_accuracy, label = "training accuracy")
plt.plot(neighbors_settings, test_accuracy, label = "test accuracy")
plt.ylabel("Accuracy")
plt.xlabel("n_neighbors")
plt.legend()

# Printing highest test accuracy
print(test_accuracy.index(max(test_accuracy)) + 1)

########################
# The best results occur when k = 15.
########################

# Building a model with k = 15
knn_reg = KNeighborsRegressor(algorithm = 'auto',
                              n_neighbors = 22)
# Fitting the model based on the training data
knn_reg_fit = knn_reg.fit(X_train, y_train)
# Scoring the model
y_score_knn_optimal = knn_reg.score(X_test, y_test)
# The score is directly comparable to R-Square
print(y_score_knn_optimal)







###############################################################################
# Hyperparameter Tuning with Logistic Regression
###############################################################################


logreg = LogisticRegression(C = 100,
                            solver = 'lbfgs')


logreg_fit = logreg.fit(X_train, y_train)


logreg_pred = logreg_fit.predict(X_test)


# Let's compare the testing score to the training score.
print('Training Score', logreg_fit.score(X_train, y_train).round(4))
print('Testing Score:', logreg_fit.score(X_test, y_test).round(4))


"""
Prof. Chase:
    The hyperparameter C helps is regularize our model. In general, this means
    that if we increase C, our model will perform better on the training data
    (good if a model is underfit). Also, if we decrease C, our model will
    perform better on the testing data (good if a model is overfit).
"""


########################################################################
# Adjusting the hyperparameter C to 100
########################################################################

logreg_100 = LogisticRegression(C = 100,
                                solver = 'lbfgs')


logreg_100_fit = logreg_100.fit(X_train, y_train)


logreg_pred = logreg_100_fit.predict(X_test)


# Let's compare the testing score to the training score.
print('Training Score', logreg_100_fit.score(X_train, y_train).round(4))
print('Testing Score:', logreg_100_fit.score(X_test, y_test).round(4))









from sklearn.ensemble import RandomForestClassifier

# Loading other libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

###############################################################################
# Random Forest in scikit-learn
###############################################################################

# Following the same procedure as other scikit-learn modeling techniques

full_forest_gini = RandomForestClassifier(n_estimators = 1000,
                                     criterion = 'gini',
                                     max_depth = None,
                                     min_samples_leaf = 15,
                                     bootstrap = True,
                                     warm_start = False,
                                     random_state = 508)



# Fitting the models
full_gini_fit = full_forest_gini.fit(X_train, y_train)

# Scoring the gini model
print('Training Score', full_gini_fit.score(X_train, y_train).round(4))
print('Testing Score:', full_gini_fit.score(X_test, y_test).round(4))


# Scoring the entropy model
print('Training Score', full_entropy_fit.score(X_train, y_train).round(4))
print('Testing Score:', full_entropy_fit.score(X_test, y_test).round(4))






########################
# Parameter tuning with GridSearchCV
########################

from sklearn.model_selection import GridSearchCV


# Creating a hyperparameter grid
estimator_space = pd.np.arange(100, 1350, 250)
leaf_space = pd.np.arange(1, 150, 15)
criterion_space = ['gini', 'entropy']
bootstrap_space = [True, False]
warm_start_space = [True, False]



param_grid = {'n_estimators' : estimator_space,
              'min_samples_leaf' : leaf_space,
              'criterion' : criterion_space,
              'bootstrap' : bootstrap_space,
              'warm_start' : warm_start_space}



# Building the model object one more time
full_forest_grid = RandomForestClassifier(max_depth = None,
                                          random_state = 508)


# Creating a GridSearchCV object
full_forest_cv = GridSearchCV(full_forest_grid, param_grid, cv = 3)



# Fit it to the training data
full_forest_cv.fit(X_train, y_train)


########################
# Building Random Forest Model Based on Best Parameters
########################

rf_optimal = RandomForestClassifier(bootstrap = False,
                                    criterion = 'entropy',
                                    min_samples_leaf = 1,
                                    n_estimators = 100,
                                    warm_start = True)

rf_optimal.fit(X_train, y_train)
rf_optimal_pred = rf_optimal.predict(X_test)


print('Training Score', rf_optimal.score(X_train, y_train).round(4))
print('Testing Score:', rf_optimal.score(X_test, y_test).round(4))


rf_optimal_train = rf_optimal.score(X_train, y_train)
rf_optimal_test  = rf_optimal.score(X_test, y_test)



# Print the optimal parameters and best score
print("Tuned Logistic Regression Parameter:", full_forest_cv.best_params_)
print("Tuned Logistic Regression Accuracy:", full_forest_cv.best_score_.round(4))




















#################
# My Model
#################
# Split first

got2 = pd.concat(
        [got['isAlive'], got['name'],got['m_dateOfBirth'],got['m_age'],
         got['m_mother'], got['male'], got['house_Frey'], got['house_Targ'], 
         got['old'], got['all_books'],got['no_books'],got['date'],
         got['brotherhood'],got['book5_A_Dance_with_Dragons'], got['book4_A_Feast_For_Crows'],got['lord'],
         got['isNoble'],got['unpop'], got['pop'],
         got['only_book']],
         axis = 1)

got_data = got2.drop(['name','isAlive'],
                                axis = 1)
got_target = got2.loc[:, 'isAlive']


X_train, X_test, y_train, y_test = train_test_split(
            got_data,
            got_target,
            test_size = 0.10,
            random_state = 508)



########################################################################
# Adjusting the hyperparameter C to 100
########################################################################

logreg_100 = LogisticRegression(C = 100,
                                solver = 'lbfgs')


logreg_100_fit = logreg_100.fit(X_train, y_train)


logreg_pred = logreg_100_fit.predict(X_test)


# Let's compare the testing score to the training score.
print('Training Score', logreg_100_fit.score(X_train, y_train).round(4))
print('Testing Score:', logreg_100_fit.score(X_test, y_test).round(4))





from sklearn.ensemble import RandomForestClassifier
# Loading other libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


###############################################################################
# Random Forest in scikit-learn
###############################################################################
# Following the same procedure as other scikit-learn modeling techniques
full_forest_gini = RandomForestClassifier(n_estimators = 1000,
                                     criterion = 'gini',
                                     max_depth = None,
                                     min_samples_leaf = 15,
                                     bootstrap = True,
                                     warm_start = False,
                                     random_state = 508)

# Fitting the models
full_gini_fit = full_forest_gini.fit(X_train, y_train)

# Scoring the gini model
print('Training Score', full_gini_fit.score(X_train, y_train).round(4))
print('Testing Score:', full_gini_fit.score(X_test, y_test).round(4))





cv_lr_3 = cross_val_score(logreg_100, got_data, got_target, cv = 3)
print(pd.np.mean(cv_lr_3))







# ! # ! # ! # ! # ! # ! # ! # # ! # ! # ! #

#######################################################
# Parameter tuning with GridSearchCV
#######################################################

from sklearn.model_selection import GridSearchCV


# Creating a hyperparameter grid
estimator_space = pd.np.arange(100, 1350, 250)
leaf_space = pd.np.arange(1, 150, 15)
criterion_space = ['gini', 'entropy']
bootstrap_space = [True, False]
warm_start_space = [True, False]



param_grid = {'n_estimators' : estimator_space,
              'min_samples_leaf' : leaf_space,
              'criterion' : criterion_space,
              'bootstrap' : bootstrap_space,
              'warm_start' : warm_start_space}



# Building the model object one more time
full_forest_grid = RandomForestClassifier(max_depth = None,
                                          random_state = 508)


# Creating a GridSearchCV object
full_forest_cv = GridSearchCV(full_forest_grid, param_grid, cv = 3)



# Fit it to the training data
full_forest_cv.fit(X_train, y_train)


########################
# Building Random Forest Model Based on Best Parameters
########################
#GridSearchCV(cv=3, error_score='raise-deprecating',
 #      estimator=RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
  #          max_depth=None, max_features='auto', max_leaf_nodes=None,
   #         min_impurity_decrease=0.0, min_impurity_split=None,
    #        min_samples_leaf=1, min_samples_split=2,
     #       min_weight_fraction_leaf=0.0, n_estimators='warn', n_jobs=None,
      #      oob_score=False, random_state=508, verbose=0, warm_start=False),
       #fit_params=None, iid='warn', n_jobs=None,
       #param_grid={'n_estimators': array([ 100,  350,  600,  850, 1100]), 'min_samples_leaf': array([  1,  16,  31,  46,  61,  76,  91, 106, 121, 136]), 'criterion': ['gini', 'entropy'], 'bootstrap': [True, False], 'warm_start': [True, False]},
       #pre_dispatch='2*n_jobs', refit=True, return_train_score='warn',
       #scoring=None, verbose=0)

rf_optimal = RandomForestClassifier(bootstrap = True,
                                    criterion = 'gini',
                                    min_samples_leaf = 1,
                                    min_samples_split=2,
                                    n_estimators = 500,
                                    warm_start = False)

rf_optimal.fit(X_train, y_train)
rf_optimal_pred = rf_optimal.predict(X_test)


print('Training Score', rf_optimal.score(X_train, y_train).round(4))
print('Testing Score:', rf_optimal.score(X_test, y_test).round(4))


rf_optimal_train = rf_optimal.score(X_train, y_train)
rf_optimal_test  = rf_optimal.score(X_test, y_test)



# Print the optimal parameters and best score
print("Tuned Logistic Regression Parameter:", full_forest_cv.best_params_)
print("Tuned Logistic Regression Accuracy:", full_forest_cv.best_score_.round(4))


