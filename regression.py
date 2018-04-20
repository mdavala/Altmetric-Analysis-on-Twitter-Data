# -*- coding: utf-8 -*-
"""
Created on Tue Apr 17 13:43:18 2018

@author: mdavala
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix
from sklearn.linear_model import LinearRegression
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from scipy.stats import pearsonr,spearmanr
from sklearn.linear_model import LassoLarsCV
from sklearn import preprocessing


#data = pd.read_excel('Psyregression_dataset.xlsx')
data = pd.read_excel('Medregression_dataset.xlsx')
data = data.dropna(axis = 0, how='any')
X = data.drop(['Tweet Count'], axis=1)
y = data['Tweet Count']


'''
OLS
'''
X=sm.add_constant(X)
olsmodel = sm.OLS(y, X).fit()
olsmodel.summary()


To find different correlations in data
data.corr(method='pearson')
data.corr(method='kendall')
data.corr(method='spearman')


X['academic tweeters'] = preprocessing.scale(X['academic tweeters'].astype('float64'))
X['non-academic tweeters'] = preprocessing.scale(X['non-academic tweeters'].astype('float64'))
X['individual tweeters'] = preprocessing.scale(X['individual tweeters'].astype('float64'))
X['organization tweeters'] = preprocessing.scale(X['organization tweeters'].astype('float64'))
X['academic followers'] = preprocessing.scale(X['academic followers'].astype('float64'))
X['non-academic followers'] = preprocessing.scale(X['non-academic followers'].astype('float64'))
X['individual followers'] = preprocessing.scale(X['individual followers'].astype('float64'))
X['organization followers'] = preprocessing.scale(X['organization followers'].astype('float64'))
X['article retweets'] = preprocessing.scale(X['article retweets'].astype('float64'))
X['article likes'] = preprocessing.scale(X['article likes'].astype('float64'))
X['individual academic'] = preprocessing.scale(X['individual academic'].astype('float64'))
X['individual non-acad'] = preprocessing.scale(X['individual non-acad'].astype('float64'))

X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size = 0.2, 
                                                    random_state = 0)

lassomodel = LassoLarsCV(cv=10, precompute= False).fit(X_train, y_train)
dict(zip(X.columns, lassomodel.coef_))


#Plot coefficients progression
m_log_alphas = -np.log10(lassomodel.alphas_)
ax = plt.gca()
plt.plot(m_log_alphas, lassomodel.coef_path_.T)
plt.axvline(-np.log10(lassomodel.alpha_), linestyle='--',color='k',
            label='alpha CV')
plt.ylabel('Regression Coefficients')
plt.xlabel('-log(alpha)')
plt.title('Regression Coefficients Progression for Lasso Paths')


# plot mean square error for each fold
m_log_alphascv = -np.log10(lassomodel.cv_alphas_)
plt.figure()
plt.plot(m_log_alphascv, lassomodel.cv_mse_path_, ':')
plt.plot(m_log_alphascv, lassomodel.cv_mse_path_.mean(axis=-1), 'k',
         label='Average across the folds', linewidth=2)
plt.axvline(-np.log10(lassomodel.alpha_), linestyle='--', color='k',
            label='alpha CV')
plt.legend()
plt.xlabel('-log(alpha)')
plt.ylabel('Mean squared error')
plt.title('Mean squared error on each fold')


# MSE from training and test data
from sklearn.metrics import mean_squared_error
train_error = mean_squared_error(y_train, lassomodel.predict(X_train))
test_error = mean_squared_error(y_test, lassomodel.predict(X_test))
print ('training data MSE')
print(train_error)
print ('test data MSE')
print(test_error)

# R-square from training and test data
rsquared_train= lassomodel.score(X_train,y_train)
rsquared_test= lassomodel.score(X_test,y_test)
print ('training data R-square')
print(rsquared_train)
print ('test data R-square')
print(rsquared_test)



