# -*- coding: utf-8 -*-
"""
Created on Tue Mar 27 23:37:13 2018

@author: mdavala
"""

import pandas as pd
import numpy as np
import re
from sklearn.pipeline import Pipeline
from sklearn.cross_validation import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import GridSearchCV

acadTags = pd.read_csv('labelled sets/Academic_3000.csv', sep=",", names = ["Labels","Text"],
                       encoding = 'Latin-1')

indivTags = pd.read_csv('labelled sets/Individual_3000.csv', sep=",", names = ["Labels","Text"],
                       encoding = 'Latin-1')

acadDocuments = acadTags['Text']
acadLabels = acadTags['Labels'] 

indivDocuments = indivTags['Text']
indivLabels = indivTags['Labels']

'''
Academic Classifier
'''


acadTrain, acadTest, acadTrainLabels, acadTestLabels = train_test_split(acadDocuments, 
                                                                        acadLabels, 
                                                                        test_size=0.25, 
                                                                        random_state=101,
                                                                        stratify=acadLabels)

# Create pipeline to create a model
acad_text_clf1 = Pipeline([('vect', CountVectorizer(stop_words = 'english')),
                     ('tfidf', TfidfTransformer()),
                     ('clf', MultinomialNB())])

#Train the model with academic document tags
acad_text_clf1 = acad_text_clf1.fit(acadTrain, acadTrainLabels)

# Predict Academic or non-academic
predicted_clf1 = acad_text_clf1.predict(acadTest)

# Check how many predicted academic is correct
acad_accuracy_clf1 = np.mean(predicted_clf1 == acadTestLabels)

parameters_clf1 = {'vect__ngram_range': [(1, 1), (1, 2)],
                   'tfidf__use_idf': (True, False),
                   'clf__alpha': (1e-2, 1e-3)}

gs_acad_clf1 = GridSearchCV(acad_text_clf1, parameters_clf1, n_jobs=-1)
gs_acad_clf1 = gs_acad_clf1.fit(acadTrain, acadTrainLabels)

gs_acad_pred_clf1 = gs_acad_clf1.predict (acadTest)

# Check how many predicted academic is correct
gs_acad_accuracy_clf1 = np.mean(gs_acad_pred_clf1 == acadTestLabels)

#gs_clf1.best_score_
#gs_clf1.best_params_


# SGD Classifier
acad_text_clf2 = Pipeline([('vect', CountVectorizer(stop_words='english')),
                         ('tfidf', TfidfTransformer()),
                         ('clf-svm', SGDClassifier(loss='hinge', penalty='l2',
                         alpha=1e-3, n_iter=5, random_state=42))])

acad_text_clf2 = acad_text_clf2.fit(acadTrain, acadTrainLabels)

predicted_clf2 = acad_text_clf2.predict (acadTest)

acad_accuracy_clf2 = np.mean(predicted_clf2 == acadTestLabels)

# Use Grid Search CV
parameters_clf2 = {'vect__ngram_range': [(1, 1), (1, 2)],
                  'tfidf__use_idf': (True, False),
                  'clf-svm__alpha': (1e-2, 1e-3)}

gs_acad_clf2 = GridSearchCV(acad_text_clf2, parameters_clf2, n_jobs=-1)
gs_acad_clf2 = gs_acad_clf2.fit(acadTrain, acadTrainLabels)

gs_acad_pred_clf2 = gs_acad_clf2.predict(acadTest)

# Check how many predicted academic is correct
gs_acad_accuracy_clf2 = np.mean(gs_acad_pred_clf2 == acadTestLabels)

#gs_clf2.best_score_
#gs_clf2.best_params_

'''
Classifier for individual and organization
'''
indivTrain, indivTest, indivTrainLabels, indivTestLabels = train_test_split (indivDocuments,
                                                                             indivLabels,
                                                                             test_size = 0.25,
                                                                             random_state=101,
                                                                             stratify= indivLabels)

# Create pipeline to create a model
indiv_text_clf1 = Pipeline([('vect', CountVectorizer(stop_words='english')),
                     ('tfidf', TfidfTransformer()),
                     ('clf', MultinomialNB())])

#Train the model with academic document tags
indiv_text_clf1 = indiv_text_clf1.fit(indivTrain, indivTrainLabels)

# Predict Academic or non-academic
indiv_predicted_clf1 = indiv_text_clf1.predict (indivTest)

# Check how many predicted academic is correct
indiv_accuracy_clf1 = np.mean(predicted_clf1 == indivTestLabels)

parameters_clf1 = {'vect__ngram_range': [(1, 1), (1, 2)],
                   'tfidf__use_idf': (True, False),
                   'clf__alpha': (1e-2, 1e-3)}

gs_indiv_clf1 = GridSearchCV(indiv_text_clf1, parameters_clf1, n_jobs=-1)
gs_indiv_clf1 = gs_indiv_clf1.fit(indivTrain, indivTrainLabels)

gs_indiv_pred_clf1 = gs_indiv_clf1.predict (indivTest)

# Check how many predicted academic is correct
gs_indiv_accuracy_clf1 = np.mean(gs_indiv_pred_clf1 == indivTestLabels)

#gs_clf1.best_score_
#gs_clf1.best_params_

# SGD Classifier
indiv_text_clf2 = Pipeline([('vect', CountVectorizer(stop_words='english')),
                         ('tfidf', TfidfTransformer()),
                         ('clf-svm', SGDClassifier(loss='hinge', penalty='l2',
                         alpha=1e-3, n_iter=5, random_state=42))])

indiv_text_clf2 = indiv_text_clf2.fit(indivTrain, indivTrainLabels)

indiv_predicted_clf2 = indiv_text_clf2.predict (indivTest)

indiv_accuracy_clf2 = np.mean(predicted_clf2 == indivTestLabels)

# Use Grid Search CV
parameters_clf2 = {'vect__ngram_range': [(1, 1), (1, 2)],
                  'tfidf__use_idf': (True, False),
                  'clf-svm__alpha': (1e-2, 1e-3)}

gs_indiv_clf2 = GridSearchCV(indiv_text_clf2, parameters_clf2, n_jobs=-1)
gs_indiv_clf2 = gs_indiv_clf2.fit(indivTrain, indivTrainLabels)

gs_indiv_pred_clf2 = gs_indiv_clf2.predict(indivTest)

# Check how many predicted academic is correct
gs_indiv_accuracy_clf2 = np.mean(gs_indiv_pred_clf2 == indivTestLabels)

#gs_clf2.best_score_
#gs_clf2.best_params_

'''
Accuracies:
    1. gs_indiv_accuracy_clf1 ~ 0.9
    2. gs_indiv_accuracy_clf2 ~ 0.876
    3. gs_acad_accuracy_clf1 ~ 0.796
    4. gs_acad_accuracy_clf2 ~ 0.9253 


Use academic predictor: gs_acad_pred_clf2
Use individual predictor: gs_indiv_pred_clf1
'''









