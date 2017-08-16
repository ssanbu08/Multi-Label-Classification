# -*- coding: utf-8 -*-
"""
Created on Mon Aug 14 14:40:12 2017

@author: anbarasan.selvarasu
"""


from utility_func import Utility
from evaluation import EvaluationMetrics

from time import time
import pandas as pd
from sklearn.naive_bayes import MultinomialNB


class BinaryRelevance(object):
    
    def __init__(self):
        pass
    
    def build_model(self,train_x, train_y, test_x, test_y):
        util  = Utility()
        eval_metrics = EvaluationMetrics()
        br_predictions_label = pd.DataFrame()
        br_predictions_score = pd.DataFrame()
        br_classifiers = []
        test_y_br = test_y.drop(['E','P'], axis = 1)
        for i_col in test_y_br.columns:
            print(i_col)
            # i_col = '8'
            grid_search_dict = {}
            grid_search_dict['estimator'] = MultinomialNB()
            grid_search_dict['cvalidator'] = 3
            grid_search_dict['params'] = [{'alpha':[0.7,1.0]}]  
            grid_search_dict['loss_fun'] = 'neg_log_loss'
        
            t1 = time()
            classifier = util.build_model(train_x, train_y[i_col], grid_search_dict)
            print('Classifier {}, completed in {} '.format(i_col, time() - t1))
            
            pred_labels =  classifier.predict( test_x)
            pred_score = util.predict_score(classifier, test_x)
            
            eval_labels = eval_metrics.get_classification_report_1(test_y[i_col]
                                                    , pred_labels
                                                    , verbose = 1)    
            eval_score = eval_metrics.get_classification_report_3(test_y[i_col]
                                                    , pred_labels
                                                    , verbose = 1)
            br_predictions_label[i_col] =  pred_labels
            br_predictions_score[i_col] = pred_score[:,1]  
            br_classifiers.append(classifier) 
        return br_predictions_label, br_predictions_score, br_classifiers