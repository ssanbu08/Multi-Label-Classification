# -*- coding: utf-8 -*-
"""
Created on Mon Aug 14 10:29:57 2017

@author: anbarasan.selvarasu
"""
import warnings
warnings.filterwarnings("ignore",category=DeprecationWarning)
from sklearn.cross_validation import StratifiedShuffleSplit
    
from utility_func import Utility
from evaluation import EvaluationMetrics

from time import time
import pandas as pd
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC


class LabelPowerSet(object):
    
    def __init__(self,train_x, train_y, test_x, test_y, frequent_items_list,classifier):
        self.train_x  = train_x
        self.train_y = train_y
        self.test_x = test_x
        self.test_y = test_y
        self.frequent_items_list =  frequent_items_list
        self.classifier = classifier
        
        
    
    def build_model_lp(self):
        util = Utility()
        eval_met = EvaluationMetrics()
        model_list = []
        metrics_labels = []
        metrics_score = []
        prediction_list = []
        t0 = time()
        for items in self.frequent_items_list:
            #items = 0
            label_subset_train = self.train_y[items]
            multiclass_labels = util.transform(label_subset_train)
            sss_cv = StratifiedShuffleSplit(multiclass_labels
                                               ,n_iter = 3 #3
                                               ,test_size = 0.3 #0.3
                                               ,random_state = 123) 
            grid_search_dict = {}
            if(self.classifier =='nb'):
                grid_search_dict['estimator'] = MultinomialNB()
                grid_search_dict['cvalidator'] = sss_cv
                grid_search_dict['params'] = [{'alpha':[0.7,1.0]}]  
                grid_search_dict['loss_fun'] = 'neg_log_loss'
            
            elif(self.classifier =='svc'):
                grid_search_dict['estimator'] = SVC(probability  = True
                                                    , kernel = 'rbf'
                                                    , gamma = 0.001)
                grid_search_dict['cvalidator'] = sss_cv
                grid_search_dict['params'] = [{'C':[100,1000]}]  
                grid_search_dict['loss_fun'] = 'neg_log_loss'
                
            t1 = time()
            print('Classifier {}, Started {} seconds '.format(items, time() - t1))
            classifier = util.build_model(self.train_x, multiclass_labels, grid_search_dict)
            print('Classifier {}, completed in {} seconds '.format(items, time() - t1))
            
            pred_labels =  util.predict_label(classifier, self.test_x)
            pred_labels.columns = label_subset_train.columns.tolist()
            pred_score = util.predict_score(classifier, self.test_x)
            
            label_subset_test = self.test_y[items]
            
            eval_labels = eval_met.get_classification_report_1(label_subset_test
                                                             , pred_labels
                                                             , verbose = 1)
            
            transformed_test_labels = util.transform(label_subset_test)
            dummy_trans_test_labels = pd.get_dummies(transformed_test_labels)    
            eval_score = eval_met.get_classification_report_2(dummy_trans_test_labels
                                                            , pred_score
                                                            , verbose = 1)
                
            
            model_list.append(classifier)
            metrics_labels.append(eval_labels)
            metrics_score.append(eval_score)
            prediction_list.append(pred_labels)
        print('Label Powerset Method Completed in {}'.format(time()-t0))
        return model_list, metrics_labels, metrics_score, prediction_list