# -*- coding: utf-8 -*-
"""
Created on Mon Aug 14 10:27:28 2017

@author: anbarasan.selvarasu
"""
import warnings
warnings.filterwarnings("ignore",category=DeprecationWarning)


import pandas as pd
from sklearn.grid_search import GridSearchCV

import numpy as np

from skmultilearn.problem_transform import LabelPowerset

from sklearn.cross_validation import StratifiedShuffleSplit

from orangecontrib.associate.fpgrowth import *
from scipy.sparse import lil_matrix

class Utility(object):
    
    def __init__(self):
        pass

    def transform(self, multilabel):
        '''
        Multi Label to MultiCLass Mapping
        
        Input:
            multilabel : DataFrame containing multi labels
        
        Output:
            multiclass_label : Series of Multiclass Labels (MC)    
        '''
        multilabel = multilabel.astype(int)
        multilabel = multilabel.astype(str)
        multilabel['combined_label'] = multilabel.iloc[:,0].str.\
                                        cat(others=[multilabel[col] 
                                                    for col in multilabel.columns[1:]]
                                                    , sep = ''
                                                    , na_rep='')
        multilabel['transformed_label'] = multilabel['combined_label'].apply(lambda x : int(x,2))
        return multilabel['transformed_label']
                                                      
    def inverse_transform(self, multiclass_labels):
        '''
        MultiCLass to Multi Label Mapping
        
        Input:
            class_labels : Multiclass Labels
        
        Output:
            multi_label : Multi Labels
        '''
        multi_label = pd.Series(multiclass_labels)\
                            .apply(lambda x: pd.Series(list(format(int(x), '03b'))))
        return multi_label
        
    def filter_rare_classes(self, feature_matrix, target_matrix):
        '''
        In order to perform stratified split between train and test,there
        should be atleast 2 instances present in the data. Hence, filter 
        label combinations that occurs only once in the entire dataset.
        Input : 
            Feature Matrix : matrix of features
            Target Matrix : matrix containing the the target labels
        Output :
            Feature Matrix : Filtered 
            Target Matrix : Filtered    
        
        '''
        lp = LabelPowerset()
        multi_class_target_labels = lp.transform(target_matrix)
        classes_vc = np.asarray(np.unique(multi_class_target_labels
                               , return_counts= True)).T # 1635 unique classes
        class_to_keep = classes_vc[np.where(classes_vc[:,1]>1)][:,0]
        mask = [True 
                if (multi_class_target_labels[i] in (class_to_keep)) 
                else False
                for i in range(len(multi_class_target_labels))]
        feature_matrix = feature_matrix[mask]        
        target_matrix = target_matrix[mask]
        
        return feature_matrix, target_matrix
        
    def train_test_split(self, feature_matrix, target_matrix, test_size = 0.2):
        '''
        Stratified Shuffle split technique is used to split train and test set,
        to have the equal proportion of classes in train and test.
        
        Input:
            feature_matrix : Feature matrix with rare classes filtered out
            target_matrix : Target matrix with rare classes filtered out
            test_size: default is  20%
        
        Output:
            train_x, train_y, test_x, test_y
        '''
        lp = LabelPowerset()
        sss_level_1 = StratifiedShuffleSplit(lp.transform(target_matrix)
                                    ,n_iter = 1
                                    ,test_size = 0.2
                                    ,random_state = 123) 
        for train_ix, test_ix in sss_level_1:
            
            train_x = feature_matrix.iloc[train_ix,:]
            train_y = target_matrix.iloc[train_ix,:]
            
            test_x = feature_matrix.iloc[test_ix,:]
            test_y = target_matrix.iloc[test_ix,:]
            
        return train_x, train_y, test_x, test_y
        
    def find_frequent_itemsets(self, target_lil, col_mapping, supp, item_size):
        
        '''
        Input:
            col_mapping : Dictionary of mapping between Column Number 
                          and Column Name.
            supp : minimum support threshold required for frequent itemset 
                   mining.
            item_size : size of Frequwnt itemsets to be mined.
            
        Output:
            frequent_items_list : List containing list of frequent itemsets
        '''
        
        freq_itemset = [(itemset, support) 
                        for itemset, support in frequent_itemsets(target_lil,supp)
                        if len(itemset) == item_size]
             
        frequent_items_list = []     
        
        for itemsets, num_instances in freq_itemset:
            frequent_items = [col_mapping[i] for i in itemsets]
            frequent_items_list.append(frequent_items)
            print(', '.join(col_mapping[i] for i in itemsets)
                  , '(Num Instances: {})'.format(num_instances))
        
        return frequent_items_list
    
    def build_model(self, train_x, train_y\
                  , grid_search_dict):  
        """
        Input:
             train_x : feature matrix
             train_y : Frequent Subset of multilabel tranformed as 
                       multiclass label
             grid_search_dict : arguments to be passed for grid search
        Output:
             Classifier trained with best parameters.
        """
        clf = GridSearchCV(estimator = grid_search_dict['estimator']
                                , cv = grid_search_dict['cvalidator']
                                , param_grid = grid_search_dict['params']
                                , scoring = grid_search_dict['loss_fun']
                                , n_jobs = -1)  
        clf.fit(train_x, train_y)
        print(clf.best_params_, clf.best_score_ ) 
        return clf
        
    def predict_label(self, classifier, test_x):
        '''
        Input:
            Classifier: Multi_class classifier fitted on training set
            test_x : Test Feature Matrix
            
        Output:
            multilabel_pred_label : predicted Multiclass transformed to Multilabel
        '''
        
        multiclass_pred_label = classifier.predict(test_x)
        multilabel_pred_label = self.inverse_transform(multiclass_pred_label)
        return multilabel_pred_label.astype(int)
        
    def predict_score(self, classifier, test_x):
        '''
        Input:
             Classifier: Multi_class classifier fitted on training set
             test_x : Test Feature Matrix
            
        Output:
            probscore_multiclass : predicted Multiclass transformed to Multilabel.
                                   (train_x.shape[0] x n_multiclass)
        
        '''
        probscore_multiclass =  classifier.predict_proba(test_x)
        
        return probscore_multiclass
        
                           
             