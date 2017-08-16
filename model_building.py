# -*- coding: utf-8 -*-
"""
Created on Wed Aug 16 00:57:23 2017

@author: anbarasan.selvarasu
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Aug 14 09:08:09 2017

@author: anbarasan.selvarasu
"""
import warnings

import sys

import pandas as pd
import numpy as np
import pickle
from collections import Counter
from time import time
import gc



from scipy.sparse import lil_matrix
from data_loading import DataLoading
from text_processing import TextProcessing
from utility_func import Utility
from lp_classification import LabelPowerSet
from br_classification import BinaryRelevance
from evaluation import EvaluationMetrics


def main():
    input_directory  = sys.argv[1]
    train_size = int(sys.argv[2])
    test_size = (100- train_size) / 100
    
    
      
    
    ##### Step 1: Data Loading and Basic stats #####
    t0 = time()
    
    print()
    print('** STEP 1: Data Loading **')
    dl_obj = DataLoading()
    base_df = dl_obj.clean_data(input_directory)
    #prodid_ix = base_df.id.values
    #base_df = base_df.reindex(prodid_ix)
    
    ## This line should be removed ##
    #print('Only 1000 rows are loaded')
    #base_df = base_df.sample(10000, random_state = 123)
    
    target_matrix = dl_obj.get_multilabel(base_df)
    #target_matrix = target_matrix.reindex(prodid_ix)
    
    dl_obj.get_label_info(target_matrix)
    
    #### Step 2: feature Engineering #####
    
    print()
    print('** STEP 2: Text Processing **')
    tp_obj = TextProcessing()
    cnt_vectorizer, feature_matrix = tp_obj.run_textprocessing(base_df)
    
    feature_matrix  = pd.DataFrame(feature_matrix.toarray())
    feature_matrix = feature_matrix.join(base_df[['vegetarian','spicy', 'garlic', 'fish']])  
    feature_matrix.fillna(0, inplace = True)  
    
    #### Step 3: 
    ### STEP 1: Normalize the labels ###
    print()
    print('** Filter Rare Labels combination **')
    util = Utility()
    print("Feature Matrix Shape:{} Target Matrix.shape: {}"\
            .format(feature_matrix.shape, target_matrix.shape))    
    feature_matrix_fltrd,target_matrix_fltrd =  util.filter_rare_classes(feature_matrix
                                                        , target_matrix)
    print("Feature Matrix Shape:{} Target Matrix.shape: {}"\
            .format(feature_matrix_fltrd.shape, target_matrix_fltrd.shape))# (18340,3763)
            
    ### STEP 2: Train Test Split using StratifiedShuffleSplit #####
    print()
    print('** Train test split **')
    train_x, train_y, test_x, test_y  = util.train_test_split(feature_matrix_fltrd
                                                        , target_matrix_fltrd
                                                        , test_size = test_size)
    print("Train_x Shape:{} \n Train_y.shape: {}"\
            .format(train_x.shape, train_y.shape)) # 14672
    print("Test_x Shape:{} \n Test_y.shape: {}"\
            .format(test_x.shape, test_y.shape)) # 3668 
   
          ### Delete unnecssary files from memory ##
    
            
    ### STEP 3: Find Frequnet Itemsets on training target matrix ####
    print()
    print('** STEP 3: Frequent Itemset **')
    
    col_mapping = {}
    for i_col, col_name in enumerate(target_matrix.columns.tolist()):
        col_mapping[i_col] = col_name    
    
    supp = 0.05
    item_size = 3
    train_y_lil = lil_matrix(train_y)
    frequent_items_list = util.find_frequent_itemsets(train_y_lil
                                                , col_mapping
                                                , supp
                                                , item_size)
    print('No of {} frequent itemsets with support {}: {} '\
           .format(item_size
                  , supp
                  , len(frequent_items_list))) #21 itemsets
    
    freq_additives_list = [items for itemset in frequent_items_list for items in itemset]
    freq_additives_set = list(set([items for itemset in frequent_items_list for items in itemset]))
    freq_additives_cnt_dict = dict(Counter(freq_additives_list).items())
    
    #del base_df,target_matrix,target_matrix_fltrd, feature_matrix, feature_matrix_fltrd
    #gc.collect()
    
    ### STEP 4.1: Build 21 classifiers using Naive Bayes ####
    print()
    print('** STEP 4: LabelPowerSet Classifiers**')
    lp = LabelPowerSet(train_x, train_y, test_x, test_y, frequent_items_list,'nb')
    
    
    model_list, metrics_labels, metrics_score, prediction_list = lp.build_model_lp()
    index_value = [''.join(items) for items in frequent_items_list]                                                                       
    metrics_labels_df = pd.DataFrame(metrics_labels
                                    , columns = ['Accuracy', 'HammingLoss', 'JaccardSim']
                                    , index = index_value)                                                                     
    metrics_score_df = pd.DataFrame(metrics_score
                                    , columns = ['CoverageError','LblRankAvgPrec'
                                                ,'LblRankLoss','LogLoss']
                                    , index = index_value
                                    )
    pickle.dump(model_list, open('LP_NB_21FSS.pkl', 'wb')) 
    del model_list,lp  
    metrics_labels_df.to_csv(input_directory +'LP_NB_metrics_labels.csv')
    metrics_score_df.to_csv(input_directory +'LP_NB_metrics_score.csv')                          
    
    
    
    ####### STEP 4.1: stack the predictions ############
    final_predictions = pd.DataFrame(np.zeros(test_y[freq_additives_set].shape)
                                     ,columns = freq_additives_set)   
    for i_model in range(len(prediction_list )):
        #i_model = 0
        prediction = prediction_list[i_model]
        for col in prediction.columns:
            final_predictions[col] = final_predictions[col] + prediction[col]
            
    
    final_predictions_2= final_predictions.apply(lambda x : x/freq_additives_cnt_dict[x.name])
    final_predictions_2 = final_predictions_2.applymap(lambda x: 1 if x>=0.5 else 0)
    
    print()
    print('** Evaluation metrics : Majority Voting**')
    eval_metrics = EvaluationMetrics()
    eval_final = eval_metrics.get_classification_report_1(test_y[freq_additives_set], final_predictions_2, verbose = 1)
    
    
    #### STEP 5: Build Binary Relevance models ####
    
    print()
    print('** STEP 5 : Binary Relevance Classifiers **')
    br =  BinaryRelevance()
    label_df, score_df, classifier_list = br.build_model(train_x, train_y, test_x, test_y)
    pickle.dump(classifier_list
              , open('BR_NB_classifiersList.pickle','wb'))
    
    print()
    print('** Evaluation Metrics for BR Classfiers **')
    eval_metrics.get_classification_report_1(test_y[label_df.columns], label_df )
    # Accurcay : 0.42 Hamming Loss: 0.05, Jaccard Similarity :0.62
    
    eval_metrics.get_classification_report_2(test_y[label_df.columns], score_df)
    # CoverageError : 5.61, LabelRankingAvgPrec :0.83, LabelRankingLoss : 0.04, Log_loss = 6.7
    
    ######## Binary Relevance predictions for frequent labels #####
    print()
    print('** BR classifiers evaluation for labels in frequentitemset **')
    eval_metrics.get_classification_report_1(test_y[freq_additives_set],label_df[freq_additives_set])
    
    
    
    ### STEP 6: Final Predictions #########
    print()
    print('** STEP 6 : Final Predictions **')
    final_predictions_3 = pd.DataFrame(np.zeros(label_df.shape)
                                     ,columns = label_df.columns
                                     )   
    
    
    ### Binary Relevance + LabelPowerset #####
    for col in final_predictions_3.columns:
        if col in freq_additives_set:
            final_predictions_3[col] = final_predictions_2[col]
        else:
            final_predictions_3[col] = label_df[col]
            
    print()
    print('** Evaluation Metrics for Final Predcition **')
    print('test_ shape', test_y[label_df.columns].shape)
    print('final predictions',final_predictions_3.shape)
    eval_final_2 = eval_metrics.get_classification_report_1(test_y[label_df.columns], final_predictions_3, verbose = 1)
    
    
    ### STEP 7: Dumping Predictions ##########
    print()
    print('** STEP 7 : Saving Predictions **')
    test_y.to_csv('test_actual_labels.csv')
    final_predictions_3.to_csv('test_final_predicted_labels.csv')
    score_df.to_csv('test_scoring_from_br.csv')
    
    print('Entire Process completed in {} seconds'.format(time()-t0))


if __name__ == '__main__':
    warnings.filterwarnings("ignore",)
    main()

