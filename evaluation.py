# -*- coding: utf-8 -*-
"""
Created on Mon Aug 14 10:19:41 2017

@author: anbarasan.selvarasu
"""

from sklearn import metrics
import pandas as pd
import numpy as np

class EvaluationMetrics(object):
    
    def __init__(self):
        pass

    def get_classification_report_1(self, train_y, predicted_label_spm, verbose = 1):
        acc = metrics.accuracy_score(np.array(train_y), predicted_label_spm)
        ham_loss = metrics.hamming_loss(np.array(train_y), predicted_label_spm)
        jac_sim = metrics.jaccard_similarity_score(np.array(train_y), predicted_label_spm)
        if(verbose):
            print('Accuracy', acc)
            print('Hamming Loss', ham_loss)
            print('Jaccard Similarity',jac_sim)
        return [acc,ham_loss, jac_sim]
        
    def get_classification_report_2(self,train_y, predicted_score, verbose = 1):
        cov_err = metrics.coverage_error(train_y,predicted_score)
        label_rank_avg_prec = metrics.label_ranking_average_precision_score(train_y, predicted_score)
        rank_loss = metrics.label_ranking_loss(train_y, predicted_score)
        log_loss = metrics.log_loss(train_y, predicted_score)
        if(verbose):
            print('CoverageError', cov_err)
            print('LabelRankingAvgPrec', label_rank_avg_prec)
            print('LabelRankingLoss', rank_loss)
            print('log_loss', log_loss)
        return [cov_err, label_rank_avg_prec, rank_loss, log_loss]
        
    def get_classification_report_3(self, train_y, predicted_score, verbose = 1):
        cov_err = metrics.precision_score(train_y,predicted_score)
        label_rank_avg_prec = metrics.recall_score(train_y, predicted_score)
        rank_loss = metrics.f1_score(train_y, predicted_score)
        log_loss = metrics.log_loss(train_y, predicted_score)
        auc = metrics.roc_auc_score(train_y, predicted_score)
        if(verbose):
            print('Precision', cov_err)
            print('Recall', label_rank_avg_prec)
            print('F1-Score', rank_loss)
            print('log_loss', log_loss)
            print('AUC', auc)
        return [cov_err, label_rank_avg_prec, rank_loss, log_loss, auc]