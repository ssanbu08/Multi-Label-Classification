# -*- coding: utf-8 -*-
"""
Created on Mon Aug 14 08:51:46 2017

@author: anbarasan.selvarasu
"""

from pandas import DataFrame, Series
import pandas as pd
import numpy as np



class DataLoading(object):
    
    def __init__(self):
        pass
    
    def get_multilabel(self, base_df):
    
        additives_list = [x.split(',') if isinstance(x,str) else [x] for x in base_df.additives]
        additives_list = set([y for x in additives_list for y in x])    
        num_rows = base_df.shape[0]
        num_cols = len(additives_list)
        print('Dimension of Dummy Matrix', num_rows,num_cols)        
        ###create a dataframe to hold the target labels ####
        target_dummies = DataFrame(np.zeros((num_rows, num_cols)), columns = additives_list)        
        for i,a_a in enumerate(base_df.additives):
            if isinstance(a_a,str) :
                target_dummies.ix[i, a_a.split(',')] = 1
            else:
                target_dummies.ix[i, a_a] = 1
        return target_dummies
        
    def clean_data(self,input_directory):    
        base_df = pd.read_csv(input_directory, sep =';')
        
        ## Preprocessing ##
        base_df.description.fillna(base_df.name, inplace=True)
        base_df = base_df[base_df.additives.notnull()]
                
        print('Dimension of Dataset', base_df.shape)
        print('Unique A&A ',len(base_df['additives'].unique()))
        return base_df

    def get_label_info(self, target_matrix):
        print()
        print('Label Cardinality',target_matrix.sum(axis = 1).mean()) 
        print()
        print('Label Cardinality gives an idea about number of frequnet itemset to be mined')
        print()
        print('Label Distribution',target_matrix.sum(axis = 0)/target_matrix.shape[0])

        co_occurence = np.dot(target_matrix.T, target_matrix)
        np.fill_diagonal(co_occurence, 0 )

        labels = target_matrix.columns
        labels_dict = {}
        for i, label in enumerate(labels):
            labels_dict[i] = label
        
        
        
#==============================================================================
#     def show_graph_with_labels(self, adjacency_matrix, mylabels):
#         rows, cols = np.where(adjacency_matrix>0)
#         print(set(cols))
#         drop_labels = [col for col in list(mylabels.keys()) if col not in set(cols)]
#         print(drop_labels)
#         
#         if len(drop_labels)>0:
#             for drop in drop_labels:
#                 del mylabels[drop]
#         print(mylabels)            
#         edges = zip(rows.tolist(), cols.tolist())
#         gr = nx.Graph()
#         gr.add_edges_from(edges)
#         #elist=[('a','b',5.0),('b','c',3.0),('a','c',1.0),('c','d',7.3)]
#         #G.add_weighted_edges_from(elist)
#         
#         nx.draw(gr, node_size=500, labels=mylabels, with_labels=True)
#         plt.show()
#         
#==============================================================================
    
    



