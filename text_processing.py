# -*- coding: utf-8 -*-
"""
Created on Mon Aug 14 08:42:11 2017

@author: anbarasan.selvarasu
"""

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import nltk 
nltk.download('punkt')
nltk.download('stopwords')
import re
import string

class TextProcessing(object):
    
    def __init__(self):
        pass
    
    def build_feature_matrix(self,documents, feature_type='frequency',
                                 ngram_range=(1, 1), min_df=1, max_df=1.0):
        feature_type = feature_type.lower().strip()  
        
        if feature_type == 'binary':
            vectorizer = CountVectorizer(binary=True, min_df=min_df,
                                         max_df=max_df, ngram_range=ngram_range)
        elif feature_type == 'frequency':
            vectorizer = CountVectorizer(binary=False, min_df=min_df,
                                         max_df=max_df, ngram_range=ngram_range)
        elif feature_type == 'tfidf':
            print("mindf",min_df)
            print("max_df", max_df)
            vectorizer = TfidfVectorizer(min_df=min_df, max_df=max_df, 
                                         ngram_range=ngram_range)
        else:
            raise Exception("Wrong feature type entered. Possible values: 'binary', 'frequency', 'tfidf'")
    
        feature_matrix = vectorizer.fit_transform(documents).astype(float)
        return vectorizer, feature_matrix
     
          
    def remove_stopwords(self, tokens):
        
        stopword_list = nltk.corpus.stopwords.words('german')
        filtered_tokens = [token for token in tokens if token not in stopword_list]
        filtered_text = ' '.join(filtered_tokens)    
        return filtered_text
        
    def tokenize_text(self, text):
        text = str(text)
        tokens = nltk.word_tokenize(text) 
        tokens = [token.strip() for token in tokens]
        tokens = [token.lower() for token in tokens]
        return tokens
        
    def keep_text_characters(self, text):
        tokens = self.tokenize_text(text)
        filtered_tokens = []
        for token in tokens:
            if re.search('[a-zA-Z]', token):
                filtered_tokens.append(token)
        filtered_text = ' '.join(filtered_tokens)
        return filtered_text            

    def run_textprocessing(self, base_df):    
        base_df['description_tokenized'] = base_df.description.apply(self.tokenize_text)    
        base_df['description_tokenized'] = base_df.description_tokenized.apply(self.remove_stopwords)    
        base_df['description_tokenized'] = base_df.description_tokenized.apply(self.keep_text_characters)    
        
        cnt_vectorizer, feature_matrix = self.build_feature_matrix(base_df['description_tokenized'] 
                                                            , feature_type='binary'
                                                            , ngram_range=(1, 1)
                                                            , min_df= 1 , max_df= 1.0)
        print("Feature Matrix shape", feature_matrix.shape)  
        return cnt_vectorizer, feature_matrix

def main():
    print('')
    

if __name__ == "main":
    main()