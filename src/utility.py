from sklearn.ensemble import AdaBoostClassifier
from catboost import CatBoostClassifier 
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler      

import pandas as pd
import numpy as np
import re
import ssl #Secure Sockets Layer
ssl._create_default_https_context = ssl._create_unverified_context
from sklearn.model_selection import train_test_split,cross_val_score, StratifiedKFold,GridSearchCV
from sklearn.pipeline import Pipeline

from sklearn.metrics import accuracy_score
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score  # Ensure recall_score is imported
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
from sklearn.metrics import log_loss
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import balanced_accuracy_score



class data_profiling:
    def __init__(self,*args):
        self.df       = args[0]
        if len(args) >1 and args[1]:
            self.cat_col  = args[1]
        else:
            self.cat_col = [col for col in self.df.columns if len(self.df[col].unique()) < 40 ]
        self.float_column = [col for col in self.df.columns if self.df[col].dtype == float]
        self.int_column = [col for col in self.df.columns if self.df[col].dtype == int]
        self.date_columns  = []
        self.rest_columns  = [col for col in self.df.columns if col not in (self.float_column + self.int_column + self.cat_col)]
        self.col_min_char = pd.DataFrame(self.df.astype(str).apply(lambda x:x.str.len().min()),columns=['min_char'])
        self.col_max_char = pd.DataFrame(self.df.astype(str).apply(lambda x:x.str.len().max()),columns=['max_char'])
        self.term_deposit_non_ascii     = pd.DataFrame(self.df.apply(lambda x: sum(ord(char)>127 for chars in x for char in str(chars)),axis=0),columns=['non-ascii character'])
        self.term_deposit_null_value    = pd.DataFrame(self.df.isna().sum(),columns=['number of nan and none values'])
        self.size_mega        = self.df.memory_usage(deep=True).sum() / (1024 * 1024)
        self.number_of_duplicated_rows  = sum(self.df.duplicated())
        self.space = []
        

        for col in self.df.columns:
            if self.df[col].dtype == 'object':  # Check if the column is of string type
                total_spaces = self.df[col].apply(lambda x: self.count_empty_space(x) if x is not (np.nan or None) else 0).sum() 
            else:total_spaces=0
            self.space.append(total_spaces)    
        self.empty_string =[]
        
    def count_empty_space(self,x):
        return len(re.findall(r' +', x))
    def __call__(self):
        le = LabelEncoder()
        term_deposit_encoded = self.df.copy()
        for col in self.cat_col:
            term_deposit_encoded[col] = le.fit_transform(self.df[col])
        self.term_deposit_describe = pd.DataFrame(term_deposit_encoded.describe().transpose())
        self.duplicated_rows = pd.DataFrame({"duplicated_rows":[sum(self.df.duplicated())]*self.df.shape[1]},index=self.df.columns.to_list())
        self.duplicated_cols = pd.DataFrame({"duplicated_columnss":[sum(self.df.transpose().duplicated())]*self.df.shape[1]},index=self.df.columns.to_list())

        self.shape_size = pd.DataFrame({"shape and size":[[self.df.shape]+["{:.2f} Mb".format(self.size_mega)]]*self.df.shape[1]},index=self.df.columns.to_list())
        self.empty_string_total = pd.DataFrame({"Completeness_Empty":[self.empty_string]},index=self.df.columns.to_list())
        self.empty_space_total =  pd.DataFrame({"Completeness_Space":[ self.space]*self.df.shape[1]},index=self.df.columns.to_list())

        self.float_col =  pd.DataFrame({"float_col":[ self.float_column]*self.df.shape[1]},index=self.df.columns.to_list())
        self.float_col_length =  pd.DataFrame({"float_col_legth":[ len(self.float_column)]*self.df.shape[1]},index=self.df.columns.to_list())

        self.int_col =  pd.DataFrame({"int_col":[ self.int_column]*self.df.shape[1]},index=self.df.columns.to_list())
        self.int_col_length =  pd.DataFrame({"int_col_length":[ len(self.int_column)]*self.df.shape[1]},index=self.df.columns.to_list())
    
        return pd.concat([self.term_deposit_describe,self.term_deposit_null_value,self.col_min_char,self.col_max_char,self.term_deposit_non_ascii,
                          self.duplicated_rows,self.duplicated_cols,self.shape_size,self.empty_string_total, self.empty_space_total,self.float_col,
                          self.float_col_length,self.int_col,self.int_col_length],axis=1)
    
    def __repr__(self):
        """
        stands for representation
        print(repr(data_profiling))
        """
        return self.__class__.__name__



model_parameters_classification = {
    'decisiontree_classifier':{
        'model': DecisionTreeClassifier(random_state=42),
        'params': {
            'classifier__max_depth': [None, 5], # 
            'classifier__min_samples_split': [2, 5],
            'classifier__min_samples_leaf': [1, 2]
        }
    },
    'randomforest_classifier':{
        'model': RandomForestClassifier(random_state=42),
        'params': {
            'classifier__n_estimators': [10, 100],
            'classifier__criterion': ['gini', 'entropy'],
            'classifier__max_depth': [None, 10],
            'classifier__bootstrap': [True, False],
            'classifier__max_features': ['sqrt',  None],
            'classifier__max_samples': [0.0, 0.2]
        }
    },
    'adaboost_classifier':{
        'model': AdaBoostClassifier(),
        'params': {
            'classifier__n_estimators': [50, 100],
            'classifier__learning_rate': [0.01, 0.1],
            'classifier__algorithm': ['SAMME']
        }
    },
    'catboost_classifier':{
        'model': CatBoostClassifier(silent=True),
        'params': {
            'classifier__depth': [4, 6],
            'classifier__learning_rate': [0.01, 0.03],
            'classifier__l2_leaf_reg': [1, 3]
        }
    },
    'xgboost':{
        'model': xgb.XGBClassifier(),
        'params': {
            'classifier__max_depth': [3, 5],
            'classifier__min_child_weight': [1, 3],
            'classifier__gamma': [0, 0.1],
            'classifier__subsample': [0.8, 1.0],
            'classifier__colsample_bytree': [0.8, 1],
            'classifier__eta': [0.01, 0.1]
        }
    },
    'lgb_classifier':{
        'model': LGBMClassifier(verbose=-1),
        'params': {
            'classifier__num_leaves': [15, 31],
            'classifier__max_depth': [3, 5],
            'classifier__learning_rate': [0.01, 0.05],
            'classifier__min_child_samples': [10, 20],
            'classifier__feature_fraction': [0.5, 0.7],
             'classifier__colsample_bytree': [None],
            'classifier__bagging_fraction': [0.5, 0.7]
        }
    },
    'mlp_classifier':{
        'model': MLPClassifier(max_iter=500,tol=1e-9, random_state=42, early_stopping=True),
        'params': {
            'classifier__hidden_layer_sizes': [(50,), (100,), (50, 30)],
            'classifier__activation': ['relu', 'tanh'],
            'classifier__solver': ['adam', 'sgd'],
            'classifier__alpha': [1e-5, 1e-4],
            'classifier__learning_rate_init': [0.001, 0.01],
            'classifier__max_iter': [200, 300]
        }
    }
}
