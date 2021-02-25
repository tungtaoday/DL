import numpy as np
import tensorflow as tf
import pandas as pd
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK, STATUS_FAIL
from hyperopt.pyll.base import scope
from sklearn.multioutput import ClassifierChain
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

X, Y = fetch_openml('yeast', version=4, return_X_y=True)
Y = Y.astype(int)

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=42)

#Base line model
params_baseline = {
	'n_estimators' : 100,
	'criterion':'gini',
	'max_depth':None,
	'min_samples_split':2,
	'min_samples_leaf':1,
	'min_weight_fraction_leaf':0.0,
	'max_features':'auto',
	'max_leaf_nodes':None,
	'min_impurity_decrease':0.0,
	'min_impurity_split':None,
	'bootstrap':True,
	'oob_score':False,
	'n_jobs':None,
	'random_state':None,
	'verbose':0,
	'class_weight':None,
	'ccp_alpha':0.0,
	'max_samples':None
}
y_train_test = y_train['Class1']
# Train model.
model = tf.estimator.BoostedTreesClassifier(feature_columns,**params_baseline)
model.train(X_train,y_train_test, max_steps=100)

chains=[ClassifierChain(model, order='random', random_state=42) for i in range(10)]
for chain in chains:
	chain.fit(X_train,y_train)

# Evaluation.
results = est.evaluate(eval_input_fn)
clear_output()
pd.Series(results).to_frame()
