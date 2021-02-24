import numpy as np
import tensorflow as tf
import pandas as pd
from pyspark.sql import *
from pyspark import SparkContext
from pyspark import SparkConf, SparkContext
import pyspark.sql.functions as sqlf
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK, STATUS_FAIL
from hyperopt.pyll.base import scope
tf.random.set_seed(123)

#Spark config
spark_config = SparkConf().setMaster('local')\
                .set('spark.sql.sources.partitionOverwriteMode','dynamic')\
                .set('spark.num.executors', 3) \
                .set('spark.executor.memory', '6gb')\
                .set('spark.executor.cores', 4).setAppName('Project_1')

sc = SparkContext(conf = spark_config)
sqlContext = SQLContext(sc)

# Load dataset.
dftrain = sqlContext.createDataFrame(pd.read_csv("https://storage.googleapis.com/tf-datasets/titanic/train.csv"))
dfeval = sqlContext.createDataFrame(pd.read_csv("https://storage.googleapis.com/tf-datasets/titanic/eval.csv"))
y_train = dftrain.select('survived')
y_eval = dfeval.select('survived')

CATEGORICAL_COLUMNS = ['sex', 'n_siblings_spouses', 'parch', 'class', 'deck',
                       'embark_town', 'alone']
NUMERIC_COLUMNS = ['age', 'fare']
feature_columns = []
exprs = []
for feature_name in CATEGORICAL_COLUMNS:
    # Need to one-hot encode categorical features.
    vocabulary = [feature_name + str('_') + str(i[0]) for i in dftrain.select(feature_name).distinct().rdd.collect()]
    feature_columns.extend(vocabulary)
    a = ([sqlf.when(sqlf.col(feature_name) == cat, 1).otherwise(0).alias(str(cat)) for cat in vocabulary])
    exprs.append(a)

for feature_name in NUMERIC_COLUMNS:
    feature_columns.append(feature_name)

exprs2 = []
for i in exprs:
    for j in i:
        exprs2.append(j)
dftrain2 = dftrain.select(exprs2+dftrain.columns)
dfeval2 = dfeval.select(exprs2+dfeval.columns)

#-----------------------MAKE INPUT-----------------------
def make_input(df, feature_columns,label_name, NUM_EXAMPLES=1000,n_epochs=None, shuffle=False,validate_data = False):
    X = df.select(feature_columns).toPandas()
    y = df.select(label_name).toPandas()
    dataset = tf.data.Dataset.from_tensor_slices((X, y))
    print(dataset.element_spec)
    if shuffle:
        dataset = dataset.shuffle(NUM_EXAMPLES)
        # For training, cycle thru dataset as many times as need (n_epochs=None).
    NUM_EXAMPLES = df.count()
    if validate_data==False:
        dataset = (dataset.repeat(n_epochs).batch(NUM_EXAMPLES))
    return dataset

label_name = 'survived'
train_input_fn = make_input(dftrain2,feature_columns,label_name,validate_data=False)
val_input_fn = make_input(dfeval2,feature_columns,label_name,validate_data=False)

#-----------------------MAKE MODEL-----------------------
def make_model(params, output_bias=None):
    METRICS = [
        tf.keras.metrics.BinaryAccuracy(name='accuracy'),
        tf.keras.metrics.Recall(name='recall'),
        tf.keras.metrics.AUC(name='auc'),
    ]
    metrics = METRICS
    if output_bias is not None:
        output_bias = tf.keras.initializers.Constant(output_bias)
    model = tf.keras.Sequential([
                                    tf.keras.layers.Flatten(),
                                    tf.keras.layers.Dense(units=params['units'],activation = 'relu'),
                                    tf.keras.layers.Dense(units=params['units'],activation='relu'),
                                    tf.keras.layers.Dropout(rate=params['fc_dropout_drop_proba']),
                                    tf.keras.layers.Dense(1, activation=params['activation'],bias_initializer=output_bias),
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(lr=params['lr_rate_mult']),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=metrics)

    # es=tf.keras.callbacks.EarlyStopping(monitor='val_loss',patience=10,verbose=1,mode='min',
    #                                                 restore_best_weights=True)

    #best_model=tf.keras.callbacks.ModelCheckpoint(filepath='best_model.h5',monitor='val_loss',save_best_only=True,verbose=1)

    result = model.fit(train_input_fn, validation_data = val_input_fn, epochs=100,steps_per_epoch = 200,
                              validation_steps = 50, verbose = 1)#, callbacks=[es])

    validation_loss = np.amin(result.history['val_loss'])
    return {'loss': validation_loss,
            'status': STATUS_OK,
            'model': model,
            'params': params}

space = {
    'lr_rate_mult': hp.loguniform('lr_rate_mult', -0.5, 0.5),
    # L2 weight decay:
    #'l2_weight_reg_mult': hp.loguniform('l2_weight_reg_mult', -1.3, 1.3),
    #node:
    'units': scope.int(hp.quniform('units',22,68,2)),
    # Batch size fed for each gradient update
    #'batch_size': hp.quniform('batch_size', 100, 450, 5),
    # Choice of optimizer:
    #'optimizer': hp.choice('optimizer', ['Adam', 'Nadam', 'RMSprop']),
    # Coarse labels importance for weights updates:
    # 'coarse_labels_weight': hp.uniform('coarse_labels_weight', 0.1, 0.7),
    # Uniform distribution in finding appropriate dropout values, FC layers
    'fc_dropout_drop_proba': hp.uniform('fc_dropout_drop_proba', 0.0, 0.6),
    # Use batch normalisation at more places?
    #'use_BN': hp.choice('use_BN', [False, True]),
    # Use activation everywhere?
    'activation': hp.choice('activation', ['relu', 'elu'])
}

trials = Trials()
best = fmin(make_model,
            space,
            algo=tpe.suggest,
            max_evals=50,
            trials=trials)