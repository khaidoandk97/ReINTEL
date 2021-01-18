import numpy as np 
import pandas as pd 

from sklearn.naive_bayes import GaussianNB, MultinomialNB # MutinominalNB
from sklearn.svm import SVC 
# from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

from sklearn.pipeline import Pipeline
# from sklearn.ensemble import StackingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import time

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

def models(X, y, option = 0, val_size=0.2, random_state=0):
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=val_size, random_state=random_state)

    time_init = time.time()
    if option == 0:
        model = KNeighborsClassifier()
        model = model.fit(X_train, y_train)

    elif option == 1:
        model = GaussianNB()
        model = model.fit(X_train, y_train)
    
    elif option == 2:
        model = SVC()
        model = model.fit(X_train, y_train)

    elif option == 3:
        model = DecisionTreeClassifier()
        model = model.fit(X_train, y_train)

    else:
        model = LogisticRegression()
        model = model.fit(X_train, y_train)

    start = time.time()
    y_pred_train = model.predict(X_train)
    stop_train = time.time()
    y_pred_val = model.predict(X_val)
    stop_val = time.time()

    # Report
    print(f'Report of train set with {int(len(X_train))} examples: ')
    print(f'Time to train: {round(start-time_init, 4)}s')
    print(f'Time to predict train set: {round(stop_train-start, 4)}s')
    print(classification_report(y_train, y_pred_train))
    print('\nReport of validation set:')
    print(f'Time to predict validation set {round(stop_val-start, 4)}s')
    print(classification_report(y_val, y_pred_val))

    return model

def cat_models(X, y, option = 0, val_size=0.2, random_state=0):
    pipeline = Pipeline([('count',CountVectorizer()),('tfidf',TfidfTransformer())])

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=val_size, random_state=random_state)
    X_train = pipeline.fit_transform(X_train).toarray()
    X_val = pipeline.transform(X_val).toarray()

    time_init = time.time()

    if option == 0:
        model = KNeighborsClassifier()
        model = model.fit(X_train, y_train)

    elif option == 1:
        model = GaussianNB()
        model = model.fit(X_train, y_train)
    
    elif option == 2:
        model = SVC()
        model = model.fit(X_train, y_train)

    elif option == 3:
        model = DecisionTreeClassifier()
        model = model.fit(X_train, y_train)

    else:
        model = LogisticRegression()
        model = model.fit(X_train, y_train)

    start = time.time()
    y_pred_train = model.predict(X_train)
    stop_train = time.time()
    y_pred_val = model.predict(X_val)
    stop_val = time.time()

    # Report
    print(f'Report of train set with {int(len(X_train))} examples: ')
    print(f'Time to train: {round(start-time_init, 4)}s')
    print(f'Time to predict train set: {round(stop_train-start, 4)}s')
    print(classification_report(y_train, y_pred_train))
    print('\nReport of validation set:')
    print(f'Time to predict validation set {round(stop_val-start, 4)}s')
    print(classification_report(y_val, y_pred_val))

    return model, pipeline

def predict(model, X, y, name_set='test set'):
    start = time.time()
    y_pred = model.predict(X)
    stop = time.time()

    print(f'Time to predict on the {name_set} with {len(X)} examples: {round(stop-start, 4)}s')
    print(classification_report(y, y_pred))
    
    return y_pred

def cat_predict(model, pipeline, X, y, name_set='test set'):
    X = pipeline.transform(X).toarray()
    
    start = time.time()
    y_pred = model.predict(X)
    stop = time.time()

    print(f'Time to predict on the {name_set} with {len(X)} examples: {round(stop-start, 4)}s')
    print(classification_report(y, y_pred))
    
    return y_pred