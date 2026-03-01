import numpy as np
from sklearn.metrics import confusion_matrix, classification_report

def CM(y_test,y_test_predict):
    '''
    Args:
       y_test: labels for test data
       y_test_predict: pedicted label for test data
    
    Returns:
       confusion matrix for model

    '''

    cm = confusion_matrix(y_true, y_test_predict)
    return cm

def report(y_test,y_test_predict):
    '''
    Args:
       y_test: labels for test data
       y_test_predict: pedicted label for test data
    
    Returns:
       classification_report for model
    '''
    return classification_report(y_test,y_test_predict)