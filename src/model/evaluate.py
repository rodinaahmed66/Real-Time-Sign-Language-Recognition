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

    y_pred = np.argmax(y_test_predict, axis=1)
    y_true = np.argmax(y_test, axis=1)

    cm = confusion_matrix(y_true, y_pred)
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