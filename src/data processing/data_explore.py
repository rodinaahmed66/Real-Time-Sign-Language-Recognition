import pandas as pd

def balance_or(y):
    """
    Return class counts .
    """
    return pd.Series(y.flatten()).value_counts()

def remove_unbalance(x,y,clas):
    """
    Remove all samples of a specific class.
    """
    y = y.squeeze() 
    mask = y != clas
    # apply mask
    x = x[mask]
    y = y[mask]
    return x,y