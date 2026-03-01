import numpy as np

def data_loader(img_path,label_path):
        
    """
    Load dataset from .npy files.
    
    Args:
        img_path (str): path to X.npy
        label_path (str): path to Y.npy

    Returns:
        x (np.ndarray): image array
        y (np.ndarray): label array
    """
    x=np.load(img_path)
    y=np.load(label_path)
    return x,y

