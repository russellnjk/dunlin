import numpy as np

import dunlin.utils as ut

###############################################################################
#Indexing Functions
###############################################################################    
def index(array, idx):
    '''
    Gets the value of the array at a particular index.

    Parameters
    ----------
    array : ndarray
        The array of values.
    idx : int
        The position.

    Returns
    -------
    float
        The value at that index. If the array is a scalar, the scalar value is 
        returned and the indexis ignored.

    '''
    if ut.isnum(array):
        return float(array)
    else:
        return array[idx]

###############################################################################
#Max/Min Functions
############################################################################### 
def where(cond, array2):
    '''
    Gets the first element of array2 that satisfies a particular condition.

    Parameters
    ----------
    cond : ndarray of boolean
        An array of True and False. e.g. time > 250. In this case, the first True 
        corresponds to the the first time point after 250.
    array2 : ndarray
        The array to extract the value from. 

    Returns
    -------
    float
        The value of array2 where the index corresponds to the index of the first 
        True in the condition. For example, if the condition is time > 250 and 
        array2 is state x0, then the return value is the value of x0 just after 
        time=250.

    '''
    return _argmax(cond, array2)

def max(array1, array2=None):
    if array2 is None:
        if ut.isnum(array1):
            return float(array1)
        else:
            return np.nanmax(array1)
    else:
        return _argmax(array1, array2)

def min(array1, array2=None):
    if array2 is None:
        if ut.isnum(array1):
            return float(array1)
        else:
            return np.nanmin(array1)
    else:
        return _argmin(array1, array2)
    
def _argmax(array1, array2):
    if ut.isnum(array1):
        idx = 0
    else:
        idx = np.nanargmax(array1)
    
    if ut.isnum(array2):
        return array2
    else:
        return array2[idx]

def _argmin(array1, array2):
    if ut.isnum(array1):
        idx = 0
    else:
        idx = np.nanargmin(array1)
    
    if ut.isnum(array2):
        return array2
    else:
        return array2[idx]

###############################################################################
#For External Use
############################################################################### 
exf = {'__where': where, '__max': max, '__min': min, '__index': index}

# time = np.linspace(0, 1, 11)
# x0   = np.linspace(0, 10, 11)
# r = where(time > 0.5, x0)

# print(r)