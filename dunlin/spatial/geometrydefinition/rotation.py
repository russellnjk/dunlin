import numpy as np
from numba import njit
###############################################################################
#Rotation
###############################################################################
def rotate2D(points, radians, x=0, y=0):
    '''
    Rotates a set of points around a particular point.

    Parameters
    ----------
    points : np.ndarray
        An array of points in the form `[pt0, pt1 ...]` where each point is 
        in the form `[x, y]`.
    radians : float
        The angle of rotation in radians.
    x : float
        The x coordinate of the point of rotation. The default is 0.
    y : float, optional
        The y coordinate of the point of rotation. The default is 0.

    Returns
    -------
    new_points : np.ndarray
        An array in the same form as the `points` argument.

    '''
    #Reframe the coordinates
    points = np.array(points, dtype=np.float64)
    offset = np.array([x, y], dtype=np.float64)
    
    new_points = _rotate2D(points, radians, offset)
    
    return new_points

@njit
def _rotate2D(points, radians, offset):
    '''
    Rotates a set of points around the offset.

    Parameters
    ----------
    points : np.ndarray
        An array of points in the form `[pt0, pt1 ...]` where each point is 
        in the form `[x, y]`.
    radians : float
        The angle of rotation in radians.
    offset : [float, float]
        The displacement of point of rotation from the center.

    Returns
    -------
    new_points : np.ndarray
        An array in the same form as the `points` argument.

    '''
    #Translate points
    points = points - offset
    
    #Convert to vectors
    X, Y = points.T
    
    #Use complex number
    #IMPT: The angle for complex numbers goes anti-clockwise
    complex_format = X + 1j*Y
    R     = np.abs(complex_format)
    Theta = np.angle(complex_format, deg=False) + radians
    
    #Convert back to cartesian
    X = R*np.cos(Theta)
    Y = R*np.sin(Theta)
    
    #Convert to points
    new_points = np.stack((X, Y), axis=1)
    
    #Untranslate points
    new_points = new_points + offset
    
    return new_points

###############################################################################
#Quaternion and 3D Operations
###############################################################################
@njit
def convert_to_quaternion(radians, x, y, z):
    '''Assumes `x, y, z` make a unit vector.
    '''
    q0 = np.cos(radians/2)
    q1 = np.sin(radians/2)*x
    q2 = np.sin(radians/2)*y
    q3 = np.sin(radians/2)*z
    
    return np.array([q0, q1, q2, q3])

@njit
def convert_from_quaternion(q0, q1, q2, q3):
    '''Assumes `x, y, z` make a unit vector.
    '''
    radians = np.arccos(q0)*2
    
    x = q1/np.sin(radians/2)
    y = q2/np.sin(radians/2)
    z = q3/np.sin(radians/2)
    
    return np.array([radians, x, y, z])

@njit
def rotate3D(points, radians, x, y, z):
    '''
    Rotates a set of points around a unit vector by a particular angle.

    Parameters
    ----------
    points : np.ndarray
        An array of points in the form `[pt0, pt1 ...]` where each point is 
        in the form `[x, y, z]`.
    radians : float
        The angle of rotation in radians.
    x : float
        The x component of the unit vector.
    y : float
        The y component of the unit vector.
    z : float
        The z component of the unit vector.
    
    Notes
    -----
    Refer to https://stackoverflow.com/questions/6721544/circular-rotation-around-an-arbitrary-axis
    
    Returns
    -------
    new_points : np.ndarray
        An array in the same form as the `points` argument.

    '''
    #Normalize
    vct = np.array([x, y, z], dtype=np.float64)
    m   = np.sum(vct**2)**.5
    
    if m == 0:
        return points
    else:
        x, y, z = vct/m

    #Convert to quaternion
    q0 = np.cos(radians/2)
    q1 = np.sin(radians/2)*x
    q2 = np.sin(radians/2)*y
    q3 = np.sin(radians/2)*z
    
    q0q0 = q0*q0
    q0q1 = q0*q1
    q0q2 = q0*q2
    q0q3 = q0*q3
    
    q1q1 = q1*q1
    q1q2 = q1*q2
    q1q3 = q1*q3
    
    q2q2 = q2*q2
    q2q3 = q2*q3
    
    q3q3 = q3*q3
    
    M = np.array([[q0q0 + q1q1 - q2q2 - q3q3, 2*(q1q2 - q0q3), 2*(q1q3 + q0q2)],
                  [2*(q1q2 + q0q3), q0q0 - q1q1 + q2q2 - q3q3, 2*(q2q3 - q0q1)],
                  [2*(q1q3 - q0q2), 2*(q2q3 + q0q1), q0q0 - q1q1 - q2q2 + q3q3]
                  ])
    
    new_points = M @ points.T
    new_points = new_points.T
    return new_points

def add_rotations(rot0, rot1):
    '''rot0 and rot1 must be unit vectors
    '''
    #Ensure array
    rot0 = np.array(rot0, dtype=np.float64)
    rot1 = np.array(rot1, dtype=np.float64)
    
    #Normalize
    vct0 = np.array(rot0[1:], dtype=np.float64)
    m0   = np.sum(vct0**2)**.5
    
    vct1 = np.array(rot1[1:], dtype=np.float64)
    m1   = np.sum(vct1**2)**.5
    
    theta0 = rot0[0] if any(vct0) else 0
    theta1 = rot1[0] if any(vct1) else 0

    #Special cases 
    if m0 == 0 and m1 == 0:
        return np.array([0, 0, 0, 0], dtype=np.float64)
    elif m0 == 0:
        new_vct1 = vct1/m1
        return np.array([theta1, *new_vct1], dtype=np.float64)
    elif m1 == 0:
        new_vct0 = vct0/m0
        return np.ndarray([theta0, *new_vct0], dtype=np.float64)
    else:
        new_vct0 = vct0/m0
        new_vct1 = vct1/m1
        rot0 = np.array([theta0, *new_vct0], dtype=np.float64)
        rot1 = np.array([theta1, *new_vct1], dtype=np.float64)
        
    #Convert to quaternions
    Q0 = convert_to_quaternion(*rot0)
    Q1 = convert_to_quaternion(*rot1)
    
    #Multiply in reverse order
    Q2 = multiply_quaternion(Q1, Q0)
    
    #Convert back to orientation
    rot2 = convert_from_quaternion(*Q2)
    vct  = rot2[1:]
    m    = np.sum(vct**2)**.5
    rot2[1:] = vct/m
    
    return rot2

@njit
def multiply_quaternion(Q0, Q1):
    '''
    Multiplies two quaternions.

    Parameters
    ----------
    Q0 : np.ndarray
        A 4 element array containing the first quaternion (q01,q11,q21,q31) .
    Q1 : np.ndarray
        A 4 element array containing the first quaternion (q01,q11,q21,q31) .

    Returns
    -------
    final_quaternion : np.ndarray
        A 4 element array containing the final quaternion (q03,q13,q23,q33).

    '''

    # Extract the values from Q0
    w0 = Q0[0]
    x0 = Q0[1]
    y0 = Q0[2]
    z0 = Q0[3]

    # Extract the values from Q1
    w1 = Q1[0]
    x1 = Q1[1]
    y1 = Q1[2]
    z1 = Q1[3]

    # Computer the product of the two quaternions, term by term
    Q0Q1_w = w0 * w1 - x0 * x1 - y0 * y1 - z0 * z1
    Q0Q1_x = w0 * x1 + x0 * w1 + y0 * z1 - z0 * y1
    Q0Q1_y = w0 * y1 - x0 * z1 + y0 * w1 + z0 * x1
    Q0Q1_z = w0 * z1 + x0 * y1 - y0 * x1 + z0 * w1

    # Create a 4 element array containing the final quaternion
    final_quaternion = np.array([Q0Q1_w, Q0Q1_x, Q0Q1_y, Q0Q1_z])

    # Return a 4 element array containing the final quaternion (q02,q12,q22,q32) 
    return final_quaternion
    
