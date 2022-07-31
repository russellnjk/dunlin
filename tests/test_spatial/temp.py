import numpy as np

def convert_to_quaternion(radians, x, y, z):
    q0 = np.cos(radians/2)
    q1 = np.sin(radians/2)*x
    q2 = np.sin(radians/2)*y
    q3 = np.sin(radians/2)*z
    
    return np.array([q0, q1, q2, q3])

def convert_from_quaternion(q0, q1, q2, q3):
    radians = np.arccos(q0)*2
    
    x = q1/np.sin(radians/2)
    y = q2/np.sin(radians/2)
    z = q3/np.sin(radians/2)
    
    return np.array([radians, x, y, z])

def rotate_quaternion(points, radians, x, y, z):
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

def rotate_quaternion2(points, radians, x, y, z):
    q0 = np.cos(radians)/2
    q1 = np.sin(radians/2)*x
    q2 = np.sin(radians/2)*y
    q3 = np.sin(radians/2)*z
    
    r00 = 1 - 2*(q2**2 + q3**2)
    r01 = 2*(q1*q2 - q3*q0)
    r02 = 2*(q1*q3 + q2*q0)
    
    r10 = 2*(q1*q2 + q3*q0)
    r11 = 1 - 2*(q1**2 + q3**2)
    r12 = 2*(q2*q3 - q1*q0)
    
    r20 = 2*(q1*q3 - q2*q0)
    r21 = 2*(q2*q3 + q1*q0)
    r22 = 1 - 2*(q1**2 + q2**2)
    
    # First row of the rotation matrix
    r00 = 2 * (q0 * q0 + q1 * q1) - 1
    r01 = 2 * (q1 * q2 - q0 * q3)
    r02 = 2 * (q1 * q3 + q0 * q2)
     
    # Second row of the rotation matrix
    r10 = 2 * (q1 * q2 + q0 * q3)
    r11 = 2 * (q0 * q0 + q2 * q2) - 1
    r12 = 2 * (q2 * q3 - q0 * q1)
     
    # Third row of the rotation matrix
    r20 = 2 * (q1 * q3 - q0 * q2)
    r21 = 2 * (q2 * q3 + q0 * q1)
    r22 = 2 * (q0 * q0 + q3 * q3) - 1
    
    mat = np.array([[r00, r01, r02],
                    [r10, r11, r12],
                    [r20, r21, r22]
                    ])
    
    new_points = mat @ points.T
    new_points = new_points.T
    return new_points

def quaternion_multiply(Q0,Q1):
    """
    Multiplies two quaternions.

    Input
    :param Q0: A 4 element array containing the first quaternion (q01,q11,q21,q31) 
    :param Q1: A 4 element array containing the second quaternion (q02,q12,q22,q32) 

    Output
    :return: A 4 element array containing the final quaternion (q03,q13,q23,q33) 

    """
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
    


points = [[0, 0, 0], [1, 0, 0], [0, 1, 0], [-1, 0, 0], [0, -1, 0]]
points = np.array(points, dtype=np.float64)
rotations = [[np.pi/2, 0, 0, 1]]
new_points = points
for r in rotations:
    new_points = rotate_quaternion(new_points, *r)

assert all(np.isclose(points[0], new_points[0], atol=1e-12))
assert all(np.isclose(points[1], new_points[4], atol=1e-12))
assert all(np.isclose(points[2], new_points[1], atol=1e-12))
assert all(np.isclose(points[3], new_points[2], atol=1e-12))
assert all(np.isclose(points[4], new_points[3], atol=1e-12))

points = [[0, 0, 0], [1, 0, 0], [0, 1, 0], [-1, 0, 0], [0, -1, 0]]
points = np.array(points, dtype=np.float64)
rotations = [[np.pi/2, 0, 0, 1], [np.pi/2, 1, 0, 0]]
new_points = points
for r in rotations:
    new_points = rotate_quaternion(new_points, *r)

print(new_points)

R0, R1 = rotations
Q0 = convert_to_quaternion(*R0)
Q1 = convert_to_quaternion(*R1)
Q2 = quaternion_multiply(Q1, Q0)
r2 = convert_from_quaternion(*Q2)

rotations = [r2]
for r in rotations:
    new_points = rotate_quaternion(points, *r)

print(new_points)


# a = b = np.pi/4
# b_a = 0
# B = np.array([1, 0, 0])/np.sin(b)
# A = np.array([0, 0, 1])/np.sin(a)
# B_A = np.array([0, -1, 0])
# gamma = 2* np.arccos(np.cos(b)*np.cos(a) - np.sin(b)*np.sin(a)*b_a)
# D     = np.sin(b)*np.cos(a)*B + np.sin(a)*np.cos(b)*A + np.sin(b)*np.sin(a)*B_A

# D_ = D/(2*np.sin(gamma/2))
