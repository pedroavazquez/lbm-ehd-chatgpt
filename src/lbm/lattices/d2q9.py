import numpy as np

# Discrete velocities c_i = (c_ix, c_iy)
C = np.array(
    [
        [0, 0],  # 0: rest
        [1, 0],  # 1: east
        [0, 1],  # 2: north
        [-1, 0],  # 3: west
        [0, -1],  # 4: south
        [1, 1],  # 5: north-east
        [-1, 1],  # 6: north-west
        [-1, -1],  # 7: south-west
        [1, -1],  # 8: south-east
    ],
    dtype=np.int8,
)


# Quadrature weights
W = np.array(
    [
        4.0 / 9.0,
        1.0 / 9.0,
        1.0 / 9.0,
        1.0 / 9.0,
        1.0 / 9.0,
        1.0 / 36.0,
        1.0 / 36.0,
        1.0 / 36.0,
        1.0 / 36.0,
    ],
    dtype=np.float64,
)


# Opposite population index:
# c_opposite[i] = -c_i
OPPOSITE = np.array(
    [
        0,  # 0 <-> 0
        3,  # 1 <-> 3
        4,  # 2 <-> 4
        1,
        2,
        7,  # 5 <-> 7
        8,  # 6 <-> 8
        5,
        6,
    ],
    dtype=np.int8,
)


Q = 9
D = 2

CS2 = 1.0 / 3.0
CS4 = CS2**2
