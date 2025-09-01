
from transforms3d.euler import quat2euler
import math
import numpy as np


quat = np.array([ #-110
    -0.3845455379989285,
    -0.05579479747159301,
    -0.015257206647794659,
    0.9212919664392636
])



quat90 = np.array([ #90
    0.7167434086564841,
    0.00639306108434205,
    -0.06579249697685942,
    0.6941969189351146
])



quat180 = np.array([ #180
    0.01466425603730019,
    -0.05069463124905456,
    -0.04154600088253137,
    0.9977419224268687
])


quat90 = np.array([# -90?
    0.7234278919720631, 
    0.047942458038643665,
    -0.01158345510615802,
    -0.6886359193375505
])


quat0 = np.array([ #zero
    0.9977920171252104,
    0.040539874272696216,
    -0.0523950665702471,
    0.004729286865607553
])

norm = np.linalg.norm(quat)
quat_normalized = quat / norm
roll, pitch, yaw = quat2euler(quat_normalized, axes='sxyz')
d_yaw = math.degrees(yaw)
print(yaw, d_yaw)
