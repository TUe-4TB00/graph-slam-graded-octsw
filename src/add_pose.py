
import math
import numpy as np
import gtsam
from gtsam.symbol_shorthand import L, X

PRIOR_NOISE = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.1, 0.1, 0.05]))  # (x, y, theta)
ODOMETRY_NOISE = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.2, 0.2, 0.1]))  # (dx, dy, dtheta)
MEASUREMENT_NOISE = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.05, 0.1]))  # (bearing, range)

def add_pose(graph, initial_estimate):
    angle_45 = math.pi / 4
    dx = 2.0 * math.cos(angle_45)
    dy = 2.0 * math.sin(angle_45)
    dtheta = math.pi / 2 
    
    odometry = gtsam.Pose2(2.0, 0.0, math.pi / 2)

    graph.add(gtsam.BetweenFactorPose2(X(3), X(4), odometry, ODOMETRY_NOISE))

    last_pose = initial_estimate.atPose2(X(3))
    new_pose_estimate = last_pose.compose(odometry)
    
    if not initial_estimate.exists(X(4)):
        initial_estimate.insert(X(4), new_pose_estimate)
    else:
        initial_estimate.update(X(4), new_pose_estimate)
    
    return graph, initial_estimate