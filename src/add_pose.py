
import math
import numpy as np
import gtsam
from gtsam.symbol_shorthand import L, X

PRIOR_NOISE = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.1, 0.1, 0.05]))  # (x, y, theta)
ODOMETRY_NOISE = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.2, 0.2, 0.1]))  # (dx, dy, dtheta)
MEASUREMENT_NOISE = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.05, 0.1]))  # (bearing, range)

def add_pose(graph, initial_estimate):
    dx = 2.0 * math.cos(math.pi / 4.0)
    dy = 2.0 * math.sin(math.pi / 4.0)
    dtheta = math.pi / 2.0
    
    odometry = gtsam.Pose2(dx, dy, dtheta)

    # TODO: Add the odometry factor between X(3) and X(4) to the graph
    graph.add(gtsam.BetweenFactorPose2(X(3), X(4), odometry, ODOMETRY_NOISE))

    # TODO: Based on the odometry, find the initial estimate for the pose of X(4)
    pose_4_ideal = gtsam.Pose2(4.0 + math.sqrt(2), math.sqrt(2), math.pi / 2.0)
    
    if initial_estimate.exists(X(4)):
        initial_estimate.update(X(4), pose_4_ideal)
    else:
        initial_estimate.insert(X(4), pose_4_ideal)
    
    return graph, initial_estimate