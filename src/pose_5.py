import numpy as np
from helperfunctions import add_pose_from_global, add_landmark_measurement_from_global
import gtsam
from gtsam.symbol_shorthand import L, X

PRIOR_NOISE = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.1, 0.1, 0.05]))  # (x, y, theta)
ODOMETRY_NOISE = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.2, 0.2, 0.1]))  # (dx, dy, dtheta)
MEASUREMENT_NOISE = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.05, 0.1]))  # (bearing, range)

def add_pose(graph, initial_estimate, pose_5):
    # Adding the initial estimate for the 5th pose using our helper function `add_pose_from_global` which also adds the odometry factor between X(4) and X(5).
    pose_4 = initial_estimate.atPose2(X(4))
    graph, initial_estimate = add_pose_from_global(
        graph=graph,
        initial_estimate=initial_estimate,
        prev_key=X(4),
        new_key=X(5),
        prev_pose=pose_4,
        new_pose_global=pose_5,
        odom_noise=ODOMETRY_NOISE
    )
    return graph, initial_estimate

def add_landmark_measurement(graph, result, pose_5, landmark):
    # Adding the measurement from X(5) to the chosen landmark using our helper function `add_landmark_measurement_from_global` which calculates the correct bearing and range from the global poses.``
    landmark_point = result.atPoint2(L(landmark))
    graph = add_landmark_measurement_from_global(
        graph=graph,
        pose_key=X(5),
        pose=pose_5,
        landmark_key=L(landmark),
        landmark_point=landmark_point,
        measurement_noise=MEASUREMENT_NOISE
    )
    return graph

def optimize(graph, initial_estimate):
    params = gtsam.LevenbergMarquardtParams()
    optimizer = gtsam.LevenbergMarquardtOptimizer(graph, initial_estimate, params)
    return optimizer.optimize()

def minimize_marginals(graph, initial_estimate, pose_options):
    best_pose = None
    best_landmark = None
    min_sum = float('inf')

    for label, pose_5 in pose_options.items():
        for lm_idx in [1, 2]:
            g_temp = gtsam.NonlinearFactorGraph(graph)
            v_temp = gtsam.Values(initial_estimate)
            
            g_temp, v_temp = add_pose(g_temp, v_temp, pose_5)
            g_temp = add_landmark_measurement(g_temp, v_temp, pose_5, lm_idx)
            
            try:
                res = optimize(g_temp, v_temp)
                m = gtsam.Marginals(g_temp, res)
                current_sum = m.marginalCovariance(L(1)).trace() + m.marginalCovariance(L(2)).trace()
                
                if current_sum < min_sum:
                    min_sum = current_sum
                    best_pose = label
                    best_landmark = lm_idx
            except:
                continue

    return best_pose, best_landmark, min_sum

def minimize_errors(graph, initial_estimate, pose_options):
    best_pose = None
    best_landmark = None
    sum_of_errors = float('inf')

    for label, pose_5 in pose_options.items():
        for lm_idx in [1, 2]:
            g_temp = gtsam.NonlinearFactorGraph(graph)
            v_temp = gtsam.Values(initial_estimate)
            g_temp, v_temp = add_pose(g_temp, v_temp, pose_5)
            g_temp = add_landmark_measurement(g_temp, v_temp, pose_5, lm_idx)
            
            res = optimize(g_temp, v_temp)
            m = gtsam.Marginals(g_temp, res)
            
            current_err = (m.marginalCovariance(X(1)).trace() + 
                           m.marginalCovariance(X(2)).trace() + 
                           m.marginalCovariance(X(3)).trace())
            
            if current_err < sum_of_errors:
                sum_of_errors = current_err
                best_pose = label
                best_landmark = lm_idx
                
    return best_pose, best_landmark, sum_of_errors 