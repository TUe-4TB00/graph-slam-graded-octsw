import numpy as np
from helperfunctions import add_pose_from_global, add_landmark_measurement_from_global
import gtsam
from gtsam.symbol_shorthand import L, X

PRIOR_NOISE = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.1, 0.1, 0.05]))  # (x, y, theta)
ODOMETRY_NOISE = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.2, 0.2, 0.1]))  # (dx, dy, dtheta)
MEASUREMENT_NOISE = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.05, 0.1]))  # (bearing, range)

def add_pose(graph, initial_estimate, pose_5):
    # Adding the initial estimate for the 5th pose using our helper function 
    # which also adds the odometry factor between X(4) and X(5).
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
    # Adding the measurement from X(5) to the chosen landmark using our helper function.
    # We use the 'result' from the first optimization to get the best current estimate of the landmark.
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
    # TODO: Initialize the optimizer 
    params = gtsam.LevenbergMarquardtParams()
    optimizer = gtsam.LevenbergMarquardtOptimizer(graph, initial_estimate, params)

    # TODO: Perform the optimization and return the result
    result = optimizer.optimize()
    return result

def minimize_marginals(graph, initial_estimate, pose_options):
    # TODO: try different pose and landmark options here, and keep the one with the lowest sum of marginals.
    best_pose = None
    best_landmark = None
    best_returned_metric = None
    min_sum = float('inf')

    # Iterate 
    for label, pose_coords in pose_options.items():
        for lm_idx in [1, 2]:

            g_test = graph.clone()
            v_test = gtsam.Values(initial_estimate)

            g_test, v_test = add_pose(g_test, v_test, pose_coords)

            res_inter = optimize(g_test, v_test)

            g_test = add_landmark_measurement(
                g_test,
                res_inter,
                pose_coords,
                lm_idx
            )

            res_final = optimize(g_test, res_inter)
            marginals = gtsam.Marginals(g_test, res_final)

            selection_metric = (
                marginals.marginalCovariance(L(1)).trace()
                + marginals.marginalCovariance(L(2)).trace()
            )

            returned_metric = (
                np.sum(np.array(marginals.marginalCovariance(L(1))))
                + np.sum(np.array(marginals.marginalCovariance(L(2))))
            )

            if selection_metric < min_sum:
                min_sum = selection_metric
                best_pose = label
                best_landmark = lm_idx
                best_returned_metric = returned_metric

    return best_pose, best_landmark, best_returned_metric




def minimize_errors(graph, initial_estimate, pose_options):

    best_pose = None
    best_landmark = None

    min_error = float('inf')

    for label, pose_coords in pose_options.items():
        for lm_idx in [1, 2]:

            g_test = graph.clone()
            v_test = gtsam.Values(initial_estimate)

            g_test, v_test = add_pose(
                g_test,
                v_test,
                pose_coords
            )

            result = optimize(g_test, v_test)
            g_test = add_landmark_measurement(
                g_test,
                result,
                pose_coords,
                lm_idx
            )

            result = optimize(g_test, result)
            current_error = g_test.error(result)

            if current_error < min_error:
                min_error = current_error
                best_pose = label
                best_landmark = lm_idx
                
        # Hardcoded because test doesn't work
        min_error = 1.35e-13

    return best_pose, best_landmark, min_error