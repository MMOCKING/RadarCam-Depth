import numpy as np
import time
from scipy.optimize import minimize_scalar

def compute_scale_and_shift_ls(prediction, target, mask):
    # tuple specifying with axes to sum
    sum_axes = (0, 1)

    # system matrix: A = [[a_00, a_01], [a_10, a_11]]
    a_00 = np.sum(mask * prediction * prediction, sum_axes)
    a_01 = np.sum(mask * prediction, sum_axes)
    a_11 = np.sum(mask, sum_axes)

    # right hand side: b = [b_0, b_1]
    b_0 = np.sum(mask * prediction * target, sum_axes)
    b_1 = np.sum(mask * target, sum_axes)

    # solution: x = A^-1 . b = [[a_11, -a_01], [-a_10, a_00]] / (a_00 * a_11 - a_01 * a_10) . b
    x_0 = np.zeros_like(b_0)
    x_1 = np.zeros_like(b_1)

    det = a_00 * a_11 - a_01 * a_01
    # A needs to be a positive definite matrix.
    valid = det > 0

    x_0[valid] = (a_11[valid] * b_0[valid] - a_01[valid] * b_1[valid]) / det[valid]
    x_1[valid] = (-a_01[valid] * b_0[valid] + a_00[valid] * b_1[valid]) / det[valid]

    return x_0, x_1



def compute_scale_and_shift_ransac(prediction, target, mask,
                                   num_iterations, sample_size,
                                   inlier_threshold, inlier_ratio_threshold):
    # start = time.time()
    best_scale = 0.0
    best_shift = 0.0
    best_inlier_count = 0

    valid_indices = np.where(mask)
    valid_count = len(valid_indices[0])
    # print('valid_count: ', valid_count)

    for _ in range(num_iterations):
        if valid_count < sample_size:
            break

        # Randomly sample from valid indices
        indices = np.random.choice(valid_count, size=sample_size, replace=False)
        mask_sample = np.zeros_like(mask)
        mask_sample[valid_indices[0][indices], valid_indices[1][indices]] = 1

        # Calculate x_0 and x_1 for the sampled data
        sum_axes = (0, 1)
        a_00 = np.sum(mask_sample * prediction * prediction, sum_axes)
        a_01 = np.sum(mask_sample * prediction, sum_axes)
        a_11 = np.sum(mask_sample, sum_axes)
        b_0 = np.sum(mask_sample * prediction * target, sum_axes)
        b_1 = np.sum(mask_sample * target, sum_axes)
        det = a_00 * a_11 - a_01 * a_01
        valid = det > 0
        x_0 = np.zeros_like(b_0)
        x_1 = np.zeros_like(b_1)
        x_0[valid] = (a_11[valid] * b_0[valid] - a_01[valid] * b_1[valid]) / det[valid]
        x_1[valid] = (-a_01[valid] * b_0[valid] + a_00[valid] * b_1[valid]) / det[valid]

        # Calculate residuals and count inliers
        residuals = np.abs(mask * prediction * x_0 + x_1 - mask * target)
        residuals = residuals[mask]

        inlier_count = np.sum(residuals < inlier_threshold)

        # Update best model if current model has more inliers
        if inlier_count > best_inlier_count:
            best_scale = x_0
            best_shift = x_1
            best_inlier_count = inlier_count
            inlier_ratio = inlier_count / valid_count
            if inlier_ratio > inlier_ratio_threshold:
                break

    print('best_inlier_count: ', best_inlier_count)
    print('inlier_ratio: ', best_inlier_count / valid_count)
    # print('best_scale: ', best_scale)
    # print('best_shift: ', best_shift)
    # print('time', time.time() - start)
    return best_scale, best_shift



class LeastSquaresEstimator(object):
    def __init__(self, estimate, target, valid):
        self.estimate = estimate
        self.target = target
        self.valid = valid

        # to be computed
        self.scale = 1.0
        self.shift = 0.0
        self.output = None

    def compute_scale_and_shift_ran(self,
                                num_iterations=60, sample_size=5,
                                inlier_threshold=0.02, inlier_ratio_threshold=0.8):
        self.scale, self.shift = compute_scale_and_shift_ransac(self.estimate, self.target, self.valid,
                                                                num_iterations, sample_size,
                                                                inlier_threshold, inlier_ratio_threshold)

    def compute_scale_and_shift(self):
        self.scale, self.shift = compute_scale_and_shift_ls(self.estimate, self.target, self.valid)


    def apply_scale_and_shift(self):
        self.output = self.estimate * self.scale + self.shift

    def clamp_min_max(self, clamp_min=None, clamp_max=None):
        if clamp_min is not None:
            if clamp_min > 0:
                clamp_min_inv = 1.0/clamp_min
                self.output[self.output > clamp_min_inv] = clamp_min_inv
                assert np.max(self.output) <= clamp_min_inv
            else: # divide by zero, so skip
                pass
        if clamp_max is not None:
            clamp_max_inv = 1.0/clamp_max
            self.output[self.output < clamp_max_inv] = clamp_max_inv



def objective_function(x_0, prediction, target, mask):
    # Calculate x_0 * prediction
    x_0_prediction = x_0 * prediction
    # Calculate the error between x_0 * prediction and target, using the mask
    error = np.sum(mask * abs(x_0_prediction - target))
    return error



class Optimizer(object):
    def __init__(self, estimate, target, valid, depth_type):
        self.estimate = estimate
        self.target = target
        self.valid = valid
        self.depth_type = depth_type
        # to be computed
        self.scale = 1.0
        self.output = None

    def optimize_scale(self):
        if self.depth_type == 'inv':
            bounds = (0.0003, 0.01)
        else:
            bounds = (0.5, 1.6) # pos

        # Minimize the objective function using scipy.optimize.minimize_scalar
        result = minimize_scalar(
            objective_function, args=(self.estimate, self.target, self.valid),
            bounds=bounds
        )

        # Extract the optimized x_0 value from the result
        optimized_x_0 = result.x
        self.scale = optimized_x_0

    def apply_scale(self):
        self.output = self.estimate * self.scale

    def clamp_min_max(self, clamp_min=None, clamp_max=None):
        if clamp_min is not None:
            if clamp_min > 0:
                clamp_min_inv = 1.0/clamp_min
                self.output[self.output > clamp_min_inv] = clamp_min_inv
                assert np.max(self.output) <= clamp_min_inv
            else: # divide by zero, so skip
                pass
        if clamp_max is not None:
            clamp_max_inv = 1.0/clamp_max
            self.output[self.output < clamp_max_inv] = clamp_max_inv

    def clamp_min_max_pos(self, clamp_min=None, clamp_max=None):
        if clamp_min is not None:
            if clamp_min >= 0:
                self.output[self.output < clamp_min] = clamp_min
            else:
                pass
        if clamp_max is not None:
            self.output[self.output > clamp_max] = clamp_max