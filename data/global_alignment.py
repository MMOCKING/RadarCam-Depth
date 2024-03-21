import numpy as np
import os, time
import multiprocessing as mp
from modules import estimator
from data.SML_dataset import load_sparse_depth
import data.data_utils as data_utils


def process_scale(sparse_depth_root, mono_pred_root, sparse_depth_name,
                  max_pred = 255.0, min_pred = 0.0,
                  max_depth = 80.0, min_depth = 2.0,
                  save_path = '/media/lh/lh2/ZJU-4DRadarCam/data/scale_dpt_ls',
                  depth_type = 'pos'):

    sparse_depth_file = os.path.join(sparse_depth_root, sparse_depth_name)
    mono_pred_file = os.path.join(mono_pred_root, sparse_depth_name)

    sparse_depth = load_sparse_depth(sparse_depth_file).astype(np.float32)
    mono_pred = load_sparse_depth(mono_pred_file).astype(np.float32)

    sparse_depth_valid = (sparse_depth < max_depth) * (sparse_depth > min_depth)
    sparse_depth_valid = sparse_depth_valid.astype(bool)

    if depth_type == 'inv':
        sparse_depth[~sparse_depth_valid] = np.inf
        sparse_depth = 1.0 / sparse_depth
    else:
        sparse_depth[~sparse_depth_valid] = 0.0


    # global scale and shift alignment
    GlobalAlignment = estimator.LeastSquaresEstimator(
        estimate=mono_pred,
        target=sparse_depth,
        valid=sparse_depth_valid
    )
    timestart = time.time()
    # # ransac
    # GlobalAlignment.compute_scale_and_shift_ran(num_iterations=400, sample_size=5,
    #                                         inlier_threshold=6, inlier_ratio_threshold=0.9)
    # least square
    GlobalAlignment.compute_scale_and_shift()
    time_use = time.time() - timestart
    GlobalAlignment.apply_scale_and_shift()

    # # only global scale alignment
    # GlobalAlignment = estimator.Optimizer(
    #     estimate=mono_pred,
    #     target=sparse_depth,
    #     valid=sparse_depth_valid,
    #     depth_type=depth_type
    # )
    # timestart = time.time()
    # GlobalAlignment.optimize_scale()
    # time_use = time.time() - timestart
    # GlobalAlignment.apply_scale()

    GlobalAlignment.clamp_min_max(clamp_min=min_pred, clamp_max=max_pred)
    int_depth = GlobalAlignment.output.astype(np.float32)
    if depth_type == 'inv':
        int_depth = 1.0 / int_depth

    if not os.path.exists(save_path):
        os.makedirs(save_path)
    data_utils.save_depth(int_depth, os.path.join(save_path, sparse_depth_name))

    return time_use


if __name__ == '__main__':
    sparse_depth_root = '/media/lh/lh2/ZJU-4DRadarCam/data/radar_png'
    mono_pred_root ='/media/lh/lh2/ZJU-4DRadarCam/data/dpt'
    run_list = '/media/lh/lh2/ZJU-4DRadarCam/data/full.txt'

    sparse_depth_files = np.loadtxt(run_list, dtype=str)
    sparse_depth_files = [x + '.png' for x in sparse_depth_files]
    # sparse_depth_files = sorted(os.listdir(sparse_depth_root))

    num_processes = 1 # mp.cpu_count()-4

    pool = mp.Pool(num_processes)
    sum = 0.0
    cnt = 0

    for sparse_depth_file in sparse_depth_files:
        result = pool.apply_async(process_scale, args=(sparse_depth_root, mono_pred_root, sparse_depth_file))
        sum += result.get()
        cnt += 1

    pool.close()
    pool.join()

    # average_scale
    average_scale = sum / cnt
    print('sum: ', sum)
    print('average: ', average_scale)

    # dpt 0.8117274480821013
    # midas 0.0013031250057613435
