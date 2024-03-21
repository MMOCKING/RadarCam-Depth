import os, datetime
import data.data_utils as data_utils
from sml_main import train


if __name__ == '__main__':
    train_root = '/media/lh/lh2/ZJU-4DRadarCam/data'
    result_root = '/media/lh/lh2/ZJU-4DRadarCam/result'
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    image_path = os.path.join(train_root, 'image/')
    radar_path = os.path.join(train_root, 'radar_png/')
    gt_path = os.path.join(train_root, 'gt_interp/')
    sparse_gt_path = os.path.join(train_root, 'gt/')
    mono_ga_path = os.path.join(result_root, 'global_aligned_mono/dpt_var/')
    rcnet_path = os.path.join(result_root, 'rcnet/depth_predicted/')
    train_list = os.path.join(train_root, 'train.txt')

    train_image_paths = data_utils.load_data_path(image_path, train_list, '.png')
    train_radar_paths = data_utils.load_data_path(radar_path, train_list, '.png')
    train_gt_paths = data_utils.load_data_path(gt_path, train_list, '.png')
    train_sparse_gt_paths = data_utils.load_data_path(sparse_gt_path, train_list, '.png')
    train_rcnet_paths = data_utils.load_data_path(rcnet_path, train_list, '.png')
    train_mono_ga_paths = data_utils.load_data_path(mono_ga_path, train_list, '.png')


    train(
        # data input
        train_image_paths = train_image_paths,
        train_radar_paths = train_radar_paths,
        train_gt_paths = train_gt_paths,
        train_sparse_gt_paths = train_sparse_gt_paths,
        train_rcnet_paths = train_rcnet_paths,
        train_mono_ga_paths = train_mono_ga_paths,

        # training
        learning_rates = [2e-4,1e-4],
        learning_schedule = [20,80],
        batch_size = 12,
        n_step_per_summary = 20,
        n_step_per_checkpoint = 2000,

        # Loss settings
        loss_func = 'smoothl1',
        w_smoothness = 0.0,
        w_lidar_loss = 3.0,
        w_weight_decay = 0.0,
        loss_smoothness_kernel_size = -1,
        ground_truth_outlier_removal_kernel_size = -1,
        ground_truth_outlier_removal_threshold = 1.5,
        ground_truth_dilation_kernel_size = -1,

        # model
        restore_path ='', #../weights/sml_model.dpredictor.dpt_hybrid.nsamples.150.ckpt
        min_pred_depth = 0.1,
        max_pred_depth = 255.0,
        min_radar_valid_depth = 0.0,
        max_radar_valid_depth = 80.0,
        checkpoint_dirpath = os.path.join('/media/lh/lh2/ZJU-4DRadarCam/log/SML', current_time),
        n_threads=6,
    )