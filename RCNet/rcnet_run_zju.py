import os
from rcnet_main import run
import data.data_utils as data_utils


if __name__ == '__main__':
    root = '/media/lh/lh2/ZJU-4DRadarCam'
    save_root = '/media/lh/lh2/ZJU-4DRadarCam/result/rcnet'
    image_path = os.path.join(root, 'data/image/')
    radar_path = os.path.join(root, 'data/radar/')
    gt_path = os.path.join(root, 'data/gt_interp/')
    data_list = os.path.join(root, 'data/full.txt')

    image_paths = data_utils.load_data_path(image_path, data_list, '.png')
    radar_paths = data_utils.load_data_path(radar_path, data_list, '.npy')
    gt_paths = data_utils.load_data_path(gt_path, data_list, '.png')

    run(save_root = save_root,
        image_paths = image_paths,
        radar_paths = radar_paths,
        gt_paths = gt_paths,

        restore_path = './weights/rcnet/model-6000.pth',
        patch_size = [300, 100],
        normalized_image_range = [0, 1],

        encoder_type = ['rcnet', 'batch_norm'],
        n_filters_encoder_image = [32, 64, 128, 128, 128],
        n_neurons_encoder_depth = [32, 64, 128, 128, 128],
        decoder_type = ['multiscale', 'batch_norm'],
        n_filters_decoder = [256, 128, 64, 32, 16],
        weight_initializer = 'kaiming_uniform',
        activation_func = 'leaky_relu',
        response_thr = 0.5)
    




