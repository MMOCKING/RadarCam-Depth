from rcnet_main import train
import datetime
import os
import data.data_utils as data_utils


if __name__ == '__main__':
    root = '/media/lh/lh2/ZJU-4DRadarCam'
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    image_path = os.path.join(root, 'data', 'image/')
    radar_path = os.path.join(root, 'data', 'radar/')
    gt_path = os.path.join(root, 'data', 'gt_interp/')
    train_list = os.path.join(root, 'data', 'train.txt')

    train_image_paths = data_utils.load_data_path(image_path, train_list, '.png')
    train_radar_paths = data_utils.load_data_path(radar_path, train_list, '.npy')
    train_ground_truth_paths = data_utils.load_data_path(gt_path, train_list, '.png')


    train(train_image_paths = train_image_paths,
          train_radar_paths = train_radar_paths,
          train_ground_truth_paths = train_ground_truth_paths,

          # Input settings
          batch_size = 8,
          patch_size = [300, 100],
          total_points_sampled = 40,
          sample_probability_of_lidar = 0.10,
          normalized_image_range = [0, 1],
          # Network settings
          encoder_type = ['rcnet', 'batch_norm'],
          n_filters_encoder_image = [32, 64, 128, 128, 128],
          n_neurons_encoder_depth = [32, 64, 128, 128, 128],
          decoder_type = ['multiscale', 'batch_norm'],
          n_filters_decoder = [256, 128, 64, 32, 16],
          # Weight settings
          weight_initializer = 'kaiming_uniform',
          activation_func = 'leaky_relu',

          # Training settings
          learning_rates = [2e-4],
          learning_schedule = [200],
          augmentation_probabilities = [1.00],
          augmentation_schedule = [-1],
          augmentation_random_brightness = [0.80, 1.20],
          augmentation_random_contrast = [0.80, 1.20],
          augmentation_random_saturation = [0.80, 1.20],
          augmentation_random_noise_type = ['none'],
          augmentation_random_noise_spread = -1,
          augmentation_random_flip_type = ['horizontal'],
          # Loss settings
          w_weight_decay = 0.0,
          w_positive_class = 2.5,
          max_distance_correspondence = 0.5,
          set_invalid_to_negative_class = False,
          # Checkpoint and summary settings
          checkpoint_dirpath = os.path.join(root, 'log', 'rcnet', current_time),
          n_step_per_summary = 100,
          n_step_per_checkpoint = 2000,
          restore_path = '',
          # Hardware settings
          n_thread = 8,

          # Output settings
          response_thr=0.5)
