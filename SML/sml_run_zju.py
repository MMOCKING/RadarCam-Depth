import os
import torch.utils.data
import numpy as np
import data.data_utils as data_utils
from modules.midas.midas_net_custom import MidasNet_small_videpth
from SML.sml_main import validate


if __name__ == "__main__":
    root = '/media/lh/lh2/ZJU-4DRadarCam/data'
    result_root = '/media/lh/lh2/ZJU-4DRadarCam/result'
    checkpoint_dirpath = './weights/sml_dpt_var'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ScaleMapLearner = MidasNet_small_videpth(
        device = device,
        min_pred = 0.1,
        max_pred = 100,
    )

    ScaleMapLearner.eval()
    ScaleMapLearner.to(device)
    ScaleMapLearner = torch.nn.DataParallel(ScaleMapLearner)

    best_results = {
        'step': -1,
        'mae': np.infty,
        'rmse': np.infty,
        'imae': np.infty,
        'irmse': np.infty,
        'abs_rel': np.infty,
        'sq_rel': np.infty,
        'delta1': 0,
    }

    for checkpoint_filename in sorted(os.listdir(checkpoint_dirpath), reverse=True):
        if not checkpoint_filename.endswith('.pth'):
            continue

        step = int(checkpoint_filename.split('/')[-1].split('.')[0].split('-')[-1])
        # if step != 48000:
        #     continue

        checkpoint_filepath = os.path.join(checkpoint_dirpath, checkpoint_filename)
        log_path = os.path.join(checkpoint_dirpath, 'results.txt')
        ScaleMapLearner.module.load(checkpoint_filepath)
        print("Model weights loaded from {}".format(checkpoint_filename))

        # load data
        run_list = os.path.join(root, 'test.txt')
        image_paths = data_utils.load_data_path(os.path.join(root, 'image/'), run_list, '.png')
        radar_paths = data_utils.load_data_path(os.path.join(root, 'radar_png/'), run_list, '.png')
        sparse_gt_paths = data_utils.load_data_path(os.path.join(root, 'gt/'), run_list, '.png')
        rcnet_paths = data_utils.load_data_path(os.path.join(result_root, 'rcnet/depth_predicted/'), run_list, '.png')
        mono_ga_paths = data_utils.load_data_path(os.path.join(result_root, 'global_aligned_mono/dpt_var/'), run_list, '.png')

        with torch.no_grad():
            best_results = validate(
                image_paths = image_paths,
                radar_paths = radar_paths,
                gt_paths = sparse_gt_paths,
                sparse_gt_paths = sparse_gt_paths,
                rcnet_paths = rcnet_paths,
                mono_ga_paths = mono_ga_paths,

                best_results=best_results,
                ScaleMapLearner=ScaleMapLearner,
                step = step,

                min_radar_valid_depth = 0.0,
                max_radar_valid_depth = 80.0,
                min_eval_depth = 0.0,
                max_eval_depth = 80.0,

                output_path = result_root,
                log_path = log_path,
                random_sample=False,
                random_sample_size=500,
                save_output=False,
            )

            

