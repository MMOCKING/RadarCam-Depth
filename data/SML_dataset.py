import torch.utils.data
import numpy as np
import modules.midas.utils as utils
from PIL import Image

def load_input_image(input_image_fp):
    return utils.read_image(input_image_fp)


def load_sparse_depth(input_sparse_depth_fp):
    input_sparse_depth = np.array(Image.open(input_sparse_depth_fp), dtype=np.float32) / 256.0
    input_sparse_depth[input_sparse_depth <= 0] = 0.0
    return input_sparse_depth


class SML_dataset(torch.utils.data.Dataset):
    def __init__(self,
                 image_paths,
                 radar_paths,
                 gt_paths,
                 sparse_gt_paths,
                 rcnet_paths,
                 mono_pred_paths = None,
                 mono_ga_paths = None,
                 ):

        self.n_sample = len(image_paths)

        for paths in [image_paths, radar_paths, gt_paths, sparse_gt_paths,
                      rcnet_paths, mono_pred_paths, mono_ga_paths]:
            if paths is not None:
                assert len(paths) == self.n_sample

        self.image_paths = image_paths
        self.radar_paths = radar_paths
        self.gt_paths = gt_paths
        self.sparse_gt_paths = sparse_gt_paths
        self.rcnet_paths = rcnet_paths
        self.mono_pred_paths = mono_pred_paths
        self.mono_ga_paths = mono_ga_paths


    def __getitem__(self, index):
        image = load_input_image(self.image_paths[index])
        radar = load_sparse_depth(self.radar_paths[index])
        gt = load_sparse_depth(self.gt_paths[index])
        sparse_gt = load_sparse_depth(self.sparse_gt_paths[index])
        rcnet = load_sparse_depth(self.rcnet_paths[index])

        image, radar, gt, sparse_gt, rcnet = [
            T.astype(np.float32)
            for T in [image, radar, gt, sparse_gt, rcnet]
        ]

        # Crop the image for ZJU dataset
        if image.shape[0] == 720:
            image = image[720 // 3: 720 // 4 * 3, :, :]
            radar = radar[720 // 3: 720 // 4 * 3, :]
            gt = gt[720 // 3: 720 // 4 * 3, :]
            sparse_gt = sparse_gt[720 // 3: 720 // 4 * 3, :]


        if self.mono_ga_paths is not None:
            mono_pred = load_sparse_depth(self.mono_ga_paths[index])
            mono_pred = mono_pred.astype(np.float32)
            if mono_pred.shape[0] == 720:
                mono_pred = mono_pred[720 // 3: 720 // 4 * 3, :]
        else:
            mono_pred = None

        if self.mono_ga_paths is not None:
            mono_ga = load_sparse_depth(self.mono_ga_paths[index])
            mono_ga = mono_ga.astype(np.float32)
            if mono_ga.shape[0] == 720:
                mono_ga = mono_ga[720 // 3: 720 // 4 * 3, :]
        else:
            mono_ga = None

        return image, mono_pred, radar, gt, sparse_gt, rcnet, mono_ga


    def __len__(self):
        return self.n_sample