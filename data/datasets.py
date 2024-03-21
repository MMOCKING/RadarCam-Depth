import torch
import torch.utils.data
from torch.utils.data import Dataset
import numpy as np
import data.data_utils as data_utils
import random
import os
from PIL import Image
from data.data_utils import load_depth


def random_sample(T):
    '''
    Arg(s):
        T : numpy[float32]
            C x N array
    Returns:
        numpy[float32] : random sample from T
    '''

    index = np.random.randint(0, T.shape[0])
    return T[index, :]


def random_crop(inputs, shape, crop_type=['none']):
    '''
    Apply crop to inputs e.g. images, depth

    Arg(s):
        inputs : list[numpy[float32]]
            list of numpy arrays e.g. images, depth, and validity maps
        shape : list[int]
            shape (height, width) to crop inputs
        crop_type : str
            none, horizontal, vertical, anchored, top, bottom, left, right, center
    Return:
        list[numpy[float32]] : list of cropped inputs
    '''

    n_height, n_width = shape
    _, o_height, o_width = inputs[0].shape

    # Get delta of crop and original height and width

    d_height = o_height - n_height
    d_width = o_width - n_width

    # By default, perform center crop
    y_start = d_height // 2
    x_start = d_width // 2

    # If left alignment, then set starting height to 0
    if 'left' in crop_type:
        x_start = 0

    # If right alignment, then set starting height to right most position
    elif 'right' in crop_type:
        x_start = d_width

    elif 'horizontal' in crop_type:

        # Select from one of the pre-defined anchored locations
        if 'anchored' in crop_type:
            # Create anchor positions
            crop_anchors = [
                0.0, 0.50, 1.0
            ]

            widths = [
                anchor * d_width for anchor in crop_anchors
            ]
            x_start = int(widths[np.random.randint(low=0, high=len(widths))])

        # Randomly select a crop location
        else:
            x_start = np.random.randint(low=0, high=d_width)

    # If top alignment, then set starting height to 0
    if 'top' in crop_type:
        y_start = 0

    # If bottom alignment, then set starting height to lowest position
    elif 'bottom' in crop_type:
        y_start = d_height

    elif 'vertical' in crop_type and np.random.rand() <= 0.30:

        # Select from one of the pre-defined anchored locations
        if 'anchored' in crop_type:
            # Create anchor positions
            crop_anchors = [
                0.0, 0.50, 1.0
            ]

            heights = [
                anchor * d_height for anchor in crop_anchors
            ]
            y_start = int(heights[np.random.randint(low=0, high=len(heights))])

        # Randomly select a crop location
        else:
            y_start = np.random.randint(low=0, high=d_height)

    elif 'center' in crop_type:
        pass

    # Crop each input into (n_height, n_width)
    y_end = y_start + n_height
    x_end = x_start + n_width

    outputs = [
        T[:, y_start:y_end, x_start:x_end] for T in inputs
    ]

    return outputs


class RCNetTrainingDataset(torch.utils.data.Dataset):
    '''
    Dataset for fetching:
        (1) image
        (2) radar point
        (3) ground truth
        (4) bounding boxes for the points
        (5) image crops for summary part of the code

    Arg(s):
        image_paths : list[str]
            paths to images
        radar_paths : list[str]
            paths to radar points
        ground_truth_paths : list[str]
            paths to ground truth depth maps
        crop_width : int
            width of crop centered at the radar point
        total_points_sampled: int
            total number of points sampled from the total radar points available. Repeats the same points multiple times if total points in the frame is less than total sampled points
        sample_probability_of_lidar: int
            randomly sample lidar with this probability and add noise to it instead of using radar points
    '''

    def __init__(self,
                 image_paths,
                 radar_paths,
                 ground_truth_paths,
                 patch_size,
                 total_points_sampled,
                 sample_probability_of_lidar):

        self.n_sample = len(image_paths)

        assert self.n_sample == len(ground_truth_paths)
        assert self.n_sample == len(radar_paths)

        self.image_paths = image_paths
        self.radar_paths = radar_paths
        self.ground_truth_paths = ground_truth_paths

        self.patch_size = patch_size
        self.pad_size_x = patch_size[1] // 2
        self.padding = ((0, 0), (0, 0), (self.pad_size_x, self.pad_size_x))

        self.data_format = 'CHW'
        self.total_points_sampled = total_points_sampled
        self.sample_probability_of_lidar = sample_probability_of_lidar

    def __getitem__(self, index):

        # Load image
        image = data_utils.load_image(
            self.image_paths[index],
            normalize=False,
            data_format=self.data_format)

        height, width = image.shape[1:]
        if height == 720: # ZJU dataset
            image = image[:, 720 // 3: 720 // 4 * 3, :]

        image = np.pad(
            image,
            pad_width=self.padding,
            mode='edge')

        # Load radar points N x 3
        radar_points = np.load(self.radar_paths[index])

        if height == 720:
            radar_points = radar_points[radar_points[:, 1] < 720 // 4 * 3]
            radar_points[:, 1] = radar_points[:, 1] - 720 // 3
            radar_points = radar_points[radar_points[:, 1] >= 0]

        if radar_points.ndim == 1:
            # Only one point (,3), expand to 1 x 3
            radar_points = np.expand_dims(radar_points, axis=0)

        # Store bounding boxes for all radar points
        bounding_boxes_list = []

        # randomly sample radar points to output
        if radar_points.shape[0] <= self.total_points_sampled:
            radar_points = np.repeat(radar_points, 100, axis=0)
        random_idx = np.random.randint(radar_points.shape[0], size=self.total_points_sampled)
        radar_points = radar_points[random_idx, :]

        # Load ground truth depth
        ground_truth = data_utils.load_depth(
            self.ground_truth_paths[index],
            data_format=self.data_format)

        if height == 720:
            ground_truth = ground_truth[:, 720 // 3: 720 // 4 * 3]

        if random.random() < self.sample_probability_of_lidar:
            ground_truth_for_sampling = np.copy(ground_truth)
            ground_truth_for_sampling = ground_truth_for_sampling.squeeze()
            # Find lidar points with depth greater than 1
            idx_lidar_samples = np.where(ground_truth_for_sampling > 1)

            # randomly sample total_points_sampled number of points from the lidar
            random_indices = random.sample(range(0, len(idx_lidar_samples[0])), self.total_points_sampled)

            points_x = idx_lidar_samples[1][random_indices]
            points_y = idx_lidar_samples[0][random_indices]
            points_z = ground_truth_for_sampling[points_y, points_x]

            noise_for_fake_radar_x = np.random.normal(0, 25, radar_points.shape[0])
            noise_for_fake_radar_z = np.random.uniform(low=0.0, high=0.4, size=radar_points.shape[0])

            fake_radar_points = np.copy(radar_points)
            fake_radar_points[:, 0] = points_x + noise_for_fake_radar_x
            fake_radar_points[:, 0] = np.clip(fake_radar_points[:, 0], 0, ground_truth_for_sampling.shape[1])
            fake_radar_points[:, 2] = points_z + noise_for_fake_radar_z
            # we keep the y as the same it is since it is erroneous

            # convert x and y indices back to int after adding noise
            fake_radar_points[:, 0] = fake_radar_points[:, 0].astype(int)
            fake_radar_points[:, 1] = fake_radar_points[:, 1].astype(int)

            radar_points = np.copy(fake_radar_points)

        # get the shifted radar points after padding
        for radar_point_idx in range(0, radar_points.shape[0]):
            # Set radar point to the center of the patch
            radar_points[radar_point_idx, 0] = radar_points[radar_point_idx, 0] + self.pad_size_x

            bounding_box = [0, 0, 0, 0]
            bounding_box[0] = radar_points[radar_point_idx, 0] - self.pad_size_x
            bounding_box[1] = 0
            bounding_box[2] = radar_points[radar_point_idx, 0] + self.pad_size_x
            bounding_box[3] = self.patch_size[0]
            bounding_boxes_list.append(np.asarray(bounding_box))

        ground_truth = np.pad(
            ground_truth,
            pad_width=self.padding,
            mode='constant',
            constant_values=0)

        ground_truth_crops = []

        # Crop image and ground truth
        for radar_point_idx in range(0, radar_points.shape[0]):
            start_x = int(radar_points[radar_point_idx, 0] - self.pad_size_x)
            end_x = int(radar_points[radar_point_idx, 0] + self.pad_size_x)
            start_y = image.shape[-2] - self.patch_size[0]

            ground_truth_cropped = ground_truth[:, start_y:, start_x:end_x]
            ground_truth_crops.append(ground_truth_cropped)

        image = image[:, start_y:, ...]

        ground_truth = np.asarray(ground_truth_crops)

        # Convert to float32
        image, radar_points, ground_truth = [
            T.astype(np.float32)
            for T in [image, radar_points, ground_truth]
        ]

        bounding_boxes_list = [T.astype(np.float32) for T in bounding_boxes_list]

        bounding_boxes_list = np.stack(bounding_boxes_list, axis=0)

        return image, radar_points, bounding_boxes_list, ground_truth

    def __len__(self):
        return self.n_sample


class RCNetInferenceDataset(torch.utils.data.Dataset):
    '''
    Dataset for fetching:
        (1) image
        (2) radar points
        (3) ground truth (if available)

    Arg(s):
        image_paths : list[str]
            paths to images
        radar_paths : list[str]
            paths to radar points
        ground_truth_paths : list[str]
            paths to ground truth paths
    '''

    def __init__(self, image_paths, radar_paths, ground_truth_paths=None):

        self.n_sample = len(image_paths)

        assert self.n_sample == len(radar_paths)

        self.image_paths = image_paths
        self.radar_paths = radar_paths

        if ground_truth_paths is not None and None not in ground_truth_paths:
            assert self.n_sample == len(ground_truth_paths)
            self.ground_truth_available = True
        else:
            self.ground_truth_available = False

        self.ground_truth_paths = ground_truth_paths

        self.data_format = 'CHW'

    def __getitem__(self, index):

        # Load image
        image = data_utils.load_image(
            self.image_paths[index],
            normalize=False,
            data_format=self.data_format)

        height, width = image.shape[1:]
        if height == 720: # ZJU dataset
            image = image[:, 720 // 3: 720 // 4 * 3, :]

        # Load radar points N x 3
        radar_points = np.load(self.radar_paths[index])

        if height == 720:
            radar_points = radar_points[radar_points[:, 1] < 720 // 4 * 3]
            radar_points[:, 1] = radar_points[:, 1] - 720 // 3
            radar_points = radar_points[radar_points[:, 1] >= 0]

        if radar_points.ndim == 1:
            # Expand to 1 x 3
            radar_points = np.expand_dims(radar_points, axis=0)

        inputs = [image, radar_points]

        if self.ground_truth_available:
            # Load ground truth depth
            ground_truth = data_utils.load_depth(
                self.ground_truth_paths[index],
                data_format=self.data_format)
            if height == 720:
                ground_truth = ground_truth[:, 720 // 3: 720 // 4 * 3]

            inputs.append(ground_truth)

        # Convert to float32
        inputs = [
            T.astype(np.float32)
            for T in inputs
        ]

        return inputs

    def __len__(self):
        return self.n_sample

