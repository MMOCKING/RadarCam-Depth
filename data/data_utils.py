import numpy as np
from scipy.interpolate import LinearNDInterpolator
from PIL import Image
import matplotlib.pyplot as plt



def load_data_path(root, file_name_txt, data_type):
    with open(file_name_txt, 'r') as f:
        data_path = f.readlines()
    data_path = [root + x.strip() + data_type for x in data_path]
    return data_path


def load_data_path_nu(root, name_list, data_type):
    data_path = [root + x.strip() + data_type for x in name_list]
    return data_path


def read_paths(filepath):
    '''
    Reads a newline delimited file containing paths

    Arg(s):
        filepath : str
            path to file to be read
    Return:
        list[str] : list of paths
    '''

    path_list = []
    with open(filepath) as f:
        while True:
            path = f.readline().rstrip('\n')

            # If there was nothing to read
            if path == '':
                break

            path_list.append(path)

    return path_list


def write_paths(filepath, paths):
    '''
    Stores line delimited paths into file

    Arg(s):
        filepath : str
            path to file to save paths
        paths : list[str]
            paths to write into file
    '''

    with open(filepath, 'w') as o:
        for idx in range(len(paths)):
            o.write(paths[idx] + '\n')


def load_image(path, normalize=False, data_format='HWC'):
    '''
    Loads an RGB image

    Arg(s):
        path : str
            path to RGB image
        normalize : bool
            if set, then normalize image between [0, 1]
        data_format : str
            'CHW', or 'HWC'
    Returns:
        numpy[float32] : H x W x C or C x H x W image
    '''

    # Load image
    image = Image.open(path).convert('RGB')

    # Convert to numpy
    image = np.asarray(image, np.float32)

    if data_format == 'HWC':
        pass
    elif data_format == 'CHW':
        image = np.transpose(image, (2, 0, 1))
    else:
        raise ValueError('Unsupported data format: {}'.format(data_format))

    # Normalize
    image = image / 255.0 if normalize else image #255.0

    return image



def load_depth(path, multiplier=256.0, data_format='HW'):
    '''
    Loads a depth map from a 16-bit PNG file

    Arg(s):
        path : str
            path to 16-bit PNG file
        multiplier : float
            multiplier for encoding float as 16/32 bit unsigned integer
        data_format : str
            HW, CHW, HWC
    Returns:
        numpy[float32] : depth map
    '''

    # Loads depth map from 16-bit PNG file
    z = np.array(Image.open(path), dtype=np.float32)

    # Assert 16-bit (not 8-bit) depth map
    z = z / multiplier
    z[z <= 0] = 0.0

    if data_format == 'HW':
        pass
    elif data_format == 'CHW':
        z = np.expand_dims(z, axis=0)
    elif data_format == 'HWC':
        z = np.expand_dims(z, axis=-1)
    else:
        raise ValueError('Unsupported data format: {}'.format(data_format))

    return z


def save_depth(z, path, multiplier=256.0):
    '''
    Saves a depth map to a 16-bit PNG file

    Arg(s):
        z : numpy[float32]
            depth map
        path : str
            path to store depth map
        multiplier : float
            multiplier for encoding float as 16/32 bit unsigned integer
    '''

    z = np.uint32(z * multiplier)
    z = Image.fromarray(z, mode='I')
    z.save(path)


def save_color_depth(z, path):
    '''
    Saves a color depth map to a 16-bit PNG file

    Arg(s):
        z : numpy[float32]
            depth map
        path : str
            path to store depth map
        multiplier : float
            multiplier for encoding float as 16/32 bit unsigned integer
    '''

    # Normalize depth map to the range [0, 1]
    z_normalized = (z - np.min(z)) / (np.max(z) - np.min(z))

    # Convert depth map to color
    # colormap = plt.cm.jet  # Choose a colormap (e.g., jet)
    colormap = plt.cm.viridis
    z_color = colormap(z_normalized)

    # Scale color values to the range [0, 255] and convert to uint8
    z_color = np.uint8(z_color * 255)

    # Save color depth map as an image
    image = Image.fromarray(z_color)
    image.save(path)


def load_response(path, multiplier=2**14, data_format='HW'):
    '''
    Loads a response map from a 16-bit PNG file

    Arg(s):
        path : str
            path to 16-bit PNG file
        multiplier : float
            multiplier for encoding float as 16/32 bit unsigned integer
        data_format : str
            HW, CHW, HWC
    Returns:
        numpy[float32] : response map
    '''

    # Loads response map from 16-bit PNG file
    response = np.array(Image.open(path), dtype=np.float32)

    # Convert using encodering multiplier
    response = response / multiplier

    if data_format == 'HW':
        pass
    elif data_format == 'CHW':
        response = np.expand_dims(response, axis=0)
    elif data_format == 'HWC':
        response = np.expand_dims(response, axis=-1)
    else:
        raise ValueError('Unsupported data format: {}'.format(data_format))

    return response


def save_response(response, path, multiplier=2**14):
    '''
    Saves a response map to a 16-bit PNG file

    Arg(s):
        response : numpy[float32]
            depth map
        path : str
            path to store depth map
        multiplier : float
            multiplier for encoding float as 16/32 bit unsigned integer
    '''

    response = np.uint32(response * multiplier)
    response = Image.fromarray(response, mode='I')
    response.save(path)


def interpolate_depth(depth_map, validity_map, log_space=False):
    '''
    Interpolate sparse depth with barycentric coordinates

    Arg(s):
        depth_map : np.float32
            H x W depth map
        validity_map : np.float32
            H x W depth map
        log_space : bool
            if set then produce in log space
    Returns:
        np.float32 : H x W interpolated depth map
    '''

    assert depth_map.ndim == 2 and validity_map.ndim == 2

    rows, cols = depth_map.shape
    data_row_idx, data_col_idx = np.where(validity_map)
    depth_values = depth_map[data_row_idx, data_col_idx]

    # Perform linear interpolation in log space
    if log_space:
        depth_values = np.log(depth_values)

    interpolator = LinearNDInterpolator(
        # points=Delaunay(np.stack([data_row_idx, data_col_idx], axis=1).astype(np.float32)),
        points=np.stack([data_row_idx, data_col_idx], axis=1),
        values=depth_values,
        fill_value=0 if not log_space else np.log(1e-3))

    query_row_idx, query_col_idx = np.meshgrid(
        np.arange(rows), np.arange(cols), indexing='ij')

    query_coord = np.stack(
        [query_row_idx.ravel(), query_col_idx.ravel()], axis=1)

    Z = interpolator(query_coord).reshape([rows, cols])

    if log_space:
        Z = np.exp(Z)
        Z[Z < 1e-1] = 0.0

    return Z


def interpolate_depth_ZJU(depth_map, validity_map=None, log_space=False, window_size=12):
    '''
    Interpolate sparse depth with barycentric coordinates
    Args:
    depth_map : np.float32
      H x W depth map
    validity_map : np.float32
      H x W depth map
    log_space : bool
      if set then produce in log space
    window_size : int
      size of the window for checking validity
    Returns:
    np.float32 : H x W interpolated depth map
    '''
    assert depth_map.ndim == 2
    if validity_map is None:
        validity_map = depth_map > 0.0
    rows, cols = depth_map.shape
    data_row_idx, data_col_idx = np.where(validity_map)
    depth_values = depth_map[data_row_idx, data_col_idx]
    # Perform linear interpolation in log space
    if log_space:
        depth_values = np.log(depth_values)
    interpolator = LinearNDInterpolator(
        points=np.stack([data_row_idx, data_col_idx], axis=1),
        values=depth_values,
        fill_value=0 if not log_space else np.log(1e-3))
    query_row_idx, query_col_idx = np.meshgrid(np.arange(rows), np.arange(cols), indexing='ij')
    Z = np.zeros_like(depth_map)

    # Create window indices for each query point
    query_indices = np.stack([query_row_idx.ravel(), query_col_idx.ravel()], axis=1)
    window_indices = np.indices((window_size, window_size)).reshape(2, -1) - window_size // 2

    # Calculate window indices for each query point
    window_row_indices = np.clip(query_indices[:, 0, None] + window_indices[0], 0, rows - 1)
    window_col_indices = np.clip(query_indices[:, 1, None] + window_indices[1], 0, cols - 1)

    # Get window values and check validity
    window_values = depth_map[window_row_indices, window_col_indices]
    valid_indices = np.any(window_values > 0, axis=1)

    # Interpolate for valid query points
    valid_query_indices = np.where(valid_indices)[0]
    valid_query_coords = query_indices[valid_query_indices]
    Z.ravel()[valid_query_indices] = interpolator(valid_query_coords)

    if log_space:
        Z = np.exp(Z)
        Z[Z < 1e-1] = 0.0

    return Z