import os, time
import cv2
import numpy as np
import torch, torchvision
import torch.utils.data
from torch.utils.tensorboard import SummaryWriter

import data.data_utils as data_utils
from utils.log_utils import log
import data.SML_dataset as UTV
from utils.net_utils import OutlierRemoval
import utils.log_utils as log_utils
import utils.eval_utils as eval_utils
from utils.loss import compute_loss

from modules.midas.midas_net_custom import MidasNet_small_videpth
from modules.estimator import LeastSquaresEstimator, Optimizer

import modules.midas.transforms as transforms
import modules.midas.utils as utils

def train(
        # data input
        train_image_paths,
        train_radar_paths,
        train_gt_paths,
        train_sparse_gt_paths,
        train_rcnet_paths,

        # training
        learning_rates,
        learning_schedule,
        batch_size,
        n_step_per_summary,
        n_step_per_checkpoint,

        # loss
        loss_func,
        w_smoothness,
        w_weight_decay,
        loss_smoothness_kernel_size,
        w_lidar_loss,
        ground_truth_outlier_removal_kernel_size,
        ground_truth_outlier_removal_threshold,
        ground_truth_dilation_kernel_size,

        # model
        restore_path,
        min_pred_depth,
        max_pred_depth,
        min_radar_valid_depth,
        max_radar_valid_depth,
        checkpoint_dirpath,

        train_mono_pred_paths = None,
        train_mono_ga_paths = None,
        n_threads = 10,
        ):

    if not os.path.exists(checkpoint_dirpath):
        os.makedirs(checkpoint_dirpath)

    depth_model_checkpoint_path = os.path.join(checkpoint_dirpath, 'model-{}.pth')
    log_path = os.path.join(checkpoint_dirpath, 'results.txt')
    event_path = os.path.join(checkpoint_dirpath, 'events')

    log_utils.log_params(log_path, locals())
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    n_train_sample = len(train_gt_paths)
    n_train_step = learning_schedule[-1] * np.ceil(n_train_sample / batch_size).astype(np.int32)

    train_dataloader = torch.utils.data.DataLoader(
        UTV.SML_dataset(
            image_paths = train_image_paths,
            radar_paths = train_radar_paths,
            gt_paths = train_gt_paths,
            sparse_gt_paths = train_sparse_gt_paths,
            rcnet_paths = train_rcnet_paths,
            mono_pred_paths = train_mono_pred_paths,
            mono_ga_paths = train_mono_ga_paths,
        ),
        batch_size=1,
        shuffle=False,
        num_workers=1)

    # transform
    ScaleMapLearner_transform = transforms.get_transforms('dpt_hybrid', 'void', '150')

    # Initialize ground truth outlier removal
    if ground_truth_outlier_removal_kernel_size > 1 and ground_truth_outlier_removal_threshold > 0:
        ground_truth_outlier_removal = OutlierRemoval(
            kernel_size=ground_truth_outlier_removal_kernel_size,
            threshold=ground_truth_outlier_removal_threshold)
    else:
        ground_truth_outlier_removal = None

    # Initialize ground truth dilation
    if ground_truth_dilation_kernel_size > 1:
        ground_truth_dilation = torch.nn.MaxPool2d(
            kernel_size=ground_truth_dilation_kernel_size,
            stride=1,
            padding=ground_truth_dilation_kernel_size // 2)
    else:
        ground_truth_dilation = None


    # build model
    ScaleMapLearner = MidasNet_small_videpth(
        device = device,
        min_pred = min_pred_depth,
        max_pred = max_pred_depth,
    )

    '''
    Train model
    '''
    # Initialize optimizer with starting learning rate
    learning_schedule_pos = 0
    learning_rate = learning_rates[0]

    # Initialize optimizer with starting learning rate
    parameters_model = list(ScaleMapLearner.parameters())
    optimizer = torch.optim.Adam([
        {
            'params': parameters_model,
            'weight_decay': w_weight_decay
        }],
        lr=learning_rate)

    # Set up tensorboard summary writers
    train_summary_writer = SummaryWriter(event_path + '-train')

    # Start training
    train_step = 0

    if restore_path is not None and restore_path != '':
        ScaleMapLearner.load(restore_path)

    for g in optimizer.param_groups:
        g['lr'] = learning_rate

    time_start = time.time()

    print('Begin training...', log_path)
    for epoch in range(1, learning_schedule[-1] + 1):
        print('Epoch: ', epoch)
        # Set learning rate schedule
        if epoch > learning_schedule[learning_schedule_pos]:
            learning_schedule_pos = learning_schedule_pos + 1
            learning_rate = learning_rates[learning_schedule_pos]

            # Update optimizer learning rates
            for g in optimizer.param_groups:
                g['lr'] = learning_rate

           # Train model for an epoch
        for batch_data in train_dataloader:
            train_step = train_step + 1
            batch_data = [
                in_.to(device) for in_ in batch_data
            ]

            image, _, sparse_depth, gt, sparse_gt, rcnet, mono_ga_pos = batch_data

            # sparse radar points depth
            sparse_depth_valid = (sparse_depth < max_radar_valid_depth) * (sparse_depth > min_radar_valid_depth)
            sparse_depth_valid = sparse_depth_valid.bool()
            sparse_depth[~sparse_depth_valid] = np.inf  # set invalid depth
            sparse_depth = 1.0 / sparse_depth

            # RCNet output preprocessing
            rcnet_valid = (rcnet < max_radar_valid_depth) * (rcnet > min_radar_valid_depth)
            rcnet_valid = rcnet_valid.bool()
            rcnet[~rcnet_valid] = np.inf  # set invalid depth
            rcnet = 1.0 / rcnet

            # global aligned mono-pred
            mono_ga = 1.0 / mono_ga_pos

            batch_size = sparse_depth.shape[0]  # 获取批量大小

            # empty batch
            batch_x = []
            batch_d = []
            batch_image = []
            batch_gt = []
            batch_sparse_gt = []

            for i in range(batch_size):
                # single sample in batch
                sparse_depth_i = sparse_depth[i].squeeze().cpu().numpy()
                sparse_depth_valid_i = sparse_depth_valid[i].squeeze().cpu().numpy()
                rcnet_i = rcnet[i].squeeze().cpu().numpy()
                rcnet_valid_i = rcnet_valid[i].squeeze().cpu().numpy()
                int_depth_i = mono_ga[i].squeeze().cpu().numpy()

                int_scales_i = np.ones_like(int_depth_i)
                int_scales_i[rcnet_valid_i] = rcnet_i[rcnet_valid_i] / int_depth_i[rcnet_valid_i]
                int_scales_i = utils.normalize_unit_range(int_scales_i.astype(np.float32))

                # transforms
                sample = {'image': image[i].squeeze().cpu().numpy(),
                          'gt': gt[i].squeeze().cpu().numpy(),
                          'sparse_gt': sparse_gt[i].squeeze().cpu().numpy(),
                          'int_depth': int_depth_i,
                          'int_scales': int_scales_i,
                          'int_depth_no_tf': int_depth_i}

                sample = ScaleMapLearner_transform(sample)

                x = torch.cat([sample['int_depth'], sample['int_scales']], 0)
                x = x.to(device)
                d = sample['int_depth_no_tf'].to(device)
                batch_x.append(x)
                batch_d.append(d)
                batch_image.append(sample['image'].to(device))
                batch_gt.append(sample['gt'].to(device))
                batch_sparse_gt.append(sample['sparse_gt'].to(device))

            x = torch.stack(batch_x, dim=0)
            d = torch.stack(batch_d, dim=0)
            batch_image = torch.stack(batch_image, dim=0)
            batch_gt = torch.stack(batch_gt, dim=0)
            batch_sparse_gt = torch.stack(batch_sparse_gt, dim=0)
            # Forward pass
            sml_pred, sml_scales = ScaleMapLearner.forward(x, d)
            # inverse depth to depth
            d = 1.0 / d
            sml_pred = 1.0 / sml_pred

            # Compute loss function
            if ground_truth_dilation is not None:
                batch_gt = ground_truth_dilation(batch_gt)

            if ground_truth_outlier_removal is not None:
                batch_gt = ground_truth_outlier_removal.remove_outliers(batch_gt)

            validity_map_loss_smoothness = torch.where(
                batch_gt > 0,
                torch.zeros_like(batch_gt),
                torch.ones_like(batch_gt))

            loss, loss_info = compute_loss(
                image=d,
                output_depth=sml_pred,
                ground_truth=batch_gt,
                lidar_map=batch_sparse_gt,
                loss_func=loss_func,
                w_smoothness=w_smoothness,
                loss_smoothness_kernel_size=loss_smoothness_kernel_size,
                validity_map_loss_smoothness=validity_map_loss_smoothness,
                w_lidar_loss=w_lidar_loss)
            print('{}/{} epoch:{}: {}'.format(train_step % n_train_step, n_train_step, epoch, loss.item()))

            # Compute gradient and backpropagate
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (train_step % n_step_per_summary) == 0:
                with torch.no_grad():
                    # Log tensorboard summary
                    log_summary(
                        summary_writer=train_summary_writer,
                        tag='train',
                        step=train_step,
                        max_predict_depth=max_pred_depth,
                        image=batch_image,
                        input_depth=d,
                        output_depth=sml_pred,
                        ground_truth=batch_gt,
                        scalars=loss_info,
                        n_display=min(batch_size, 4))

            # Log results and save checkpoints
            if (train_step % n_step_per_checkpoint) == 0:
                time_elapse = (time.time() - time_start) / 3600
                time_remain = (n_train_step - train_step) * time_elapse / train_step

                print('Step={:6}/{}  Loss={:.5f}  Time Elapsed={:.2f}h  Time Remaining={:.2f}h'.format(
                    train_step, n_train_step, loss.item(), time_elapse, time_remain),
                    log_path)
                # Save checkpoints
                ScaleMapLearner.save(depth_model_checkpoint_path.format(train_step))

    # Save checkpoints
    ScaleMapLearner.save(depth_model_checkpoint_path.format(train_step))






def validate(
        image_paths,
        radar_paths,
        gt_paths,
        sparse_gt_paths,
        rcnet_paths,

        best_results,
        ScaleMapLearner,
        step,
        min_radar_valid_depth,
        max_radar_valid_depth,
        min_eval_depth,
        max_eval_depth,
        output_path,

        mono_pred_paths = None,
        mono_ga_paths = None,

        save_output = False,
        random_sample = False,
        random_sample_size = 1000,
        log_path = None,
        depth_predictor = 'dpt_hybrid',
        ):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if random_sample:
        random_sample_idx = np.random.choice(len(image_paths), random_sample_size, replace=False)
        image_paths = [image_paths[idx] for idx in random_sample_idx]
        radar_paths = [radar_paths[idx] for idx in random_sample_idx]
        gt_paths = [gt_paths[idx] for idx in random_sample_idx]
        sparse_gt_paths = [sparse_gt_paths[idx] for idx in random_sample_idx]
        rcnet_paths = [rcnet_paths[idx] for idx in random_sample_idx]
        if mono_ga_paths is not None:
            mono_pred_paths = [mono_pred_paths[idx] for idx in random_sample_idx]
        if mono_ga_paths is not None:
            mono_ga_paths = [mono_ga_paths[idx] for idx in random_sample_idx]

    val_dataloader = torch.utils.data.DataLoader(
        UTV.SML_dataset(
            image_paths = image_paths,
            radar_paths = radar_paths,
            gt_paths = gt_paths,
            sparse_gt_paths = sparse_gt_paths,
            rcnet_paths = rcnet_paths,
            mono_pred_paths = mono_pred_paths,
            mono_ga_paths = mono_ga_paths,
        ),
        batch_size=1,
        shuffle=False,
        num_workers=1)

    n_sample = len(val_dataloader)
    mae = np.zeros(n_sample)
    rmse = np.zeros(n_sample)
    imae = np.zeros(n_sample)
    irmse = np.zeros(n_sample)
    abs_rel = np.zeros(n_sample)
    sq_rel = np.zeros(n_sample)
    delta1 = np.zeros(n_sample)

    save_file_name = os.path.join(output_path, 'RadarCam-Depth')
    if save_output:
        os.makedirs(save_file_name, exist_ok=True)
        os.makedirs(os.path.join(save_file_name, 'sml_depth'), exist_ok=True)
        os.makedirs(os.path.join(save_file_name, 'sml_depth_color'), exist_ok=True)

    time_start = time.time()

    for idx, inputs in enumerate(val_dataloader):
        inputs = [in_.to(device) for in_ in inputs]

        image, _, _, _, sparse_gt, rcnet, mono_ga_pos = inputs
        input_height, input_width = image.shape[1:3]

        # transform
        ScaleMapLearner_transform = transforms.get_transforms(depth_predictor, 'void', '150')

        rcnet_valid = (rcnet < max_radar_valid_depth) * (rcnet > min_radar_valid_depth)
        rcnet_valid = rcnet_valid.bool()
        rcnet[~rcnet_valid] = np.inf  # set invalid depth
        rcnet = 1.0 / rcnet
        mono_ga = 1.0 / mono_ga_pos

        rcnet = rcnet.squeeze().cpu().numpy()
        rcnet_valid = rcnet_valid.squeeze().cpu().numpy()
        mono_ga_pos = mono_ga_pos.squeeze().cpu().numpy()
        int_depth = mono_ga.squeeze().cpu().numpy()

        int_scales = np.ones_like(int_depth)
        int_scales[rcnet_valid] = rcnet[rcnet_valid] / int_depth[rcnet_valid]
        int_scales = utils.normalize_unit_range(int_scales.astype(np.float32))

        # transforms
        sample = {'image': image.squeeze().cpu().numpy(),
                  'int_depth': int_depth,
                  'int_scales': int_scales,
                  'int_depth_no_tf': int_depth}

        sample = ScaleMapLearner_transform(sample)

        x = torch.cat([sample['int_depth'], sample['int_scales']], 0)
        x = x.to(device)
        d = sample['int_depth_no_tf'].to(device)

        with torch.no_grad():
            sml_pred, sml_scales = ScaleMapLearner.forward(x.unsqueeze(0), d.unsqueeze(0))
            sml_pred = (
                torch.nn.functional.interpolate(
                    1.0 / sml_pred,
                    size=(input_height, input_width),
                    mode="bicubic",
                    align_corners=False,
                )
                .squeeze()
                .cpu()
                .numpy()
            )

        sparse_gt = np.squeeze(sparse_gt.cpu().numpy())
        validity_map = np.where(sparse_gt > 0, 1, 0)

        # Select valid regions to evaluate
        validity_mask = np.where(validity_map > 0, 1, 0)
        min_max_mask = np.logical_and(
            sparse_gt > min_eval_depth,
            sparse_gt < max_eval_depth)
        mask = np.where(np.logical_and(validity_mask, min_max_mask) > 0)
        output_depth = sml_pred[mask]
        sparse_gt = sparse_gt[mask]

        # Compute validation metrics
        mae[idx] = eval_utils.mean_abs_err(1000.0 * output_depth, 1000.0 * sparse_gt)
        rmse[idx] = eval_utils.root_mean_sq_err(1000.0 * output_depth, 1000.0 * sparse_gt)
        imae[idx] = eval_utils.inv_mean_abs_err(0.001 * output_depth, 0.001 * sparse_gt)
        irmse[idx] = eval_utils.inv_root_mean_sq_err(0.001 * output_depth, 0.001 * sparse_gt)
        abs_rel[idx] = eval_utils.mean_abs_rel_err(1000.0 * output_depth, 1000.0 * sparse_gt)
        sq_rel[idx] = eval_utils.mean_sq_rel_err(1000.0 * output_depth, 1000.0 * sparse_gt)
        delta1[idx] = eval_utils.thr_acc(output_depth, sparse_gt)
        print(mae[idx], rmse[idx], imae[idx], irmse[idx], abs_rel[idx], sq_rel[idx], delta1[idx])

        if save_output:
            basename = os.path.basename(image_paths[idx]).split('.')[0] + '.png'
            print('Saving output {}'.format(basename))
            sky_mask = mono_ga_pos >= 200
            sml_pred[sky_mask] = mono_ga_pos[sky_mask]
            data_utils.save_depth(sml_pred, os.path.join(save_file_name, 'sml_depth', basename))
            data_utils.save_color_depth(sml_pred, os.path.join(save_file_name, 'sml_depth_color', basename))

    time_end = time.time()
    print('Time taken: {:.4f} seconds'.format(time_end - time_start))
    print('average time per sample: {:.4f} seconds'.format((time_end - time_start) / len(image_paths)))

    # Compute mean metrics
    mae = np.mean(mae)
    rmse = np.mean(rmse)
    imae = np.mean(imae)
    irmse = np.mean(irmse)
    abs_rel = np.mean(abs_rel)
    sq_rel = np.mean(sq_rel)
    delta1 = np.mean(delta1)

    # Print validation results to console
    log_evaluation_results(
        title='Validation results',
        mae=mae,
        rmse=rmse,
        imae=imae,
        irmse=irmse,
        abs_rel=abs_rel,
        sq_rel=sq_rel,
        delta1=delta1,
        step=step,
        log_path=log_path)

    n_improve = 0
    if np.round(mae, 4) < np.round(best_results['mae'], 4):
        n_improve = n_improve + 1
    if np.round(rmse, 4) < np.round(best_results['rmse'], 4):
        n_improve = n_improve + 1
    if np.round(imae, 4) < np.round(best_results['imae'], 4):
        n_improve = n_improve + 1
    if np.round(irmse, 4) < np.round(best_results['irmse'], 4):
        n_improve = n_improve + 1
    if np.round(abs_rel, 4) < np.round(best_results['abs_rel'], 4):
        n_improve = n_improve + 1
    if np.round(sq_rel, 4) < np.round(best_results['sq_rel'], 4):
        n_improve = n_improve + 1
    if np.round(delta1, 4) > np.round(best_results['delta1'], 4):
        n_improve = n_improve + 1

    if n_improve > 3:
        best_results['step'] = step
        best_results['mae'] = mae
        best_results['rmse'] = rmse
        best_results['imae'] = imae
        best_results['irmse'] = irmse
        best_results['abs_rel'] = abs_rel
        best_results['sq_rel'] = sq_rel
        best_results['delta1'] = delta1

    log_evaluation_results(
        title='Best results',
        mae=best_results['mae'],
        rmse=best_results['rmse'],
        imae=best_results['imae'],
        irmse=best_results['irmse'],
        step=best_results['step'],
        abs_rel=best_results['abs_rel'],
        sq_rel=best_results['sq_rel'],
        delta1=best_results['delta1'],
        log_path=log_path)

    return best_results






def log_summary(summary_writer,
                tag,
                step,
                max_predict_depth,
                image=None,
                input_depth=None,
                input_response=None,
                output_depth=None,
                ground_truth=None,
                scalars={},
                n_display=4):

    with torch.no_grad():

        display_summary_image = []
        display_summary_depth = []

        display_summary_image_text = tag
        display_summary_depth_text = tag

        if image is not None:
            image_summary = image[0:n_display, ...]

            display_summary_image_text += '_image'
            display_summary_depth_text += '_image'

            # Add to list of images to log
            display_summary_image.append(
                torch.cat([
                    image_summary.cpu(),
                    torch.zeros_like(image_summary, device=torch.device('cpu'))],
                    dim=-1))

            display_summary_depth.append(display_summary_image[-1])

        if output_depth is not None:
            output_depth_summary = output_depth[0:n_display, ...]

            display_summary_depth_text += '_output_depth'

            # Add to list of images to log
            n_batch, _, n_height, n_width = output_depth_summary.shape

            display_summary_depth.append(
                torch.cat([
                    log_utils.colorize(
                        (output_depth_summary / max_predict_depth).cpu(),
                        colormap='viridis'),
                    torch.zeros(n_batch, 3, n_height, n_width, device=torch.device('cpu'))],
                    dim=3))

            # Log distribution of output depth
            summary_writer.add_histogram(tag + '_output_depth_distro', output_depth, global_step=step)

        if output_depth is not None and input_depth is not None:
            input_depth_summary = input_depth[0:n_display, ...]

            display_summary_depth_text += '_input_depth-error'

            # Compute output error w.r.t. input depth
            input_depth_error_summary = \
                torch.abs(output_depth_summary - input_depth_summary)

            input_depth_error_summary = torch.where(
                input_depth_summary > 0.0,
                input_depth_error_summary / (input_depth_summary + 1e-8),
                input_depth_summary)

            # Add to list of images to log
            input_depth_summary = log_utils.colorize(
                (input_depth_summary / max_predict_depth).cpu(),
                colormap='viridis')
            input_depth_error_summary = log_utils.colorize(
                (input_depth_error_summary / 0.05).cpu(),
                colormap='inferno')

            display_summary_depth.append(
                torch.cat([
                    input_depth_summary,
                    input_depth_error_summary],
                    dim=3))

            # Log distribution of input depth
            summary_writer.add_histogram(tag + '_input_depth_distro', input_depth, global_step=step)




        if output_depth is not None and input_response is not None:
            response_summary = input_response[0:n_display, ...]

            display_summary_depth_text += '_response'

            # Add to list of images to log
            response_summary = log_utils.colorize(
                response_summary.cpu(),
                colormap='inferno')

            display_summary_depth.append(
                torch.cat([
                    response_summary,
                    torch.zeros_like(response_summary)],
                    dim=3))

            # Log distribution of input depth
            summary_writer.add_histogram(tag + '_response_distro', input_depth, global_step=step)




        if output_depth is not None and ground_truth is not None:
            ground_truth = ground_truth[0:n_display, ...]
            ground_truth = torch.unsqueeze(ground_truth[:, 0, :, :], dim=1)

            ground_truth_summary = ground_truth[0:n_display]
            validity_map_summary = torch.where(
                ground_truth > 0,
                torch.ones_like(ground_truth),
                torch.zeros_like(ground_truth))

            display_summary_depth_text += '_ground_truth-error'

            # Compute output error w.r.t. ground truth
            ground_truth_error_summary = \
                torch.abs(output_depth_summary - ground_truth_summary)

            ground_truth_error_summary = torch.where(
                validity_map_summary == 1.0,
                (ground_truth_error_summary + 1e-8) / (ground_truth_summary + 1e-8),
                validity_map_summary)

            # Add to list of images to log
            ground_truth_summary = log_utils.colorize(
                (ground_truth_summary / max_predict_depth).cpu(),
                colormap='viridis')
            ground_truth_error_summary = log_utils.colorize(
                (ground_truth_error_summary / 0.05).cpu(),
                colormap='inferno')

            display_summary_depth.append(
                torch.cat([
                    ground_truth_summary,
                    ground_truth_error_summary],
                    dim=3))

            # Log distribution of ground truth
            summary_writer.add_histogram(tag + '_ground_truth_distro', ground_truth, global_step=step)

        # Log scalars to tensorboard
        for (name, value) in scalars.items():
            summary_writer.add_scalar(tag + '_' + name, value, global_step=step)

        # Log image summaries to tensorboard
        if len(display_summary_image) > 1:
            display_summary_image = torch.cat(display_summary_image, dim=2)

            summary_writer.add_image(
                display_summary_image_text,
                torchvision.utils.make_grid(display_summary_image, nrow=n_display),
                global_step=step)

        if len(display_summary_depth) > 1:
            display_summary_depth = torch.cat(display_summary_depth, dim=2)

            summary_writer.add_image(
                display_summary_depth_text,
                torchvision.utils.make_grid(display_summary_depth, nrow=n_display),
                global_step=step)



def log_evaluation_results(title,
                           mae,
                           rmse,
                           imae,
                           irmse,
                           abs_rel=None,
                           sq_rel=None,
                           delta1=None,
                           step=-1,
                           log_path=None):

    log(title + ':', log_path)
    log('{:>8}  {:>8}  {:>8}  {:>8}  {:>8}  {:>8}  {:>8}  {:>8}'.format(
        'Step', 'MAE', 'RMSE', 'iMAE', 'iRMSE', 'Abs_Rel', 'Sq_Rel', 'Delta1'),
        log_path)
    log('{:8}  {:8.3f}  {:8.3f}  {:8.3f}  {:8.3f}  {:8.3f}  {:8.3f}  {:8.3f}'.format(
        step,
        mae,
        rmse,
        imae,
        irmse,
        abs_rel,
        sq_rel,
        delta1),
        log_path)