#!/usr/bin/env python3
# Copyright © Niantic, Inc. 2022.

import os
import argparse
from decimal import ROUND_HALF_DOWN
import logging
import math
import time
from distutils.util import strtobool
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader
from pytorch_msssim import SSIM

import dsacstar
from ace_network import Regressor
from dataset import CamLocDataset

import ace_vis_util as vutil
from ace_visualizer import ACEVisualizer
import ace_util

from gsplat_ace import GSplat

_logger = logging.getLogger(__name__)


def _strtobool(x):
    return bool(strtobool(x))

def get_color_points(rgb, H, W):
        nn_stride = 1
        nn_offset = 1//2
        rgb = rgb[nn_offset::nn_stride, nn_offset::nn_stride, :]
        # make sure the resolution fits (catch any striding mismatches)
        rgb = rgb.resize_(H, W, 3)
        color = rgb.contiguous().view(-1, 3)

        return color
    
def convert_data_NC(data):    
    # (C, H, W) -> (H*W, C)
    C, _, _ = data.shape
    data = data.permute(1, 2, 0)  # (C, H, W) -> (H, W, C)
    data = data.view(-1, C).to(torch.float32)  # (H*W, C)
    return data


if __name__ == '__main__':
    # Setup logging.
    logging.basicConfig(level=logging.INFO)
    # ctr: initialize the logger
    _logger.setLevel(logging.INFO)
    _fh = logging.FileHandler('test.log', encoding='utf8', mode='w')
    _logger.addHandler(_fh)

    parser = argparse.ArgumentParser(
        description='Test a trained network on a specific scene.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('scene', type=Path,
                        help='path to a scene in the dataset folder, e.g. "datasets/Cambridge_GreatCourt"')

    parser.add_argument('network', type=Path, help='path to a network trained for the scene (just the head weights)')
    
    parser.add_argument('--save_image', type=Path, help='path to save image')

    parser.add_argument('--encoder_path', type=Path, default=Path(__file__).parent / "ace_encoder_pretrained.pt",
                        help='file containing pre-trained encoder weights')

    parser.add_argument('--session', '-sid', default='',
                        help='custom session name appended to output files, '
                             'useful to separate different runs of a script')

    parser.add_argument('--image_resolution', type=int, default=480, help='base image resolution')

    # ACE is RGB-only, no need for this param.
    # parser.add_argument('--mode', '-m', type=int, default=1, choices=[1, 2], help='test mode: 1 = RGB, 2 = RGB-D')

    # DSACStar RANSAC parameters. ACE Keeps them at default.
    parser.add_argument('--hypotheses', '-hyps', type=int, default=64,
                        help='number of hypotheses, i.e. number of RANSAC iterations')

    parser.add_argument('--threshold', '-t', type=float, default=10,
                        help='inlier threshold in pixels (RGB) or centimeters (RGB-D)')
    
    parser.add_argument('--refine_threshold', type=float, default=5)

    parser.add_argument('--inlieralpha', '-ia', type=float, default=100,
                        help='alpha parameter of the soft inlier count; controls the softness of the '
                             'hypotheses score distribution; lower means softer')

    parser.add_argument('--maxpixelerror', '-maxerrr', type=float, default=100,
                        help='maximum reprojection (RGB, in px) or 3D distance (RGB-D, in cm) error when checking '
                             'pose consistency towards all measurements; error is clamped to this value for stability')
    
    parser.add_argument('--test_downsampling', '-td', type=float, default=1,
                    help='downsample the points fed into dsacstar to test')
    
    parser.add_argument('--test_seed', '-ts', type=float, default=2089,
                help='set up random seed for torch')

    # Params for the visualization. If enabled, it will slow down relocalisation considerably. But you get a nice video :)
    parser.add_argument('--render_visualization', type=_strtobool, default=False,
                        help='create a video of the mapping process')

    parser.add_argument('--render_target_path', type=Path, default='renderings',
                        help='target folder for renderings, visualizer will create a subfolder with the map name')

    parser.add_argument('--render_flipped_portrait', type=_strtobool, default=False,
                        help='flag for wayspots dataset where images are sideways portrait')

    parser.add_argument('--render_sparse_queries', type=_strtobool, default=False,
                        help='set to true if your queries are not a smooth video')

    parser.add_argument('--render_pose_error_threshold', type=int, default=20,
                        help='pose error threshold for the visualisation in cm/deg')

    parser.add_argument('--render_map_depth_filter', type=int, default=10,
                        help='to clean up the ACE point cloud remove points too far away')

    parser.add_argument('--render_camera_z_offset', type=int, default=4,
                        help='zoom out of the scene by moving render camera backwards, in meters')

    parser.add_argument('--render_frame_skip', type=int, default=1,
                        help='skip every xth frame for long and dense query sequences')
    
    parser.add_argument('--refine_iters', type=int, default=20)
    parser.add_argument('--refine_lr_r', type=float, default=0.0001)
    parser.add_argument('--refine_lr_t', type=float, default=0.0001)

    opt = parser.parse_args()

    device = torch.device("cuda")
    num_workers = 1

    torch.manual_seed(opt.test_seed)

    scene_path = Path(opt.scene)
    head_network_path = Path(opt.network)
    encoder_path = Path(opt.encoder_path)
    session = opt.session
    os.makedirs(opt.save_image, exist_ok=True)

    # Setup dataset.
    testset = CamLocDataset(
        scene_path / "test",
        mode=0,  # Default for ACE, we don't need scene coordinates/RGB-D.
        image_height=opt.image_resolution,
    )
    _logger.info(f'Test images found: {len(testset)}')

    # Setup dataloader. Batch size 1 by default.
    testset_loader = DataLoader(testset, shuffle=False, num_workers=num_workers)

    # Load network weights.
    encoder_state_dict = torch.load(encoder_path, map_location="cpu")
    _logger.info(f"Loaded encoder from: {encoder_path}")
    head_state_dict = torch.load(head_network_path, map_location="cpu")
    _logger.info(f"Loaded head weights from: {head_network_path}")

    # Create regressor.
    network = Regressor.create_from_split_state_dict(encoder_state_dict, head_state_dict)

    # Setup for evaluation.
    network = network.to(device)
    network.eval()

    # Save the outputs in the same folder as the network being evaluated.
    output_dir = head_network_path.parent
    scene_name = scene_path.name
    # This will contain aggregate scene stats (median translation/rotation errors, and avg processing time per frame).
    test_log_file = output_dir / f'test_{scene_name}_{opt.session}.txt'
    _logger.info(f"Saving test aggregate statistics to: {test_log_file}")
    # This will contain each frame's pose (stored as quaternion + translation) and errors.
    pose_log_file = output_dir / f'poses_{scene_name}_{opt.session}.txt'
    _logger.info(f"Saving per-frame poses and errors to: {pose_log_file}")

    # Setup output files.
    test_log = open(test_log_file, 'w', 1)
    pose_log = open(pose_log_file, 'w', 1)

    # Metrics of interest.
    avg_batch_time = 0
    num_batches = 0

    ssim = SSIM(data_range=1.0, size_average=True, channel=3)
    mse = torch.nn.MSELoss()

    # Keep track of rotation and translation errors for calculation of the median error.
    rErrs = []
    tErrs = []

    rErrs_before = []
    tErrs_before = []

    # Percentage of frames predicted within certain thresholds from their GT pose.
    pct10_5 = 0
    pct5 = 0
    pct2 = 0
    pct1 = 0

    # Generate video of training process
    if opt.render_visualization:
        # infer rendering folder from map file name
        target_path = vutil.get_rendering_target_path(
            opt.render_target_path,
            opt.network)
        ace_visualizer = ACEVisualizer(target_path,
                                       opt.render_flipped_portrait,
                                       opt.render_map_depth_filter,
                                       reloc_vis_error_threshold=opt.render_pose_error_threshold)

        # we need to pass the training set in case the visualiser has to regenerate the map point cloud
        trainset = CamLocDataset(
            scene_path / "train",
            mode=0,  # Default for ACE, we don't need scene coordinates/RGB-D.
            image_height=opt.image_resolution,
        )

        # Setup dataloader. Batch size 1 by default.
        trainset_loader = DataLoader(trainset, shuffle=False, num_workers=num_workers)

        ace_visualizer.setup_reloc_visualisation(
            frame_count=len(testset),
            data_loader=trainset_loader,
            network=network,
            camera_z_offset=opt.render_camera_z_offset,
            reloc_frame_skip=opt.render_frame_skip)
    else:
        ace_visualizer = None

    # Testing loop.
    testing_start_time = time.time()
    global_count = 0

    loss_photo_all = {
                    'gt':{
                        'l1':[], 
                        'mse': [], 
                        'gaussian': []
                    },
                    'predict':{ 
                        'l1':[], 
                        'mse': [], 
                        'gaussian': []
                    }
                }
    
    # time_00 = time.time() 
    image_pixel = ace_util.get_pixel_grid(1).to(device)
    # _logger.info(f'Time of get_pixel_grid: {(time.time()-time_00)*1000:.0f}ms')
    
    for image_B3HW, image_B1HW, _, gt_pose_B44, _, intrinsics_B33, _, _, filenames in testset_loader:
        with torch.no_grad():
            _logger.info(f'ID: {global_count}')
            time_0 = time.time()

            # if global_count<82:
            #     global_count += 1
            #     continue

            # idx = [ 147,  148,  149,  151,  152,  153,  156,  157,  215,  264,  469,
            #         470,  477,  485,  486,  487,  677,  694,  791,  794,  961,  984,
            #         987,  988, 1339, 1542]
            # if global_count not in idx:
            #     global_count += 1
            #     continue

            batch_start_time = time.time()
            batch_size = image_B1HW.shape[0]

            image_B3HW = image_B3HW.to(device, non_blocking=True)
            image_B1HW = image_B1HW.to(device, non_blocking=True)


            # Predict scene coordinates.
            with autocast(enabled=True):
                scene_coordinates_B3HW, density_scene_coordinates_B3HW, rot_maps, scale_maps, opacity_maps = network(image_B1HW)

            time_1 = time.time()
            _logger.info(f'Time of network inference: {(time_1-time_0)*1000:.0f}ms')

            # We need them on the CPU to run RANSAC.
            scene_coordinates_B3HW = scene_coordinates_B3HW.float().cpu()

            predict_poses_B44 = torch.zeros_like(gt_pose_B44).to(device)

            # Each frame is processed independently.
            for frame_idx, (scene_coordinates_3HW, gt_pose_44, intrinsics_33, frame_path) in enumerate(
                    zip(scene_coordinates_B3HW, gt_pose_B44, intrinsics_B33, filenames)):

                # Extract focal length and principal point from the intrinsics matrix.
                focal_length = intrinsics_33[0, 0].item()
                ppX = intrinsics_33[0, 2].item()
                ppY = intrinsics_33[1, 2].item()
                # We support a single focal length.
                assert torch.allclose(intrinsics_33[0, 0], intrinsics_33[1, 1])

                # Remove path from file name
                frame_name = Path(frame_path).name

                # Allocate output variable.
                out_pose = torch.zeros((4, 4))

                # Compute the pose via RANSAC.
                # sampling = ace_util.get_pixel_grid(network.OUTPUT_SUBSAMPLE)
                sampling = ace_util.get_pixel_grid(8)
                H, W = scene_coordinates_3HW.shape[1], scene_coordinates_3HW.shape[2]
                sampling = sampling[:, : H, : W]
                # rnd_H_idx = torch.randperm(H)[: int(H // opt.test_downsampling)]
                # rnd_W_idx = torch.randperm(W)[: int(W // opt.test_downsampling)]
                # sampling = sampling[:, :, rnd_W_idx]
                # sampling = sampling[:, rnd_H_idx]
                # scene_coordinates_3HW = scene_coordinates_3HW[:, :, rnd_W_idx]
                # scene_coordinates_3HW = scene_coordinates_3HW[:, rnd_H_idx]
                # H, W = scene_coordinates_3HW.shape[1], scene_coordinates_3HW.shape[2]

                inlier_count = dsacstar.forward_rgb(
                    scene_coordinates_3HW.unsqueeze(0),
                    sampling,
                    out_pose,
                    opt.hypotheses,
                    opt.threshold,
                    focal_length,
                    ppX,
                    ppY,
                    opt.inlieralpha,
                    opt.maxpixelerror,
                    network.OUTPUT_SUBSAMPLE,
                )

                time_2 = time.time()
                _logger.info(f'Time of dsacstar: {(time_2-time_1)*1000:.0f}ms')

                predict_poses_B44[frame_idx] = out_pose  # camera to world

                # Calculate translation error.
                t_err = float(torch.norm(gt_pose_44[0:3, 3] - out_pose[0:3, 3]))

                # Rotation error.
                gt_R = gt_pose_44[0:3, 0:3].numpy()
                out_R = out_pose[0:3, 0:3].numpy()

                r_err = np.matmul(out_R, np.transpose(gt_R))
                # Compute angle-axis representation.
                r_err = cv2.Rodrigues(r_err)[0]
                # Extract the angle.
                r_err = np.linalg.norm(r_err) * 180 / math.pi

                _logger.info(f"Before refine: Rotation Error: {r_err:.2f}deg, Translation Error: {t_err * 100:.2f}cm")
                # Save the errors.
                rErrs_before.append(r_err)
                tErrs_before.append(t_err * 100)

                if ace_visualizer is not None:
                    ace_visualizer.render_reloc_frame(
                        query_pose=gt_pose_44.numpy(),
                        query_file=frame_path,
                        est_pose=out_pose.numpy(),
                        est_error=max(r_err, t_err*100),
                        sparse_query=opt.render_sparse_queries)

                # # Save the errors.
                # rErrs.append(r_err)
                # tErrs.append(t_err * 100)

                # # Check various thresholds.
                # if r_err < 5 and t_err < 0.1:  # 10cm/5deg
                #     pct10_5 += 1
                # if r_err < 5 and t_err < 0.05:  # 5cm/5deg
                #     pct5 += 1
                # if r_err < 2 and t_err < 0.02:  # 2cm/2deg
                #     pct2 += 1
                # if r_err < 1 and t_err < 0.01:  # 1cm/1deg
                #     pct1 += 1

                # # Write estimated pose to pose file (inverse).
                # out_pose = out_pose.inverse()

                # # Translation.
                # t = out_pose[0:3, 3]

                # # Rotation to axis angle.
                # rot, _ = cv2.Rodrigues(out_pose[0:3, 0:3].numpy())
                # angle = np.linalg.norm(rot)
                # axis = rot / angle

                # # Axis angle to quaternion.
                # q_w = math.cos(angle * 0.5)
                # q_xyz = math.sin(angle * 0.5) * axis

                # # Write to output file. All in a single line.
                # pose_log.write(f"{frame_name} "
                #                 f"{q_w} {q_xyz[0].item()} {q_xyz[1].item()} {q_xyz[2].item()} "
                #                 f"{t[0]} {t[1]} {t[2]} "
                #                 f"{r_err} {t_err} {inlier_count}\n")

            avg_batch_time += time.time() - batch_start_time
            num_batches += 1
        
        time_3 = time.time()
        # _logger.info(f'Time of ace: {(time_3-time_0)*1000:.0f}ms')

        # Gaussian renderer
        for i in range(0, batch_size):

            # calculate reprojection error and get mask for select pixel
            # ======================== start calculate mask ========================================
            predict_inv_poses_b34 = predict_poses_B44[i].inverse()[:3, :]
            Ks_b33 = intrinsics_B33[i].to(device)
            # Back to the original shape. Convert to float32 as well.
            pred_scene_coords_b31 = density_scene_coordinates_B3HW[i].permute(1, 2, 0).flatten(0, 1).unsqueeze(-1).float()  # .cpu()

            # Make 3D points homogeneous so that we can easily matrix-multiply them.
            pred_scene_coords_b41 = ace_util.to_homogeneous(pred_scene_coords_b31, device=device)

            # Scene coordinates to camera coordinates.
            pred_cam_coords_b31 = torch.matmul(predict_inv_poses_b34, pred_scene_coords_b41)

            # Project scene coordinates.
            pred_px_b31 = torch.matmul(Ks_b33, pred_cam_coords_b31)
            pred_px_b31[:, 2].clamp_(min=0.1)
            # Dehomogenise.
            pred_px_b21 = pred_px_b31[:, :2] / pred_px_b31[:, 2, None]

            # Measure reprojection error.
            # time_00 = time.time() 
            # image_pixel = ace_util.get_pixel_grid(1).to(device)
            # _logger.info(f'Time of get_pixel_grid: {(time.time()-time_00)*1000:.0f}ms')
            _, _, H, W = image_B1HW.shape
            image_pixel_2HW = image_pixel[:, : H, : W].clone()
            image_pixel_2N = image_pixel_2HW.view(2, -1)
            target_px_b2 = image_pixel_2N.transpose(1, 0)
            reprojection_error_b2 = pred_px_b21.squeeze() - target_px_b2
            reprojection_error_b1 = torch.norm(reprojection_error_b2, dim=1, keepdim=True, p=1)

             # Predicted coordinates behind or close to camera plane.
            invalid_min_depth_b1 = pred_cam_coords_b31[:, 2] < 0.1

            # Very large reprojection errors.
            # method 1. Select the half of the coordinates with the smaller reprojection loss.
            if opt.refine_threshold < 1:
                repro_error_90precent = torch.sort(reprojection_error_b1[:, 0]).values[int(reprojection_error_b1.shape[0]*opt.refine_threshold)]
                invalid_repro_b1 = reprojection_error_b1 > repro_error_90precent
                # _logger.info(f"============================================ 20% t: {repro_error_90precent:.2f}")
            # method 2. Select the part of coordinates where the reprojection loss is less than the threshold.
            else:
                invalid_repro_b1 = reprojection_error_b1 > opt.refine_threshold


            # Predicted coordinates beyond max distance.
            invalid_max_depth_b1 = pred_cam_coords_b31[:, 2] > 1000
            # Invalid mask is the union of all these. Valid mask is the opposite.
            invalid_mask_b1 = (invalid_min_depth_b1 | invalid_repro_b1 | invalid_max_depth_b1)
            valid_mask_b1 = ~invalid_mask_b1
            valid_mask = valid_mask_b1.reshape((H, W, 1)).repeat(1,1,3)  #.to(device)

            # repro_mean = torch.mean(reprojection_error_b1[valid_mask_b1])
            # _logger.info(f"--> Before refine: Mean Repro Error: {repro_mean:.2f}")
            # mask_save = torch.ones((480, 640), dtype=torch.uint8)
            # mask_save[~valid_mask[:,:,0]] = 0
            # cv2.imwrite(f"{opt.save_image}/mask_{global_count}.png", mask_save[:, :, None].cpu().numpy() * 255)

            # ======================== end calculate mask ========================================

            time_4 = time.time()
            _logger.info(f'Time of valid mask: {(time_4-time_3)*1000:.0f}ms')

            # Prepare the data for Gaussians
            # gs_start_time = time.time()
            
            # 1.Init Gaissians per image.
            img_gt_single = image_B3HW[i].float()
            img_gt_single[~valid_mask] = 0  # valid mask
            points_xyz = convert_data_NC(density_scene_coordinates_B3HW[i]).to(device)  # (num_points, 3)
            points_color = get_color_points(img_gt_single, density_scene_coordinates_B3HW.shape[2], density_scene_coordinates_B3HW.shape[3]).to(device)  # (num_points, 3)
            gs_rot = convert_data_NC(rot_maps[i]).to(device)
            gs_scale = convert_data_NC(scale_maps[i]).to(device)
            gs_opacity = convert_data_NC(opacity_maps[i]).to(device)

            # valid mask for gaussians
            points_xyz = points_xyz[valid_mask_b1[:,0], :]
            points_color = points_color[valid_mask_b1[:,0], :]
            gs_rot = gs_rot[valid_mask_b1[:,0], :]
            gs_scale = gs_scale[valid_mask_b1[:,0], :]
            gs_opacity = gs_opacity[valid_mask_b1[:,0], :]

            gsplat = GSplat(points_xyz, points_color, gs_rot, gs_scale, gs_opacity)
            # gs_end_time = time.time()
            # _logger.info(f'gs in {gs_end_time - gs_start_time:.2f}s')

            # export gaussians as .ply
            # gsplat.save_ply("/home/yangmin/zhangfan/code/ace_gs/out_imgs/1.ply")

            time_5 = time.time()
            _logger.info(f'Time of gaussian init: {(time_5-time_4)*1000:.0f}ms')

            ''' 
            # only use gsplat to renderer
            # 2.Feed features to predict.
            camera = {}
            camera['height'], camera['width'] = img_gt_single.shape[0], img_gt_single.shape[1]
            camera['fx'] = intrinsics_B33[i, 0, 0].item()
            camera['fy'] = intrinsics_B33[i, 1, 1].item()
            camera['cx'] = intrinsics_B33[i, 0, 2].item()
            camera['cy'] = intrinsics_B33[i, 1, 2].item()
            
            # 1) render with gt pose
            camera['world_to_camera'] = gt_pose_B44[i].inverse()[:3, :]
            w2c_r, w2c_t = camera['world_to_camera'].split(3, dim=1)
            img_render_gt, alpha_gt = gsplat.forward(camera, w2c_r, w2c_t)
            img_render_gt[~valid_mask] = 0  # valid mask
            Ll1_gt = torch.abs(img_gt_single - img_render_gt).mean()
            simloss_gt = 1 - ssim(img_gt_single.permute(2, 0, 1)[None, ...], img_render_gt.permute(2, 0, 1)[None, ...])
            loss_photo_gt = ((1 - 0.2) * Ll1_gt + 0.2 * simloss_gt)
            loss_mse_gt = mse(img_gt_single, img_render_gt)

            # 2) render with predict pose
            camera['world_to_camera'] = predict_poses_B44[i].inverse()[:3, :]
            w2c_r, w2c_t = camera['world_to_camera'].split(3, dim=1)
            img_render_predict, alpha_predict = gsplat.forward(camera, w2c_r, w2c_t)
            img_render_predict[~valid_mask] = 0  # valid mask
            Ll1_predict = torch.abs(img_gt_single - img_render_predict).mean()
            simloss_predict = 1 - ssim(img_gt_single.permute(2, 0, 1)[None, ...], img_render_predict.permute(2, 0, 1)[None, ...])
            loss_photo_predict = ((1 - 0.2) * Ll1_predict + 0.2 * simloss_predict)
            loss_mse_predict = mse(img_gt_single, img_render_predict)
            
            loss_photo_all['gt']['l1'].append(Ll1_gt.item())
            loss_photo_all['gt']['mse'].append(loss_mse_gt.item())
            loss_photo_all['gt']['gaussian'].append(loss_photo_gt.item())
            loss_photo_all['predict']['l1'].append(Ll1_predict.item())
            loss_photo_all['predict']['mse'].append(loss_mse_predict.item())
            loss_photo_all['predict']['gaussian'].append(loss_photo_predict.item())

            _logger.info(f'loss_gt:      l1:{Ll1_gt:.4f}, mse:{loss_mse_gt:.4f}, gaussian:{loss_photo_gt:.4f}')
            _logger.info(f'loss_predict: l1:{Ll1_predict:.4f}, mse:{loss_mse_predict:.4f}, gaussian:{loss_photo_predict:.4f}')

            # # save images
            img_gt_single = (img_gt_single.detach().cpu().numpy() * 255).astype(np.uint8)
            img_render_gt = (img_render_gt.detach().cpu().numpy() * 255).astype(np.uint8)
            img_render_predict = (img_render_predict.detach().cpu().numpy() * 255).astype(np.uint8)
            cv2.imwrite(f"{opt.save_image}/img_{global_count}_gt.png", img_gt_single[:, :, ::-1])
            cv2.imwrite(f"{opt.save_image}/img_{global_count}_render_gt.png", img_render_gt[:, :, ::-1])
            cv2.imwrite(f"{opt.save_image}/img_{global_count}_render_predict.png", img_render_predict[:, :, ::-1])
            # alpha_predict = (alpha_predict.detach().cpu().numpy() * 255).astype(np.uint8)
            # cv2.imwrite(f"{opt.save_image}/alpha_{global_count}_render_predict.png", alpha_predict[:, :, None])
            '''

            # '''
            # use gsplat to refine pose
            # 2.Feed features to predict.
            camera = {}
            camera['height'], camera['width'] = img_gt_single.shape[0], img_gt_single.shape[1]
            camera['fx'] = intrinsics_B33[i, 0, 0].item()
            camera['fy'] = intrinsics_B33[i, 1, 1].item()
            camera['cx'] = intrinsics_B33[i, 0, 2].item()
            camera['cy'] = intrinsics_B33[i, 1, 2].item()

            # 2) render with predict pose
            camera['world_to_camera'] = predict_poses_B44[i].inverse()[:3, :].to(device)

            w2c_r, w2c_t = camera['world_to_camera'].split(3, dim=1)
            w2c_r.requires_grad = True
            w2c_t.requires_grad = True
            optimizer = torch.optim.Adam([{"params": w2c_r, "lr": opt.refine_lr_r}, {"params": w2c_t, "lr": opt.refine_lr_t}])
            iterations = opt.refine_iters
            for iter in range(iterations):
                img_render_predict, _ = gsplat.forward(camera, w2c_r, w2c_t)
                img_render_predict[~valid_mask] = 0  # valid mask
                Ll1_predict = torch.abs(img_gt_single - img_render_predict).mean()
                simloss_predict = 1 - ssim(img_gt_single.permute(2, 0, 1)[None, ...], img_render_predict.permute(2, 0, 1)[None, ...])
                loss_photo_predict = ((1 - 0.2) * Ll1_predict + 0.2 * simloss_predict)
                # loss_photo_predict = mse(img_gt_single, img_render_predict)
                optimizer.zero_grad()
                loss_photo_predict.backward()
                optimizer.step()
                if iter == 0:
                    _logger.info(f"--> Before refine: Loss: {loss_photo_predict:.4f}")
                if iter == iterations-1:
                    _logger.info(f"--> After refine: Loss: {loss_photo_predict:.4f}")
                # if t_err * 100 > 1.0:
                #     _logger.info(f"Iteration {iter + 1}/{20}, Loss: {loss_photo_predict.item():.4f}")
                # save images
                # if iter == iterations-1 or iter == 0:
                #     # print("==============", opt.save_image)
                #     img_gt_save = (img_gt_single.detach().cpu().numpy() * 255).astype(np.uint8)
                #     img_render_save = (img_render_predict.detach().cpu().numpy() * 255).astype(np.uint8)
                #     cv2.imwrite(f"/home/yangmin/zhangfan/code/ace_gs/out_imgs/0428/img_gt_{global_count}_{iter}.png", img_gt_save[:, :, ::-1])
                #     cv2.imwrite(f"/home/yangmin/zhangfan/code/ace_gs/out_imgs/0428/img_render_{global_count}_{iter}.png", img_render_save[:, :, ::-1])
            # '''

            time_6 = time.time()
            _logger.info(f'Time of gaussian iters: {(time_6-time_5)*1000:.0f}ms')

            '''
            # ============================== reprojection error ==========================================
            # Scene coordinates to camera coordinates.
            pred_cam_coords_b31 = torch.matmul(camera['world_to_camera'].detach().cpu(), pred_scene_coords_b41)
            # Project scene coordinates.
            pred_px_b31 = torch.matmul(Ks_b33, pred_cam_coords_b31)
            pred_px_b31[:, 2].clamp_(min=0.1)
            # Dehomogenise.
            pred_px_b21 = pred_px_b31[:, :2] / pred_px_b31[:, 2, None]
            # Measure reprojection error.
            image_pixel = ace_util.get_pixel_grid(1)
            image_pixel_2HW = image_pixel[:, : H, : W].clone()
            image_pixel_2N = image_pixel_2HW.view(2, -1)
            target_px_b2 = image_pixel_2N.transpose(1, 0)
            reprojection_error_b2 = pred_px_b21.squeeze() - target_px_b2
            reprojection_error_b1 = torch.norm(reprojection_error_b2, dim=1, keepdim=True, p=1)
             # Predicted coordinates behind or close to camera plane.
            invalid_min_depth_b1 = pred_cam_coords_b31[:, 2] < 0.1
            # Very large reprojection errors.
            # method 1. Select the half of the coordinates with the smaller reprojection loss.
            repro_error_90precent = torch.sort(reprojection_error_b1[:, 0]).values[int(reprojection_error_b1.shape[0]*0.2)]
            invalid_repro_b1 = reprojection_error_b1 > repro_error_90precent
            # method 2. Select the part of coordinates where the reprojection loss is less than the threshold.
            # invalid_repro_b1 = reprojection_error_b1 > opt.refine_threshold
            # Predicted coordinates beyond max distance.
            invalid_max_depth_b1 = pred_cam_coords_b31[:, 2] > 1000
            # Invalid mask is the union of all these. Valid mask is the opposite.
            invalid_mask_b1 = (invalid_min_depth_b1 | invalid_repro_b1 | invalid_max_depth_b1)
            valid_mask_b1 = ~invalid_mask_b1
            repro_mean = torch.mean(reprojection_error_b1[valid_mask_b1])
            # _logger.info(f"--> After refine: Mean Repro Error: {repro_mean:.2f}")
            # ===========================================================================================
            '''

            # Calculate translation error.
            w2c = torch.eye(4, device='cpu', dtype=torch.float32)
            w2c[:3, :] = camera['world_to_camera']
            c2w = w2c.inverse()
            t_err = float(torch.norm(gt_pose_B44[i, 0:3, 3] - c2w[0:3, 3]))
            # _logger.info(f"{gt_pose_B44[i]}, {c2w}")
            # Rotation error.
            gt_R = gt_pose_B44[i, 0:3, 0:3].detach().cpu().numpy()
            out_R = c2w[0:3, 0:3].detach().cpu().numpy()
            r_err = np.matmul(out_R, np.transpose(gt_R))
            # Compute angle-axis representation.
            r_err = cv2.Rodrigues(r_err)[0]
            # Extract the angle.
            r_err = np.linalg.norm(r_err) * 180 / math.pi
            _logger.info(f"--> After refine: Rotation Error: {r_err:.2f}deg, Translation Error: {t_err * 100:.2f}cm  <--")

            # Save the errors.
            rErrs.append(r_err)
            tErrs.append(t_err * 100)

            # Check various thresholds.
            if r_err < 5 and t_err < 0.1:  # 10cm/5deg
                pct10_5 += 1
            if r_err < 5 and t_err < 0.05:  # 5cm/5deg
                pct5 += 1
            if r_err < 2 and t_err < 0.02:  # 2cm/2deg
                pct2 += 1
            if r_err < 1 and t_err < 0.01:  # 1cm/1deg
                pct1 += 1

            # Write estimated pose to pose file (inverse).
            # Translation.
            t = w2c[0:3, 3]

            # Rotation to axis angle.
            rot, _ = cv2.Rodrigues(w2c[0:3, 0:3].numpy())
            angle = np.linalg.norm(rot)
            axis = rot / angle

            # Axis angle to quaternion.
            q_w = math.cos(angle * 0.5)
            q_xyz = math.sin(angle * 0.5) * axis

            # Write to output file. All in a single line.
            pose_log.write(f"{frame_name} "
                            f"{q_w} {q_xyz[0].item()} {q_xyz[1].item()} {q_xyz[2].item()} "
                            f"{t[0]} {t[1]} {t[2]} "
                            f"{r_err} {t_err} {inlier_count}\n")

            # # save images
            # img_gt_single = (img_gt_single.detach().cpu().numpy() * 255).astype(np.uint8)
            # img_render_predict = (img_render_predict.detach().cpu().numpy() * 255).astype(np.uint8)
            # cv2.imwrite(f"{opt.save_image}/img_{global_count}_gt.png", img_gt_single[:, :, ::-1])
            # cv2.imwrite(f"{opt.save_image}/img_{global_count}_render_predict.png", img_render_predict[:, :, ::-1])
            
            global_count += 1
            
            # gs_end_time = time.time()
            # _logger.info(f'gs in {gs_end_time - gs_start_time:.2f}s')

            time_7 = time.time()
            _logger.info(f'Time of cal pose error: {(time_7-time_6)*1000:.0f}ms')
        
        _logger.info(f'Time of per frame: {(time_7-time_0)*1000:.0f}ms')
            


    total_frames = len(rErrs)
    # assert total_frames == len(testset)

    # Compute median errors.
    tErrs.sort()
    rErrs.sort()
    median_idx = total_frames // 2
    median_rErr = rErrs[median_idx]
    median_tErr = tErrs[median_idx]

    tErrs_before.sort()
    rErrs_before.sort()
    for i in range(total_frames):
        _logger.info(f'{i}, {rErrs_before[i]}, {rErrs[i]}, {tErrs_before[i]}, {tErrs[i]}')

    # Compute average time.
    avg_time = avg_batch_time / num_batches

    # Compute final metrics.
    pct10_5 = pct10_5 / total_frames * 100
    pct5 = pct5 / total_frames * 100
    pct2 = pct2 / total_frames * 100
    pct1 = pct1 / total_frames * 100

    _logger.info("===================================================")
    _logger.info("Test complete.")

    _logger.info('Accuracy:')
    _logger.info(f"Before refine:")
    pct10_5_before = 0
    pct5_before = 0
    pct2_before = 0
    pct1_before = 0
    for i in range(total_frames):
        r_err_b = rErrs_before[i]
        t_err_b = tErrs_before[i] / 100
        if r_err_b < 5 and t_err_b < 0.1:  # 10cm/5deg
            pct10_5_before += 1
        if r_err_b < 5 and t_err_b < 0.05:  # 5cm/5deg
            pct5_before += 1
        if r_err_b < 2 and t_err_b < 0.02:  # 2cm/2deg
            pct2_before += 1
        if r_err_b < 1 and t_err_b < 0.01:  # 1cm/1deg
            pct1_before += 1
    pct10_5_before = pct10_5_before / total_frames * 100
    pct5_before = pct5_before / total_frames * 100
    pct2_before = pct2_before / total_frames * 100
    pct1_before = pct1_before / total_frames * 100
    _logger.info(f'\t10cm/5deg: {pct10_5_before:.2f}%')
    _logger.info(f'\t5cm/5deg: {pct5_before:.2f}%')
    _logger.info(f'\t2cm/2deg: {pct2_before:.2f}%')
    _logger.info(f'\t1cm/1deg: {pct1_before:.2f}%')
    _logger.info(f"Median Error: {rErrs_before[median_idx]:.2f}deg, {tErrs_before[median_idx]:.2f}cm")

    _logger.info(f"After refine:")
    _logger.info(f'\t10cm/5deg: {pct10_5:.2f}%')
    _logger.info(f'\t5cm/5deg: {pct5:.2f}%')
    _logger.info(f'\t2cm/2deg: {pct2:.2f}%')
    _logger.info(f'\t1cm/1deg: {pct1:.2f}%')
    _logger.info(f"Median Error: {median_rErr:.2f}deg, {median_tErr:.2f}cm")

    _logger.info(f"Avg. processing time: {avg_time * 1000:4.1f}ms")

    # Write to the test log file as well.
    test_log.write(f"{median_rErr} {median_tErr} {avg_time}\n")

    test_log.close()
    pose_log.close()

    
    # import matplotlib.pyplot as plt
    # x = range(len(loss_photo_all['gt']['l1']))
    # fig = plt.figure(figsize=(20, 10))
    # plt.plot(x, loss_photo_all['gt']['l1'], color='#000000', label='gt_l1')
    # plt.plot(x, loss_photo_all['predict']['l1'], color='#0000FF', label='predict_l1')
    # plt.plot(x, loss_photo_all['gt']['mse'], color='#FF0000', label='gt_mse')
    # plt.plot(x, loss_photo_all['predict']['mse'], color='#00FF00', label='predict_mse')
    # plt.plot(x, loss_photo_all['gt']['gaussian'], color='#FFFF00', label='gt_gaussian')
    # plt.plot(x, loss_photo_all['predict']['gaussian'], color='#00FFFF', label='predict_gaussian')
    # plt.legend()
    # plt.savefig(f'{opt.save_image}/loss.png', dpi=300)

    # bool_l1 = np.array(loss_photo_all['gt']['l1']) > np.array(loss_photo_all['predict']['l1'])
    # bool_mse = np.array(loss_photo_all['gt']['mse']) > np.array(loss_photo_all['predict']['mse'])
    # bool_gaussian = np.array(loss_photo_all['gt']['gaussian']) > np.array(loss_photo_all['predict']['gaussian'])
    # count_l1 = np.sum(bool_l1)
    # count_mse = np.sum(bool_mse)
    # count_gaussian = np.sum(bool_gaussian)
    # _logger.info(f"Total gt loss > predict loss: l1: {count_l1}, mse: {count_mse}, gaussian: {count_gaussian}")
