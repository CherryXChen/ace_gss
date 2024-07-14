# Copyright © Niantic, Inc. 2022.

import logging
import random
import time
import datetime
import pickle
import pathlib

import numpy as np
from pandas import options
import torch
import torch.optim as optim
import torchvision.transforms.functional as TF
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from torch.utils.data import sampler
from pytorch_msssim import SSIM
from torch.utils.tensorboard import SummaryWriter   

from ace_util import get_pixel_grid, to_homogeneous
from ace_loss import ReproLoss
from ace_network import Regressor
from dataset import CamLocDataset

import ace_vis_util as vutil
from ace_visualizer import ACEVisualizer

from gsplat_ace import GSplat

# ctr: initialize the logger
_logger = logging.getLogger(__name__)
_logger.setLevel(logging.INFO)
_fh = logging.FileHandler('train.log', encoding='utf8', mode='w')
_logger.addHandler(_fh)

# ctr: initialize the tensorboard
time_stamp = datetime.datetime.now()
tb_name = 'tb' + time_stamp.strftime('%y_%m_%d_%H_%M')
writer = SummaryWriter('./logs/tb/' + tb_name)

def set_seed(seed):
    """
    Seed all sources of randomness.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

class TrainerACE:
    def __init__(self, options):
        self.options = options

        self.device = torch.device('cuda')

        # The flag below controls whether to allow TF32 on matmul. This flag defaults to True.
        # torch.backends.cuda.matmul.allow_tf32 = False

        # The flag below controls whether to allow TF32 on cuDNN. This flag defaults to True.
        # torch.backends.cudnn.allow_tf32 = False
        
        # reset the generator
        self.reset_generator()

        self.iteration = 0
        self.training_start = None
        self.num_data_loader_workers = 12

        # Create dataset.
        self.dataset = CamLocDataset(
            root_dir=self.options.scene / "train",
            mode=0,  # Default for ACE, we don't need scene coordinates/RGB-D.
            use_half=self.options.use_half,
            image_height=self.options.image_resolution,
            augment=self.options.use_aug,
            aug_rotation=self.options.aug_rotation,
            aug_scale_max=self.options.aug_scale,
            aug_scale_min=1 / self.options.aug_scale,
            num_clusters=self.options.num_clusters,  # Optional clustering for Cambridge experiments.
            cluster_idx=self.options.cluster_idx,    # Optional clustering for Cambridge experiments.
        )

        _logger.info("Loaded training scan from: {} -- {} images, mean: {:.2f} {:.2f} {:.2f}".format(
            self.options.scene,
            len(self.dataset),
            self.dataset.mean_cam_center[0],
            self.dataset.mean_cam_center[1],
            self.dataset.mean_cam_center[2]))

        if self.options.pretrain_head_path is not None:
            encoder_state_dict = torch.load(self.options.encoder_path, map_location="cpu")
            head_state_dict = torch.load(self.options.pretrain_head_path, map_location="cpu")
            self.regressor = Regressor.create_from_split_state_dict(encoder_state_dict, head_state_dict)
            _logger.info(f"Loaded pretrained encoder from: {self.options.encoder_path}")
            _logger.info(f"Loaded pretrained head from: {self.options.pretrain_head_path}")
        else:
            # Create network using the state dict of the pretrained encoder.
            encoder_state_dict = torch.load(self.options.encoder_path, map_location="cpu")
            self.regressor = Regressor.create_from_encoder(
                encoder_state_dict,
                mean=self.dataset.mean_cam_center,
                num_head_blocks=self.options.num_head_blocks,
                use_homogeneous=self.options.use_homogeneous
            )
            _logger.info(f"Loaded pretrained encoder from: {self.options.encoder_path}")
        

        self.regressor = self.regressor.to(self.device)
        self.regressor.train()

        # Setup optimization parameters.
        # self.optimizer = optim.AdamW(self.regressor.parameters(), lr=self.options.learning_rate_min, eps=1e-5)
        self.optimizer = optim.Adam(self.regressor.parameters(), lr=self.options.learning_rate_max, eps=1e-5)

        # Setup learning rate scheduler.
        steps_per_epoch = len(self.dataset.rgb_files) // self.options.batch_size
        self.scheduler = optim.lr_scheduler.OneCycleLR(self.optimizer,
                                                       max_lr=self.options.learning_rate_max,
                                                       epochs=self.options.epochs,
                                                       steps_per_epoch=steps_per_epoch,
                                                       pct_start=0.1,
                                                       cycle_momentum=False)

        self.ssim = SSIM(data_range=1.0, size_average=True, channel=3)

        # Gradient scaler in case we train with half precision.
        self.scaler = GradScaler(enabled=self.options.use_half)

        # Generate grid of target reprojection pixel positions.
        pixel_grid_2HW = get_pixel_grid(self.regressor.OUTPUT_SUBSAMPLE)
        self.pixel_grid_2HW = pixel_grid_2HW.to(self.device)

        # Compute total number of iterations.
        self.iterations = self.options.epochs * len(self.dataset.rgb_files) // self.options.batch_size
        self.iterations_output = 100 # print loss every n iterations, and (optionally) write a visualisation frame

        # Setup reprojection loss function.
        self.repro_loss = ReproLoss(
            total_iterations=self.iterations,
            soft_clamp=self.options.repro_loss_soft_clamp,
            soft_clamp_min=self.options.repro_loss_soft_clamp_min,
            type=self.options.repro_loss_type,
            circle_schedule=(self.options.repro_loss_schedule == 'circle')
        )

        # Will be filled at the beginning of the training process.
        self.training_buffer = None

        # Generate video of training process
        if self.options.render_visualization:
            # infer rendering folder from map file name
            target_path = vutil.get_rendering_target_path(
                self.options.render_target_path,
                self.options.output_map_file)
            self.ace_visualizer = ACEVisualizer(
                target_path,
                self.options.render_flipped_portrait,
                self.options.render_map_depth_filter,
                mapping_vis_error_threshold=self.options.render_map_error_threshold)
        else:
            self.ace_visualizer = None

    def reset_generator(self):
        # Setup randomness for reproducibility.
        self.base_seed = 2089
        set_seed(self.base_seed)

        # Used to generate batch indices.
        self.batch_generator = torch.Generator()
        self.batch_generator.manual_seed(self.base_seed + 1023)

        # Dataloader generator, used to seed individual workers by the dataloader.
        self.loader_generator = torch.Generator()
        self.loader_generator.manual_seed(self.base_seed + 511)

        # Generator used to sample random features (runs on the GPU).
        self.sampling_generator = torch.Generator(device=self.device)
        self.sampling_generator.manual_seed(self.base_seed + 4095)

        # Generator used to permute the feature indices during each training epoch.
        self.training_generator = torch.Generator()
        self.training_generator.manual_seed(self.base_seed + 8191)

    def train(self):
        """
        Main training method.

        Fills a feature buffer using the pretrained encoder and subsequently trains a scene coordinate regression head.
        """

        if self.ace_visualizer is not None:

            # Setup the ACE render pipeline.
            self.ace_visualizer.setup_mapping_visualisation(
                self.dataset.pose_files,
                self.dataset.rgb_files,
                self.iterations // self.iterations_output + 1,
                self.options.render_camera_z_offset
            )

        training_time = 0.

        self.training_start = time.time()

        # ctr: ectract features
        # buffer_start_time = time.time()
        # self.ectract_features()
        # buffer_end_time = time.time()
        # creating_buffer_time += buffer_end_time - buffer_start_time
        # _logger.info(f"Extraction costs {buffer_end_time - buffer_start_time:.1f}s.")

        # Create training buffer.
        # buffer_start_time = time.time()
        # self.create_training_buffer()
        # buffer_end_time = time.time()
        # creating_buffer_time += buffer_end_time - buffer_start_time
        # _logger.info(f"Filled training buffer in {buffer_end_time - buffer_start_time:.1f}s.")

        # Train the regression head.
        # self.reset_generator()

        batch_sampler = sampler.BatchSampler(sampler.RandomSampler(self.dataset, generator=self.batch_generator),
                                             batch_size=self.options.batch_size,
                                             drop_last=True)
        def seed_worker(worker_id):
            # Different seed per epoch. Initial seed is generated by the main process consuming one random number from
            # the dataloader generator.
            worker_seed = torch.initial_seed() % 2 ** 32
            np.random.seed(worker_seed)
            random.seed(worker_seed)

        # Batching is handled at the dataset level (the dataset __getitem__ receives a list of indices, because we
        # need to rescale all images in the batch to the same size).
        self.training_dataloader = DataLoader(dataset=self.dataset,
                                         sampler=batch_sampler,
                                         batch_size=None,
                                         worker_init_fn=seed_worker,
                                         generator=self.loader_generator,
                                         pin_memory=True,
                                         num_workers=self.num_data_loader_workers,
                                         persistent_workers=self.num_data_loader_workers > 0,
                                         timeout=60 if self.num_data_loader_workers > 0 else 0,
                                         )

        for self.epoch in range(self.options.epochs):
            epoch_start_time = time.time()
            self.run_epoch()
            training_time += time.time() - epoch_start_time
            # self.save_model_epoch(pathlib.Path(f'{self.options.output_map_file}_epoch{self.epoch}'))

        # Save trained model.
        self.save_model()

        end_time = time.time()
        _logger.info(f'Done without errors. '
                     f'Training time: {training_time:.1f} seconds. '
                     f'Total time: {end_time - self.training_start:.1f} seconds.')

        if self.ace_visualizer is not None:

            # Finalize the rendering by animating the fully trained map.
            vis_dataset = CamLocDataset(
                root_dir=self.options.scene / "train",
                mode=0,
                use_half=self.options.use_half,
                image_height=self.options.image_resolution,
                augment=False) # No data augmentation when visualizing the map

            vis_dataset_loader = torch.utils.data.DataLoader(
                vis_dataset,
                shuffle=False, # Process data in order for a growing effect later when rendering
                num_workers=self.num_data_loader_workers)

            self.ace_visualizer.finalize_mapping(self.regressor, vis_dataset_loader)
    

    # def ectract_features_save(self):
    #     torch.backends.cudnn.benchmark = False
    #     batch_sampler = sampler.BatchSampler(sampler.RandomSampler(self.dataset, generator=self.batch_generator),
    #                                          batch_size=1,
    #                                          drop_last=False)
    #     def seed_worker(worker_id):
    #         # Different seed per epoch. Initial seed is generated by the main process consuming one random number from
    #         # the dataloader generator.
    #         worker_seed = torch.initial_seed() % 2 ** 32
    #         np.random.seed(worker_seed)
    #         random.seed(worker_seed)

    #     # Batching is handled at the dataset level (the dataset __getitem__ receives a list of indices, because we
    #     # need to rescale all images in the batch to the same size).
    #     training_dataloader = DataLoader(dataset=self.dataset,
    #                                      sampler=batch_sampler,
    #                                      batch_size=None,
    #                                      worker_init_fn=seed_worker,
    #                                      generator=self.loader_generator,
    #                                      pin_memory=True,
    #                                      num_workers=self.num_data_loader_workers,
    #                                      persistent_workers=self.num_data_loader_workers > 0,
    #                                      timeout=60 if self.num_data_loader_workers > 0 else 0,
    #                                      )
        
    #     _logger.info(f"Starting extracting features.")
    #     self.regressor.eval()

    #     image_cnt = 1
    #     with torch.no_grad():
    #         for image_B1HW, _, _, _, _, _, _, image_file in training_dataloader:
    #             image_B1HW = image_B1HW.to(self.device, non_blocking=True)

    #             with autocast(enabled=self.options.use_half):
    #                 features_BCHW = self.regressor.get_features(image_B1HW)
    #                 file_name = image_file[0] + str(image_B1HW.shape[2]) + '_' + str(image_B1HW.shape[3]) + '.pkl'
    #                 file_path = self.options.scene / 'train' / 'feature' / file_name
    #                 with open(file_path, 'wb') as file:
    #                     pickle.dump(features_BCHW, file)
    #                 image_cnt += 1
    #     _logger.info(f"Totally generate {image_cnt} features.")

    #     self.regressor.train()

    def extract_features_direct(self, image_B1HW):
        self.regressor.eval()

        with torch.no_grad():
            image_B1HW = image_B1HW.to(self.device, non_blocking=True)

            with autocast(enabled=self.options.use_half):
                features_BCHW, img_feat = self.regressor.get_features(image_B1HW)

        self.regressor.train()

        return features_BCHW, img_feat

    def run_epoch(self):
        """
        Run one epoch of training.
        """
        torch.backends.cudnn.benchmark = False
        # batch_sampler = sampler.BatchSampler(sampler.RandomSampler(self.dataset, generator=self.batch_generator),
        #                                      batch_size=self.options.batch_size,
        #                                      drop_last=False)
        # def seed_worker(worker_id):
        #     # Different seed per epoch. Initial seed is generated by the main process consuming one random number from
        #     # the dataloader generator.
        #     worker_seed = torch.initial_seed() % 2 ** 32
        #     np.random.seed(worker_seed)
        #     random.seed(worker_seed)

        # # Batching is handled at the dataset level (the dataset __getitem__ receives a list of indices, because we
        # # need to rescale all images in the batch to the same size).
        # training_dataloader = DataLoader(dataset=self.dataset,
        #                                  sampler=batch_sampler,
        #                                  batch_size=None,
        #                                  worker_init_fn=seed_worker,
        #                                  generator=self.loader_generator,
        #                                  pin_memory=True,
        #                                  num_workers=self.num_data_loader_workers,
        #                                  persistent_workers=self.num_data_loader_workers > 0,
        #                                  timeout=60 if self.num_data_loader_workers > 0 else 0,
        #                                  )
        for image_BHW3, image_B1HW, image_mask_B1HW, gt_pose_B44, gt_pose_inv_B44, intrinsics_B33, intrinsics_inv_B33, _, image_name in self.training_dataloader:
            # Copy to device.
            image_BHW3 = image_BHW3.to(self.device, non_blocking=True)
            image_B1HW = image_B1HW.to(self.device, non_blocking=True)
            image_mask_B1HW = image_mask_B1HW.to(self.device, non_blocking=True)
            gt_pose_inv_B44 = gt_pose_inv_B44.to(self.device, non_blocking=True)
            intrinsics_B33 = intrinsics_B33.to(self.device, non_blocking=True)
            intrinsics_inv_B33 = intrinsics_inv_B33.to(self.device, non_blocking=True)

            # extract features
            # ctr: add time consuming
            extract_start_time = time.time()
            features_BCHW, img_feat = self.extract_features_direct(image_B1HW)
            extract_end_time = time.time()
            # _logger.info(f'Image_name: {image_name[0]}, Extract features in {extract_end_time - extract_start_time:.2f}s')

            B, C, _, _ = features_BCHW.shape
            H = image_B1HW.shape[2] // self.regressor.OUTPUT_SUBSAMPLE  # Head module includes upsample
            W = image_B1HW.shape[3] // self.regressor.OUTPUT_SUBSAMPLE

            # The image_mask needs to be downsampled to the actual output resolution and cast to bool.
            image_mask_B1HW = TF.resize(image_mask_B1HW, [H, W], interpolation=TF.InterpolationMode.NEAREST)
            image_mask_B1HW = image_mask_B1HW.bool()

            # If the current mask has no valid pixels, continue.
            if image_mask_B1HW.sum() == 0:
                continue

            # Create a tensor with the pixel coordinates of every feature vector.
            pixel_positions_B2HW = self.pixel_grid_2HW[:, :H, :W].clone()  # It's 2xHxW (actual H and W) now.
            pixel_positions_B2HW = pixel_positions_B2HW[None]  # 1x2xHxW
            pixel_positions_B2HW = pixel_positions_B2HW.expand(B, 2, H, W)  # Bx2xHxW

            # Bx3x4 -> Nx3x4 (for each image, repeat pose per feature)
            gt_pose = gt_pose_B44[:, :3]
            gt_pose = gt_pose.unsqueeze(1).expand(B, H * W, 3, 4).reshape(-1, 3, 4)
            gt_pose_inv = gt_pose_inv_B44[:, :3]
            gt_pose_inv = gt_pose_inv.unsqueeze(1).expand(B, H * W, 3, 4).reshape(-1, 3, 4)


            # Bx3x3 -> Nx3x3 (for each image, repeat intrinsics per feature)
            intrinsics = intrinsics_B33.unsqueeze(1).expand(B, H * W, 3, 3).reshape(-1, 3, 3)
            intrinsics_inv = intrinsics_inv_B33.unsqueeze(1).expand(B, H * W, 3, 3).reshape(-1, 3, 3)

            def normalize_shape(tensor_in):
                """Bring tensor from shape BxCxHxW to NxC"""
                return tensor_in.transpose(0, 1).flatten(1).transpose(0, 1)

            batch_data = {
                'img_gt': image_BHW3,
                'img_gray': image_B1HW,
                'features': normalize_shape(features_BCHW),
                'img_feat': img_feat,
                'target_px': normalize_shape(pixel_positions_B2HW),
                'gt_poses': gt_pose,
                'gt_poses_inv': gt_pose_inv,
                'intrinsics': intrinsics,
                'intrinsics_inv': intrinsics_inv
            }

            # Turn image mask into sampling weights (all equal).
            image_mask_B1HW = image_mask_B1HW.float()
            image_mask_N1 = normalize_shape(image_mask_B1HW)

            # Over-sample according to image mask.
            features_to_select = self.options.samples_per_image * B

            # Sample indices uniformly, with replacement.
            sample_idxs = torch.multinomial(image_mask_N1.view(-1),
                                            features_to_select,
                                            replacement=True,
                                            generator=self.sampling_generator)
            
            # for k in batch_data:
            #     batch_data[k] = batch_data[k][sample_idxs]
            
            if not self.options.use_aug:
                torch.backends.cudnn.benchmark = True
            self.training_step(
                batch_data['img_gt'].contiguous(),
                batch_data['img_gray'].contiguous(),
                batch_data['features'].contiguous(),
                batch_data['img_feat'],
                batch_data['target_px'].contiguous(),
                batch_data['gt_poses'].contiguous(),
                batch_data['gt_poses_inv'].contiguous(),
                batch_data['intrinsics'].contiguous(),
                batch_data['intrinsics_inv'].contiguous(),
                H,
                W,
                features_BCHW.shape[2],
                features_BCHW.shape[3],
                image_name
            )
            self.iteration += 1
            

    def training_step(self, img_gt, img_gray, features_bC, img_feat, target_px_b2, gt_poses_b34, gt_inv_poses_b34, Ks_b33, invKs_b33, H, W, H_feat, W_feat, image_name):
        """
        Run one iteration of training, computing the reprojection error and minimising it.
        """

        batch_size = features_bC.shape[0]  # B*H*W
        batch_size = batch_size * 8 * 8
        channels = features_bC.shape[1]

        # Reshape to a BCHW shape, since it's faster to run through the network compared to the original shape.
        features_bCHW = features_bC[None, None, ...].view(-1, H_feat, W_feat, channels).permute(0, 3, 1, 2)

        # ctr: add time consuming.
        predict_start_time = time.time()
        with autocast(enabled=self.options.use_half):

            with torch.no_grad():
                pred_scene_coords, pred_feature = self.regressor.get_scene_coordinates(features_bCHW)
            
            pred_scene_coords_b3HW = self.regressor.upsample_scene_coordinates(pred_scene_coords, pred_feature)
            
            # regress gaussian parameters
            rot_maps, scale_maps, opacity_maps = self.regressor.get_gs_params(img_gray, pred_scene_coords_b3HW, img_feat)

        predict_end_time = time.time()
        # _logger.info(f'Predict coordinates in {predict_end_time - predict_start_time:.2f}s')

        # Back to the original shape. Convert to float32 as well.
        pred_scene_coords_b31 = pred_scene_coords_b3HW.permute(0, 2, 3, 1).flatten(0, 2).unsqueeze(-1).float()

        # Make 3D points homogeneous so that we can easily matrix-multiply them.
        pred_scene_coords_b41 = to_homogeneous(pred_scene_coords_b31)

        # Scene coordinates to camera coordinates.
        pred_cam_coords_b31 = torch.bmm(gt_inv_poses_b34, pred_scene_coords_b41)

        # Project scene coordinates.
        pred_px_b31 = torch.bmm(Ks_b33, pred_cam_coords_b31)

        # Avoid division by zero.
        # Note: negative values are also clamped at +self.options.depth_min. The predicted pixel would be wrong,
        # but that's fine since we mask them out later.
        pred_px_b31[:, 2].clamp_(min=self.options.depth_min)

        # Dehomogenise.
        pred_px_b21 = pred_px_b31[:, :2] / pred_px_b31[:, 2, None]

        # Measure reprojection error.
        reprojection_error_b2 = pred_px_b21.squeeze() - target_px_b2
        reprojection_error_b1 = torch.norm(reprojection_error_b2, dim=1, keepdim=True, p=1)

        mask_epoch = self.options.epochs // 4 * 3

        if self.epoch < mask_epoch:
            #
            # Compute masks used to ignore invalid pixels.
            #
            # Predicted coordinates behind or close to camera plane.
            invalid_min_depth_b1 = pred_cam_coords_b31[:, 2] < self.options.depth_min

            # Very large reprojection errors.
            invalid_repro_b1 = reprojection_error_b1 > self.options.repro_loss_hard_clamp
            # Predicted coordinates beyond max distance.
            invalid_max_depth_b1 = pred_cam_coords_b31[:, 2] > self.options.depth_max

            # Invalid mask is the union of all these. Valid mask is the opposite.
            invalid_mask_b1 = (invalid_min_depth_b1 | invalid_repro_b1 | invalid_max_depth_b1)
            valid_mask_b1 = ~invalid_mask_b1

            # Reprojection error for all valid scene coordinates.
            valid_reprojection_error_b1 = reprojection_error_b1[valid_mask_b1]
            # Compute the loss for valid predictions.
            loss_valid = self.repro_loss.compute(valid_reprojection_error_b1, self.iteration)

            # Handle the invalid predictions: generate proxy coordinate targets with constant depth assumption.
            pixel_grid_crop_b31 = to_homogeneous(target_px_b2.unsqueeze(2))
            target_camera_coords_b31 = self.options.depth_target * torch.bmm(invKs_b33, pixel_grid_crop_b31)

            # Compute the distance to target camera coordinates.
            invalid_mask_b11 = invalid_mask_b1.unsqueeze(2)
            loss_invalid = torch.abs(target_camera_coords_b31 - pred_cam_coords_b31).masked_select(invalid_mask_b11).sum()

            loss = loss_valid + loss_invalid
            loss /= batch_size
            loss_valid_mean = loss_valid / batch_size
            loss_invalid_mean = loss_invalid / batch_size

        else:
            #
            # Compute masks used to ignore invalid pixels.
            #
            # Predicted coordinates behind or close to camera plane.
            invalid_min_depth_b1 = pred_cam_coords_b31[:, 2] < self.options.depth_min

            # Very large reprojection errors.
            repro_loss_90precent = torch.sort(reprojection_error_b1[:, 0]).values[int(reprojection_error_b1.shape[0]*0.8)]
            invalid_repro_b1 = reprojection_error_b1 > repro_loss_90precent

            # Predicted coordinates beyond max distance.
            invalid_max_depth_b1 = pred_cam_coords_b31[:, 2] > self.options.depth_max

            # Invalid mask is the union of all these. Valid mask is the opposite.
            invalid_mask_b1 = (invalid_min_depth_b1 | invalid_repro_b1 | invalid_max_depth_b1)
            valid_mask_b1 = ~invalid_mask_b1
            valid_mask_BN = valid_mask_b1.reshape((self.options.batch_size, H*W)).to(self.device)
            valid_mask_BHW3 = valid_mask_b1.reshape((self.options.batch_size, H, W, 1)).repeat(1, 1, 1, 3).to(self.device)

            # Reprojection error for all valid scene coordinates.
            valid_reprojection_error_b1 = reprojection_error_b1[valid_mask_b1]
            # Compute the loss for valid predictions.
            loss_valid = self.repro_loss.compute(valid_reprojection_error_b1, self.iteration)

            # Handle the invalid predictions: generate proxy coordinate targets with constant depth assumption.
            pixel_grid_crop_b31 = to_homogeneous(target_px_b2.unsqueeze(2))
            target_camera_coords_b31 = self.options.depth_target * torch.bmm(invKs_b33, pixel_grid_crop_b31)

            # Compute the distance to target camera coordinates.
            invalid_mask_b11 = invalid_mask_b1.unsqueeze(2)
            loss_invalid = torch.abs(target_camera_coords_b31 - pred_cam_coords_b31).masked_select(invalid_mask_b11).sum()

            # loss = loss_valid
            loss = loss_valid / valid_mask_b1.sum()
            loss_valid_mean = loss_valid / valid_mask_b1.sum()
            loss_invalid_mean = loss_invalid / valid_mask_b1.sum()

        loss_photo_batch = 0.0
        if self.options.open_gs and self.epoch>=0:
            # Init Gaissians per batch.
            # points_xyz = torch.zeros((self.options.batch_size, H*W, 3), device=pred_scene_coords_b3HW.device)
            # points_color = torch.zeros((self.options.batch_size, H*W, 3), device=img_gt.device)
            # for i in range(0, self.options.batch_size):
            #     points_xyz[i] = self.get_xyz_points(pred_scene_coords_b3HW[i])  # (num_points, 3)
            #     points_color[i] = self.get_color_points(img_gt[i].float(), pred_scene_coords_b3HW.shape[2], pred_scene_coords_b3HW.shape[3])  # (num_points, 3)
            # points_xyz = points_xyz.view(-1, 3)
            # points_color = points_color.view(-1, 3)
            # gsplat = GSplat(points_xyz, points_color)

            for i in range(0, self.options.batch_size):
                # Prepare the data for Gaussians
                # gs_start_time = time.time()
                
                # 1.Init Gaissians per image.
                img_gt_single = img_gt[i].float()
                points_xyz = self.convert_data_NC(pred_scene_coords_b3HW[i])  # (num_points, 3)
                points_color = self.get_color_points(img_gt_single, pred_scene_coords_b3HW.shape[2], pred_scene_coords_b3HW.shape[3])  # (num_points, 3)
                gs_rot = self.convert_data_NC(rot_maps[i])
                gs_scale = self.convert_data_NC(scale_maps[i])
                gs_opacity = self.convert_data_NC(opacity_maps[i])

                if self.epoch >= mask_epoch:
                    # valid mask for gaussians
                    points_xyz = points_xyz[valid_mask_BN[i, :], :]
                    points_color = points_color[valid_mask_BN[i, :], :]
                    gs_rot = gs_rot[valid_mask_BN[i, :], :]
                    gs_scale = gs_scale[valid_mask_BN[i, :], :]
                    gs_opacity = gs_opacity[valid_mask_BN[i, :], :]

                gsplat = GSplat(points_xyz, points_color, gs_rot, gs_scale, gs_opacity)
                # gs_end_time = time.time()
                # _logger.info(f'gs in {gs_end_time - gs_start_time:.2f}s')

                # 2.Feed features to predict.
                camera = {}
                camera['height'], camera['width'] = img_gt_single.shape[0], img_gt_single.shape[1]
                camera['fx'] = Ks_b33[i, 0, 0].item()
                camera['fy'] = Ks_b33[i, 1, 1].item()
                camera['cx'] = Ks_b33[i, 0, 2].item()
                camera['cy'] = Ks_b33[i, 1, 2].item()
                # camera['camera_to_world'] = gt_poses_b34[i*H*W]
                camera['world_to_camera'] = gt_inv_poses_b34[i*H*W]

                w2c_r, w2c_t = camera['world_to_camera'].split(3, dim=1)

                img_render, _ = gsplat.forward(camera, w2c_r, w2c_t)

                if self.epoch >= mask_epoch:
                    img_gt_single[~valid_mask_BHW3[i]] = 0  # valid mask
                    img_render[~valid_mask_BHW3[i]] = 0  # valid mask
                
                # 3.photo loss
                Ll1 = torch.abs(img_gt_single - img_render).mean()
                simloss = 1 - self.ssim(img_gt_single.permute(2, 0, 1)[None, ...], img_render.permute(2, 0, 1)[None, ...])
                loss_photo = ((1 - self.options.ssim_lambda) * Ll1 + self.options.ssim_lambda * simloss)
                loss_photo_batch += loss_photo
                # _logger.info(f'loss_photo: {loss_photo:.4f}')
                # # save images
                # if self.iteration % 1000 == 0:
                #     img_gt_single = (img_gt_single.detach().cpu().numpy() * 255).astype(np.uint8)
                #     img_render = (img_render.detach().cpu().numpy() * 255).astype(np.uint8)
                #     import cv2
                #     cv2.imwrite(f"out_imgs/train_0420/img_gt_{self.iteration}.png", img_gt_single[:, :, ::-1])
                #     cv2.imwrite(f"out_imgs/train_0420/img_render_{self.iteration}.png", img_render[:, :, ::-1])
                    # print('========================')
                    
                # gs_end_time = time.time()
                # _logger.info(f'gs in {gs_end_time - gs_start_time:.2f}s')


        if self.options.open_gs:
            loss = (1 - self.options.photo_loss_lambda) * loss + self.options.photo_loss_lambda * (loss_photo_batch / self.options.batch_size * 10)

        # We need to check if the step actually happened, since the scaler might skip optimisation steps.
        old_optimizer_step = self.optimizer._step_count

        # ctr: write loss and lr to tb
        writer.add_scalar('loss_valid', loss_valid_mean, self.iteration)
        writer.add_scalar('loss_invalid', loss_invalid_mean, self.iteration)
        writer.add_scalar('loss_photo', loss_photo_batch/self.options.batch_size, self.iteration)
        # writer.add_scalar('learning_rate', self.scheduler.get_last_lr()[0], self.iteration)
        # img_render = (img_render.detach().cpu().numpy() * 255).astype(np.uint8)
        # writer.add_images('rendered_image', img_render, self.iteration, dataformats='HWC')

        # Optimization steps.
        self.optimizer.zero_grad(set_to_none=True)
        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()

        if self.iteration % self.iterations_output == 0:
            # Print status.
            time_since_start = time.time() - self.training_start
            fraction_valid = float(valid_mask_b1.sum() / batch_size)
            # median_depth = float(pred_cam_coords_b31[:, 2].median())

            _logger.info(f'Iteration: {self.iteration:6d} / Epoch {self.epoch:03d}|{self.options.epochs:03d}, '
                         f'loss_valid: {loss_valid/batch_size:.1f}, loss_invalid: {loss_invalid/batch_size:.4f}, loss_photo: {loss_photo_batch/self.options.batch_size:.4f}, '
                         f'Loss: {loss:.1f}, Valid pc: {fraction_valid * 100:.1f}%, Time: {time_since_start:.2f}s')
            

            if self.ace_visualizer is not None:
                vis_scene_coords = pred_scene_coords_b31.detach().cpu().squeeze().numpy()
                vis_errors = reprojection_error_b1.detach().cpu().squeeze().numpy()
                self.ace_visualizer.render_mapping_frame(vis_scene_coords, vis_errors)

        # Only step if the optimizer stepped and if we're not over-stepping the total_steps supported by the scheduler.
        # if old_optimizer_step < self.optimizer._step_count < self.scheduler.total_steps:
        #     self.scheduler.step()

    def get_xyz_points(self, points):    
        # (3, H, W) -> (H*W, 3)
        xyz = points.permute(1, 2, 0)  # (3, H, W) -> (H, W, 3)
        xyz = xyz.view(-1, 3)  # (H*W, 3)

        # OpenCV to OpenGL convention
        # xyz[:, 1] = -xyz[:, 1]
        # xyz[:, 2] = -xyz[:, 2]
        return xyz

    def get_color_points(self, rgb, H, W):
        nn_stride = self.regressor.OUTPUT_SUBSAMPLE
        nn_offset = self.regressor.OUTPUT_SUBSAMPLE//2
        rgb = rgb[nn_offset::nn_stride, nn_offset::nn_stride, :]
        # make sure the resolution fits (catch any striding mismatches)
        rgb = rgb.resize_(H, W, 3)
        color = rgb.contiguous().view(-1, 3)

        return color
    
    def convert_data_NC(self, data):    
        # (C, H, W) -> (H*W, C)
        C, _, _ = data.shape
        data = data.permute(1, 2, 0)  # (C, H, W) -> (H, W, C)
        data = data.view(-1, C).to(torch.float32)  # (H*W, C)
        return data

    def create_training_buffer(self):
        # Disable benchmarking, since we have variable tensor sizes.
        torch.backends.cudnn.benchmark = False

        # Sampler.
        batch_sampler = sampler.BatchSampler(sampler.RandomSampler(self.dataset, generator=self.batch_generator),
                                             batch_size=1,
                                             drop_last=True)

        # Used to seed workers in a reproducible manner.
        def seed_worker(worker_id):
            # Different seed per epoch. Initial seed is generated by the main process consuming one random number from
            # the dataloader generator.
            worker_seed = torch.initial_seed() % 2 ** 32
            np.random.seed(worker_seed)
            random.seed(worker_seed)

        # Batching is handled at the dataset level (the dataset __getitem__ receives a list of indices, because we
        # need to rescale all images in the batch to the same size).
        training_dataloader = DataLoader(dataset=self.dataset,
                                         sampler=batch_sampler,
                                         batch_size=10,
                                         worker_init_fn=seed_worker,
                                         generator=self.loader_generator,
                                         pin_memory=True,
                                         num_workers=self.num_data_loader_workers,
                                         persistent_workers=self.num_data_loader_workers > 0,
                                         timeout=60 if self.num_data_loader_workers > 0 else 0,
                                         )

        _logger.info("Starting creation of the training buffer.")

        # Create a training buffer that lives on the GPU.
        self.training_buffer = {
            'features': torch.empty((self.options.training_buffer_size, self.regressor.feature_dim),
                                    dtype=(torch.float32, torch.float16)[self.options.use_half], device=self.device),
            'target_px': torch.empty((self.options.training_buffer_size, 2), dtype=torch.float32, device=self.device),
            'gt_poses_inv': torch.empty((self.options.training_buffer_size, 3, 4), dtype=torch.float32,
                                        device=self.device),
            'intrinsics': torch.empty((self.options.training_buffer_size, 3, 3), dtype=torch.float32,
                                      device=self.device),
            'intrinsics_inv': torch.empty((self.options.training_buffer_size, 3, 3), dtype=torch.float32,
                                          device=self.device)
        }

        # Features are computed in evaluation mode.
        self.regressor.eval()

        # The encoder is pretrained, so we don't compute any gradient.
        with torch.no_grad():
            # Iterate until the training buffer is full.
            buffer_idx = 0
            dataset_passes = 0

            while buffer_idx < self.options.training_buffer_size:
                dataset_passes += 1
                for image_B1HW, image_mask_B1HW, gt_pose_B44, gt_pose_inv_B44, intrinsics_B33, intrinsics_inv_B33, _, _ in training_dataloader:

                    # Copy to device.
                    image_B1HW = image_B1HW.to(self.device, non_blocking=True)
                    image_mask_B1HW = image_mask_B1HW.to(self.device, non_blocking=True)
                    gt_pose_inv_B44 = gt_pose_inv_B44.to(self.device, non_blocking=True)
                    intrinsics_B33 = intrinsics_B33.to(self.device, non_blocking=True)
                    intrinsics_inv_B33 = intrinsics_inv_B33.to(self.device, non_blocking=True)

                    # Compute image features.
                    with autocast(enabled=self.options.use_half):
                        features_BCHW = self.regressor.get_features(image_B1HW)

                    # Dimensions after the network's downsampling.
                    B, C, H, W = features_BCHW.shape

                    # The image_mask needs to be downsampled to the actual output resolution and cast to bool.
                    image_mask_B1HW = TF.resize(image_mask_B1HW, [H, W], interpolation=TF.InterpolationMode.NEAREST)
                    image_mask_B1HW = image_mask_B1HW.bool()

                    # If the current mask has no valid pixels, continue.
                    if image_mask_B1HW.sum() == 0:
                        continue

                    # Create a tensor with the pixel coordinates of every feature vector.
                    pixel_positions_B2HW = self.pixel_grid_2HW[:, :H, :W].clone()  # It's 2xHxW (actual H and W) now.
                    pixel_positions_B2HW = pixel_positions_B2HW[None]  # 1x2xHxW
                    pixel_positions_B2HW = pixel_positions_B2HW.expand(B, 2, H, W)  # Bx2xHxW

                    # Bx3x4 -> Nx3x4 (for each image, repeat pose per feature)
                    gt_pose_inv = gt_pose_inv_B44[:, :3]
                    gt_pose_inv = gt_pose_inv.unsqueeze(1).expand(B, H * W, 3, 4).reshape(-1, 3, 4)

                    # Bx3x3 -> Nx3x3 (for each image, repeat intrinsics per feature)
                    intrinsics = intrinsics_B33.unsqueeze(1).expand(B, H * W, 3, 3).reshape(-1, 3, 3)
                    intrinsics_inv = intrinsics_inv_B33.unsqueeze(1).expand(B, H * W, 3, 3).reshape(-1, 3, 3)

                    def normalize_shape(tensor_in):
                        """Bring tensor from shape BxCxHxW to NxC"""
                        return tensor_in.transpose(0, 1).flatten(1).transpose(0, 1)

                    batch_data = {
                        'features': normalize_shape(features_BCHW),
                        'target_px': normalize_shape(pixel_positions_B2HW),
                        'gt_poses_inv': gt_pose_inv,
                        'intrinsics': intrinsics,
                        'intrinsics_inv': intrinsics_inv
                    }

                    # Turn image mask into sampling weights (all equal).
                    image_mask_B1HW = image_mask_B1HW.float()
                    image_mask_N1 = normalize_shape(image_mask_B1HW)

                    # Over-sample according to image mask.
                    features_to_select = self.options.samples_per_image * B
                    features_to_select = min(features_to_select, self.options.training_buffer_size - buffer_idx)

                    # Sample indices uniformly, with replacement.
                    sample_idxs = torch.multinomial(image_mask_N1.view(-1),
                                                    features_to_select,
                                                    replacement=True,
                                                    generator=self.sampling_generator)

                    # Select the data to put in the buffer.
                    for k in batch_data:
                        batch_data[k] = batch_data[k][sample_idxs]

                    # Write to training buffer. Start at buffer_idx and end at buffer_offset - 1.
                    buffer_offset = buffer_idx + features_to_select
                    for k in batch_data:
                        self.training_buffer[k][buffer_idx:buffer_offset] = batch_data[k]

                    buffer_idx = buffer_offset
                    if buffer_idx >= self.options.training_buffer_size:
                        break

        buffer_memory = sum([v.element_size() * v.nelement() for k, v in self.training_buffer.items()])
        buffer_memory /= 1024 * 1024 * 1024

        _logger.info(f"Created buffer of {buffer_memory:.2f}GB with {dataset_passes} passes over the training data.")
        self.regressor.train()
        
    def run_epoch_old(self):
        """
        Run one epoch of training, shuffling the feature buffer and iterating over it.
        """
        # Enable benchmarking since all operations work on the same tensor size.
        torch.backends.cudnn.benchmark = True

        # Shuffle indices.
        random_indices = torch.randperm(self.options.training_buffer_size, generator=self.training_generator)

        # Iterate with mini batches.
        for batch_start in range(0, self.options.training_buffer_size, self.options.batch_size):
            batch_end = batch_start + self.options.batch_size

            # Drop last batch if not full.
            if batch_end > self.options.training_buffer_size:
                continue

            # Sample indices.
            random_batch_indices = random_indices[batch_start:batch_end]

            # Call the training step with the sampled features and relevant metadata.
            self.training_step(
                self.training_buffer['features'][random_batch_indices].contiguous(),
                self.training_buffer['target_px'][random_batch_indices].contiguous(),
                self.training_buffer['gt_poses_inv'][random_batch_indices].contiguous(),
                self.training_buffer['intrinsics'][random_batch_indices].contiguous(),
                self.training_buffer['intrinsics_inv'][random_batch_indices].contiguous()
            )
            self.iteration += 1

    def training_step_old(self, features_bC, target_px_b2, gt_inv_poses_b34, Ks_b33, invKs_b33):
        """
        Run one iteration of training, computing the reprojection error and minimising it.
        """
        batch_size = features_bC.shape[0]
        channels = features_bC.shape[1]

        # Reshape to a "fake" BCHW shape, since it's faster to run through the network compared to the original shape.
        features_bCHW = features_bC[None, None, ...].view(-1, 16, 32, channels).permute(0, 3, 1, 2)
        with autocast(enabled=self.options.use_half):
            pred_scene_coords_b3HW = self.regressor.get_scene_coordinates(features_bCHW)

        # Back to the original shape. Convert to float32 as well.
        pred_scene_coords_b31 = pred_scene_coords_b3HW.permute(0, 2, 3, 1).flatten(0, 2).unsqueeze(-1).float()

        # Make 3D points homogeneous so that we can easily matrix-multiply them.
        pred_scene_coords_b41 = to_homogeneous(pred_scene_coords_b31)

        # Scene coordinates to camera coordinates.
        pred_cam_coords_b31 = torch.bmm(gt_inv_poses_b34, pred_scene_coords_b41)

        # Project scene coordinates.
        pred_px_b31 = torch.bmm(Ks_b33, pred_cam_coords_b31)

        # Avoid division by zero.
        # Note: negative values are also clamped at +self.options.depth_min. The predicted pixel would be wrong,
        # but that's fine since we mask them out later.
        pred_px_b31[:, 2].clamp_(min=self.options.depth_min)

        # Dehomogenise.
        pred_px_b21 = pred_px_b31[:, :2] / pred_px_b31[:, 2, None]

        # Measure reprojection error.
        reprojection_error_b2 = pred_px_b21.squeeze() - target_px_b2
        reprojection_error_b1 = torch.norm(reprojection_error_b2, dim=1, keepdim=True, p=1)

        #
        # Compute masks used to ignore invalid pixels.
        #
        # Predicted coordinates behind or close to camera plane.
        invalid_min_depth_b1 = pred_cam_coords_b31[:, 2] < self.options.depth_min
        # Very large reprojection errors.
        invalid_repro_b1 = reprojection_error_b1 > self.options.repro_loss_hard_clamp
        # Predicted coordinates beyond max distance.
        invalid_max_depth_b1 = pred_cam_coords_b31[:, 2] > self.options.depth_max

        # Invalid mask is the union of all these. Valid mask is the opposite.
        invalid_mask_b1 = (invalid_min_depth_b1 | invalid_repro_b1 | invalid_max_depth_b1)
        valid_mask_b1 = ~invalid_mask_b1

        # Reprojection error for all valid scene coordinates.
        valid_reprojection_error_b1 = reprojection_error_b1[valid_mask_b1]
        # Compute the loss for valid predictions.
        loss_valid = self.repro_loss.compute(valid_reprojection_error_b1, self.iteration)

        # Handle the invalid predictions: generate proxy coordinate targets with constant depth assumption.
        pixel_grid_crop_b31 = to_homogeneous(target_px_b2.unsqueeze(2))
        target_camera_coords_b31 = self.options.depth_target * torch.bmm(invKs_b33, pixel_grid_crop_b31)

        # Compute the distance to target camera coordinates.
        invalid_mask_b11 = invalid_mask_b1.unsqueeze(2)
        loss_invalid = torch.abs(target_camera_coords_b31 - pred_cam_coords_b31).masked_select(invalid_mask_b11).sum()

        # Final loss is the sum of all 2.
        loss = loss_valid + loss_invalid
        loss /= batch_size

        # We need to check if the step actually happened, since the scaler might skip optimisation steps.
        old_optimizer_step = self.optimizer._step_count

        # Optimization steps.
        self.optimizer.zero_grad(set_to_none=True)
        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()

        if self.iteration % self.iterations_output == 0:
            # Print status.
            time_since_start = time.time() - self.training_start
            fraction_valid = float(valid_mask_b1.sum() / batch_size)
            # median_depth = float(pred_cam_coords_b31[:, 2].median())

            _logger.info(f'Iteration: {self.iteration:6d} / Epoch {self.epoch:03d}|{self.options.epochs:03d}, '
                         f'Loss: {loss:.1f}, Valid: {fraction_valid * 100:.1f}%, Time: {time_since_start:.2f}s')

            if self.ace_visualizer is not None:
                vis_scene_coords = pred_scene_coords_b31.detach().cpu().squeeze().numpy()
                vis_errors = reprojection_error_b1.detach().cpu().squeeze().numpy()
                self.ace_visualizer.render_mapping_frame(vis_scene_coords, vis_errors)

        # Only step if the optimizer stepped and if we're not over-stepping the total_steps supported by the scheduler.
        if old_optimizer_step < self.optimizer._step_count < self.scheduler.total_steps:
            self.scheduler.step()

    def save_model(self):
        # NOTE: This would save the whole regressor (encoder weights included) in full precision floats (~30MB).
        # torch.save(self.regressor.state_dict(), self.options.output_map_file)

        # This saves just the head weights as half-precision floating point numbers for a total of ~4MB, as mentioned
        # in the paper. The scene-agnostic encoder weights can then be loaded from the pretrained encoder file.
        head_state_dict = self.regressor.heads.state_dict()
        upsample_state_dict = self.regressor.upsample.state_dict()
        gs_regressor_state_dict = self.regressor.gs_regressor.state_dict()
        head_state_dict.update(upsample_state_dict)
        head_state_dict.update(gs_regressor_state_dict)
        
        for k, v in head_state_dict.items():
            head_state_dict[k] = head_state_dict[k].half()
        if not self.options.output_map_file.parents[1].exists():
            self.options.output_map_file.parents[1].mkdir(parents=True)
        torch.save(head_state_dict, self.options.output_map_file)
        
        _logger.info(f"Saved trained head weights to: {self.options.output_map_file}")
    
    
    def save_model_epoch(self, model_name):
        # NOTE: This would save the whole regressor (encoder weights included) in full precision floats (~30MB).
        # torch.save(self.regressor.state_dict(), self.options.output_map_file)

        # This saves just the head weights as half-precision floating point numbers for a total of ~4MB, as mentioned
        # in the paper. The scene-agnostic encoder weights can then be loaded from the pretrained encoder file.
        head_state_dict = self.regressor.heads.state_dict()
        upsample_state_dict = self.regressor.upsample.state_dict()
        gs_regressor_state_dict = self.regressor.gs_regressor.state_dict()
        head_state_dict.update(upsample_state_dict)
        head_state_dict.update(gs_regressor_state_dict)

        for k, v in head_state_dict.items():
            head_state_dict[k] = head_state_dict[k].half()

        if not self.options.output_map_file.parents[1].exists():
            self.options.output_map_file.parents[1].mkdir(parents=True)
        torch.save(head_state_dict, model_name)

        
        _logger.info(f"Saved trained head weights to: {model_name}")
