from __future__ import annotations

dependencies = ['torch', 'torchvision']

import os
import cv2
import getpass
import numpy as np
import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from decimal import Decimal
from pathlib import Path
from robotdataprocess import ImageDataOnDisk, CameraData, LiDARData, CoordinateFrame, TransformationData
import torch
try:
  from mmcv.utils import Config, DictAction
except:
  from mmengine import Config, DictAction

from mono.model.monodepth_model import get_configured_monodepth_model
metric3d_dir = os.path.dirname(__file__)

MODEL_TYPE = {
  'ConvNeXt-Tiny': {
    'cfg_file': f'{metric3d_dir}/mono/configs/HourglassDecoder/convtiny.0.3_150.py',
    'ckpt_file': 'https://huggingface.co/JUGGHM/Metric3D/resolve/main/convtiny_hourglass_v1.pth',
  },
  'ConvNeXt-Large': {
    'cfg_file': f'{metric3d_dir}/mono/configs/HourglassDecoder/convlarge.0.3_150.py',
    'ckpt_file': 'https://huggingface.co/JUGGHM/Metric3D/resolve/main/convlarge_hourglass_0.3_150_step750k_v1.1.pth',
  },
  'ViT-Small': {
    'cfg_file': f'{metric3d_dir}/mono/configs/HourglassDecoder/vit.raft5.small.py',
    'ckpt_file': 'https://huggingface.co/JUGGHM/Metric3D/resolve/main/metric_depth_vit_small_800k.pth',
  },
  'ViT-Large': {
    'cfg_file': f'{metric3d_dir}/mono/configs/HourglassDecoder/vit.raft5.large.py',
    'ckpt_file': 'https://huggingface.co/JUGGHM/Metric3D/resolve/main/metric_depth_vit_large_800k.pth',
  },
  'ViT-giant2': {
    'cfg_file': f'{metric3d_dir}/mono/configs/HourglassDecoder/vit.raft5.giant2.py',
    'ckpt_file': 'https://huggingface.co/JUGGHM/Metric3D/resolve/main/metric_depth_vit_giant2_800k.pth',
  },
}



def metric3d_convnext_tiny(pretrain=False, **kwargs):
  '''
  Return a Metric3D model with ConvNeXt-Large backbone and Hourglass-Decoder head.
  For usage examples, refer to: https://github.com/YvanYin/Metric3D/blob/main/hubconf.py
  Args:
    pretrain (bool): whether to load pretrained weights.
  Returns:
    model (nn.Module): a Metric3D model.
  '''
  cfg_file = MODEL_TYPE['ConvNeXt-Tiny']['cfg_file']
  ckpt_file = MODEL_TYPE['ConvNeXt-Tiny']['ckpt_file']

  cfg = Config.fromfile(cfg_file)
  model = get_configured_monodepth_model(cfg)
  if pretrain:
    model.load_state_dict(
      torch.hub.load_state_dict_from_url(ckpt_file)['model_state_dict'], 
      strict=False,
    )
  return model

def metric3d_convnext_large(pretrain=False, **kwargs):
  '''
  Return a Metric3D model with ConvNeXt-Large backbone and Hourglass-Decoder head.
  For usage examples, refer to: https://github.com/YvanYin/Metric3D/blob/main/hubconf.py
  Args:
    pretrain (bool): whether to load pretrained weights.
  Returns:
    model (nn.Module): a Metric3D model.
  '''
  cfg_file = MODEL_TYPE['ConvNeXt-Large']['cfg_file']
  ckpt_file = MODEL_TYPE['ConvNeXt-Large']['ckpt_file']

  cfg = Config.fromfile(cfg_file)
  model = get_configured_monodepth_model(cfg)
  if pretrain:
    model.load_state_dict(
      torch.hub.load_state_dict_from_url(ckpt_file)['model_state_dict'], 
      strict=False,
    )
  return model

def metric3d_vit_small(pretrain=False, **kwargs):
  '''
  Return a Metric3D model with ViT-Small backbone and RAFT-4iter head.
  For usage examples, refer to: https://github.com/YvanYin/Metric3D/blob/main/hubconf.py
  Args:
    pretrain (bool): whether to load pretrained weights.
  Returns:
    model (nn.Module): a Metric3D model.
  '''
  cfg_file = MODEL_TYPE['ViT-Small']['cfg_file']
  ckpt_file = MODEL_TYPE['ViT-Small']['ckpt_file']

  cfg = Config.fromfile(cfg_file)
  model = get_configured_monodepth_model(cfg)
  if pretrain:
    model.load_state_dict(
      torch.hub.load_state_dict_from_url(ckpt_file)['model_state_dict'], 
      strict=False,
    )
  return model

def metric3d_vit_large(pretrain=False, **kwargs):
  '''
  Return a Metric3D model with ViT-Large backbone and RAFT-8iter head.
  For usage examples, refer to: https://github.com/YvanYin/Metric3D/blob/main/hubconf.py
  Args:
    pretrain (bool): whether to load pretrained weights.
  Returns:
    model (nn.Module): a Metric3D model.
  '''
  cfg_file = MODEL_TYPE['ViT-Large']['cfg_file']
  ckpt_file = MODEL_TYPE['ViT-Large']['ckpt_file']

  cfg = Config.fromfile(cfg_file)
  model = get_configured_monodepth_model(cfg)
  if pretrain:
    model.load_state_dict(
      torch.hub.load_state_dict_from_url(ckpt_file)['model_state_dict'], 
      strict=False,
    )
  return model

def metric3d_vit_giant2(pretrain=False, **kwargs):
  '''
  Return a Metric3D model with ViT-Giant2 backbone and RAFT-8iter head.
  For usage examples, refer to: https://github.com/YvanYin/Metric3D/blob/main/hubconf.py
  Args:
    pretrain (bool): whether to load pretrained weights.
  Returns:
    model (nn.Module): a Metric3D model.
  '''
  cfg_file = MODEL_TYPE['ViT-giant2']['cfg_file']
  ckpt_file = MODEL_TYPE['ViT-giant2']['ckpt_file']

  cfg = Config.fromfile(cfg_file)
  model = get_configured_monodepth_model(cfg)
  if pretrain:
    model.load_state_dict(
      torch.hub.load_state_dict_from_url(ckpt_file)['model_state_dict'], 
      strict=False,
    )
  return model

def infer_single_image(rgb_image: np.ndarray, intrinsics: np.ndarray, model: torch.nn.Module, gt_depth_image: np.ndarray | None = None):
  """
  Using Metric3D, infer depth (and, for ViT models, surface normals) for a single image.

  Args:
    rgb_image (np.ndarray): HxWx3 RGB image to run inference on.
    intrinsics (np.ndarray): camera intrinsics [fx, fy, cx, cy] for rgb_image.
    model (nn.Module): must be one of the ViT-based models (metric3d_vit_small,
      metric3d_vit_large, metric3d_vit_giant2), already moved to CUDA and set
      to eval mode (e.g. via metric3d_vit_small(pretrain=True).cuda().eval())
      — input_size and padding above are hardcoded to vit.raft5.*'s shared
      crop_size, which is the same for all three ViT configs, so any of them
      can be passed in. Passing a ConvNeXt model instead will silently
      produce wrong results, since ConvNeXt uses a different crop_size. Load
      this once outside any loop and pass it in, to avoid re-downloading/
      re-loading the model on every call.
    gt_depth_image (np.ndarray): optional HxW ground-truth depth map, used only to
      print the absolute relative error against the prediction. Pass None to skip.

  Returns:
    pred_depth (np.ndarray): HxW float32 metric depth (in meters).
    depth_vis (np.ndarray): HxW uint8 min-max-normalized visualization of pred_depth.
    normal_vis (np.ndarray | None): HxWx3 uint8 visualization of the predicted
      surface normals, or None (only available for ViT models).
  """

  #### Adjust input size to fit pretrained model
  input_size = (616, 1064) # for vit model
  # input_size = (544, 1216) # for convnext model
  h, w = rgb_image.shape[:2]
  scale = min(input_size[0] / h, input_size[1] / w)
  rgb = cv2.resize(rgb_image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_LINEAR)
  # remember to scale intrinsic, hold depth
  intrinsics = [intrinsics[0] * scale, intrinsics[1] * scale, intrinsics[2] * scale, intrinsics[3] * scale]
  # padding to input_size
  padding = [123.675, 116.28, 103.53]
  h, w = rgb.shape[:2]
  pad_h = input_size[0] - h
  pad_w = input_size[1] - w
  pad_h_half = pad_h // 2
  pad_w_half = pad_w // 2
  rgb = cv2.copyMakeBorder(rgb, pad_h_half, pad_h - pad_h_half, pad_w_half, pad_w - pad_w_half, cv2.BORDER_CONSTANT, value=padding)
  pad_info = [pad_h_half, pad_h - pad_h_half, pad_w_half, pad_w - pad_w_half]

  #### normalize
  mean = torch.tensor([123.675, 116.28, 103.53]).float()[:, None, None]
  std = torch.tensor([58.395, 57.12, 57.375]).float()[:, None, None]
  rgb = torch.from_numpy(rgb.transpose((2, 0, 1))).float()
  rgb = torch.div((rgb - mean), std)
  rgb = rgb[None, :, :, :].cuda()

  ###################### canonical camera space ######################
  # inference
  with torch.no_grad():
    pred_depth, confidence, output_dict = model.inference({'input': rgb})

  # un pad
  pred_depth = pred_depth.squeeze()
  pred_depth = pred_depth[pad_info[0] : pred_depth.shape[0] - pad_info[1], pad_info[2] : pred_depth.shape[1] - pad_info[3]]
  
  # upsample to original size
  pred_depth = torch.nn.functional.interpolate(pred_depth[None, None, :, :], rgb_image.shape[:2], mode='bilinear').squeeze()
  ###################### canonical camera space ######################

  #### de-canonical transform
  canonical_to_real_scale = intrinsics[0] / 1000.0 # 1000.0 is the focal length of canonical camera
  pred_depth = pred_depth * canonical_to_real_scale # now the depth is metric
  pred_depth = torch.clamp(pred_depth, 0, 300)

  #### you can now do anything with the metric depth
  # such as evaluate predicted depth
  pred_depth_np = pred_depth.cpu().numpy().astype(np.float32)
  pred_depth_vis = (pred_depth_np - pred_depth_np.min()) / (pred_depth_np.max() - pred_depth_np.min() + 1e-8)
  pred_depth_vis = (pred_depth_vis * 255).astype(np.uint8)

  if gt_depth_image is not None:
    gt_depth_tensor: torch.Tensor = torch.from_numpy(gt_depth_image).float().cuda()
    assert gt_depth_tensor.shape == pred_depth.shape
    
    mask = (gt_depth_tensor > 1e-8)
    abs_rel_err = (torch.abs(pred_depth[mask] - gt_depth_tensor[mask]) / gt_depth_tensor[mask]).mean()
    print('abs_rel_err:', abs_rel_err.item())

  #### normal are also available
  pred_normal_vis = None
  if 'prediction_normal' in output_dict: # only available for Metric3Dv2, i.e. vit model
    pred_normal = output_dict['prediction_normal'][:, :3, :, :]
    normal_confidence = output_dict['prediction_normal'][:, 3, :, :] # see https://arxiv.org/abs/2109.09881 for details
    # un pad and resize to some size if needed
    pred_normal = pred_normal.squeeze()
    pred_normal = pred_normal[:, pad_info[0] : pred_normal.shape[1] - pad_info[1], pad_info[2] : pred_normal.shape[2] - pad_info[3]]
    # you can now do anything with the normal
    # such as visualize pred_normal
    pred_normal_vis = pred_normal.cpu().numpy().transpose((1, 2, 0))
    pred_normal_vis = (pred_normal_vis + 1) / 2
    pred_normal_vis = (pred_normal_vis * 255).astype(np.uint8)

  return pred_depth_np, pred_depth_vis, pred_normal_vis

def find_nearest_timestamp_index(sorted_timestamps: np.ndarray[Decimal], query_ts: Decimal, tolerance: Decimal) -> int | None:
  """
  Find the index of the timestamp in a sorted array closest to query_ts.

  Args:
    sorted_timestamps (np.ndarray[Decimal]): Ascending array of Decimal timestamps.
    query_ts (Decimal): The timestamp to match against.
    tolerance (Decimal): Maximum allowed |difference| for a match.

  Returns:
    int | None: The closest index, or None if no timestamp is within tolerance.
  """
  idx = int(np.searchsorted(sorted_timestamps, query_ts))
  candidates = [i for i in (idx - 1, idx) if 0 <= i < len(sorted_timestamps)]
  if not candidates:
    return None

  best_idx = min(candidates, key=lambda i: abs(sorted_timestamps[i] - query_ts))
  if abs(sorted_timestamps[best_idx] - query_ts) > tolerance:
    return None
  return best_idx


def compare_lidar_to_pred_depth(pred_depth: np.ndarray, lidar_points: np.ndarray, intrinsics: np.ndarray, T_cam_lidar: np.ndarray | None, max_err_m: float = 20.0, stats_max_pred_depth_m: float = 15.0, point_radius_px: int = 3):
  """
  Project LiDAR points into the camera frame and onto the 2D image, then
  compare the LiDAR-derived depth against the predicted depth at each
  corresponding pixel.

  Args:
    pred_depth (np.ndarray): HxW predicted metric depth, at the same
      resolution/intrinsics as the original camera image (see
      infer_single_image, which upsamples back to the input image size).
    lidar_points (np.ndarray): (N, 3) LiDAR points in the LiDAR sensor frame,
      as returned by LiDARData.get_point_cloud_at_index.
    intrinsics (np.ndarray): [fx, fy, cx, cy] camera intrinsics matching
      pred_depth's resolution.
    T_cam_lidar (np.ndarray | None): 4x4 homogeneous transform giving the pose
      of the LiDAR frame with respect to the camera optical frame. Used to
      map points from the LiDAR frame into the camera optical frame.
    max_err_m (float): |pred - gt| (in meters) at or above which a pixel is
      rendered fully red in the returned error visualization.
    stats_max_pred_depth_m (float): only points whose predicted depth is
      <= this value (in meters) are included in the printed mean abs err /
      mean abs rel err stats. Does not affect the returned error_vis image,
      which colors every projected point regardless of predicted depth.
    point_radius_px (int): radius, in pixels, of the filled circle drawn for
      each projected LiDAR point in the returned error_vis image (drawn
      larger than a single pixel so sparse points are actually visible).

  Returns:
    error_vis (np.ndarray): HxWx3 uint8 BGR image, the same size as
      pred_depth. Pixels with no corresponding LiDAR point are grey.
      Each projected LiDAR point is drawn as a filled circle colored on a
      green (0 error) to red (>= max_err_m error) gradient based on
      |pred - gt|.
    stats_abs_err (np.ndarray): 1D array of |pred - gt| (meters), restricted
      to points with predicted depth <= stats_max_pred_depth_m (the same
      point set used for the printed stats). Empty if none.
    all_abs_err (np.ndarray): 1D array of |pred - gt| (meters) for every
      projected point, regardless of predicted depth. Empty if none.

  Raises:
    ValueError: If T_cam_lidar is None.
  """
  if T_cam_lidar is None:
    raise ValueError("T_cam_lidar (pose of the LiDAR frame wrt. the camera frame) must be provided.")

  grey_bgr = (128, 128, 128)
  h, w = pred_depth.shape[:2]
  error_vis = np.full((h, w, 3), grey_bgr, dtype=np.uint8)
  empty_abs_err = np.array([])

  # Drop invalid (NaN) points
  valid = np.isfinite(lidar_points).all(axis=1)
  points_lidar = lidar_points[valid]
  if points_lidar.shape[0] == 0:
    return error_vis, empty_abs_err, empty_abs_err

  # Transform points from the LiDAR frame into the camera optical frame
  R_cam_lidar = T_cam_lidar[:3, :3]
  t_cam_lidar = T_cam_lidar[:3, 3]
  points_cam = (R_cam_lidar @ points_lidar.T).T + t_cam_lidar

  # Only points in front of the camera can be projected onto the image
  in_front = points_cam[:, 2] > 1e-3
  points_cam = points_cam[in_front]
  if points_cam.shape[0] == 0:
    return error_vis, empty_abs_err, empty_abs_err

  # Project onto the image plane
  fx, fy, cx, cy = intrinsics
  gt_depth = points_cam[:, 2]
  u = (points_cam[:, 0] * fx / gt_depth) + cx
  v = (points_cam[:, 1] * fy / gt_depth) + cy

  # Keep only points that land inside the image bounds
  u_pix = np.round(u).astype(np.int64)
  v_pix = np.round(v).astype(np.int64)
  in_bounds = (u_pix >= 0) & (u_pix < w) & (v_pix >= 0) & (v_pix < h)
  u_pix, v_pix, gt_depth = u_pix[in_bounds], v_pix[in_bounds], gt_depth[in_bounds]
  if gt_depth.shape[0] == 0:
    return error_vis, empty_abs_err, empty_abs_err

  # Compare predicted depth at each projected pixel against the LiDAR (gt) depth
  pred_depth_at_points = pred_depth[v_pix, u_pix]
  depth_diff = pred_depth_at_points - gt_depth
  abs_err = np.abs(depth_diff)
  rel_err_pct = abs_err / gt_depth * 100.0

  in_stats_range = pred_depth_at_points <= stats_max_pred_depth_m
  if in_stats_range.any():
    print(f"LiDAR vs. predicted depth, pred depth <= {stats_max_pred_depth_m}m "
          f"({in_stats_range.sum()} of {gt_depth.shape[0]} points): "
          f"mean abs err={abs_err[in_stats_range].mean():.4f} m, "
          f"median abs err={np.median(abs_err[in_stats_range]):.4f} m, "
          f"mean rel err={rel_err_pct[in_stats_range].mean():.2f}%, "
          f"median rel err={np.median(rel_err_pct[in_stats_range]):.2f}%")
  else:
    print(f"LiDAR vs. predicted depth: no points with pred depth <= {stats_max_pred_depth_m}m "
          f"(out of {gt_depth.shape[0]} points)")

  # Color each projected point on a green (0 error) -> red (>= max_err_m) gradient
  t = np.clip(abs_err / max_err_m, 0.0, 1.0)
  b = np.zeros_like(t)
  g = (1.0 - t) * 255.0
  r = t * 255.0
  colors = np.stack([b, g, r], axis=-1).astype(np.uint8)  # BGR, for cv2.imwrite
  for x, y, color in zip(u_pix, v_pix, colors):
    cv2.circle(error_vis, (int(x), int(y)), point_radius_px, color.tolist(), thickness=-1)

  return error_vis, abs_err[in_stats_range], abs_err

def main_graco(enable_debug_vis: bool = False):
  """
  Args:
    enable_debug_vis (bool): whether to save the four debugging
      visualizations (depth_vis, normal_vis, error_vis, error_hist). The
      predicted depth .npy files are always saved regardless.
  """

  # Set configuration for with robot to use
  robot_name: str = "aerial-08"
  category: str = robot_name.split('-')[0] # "ground" or "aerial"

  # Load image data from the GrAco dataset
  dataset_root: Path = Path("/") / 'home' / getpass.getuser() / 'data' / 'GrAco_dataset' / 'V1.0'
  data_dir: Path = dataset_root / 'data'
  bag_dir: Path = data_dir / robot_name
  calibration_dir: Path = data_dir / (category + "-calibration")
  ros1_bag_path: Path = bag_dir / (robot_name + ".bag")
  ros2_bag_path: Path = bag_dir / robot_name
  input_images = ImageDataOnDisk.from_ros1_bag(ros1_bag_path, '/camera_left/image_raw')
  assert input_images.encoding in (
    ImageDataOnDisk.ImageEncoding.Mono8,
    ImageDataOnDisk.ImageEncoding.RGB8,
  )

  # Load camera intrinsics
  camera_info = CameraData.from_ros1_bag(ros1_bag_path, '/camera_left/camera_info')
  intrinsics = np.array([camera_info.K[0,0], camera_info.K[1,1], camera_info.K[0,2], camera_info.K[1,2]])

  # Load LiDAR data for evaluation
  lidar_data = LiDARData.from_ros2_bag(ros2_bag_path, '/velodyne/points', CoordinateFrame.ENU)

  # Pose of the LiDAR frame with respect to the camera optical frame,
  T_cam_lidar: np.ndarray = TransformationData.from_GrAco_yaml(str(calibration_dir / "stereo-lidar.yaml"), "T_cam0_Lidar").as_matrix()

  # Load the model once, reused for every frame below
  model = torch.hub.load('yvanyin/metric3d', 'metric3d_vit_small', pretrain=True)
  model.cuda().eval()

  # Make output paths
  results_root: Path = dataset_root / 'results' / 'Metric3D' / category / robot_name
  depth_dir: Path = results_root / 'depth'
  depth_vis_dir: Path = results_root / 'depth_vis'
  normal_vis_dir: Path = results_root / 'normal_vis'
  error_vis_dir: Path = results_root / 'error_vis'
  error_hist_dir: Path = results_root / 'error_hist'
  depth_dir.mkdir(parents=True, exist_ok=True)
  if enable_debug_vis:
    depth_vis_dir.mkdir(parents=True, exist_ok=True)
    normal_vis_dir.mkdir(parents=True, exist_ok=True)
    error_vis_dir.mkdir(parents=True, exist_ok=True)
    error_hist_dir.mkdir(parents=True, exist_ok=True)

  # Upper bound (in meters) of the error_vis color scale; also used as the
  # colorbar's range, so it's defined once here rather than in both places.
  error_vis_max_err_m = 20.0

  # Only points with predicted depth <= this value (in meters) are included
  # in the printed stats and the error histogram; defined once here so both
  # stay in sync.
  stats_max_pred_depth_m = 15.0

  # Run inference on every frame in the bag
  for i, timestamp in tqdm.tqdm(enumerate(input_images.timestamps), total=len(input_images.timestamps), desc="Running Metric3D inference", unit=" images"):
    image = input_images.images[i]
    if input_images.encoding == ImageDataOnDisk.ImageEncoding.Mono8:
      image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB) # replicate the single channel into R, G, B

    pred_depth, depth_vis, normal_vis = infer_single_image(image, intrinsics, model)

    # If a LiDAR scan was captured at (near) the same time as this image,
    # project it into the camera frame and compare it against pred_depth.
    if enable_debug_vis:
      lidar_idx = find_nearest_timestamp_index(lidar_data.timestamps, timestamp, Decimal("0.01"))
      error_vis, stats_abs_err, all_abs_err = None, None, None
      if lidar_idx is not None:
        lidar_points, _ = lidar_data.get_point_cloud_at_index(lidar_idx)
        error_vis, stats_abs_err, all_abs_err = compare_lidar_to_pred_depth(pred_depth, lidar_points, intrinsics, T_cam_lidar, max_err_m=error_vis_max_err_m, stats_max_pred_depth_m=stats_max_pred_depth_m)

    # Filenames encode the timestamp, matching robotdataprocess's own convention
    # (see ImageData.to_image_files) so from_npy_files can load these back in.
    stem = f"{timestamp:.9f}"
    np.save(str(depth_dir / f"{stem}.npy"), pred_depth)

    if enable_debug_vis:
      cv2.imwrite(str(depth_vis_dir / f"{stem}_rgb.png"), depth_vis)
      if normal_vis is not None:
        cv2.imwrite(str(normal_vis_dir / f"{stem}_normal.png"), normal_vis)
      if error_vis is not None:
        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        comparison_rgb = cv2.cvtColor(np.hstack([image_bgr, error_vis]), cv2.COLOR_BGR2RGB)

        fig, ax = plt.subplots(figsize=(comparison_rgb.shape[1] / 100, comparison_rgb.shape[0] / 100), dpi=100)
        ax.imshow(comparison_rgb)
        ax.axis('off')
        ax.set_title("RGB image (left) vs. LiDAR depth error (right, grey = no LiDAR point)")

        cmap = LinearSegmentedColormap.from_list('depth_error', ['green', 'red'])
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=Normalize(vmin=0, vmax=error_vis_max_err_m))
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, fraction=0.025, pad=0.02)
        cbar.set_label('Abs depth error (m)')

        fig.savefig(str(error_vis_dir / f"{stem}_error.png"), bbox_inches='tight', dpi=150)
        plt.close(fig)

      if stats_abs_err is not None and stats_abs_err.size > 0:
        fig, ax = plt.subplots(figsize=(6, 4), dpi=100)
        ax.hist(stats_abs_err, bins=30, range=(0, 5))
        ax.set_xlabel('Abs depth error (m)')
        ax.set_ylabel('Count')
        ax.set_xlim(0, 5)
        ax.set_title(f"LiDAR abs depth error histogram (pred depth <= {stats_max_pred_depth_m}m, error in [0, 5]m, n={stats_abs_err.size})")

        fig.savefig(str(error_hist_dir / f"{stem}_error_hist.png"), bbox_inches='tight', dpi=150)
        plt.close(fig)

      if all_abs_err is not None and all_abs_err.size > 0:
        fig, ax = plt.subplots(figsize=(6, 4), dpi=100)
        ax.hist(all_abs_err, bins=30)
        ax.set_xlabel('Abs depth error (m)')
        ax.set_ylabel('Count')
        ax.set_title(f"LiDAR abs depth error histogram (all points, n={all_abs_err.size})")

        fig.savefig(str(error_hist_dir / f"{stem}_error_hist_all.png"), bbox_inches='tight', dpi=150)
        plt.close(fig)

def main_airmuseum(enable_debug_vis: bool = True):
  # Set configuration for with robot to use
  robot_name: str = "drone"
  scenario: str = "Scenario5"

  # Load image data from the GrAco dataset
  dataset_root: Path = Path("/") / 'home' / getpass.getuser() / 'data' / 'AirMuseum_dataset' / scenario
  data_dir: Path = dataset_root / 'data'
  bag_dir: Path = data_dir / robot_name
  calibration_dir: Path = dataset_root.parent / 'sensors'
  bag_path: Path = bag_dir / 'cam100_imu.bag'
  input_images = ImageDataOnDisk.from_ros1_bag(bag_path, f'/{robot_name}/cam100/image_raw')
  assert input_images.encoding == ImageDataOnDisk.ImageEncoding.Mono8

  # Undistort the images
  camera_info = CameraData.from_kalibr_mono(calibration_dir / f'{robot_name}_cameras_calib.yaml', 'cam0')
  image_before_undistort = input_images.images[0].copy()
  input_images.undistort_imagery_mono(camera_info)

  # Set intrinsics after undistortion
  intrinsics = np.array([camera_info.K[0,0], camera_info.K[1,1], camera_info.K[0,2], camera_info.K[1,2]])

  # Load the model once, reused for every frame below
  model = torch.hub.load('yvanyin/metric3d', 'metric3d_vit_small', pretrain=True)
  model.cuda().eval()

  # Make output paths
  results_root: Path = dataset_root / 'results' / 'Metric3D' / robot_name
  depth_dir: Path = results_root / 'depth'
  depth_vis_dir: Path = results_root / 'depth_vis'
  normal_vis_dir: Path = results_root / 'normal_vis'
  depth_dir.mkdir(parents=True, exist_ok=True)
  if enable_debug_vis:
    depth_vis_dir.mkdir(parents=True, exist_ok=True)
    normal_vis_dir.mkdir(parents=True, exist_ok=True)

  # Run inference on every frame in the bag
  for i, timestamp in tqdm.tqdm(enumerate(input_images.timestamps), total=len(input_images.timestamps), desc="Running Metric3D inference", unit=" images"):
    image = input_images.images[i]
    if input_images.encoding == ImageDataOnDisk.ImageEncoding.Mono8:
      image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB) # replicate the single channel into R, G, B

    pred_depth, depth_vis, normal_vis = infer_single_image(image, intrinsics, model)

    # Filenames encode the timestamp, matching robotdataprocess's own convention
    # (see ImageData.to_image_files) so from_npy_files can load these back in.
    stem = f"{timestamp:.9f}"
    np.save(str(depth_dir / f"{stem}.npy"), pred_depth)

    if enable_debug_vis:
      cv2.imwrite(str(depth_vis_dir / f"{stem}_rgb.png"), depth_vis)
      if normal_vis is not None:
        cv2.imwrite(str(normal_vis_dir / f"{stem}_normal.png"), normal_vis)

def main():
  # TODO: Undistort images before passing to Metric3D as this might be a source of error. Test this with graco since we have a pseudo-GT from the LiDAR.

  dataset: str = "airmuseum"
  if dataset == "graco":
    main_graco()
  elif dataset == "airmuseum":
    main_airmuseum()
  else:
    raise NotImplementedError()

if __name__ == '__main__':
  main()