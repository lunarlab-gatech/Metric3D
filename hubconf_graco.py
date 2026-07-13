dependencies = ['torch', 'torchvision']

import os
import cv2
import getpass
import numpy as np
import tqdm
from pathlib import Path
from robotdataprocess import ImageDataOnDisk, CameraData
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
    model (nn.Module): must be a metric3d_vit_small model, already moved to CUDA
      and set to eval mode (e.g. via
      metric3d_vit_small(pretrain=True).cuda().eval()) — input_size and padding
      above are hardcoded for this specific model/config (vit.raft5.small.py's
      crop_size), so passing any other backbone (ConvNeXt, ViT-Large/giant2)
      will silently produce wrong results. Load this once outside any loop and
      pass it in, to avoid re-downloading/re-loading the model on every call.
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

def main():

  # Set configuration for with robot to use
  robot_name: str = "ground-01"
  category: str = robot_name.split('-')[0] # "ground" or "aerial"

  # Load image data from the GrAco dataset
  dataset_root: Path = Path("/") / 'home' / getpass.getuser() / 'data' / 'GrAco_dataset' / 'V1.0'
  bag_path: Path = dataset_root / 'data' / robot_name / (robot_name + ".bag")
  input_images = ImageDataOnDisk.from_ros1_bag(bag_path, '/camera_left/image_raw')
  assert input_images.encoding in (
    ImageDataOnDisk.ImageEncoding.Mono8,
    ImageDataOnDisk.ImageEncoding.RGB8,
  )

  # Load camera intrinsics
  camera_info = CameraData.from_ros1_bag(bag_path, '/camera_left/camera_info')
  intrinsics = np.array([camera_info.K[0,0], camera_info.K[1,1], camera_info.K[0,2], camera_info.K[1,2]])

  # Load the model once, reused for every frame below
  model = torch.hub.load('yvanyin/metric3d', 'metric3d_vit_small', pretrain=True)
  model.cuda().eval()

  # Make output paths
  results_root: Path = dataset_root / 'results' / 'Metric3D' / category / robot_name
  depth_dir: Path = results_root / 'depth'
  depth_vis_dir: Path = results_root / 'depth_vis'
  normal_vis_dir: Path = results_root / 'normal_vis'
  depth_dir.mkdir(parents=True, exist_ok=True)
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
    cv2.imwrite(str(depth_vis_dir / f"{stem}_rgb.png"), depth_vis)
    if normal_vis is not None:
      cv2.imwrite(str(normal_vis_dir / f"{stem}_normal.png"), normal_vis)

if __name__ == '__main__':
  main()