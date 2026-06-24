from dataclasses import dataclass
import logging
from typing import Optional, Any

import torch

from .model.flow_matching import FlowMatching
from .model.networks import MMAudio
from .model.sequence_config import CONFIG_44K
from .model.utils.features_utils import FeaturesUtils

from torchvision.transforms import v2
from fractions import Fraction
import numpy as np

log = logging.getLogger()

@dataclass
class VideoInfo:
    duration_sec: float
    fps: Fraction
    clip_frames: torch.Tensor
    sync_frames: torch.Tensor
    all_frames: Optional[list[np.ndarray]]

    @property
    def height(self):
        return self.all_frames[0].shape[0]

    @property
    def width(self):
        return self.all_frames[0].shape[1]

    @classmethod
    def from_image_info(cls, image_info: 'ImageInfo', duration_sec: float,
                        fps: Fraction) -> 'VideoInfo':
        num_frames = int(duration_sec * fps)
        all_frames = [image_info.original_frame] * num_frames
        return cls(duration_sec=duration_sec,
                   fps=fps,
                   clip_frames=image_info.clip_frames,
                   sync_frames=image_info.sync_frames,
                   all_frames=all_frames)

@dataclass
class ImageInfo:
    clip_frames: torch.Tensor
    sync_frames: torch.Tensor
    original_frame: Optional[np.ndarray]

    @property
    def height(self):
        return self.original_frame.shape[0]

    @property
    def width(self):
        return self.original_frame.shape[1]

def generate(
    clip_video: torch.Tensor,
    sync_video: torch.Tensor,
    text_encoded: Any,
    feature_utils: FeaturesUtils,
    net: MMAudio,
    fm: FlowMatching,
    rng: torch.Generator,
    cfg_strength: float,
) -> torch.Tensor:
    clip_features = feature_utils.encode_video_with_clip(clip_video, batch_size=40)
    sync_features = feature_utils.encode_video_with_sync(sync_video, batch_size=40)

    preprocessed_conditions = net.preprocess_conditions(clip_features, sync_features, text_encoded)
    empty_conditions = net.get_empty_conditions(1, negative_text_features=text_encoded)

    x0 = torch.randn(1, net.latent_seq_len, net.latent_dim, device='cuda', dtype=torch.bfloat16, generator=rng)
    cfg_ode_wrapper = lambda t, x: net.ode_wrapper(t, x, preprocessed_conditions, empty_conditions, cfg_strength)
    x1 = fm.to_data(cfg_ode_wrapper, x0)
    x1 = net.unnormalize(x1)
    spec = feature_utils.decode(x1)
    audio = feature_utils.vocode(spec)
    return audio

_CLIP_TRANSFORM = v2.Compose([
    v2.Resize((384, 384), interpolation=v2.InterpolationMode.BICUBIC, antialias=False),
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
])

_SYNC_TRANSFORM = v2.Compose([
    v2.Resize(224, interpolation=v2.InterpolationMode.BICUBIC, antialias=False),
    v2.CenterCrop(224),
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])

def load_video_comfy(images, duration_sec: float) -> VideoInfo:
    if images.dim() == 3:
        images = images.unsqueeze(0)
        
    num_frames = len(images)
    
    output_indices = [[], []]
    target_fps = [8, 25]
    
    for frame_idx in range(num_frames):
        frame_time = frame_idx * 0.04
        
        for i in range(2):
            expected_frame_idx = int(frame_time * target_fps[i])
            if expected_frame_idx >= len(output_indices[i]):
                output_indices[i].append(frame_idx)
    
    all_frames = images.cpu().numpy()
    output_frames = [all_frames[indices] for indices in output_indices]

    clip_chunk, sync_chunk = output_frames
    clip_chunk = torch.from_numpy(clip_chunk).permute(0, 3, 1, 2).to('cuda')
    sync_chunk = torch.from_numpy(sync_chunk).permute(0, 3, 1, 2).to('cuda')

    clip_frames = _CLIP_TRANSFORM(clip_chunk)
    sync_frames = _SYNC_TRANSFORM(sync_chunk)

    clip_length_sec = clip_frames.shape[0] / 8
    sync_length_sec = sync_frames.shape[0] / 25

    if clip_length_sec < duration_sec:
        duration_sec = clip_length_sec

    if sync_length_sec < duration_sec:
        duration_sec = sync_length_sec

    clip_frames = clip_frames[:int(8 * duration_sec)]
    sync_frames = sync_frames[:int(25 * duration_sec)]

    video_info = VideoInfo(
        duration_sec=duration_sec,
        fps=25,
        clip_frames=clip_frames,
        sync_frames=sync_frames,
        all_frames=all_frames
    )
    return video_info
