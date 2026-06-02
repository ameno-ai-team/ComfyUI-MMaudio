import os
import torch
import json
from accelerate import init_empty_weights
from accelerate.utils import set_module_tensor_to_device

import folder_paths
import comfy.model_management as mm
from comfy.utils import load_torch_file

script_directory = os.path.dirname(os.path.abspath(__file__))

if not "mmaudio" in folder_paths.folder_names_and_paths:
    folder_paths.add_model_folder_path("mmaudio", os.path.join(folder_paths.models_dir, "mmaudio"))

from .mmaudio.eval_utils import generate, load_video_comfy
from .mmaudio.model.flow_matching import FlowMatching
from .mmaudio.model.networks import MMAudio
from .mmaudio.model.utils.features_utils import FeaturesUtils
from .mmaudio.model.sequence_config import CONFIG_44K
from .mmaudio.ext.bigvgan_v2.bigvgan import BigVGAN as BigVGANv2
from .mmaudio.ext.synchformer import Synchformer
from .mmaudio.ext.autoencoder import AutoEncoderModule
from huggingface_hub import snapshot_download
from open_clip import CLIP
import random
import librosa

DEVICE = mm.get_torch_device()
DTYPE = torch.bfloat16

RNG = torch.Generator(device=DEVICE)
RNG.manual_seed(random.randint(0, 2**64))
FM = FlowMatching(min_sigma=0, inference_mode='euler', num_steps=25)
EMPTY_ENCODED = ''

mmaudio_model_path = folder_paths.get_full_path_or_raise("mmaudio", 'mmaudio_large_44k_v2_fp16.safetensors')
mmaudio_sd = load_torch_file(mmaudio_model_path, device=DEVICE)


with init_empty_weights():
    MODEL = MMAudio(latent_dim=40,
        clip_dim=1024,
        sync_dim=768,
        text_dim=1024,
        hidden_dim=64 * 14,
        depth=21,
        fused_depth=14,
        num_heads=14,
        latent_seq_len=345,
        clip_seq_len=64,
        sync_seq_len=192,
        v2=mmaudio_sd["t_embed.mlp.0.weight"].shape[1] == 896
    )
MODEL.load_state_dict(mmaudio_sd, strict=False, assign=True)

# materialize any buffers still on meta (computed at init, not in state dict)
MODEL = MODEL.to_empty(device=DEVICE)
# to_empty leaves garbage in buffers/params not loaded; reload real weights
MODEL.load_state_dict(mmaudio_sd, strict=False, assign=True)
MODEL = MODEL.eval().to(dtype=DTYPE)
del mmaudio_sd

MODEL.seq_cfg = CONFIG_44K

#synchformer
synchformer_path = folder_paths.get_full_path_or_raise("mmaudio", 'mmaudio_synchformer_fp16.safetensors')
synchformer_sd = load_torch_file(synchformer_path, device=DEVICE)
with init_empty_weights():
    synchformer = Synchformer().eval()

for name, _ in synchformer.named_parameters():
    # Set tensor to device
    set_module_tensor_to_device(synchformer, name, device=DEVICE, dtype=DTYPE, value=synchformer_sd[name])

#vae
download_path = folder_paths.get_folder_paths("mmaudio")[0]

nvidia_bigvgan_vocoder_path = os.path.join(download_path, "nvidia", "bigvgan_v2_44khz_128band_512x")
if not os.path.exists(nvidia_bigvgan_vocoder_path):
    snapshot_download(
        repo_id="nvidia/bigvgan_v2_44khz_128band_512x",
        ignore_patterns=["*3m*",],
        local_dir=nvidia_bigvgan_vocoder_path,
        local_dir_use_symlinks=False,
    )

bigvgan_vocoder = BigVGANv2.from_pretrained(nvidia_bigvgan_vocoder_path).eval().to(device=DEVICE, dtype=DTYPE)


vae_path = folder_paths.get_full_path_or_raise("mmaudio", 'mmaudio_vae_44k_fp16.safetensors')
vae_sd = load_torch_file(vae_path, device=DEVICE)
vae = AutoEncoderModule(
    vae_state_dict=vae_sd,
    bigvgan_vocoder=bigvgan_vocoder,
    mode='44k'
)
vae = vae.eval().to(device=DEVICE, dtype=DTYPE)

#clip

clip_model_path = folder_paths.get_full_path_or_raise("mmaudio", 'apple_DFN5B-CLIP-ViT-H-14-384_fp16.safetensors')
clip_config_path = os.path.join(script_directory, "configs", "DFN5B-CLIP-ViT-H-14-384.json")
with open(clip_config_path) as f:
        clip_config = json.load(f)
    
with init_empty_weights():
    try:
        clip_model = CLIP(**clip_config["model_cfg"]).eval()
    except:
        # for some open-clip versions
        clip_config["model_cfg"]["nonscalar_logit_scale"] = True
        clip_model = CLIP(**clip_config["model_cfg"]).eval()

clip_sd = load_torch_file(os.path.join(clip_model_path), device=DEVICE)
for name, param in clip_model.named_parameters():
    set_module_tensor_to_device(clip_model, name, device=DEVICE, dtype=DTYPE, value=clip_sd[name])
clip_model.to(device=DEVICE, dtype=DTYPE)

FEATURE_UTILS = FeaturesUtils(
    vae=vae,
    synchformer=synchformer,
    enable_conditions=True,
    clip_model=clip_model
)
FEATURE_UTILS = FEATURE_UTILS.to('cuda', torch.bfloat16).eval()
EMPTY_ENCODED = FEATURE_UTILS.encode_text([''])

#region sampling
class MMAudioSampler:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE",),
                "duration": ("FLOAT", {"default": 8, "step": 0.01, "tooltip": "Duration of the audio in seconds"}),
            },
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio", )
    FUNCTION = "sample"
    CATEGORY = "MMAudio"

    def sample(self, duration, images):
        video_info = load_video_comfy(images, duration)
        sync_frames = video_info.sync_frames.unsqueeze(0).to(DEVICE, DTYPE, non_blocking=True)
        clip_frames = video_info.clip_frames.unsqueeze(0).to(DEVICE, DTYPE, non_blocking=True)

        if MODEL.seq_cfg.duration != video_info.duration_sec:
            MODEL.seq_cfg.duration = video_info.duration_sec
            MODEL.update_seq_lengths(MODEL.seq_cfg.latent_seq_len, MODEL.seq_cfg.clip_seq_len, MODEL.seq_cfg.sync_seq_len)

        audios = generate(
            clip_frames,
            sync_frames,
            EMPTY_ENCODED,
            feature_utils=FEATURE_UTILS,
            net=MODEL,
            fm=FM,
            rng=RNG,
            cfg_strength=4.5
        )
        
        audio = audios.float().cpu()[0].squeeze().numpy()
        stretched = librosa.effects.time_stretch(audio, rate=30/25)
        stretched_audio = torch.from_numpy(stretched).unsqueeze(1)
        audio = {"waveform": stretched_audio, "sample_rate": MODEL.seq_cfg.sampling_rate}
        return (audio,)
        
NODE_CLASS_MAPPINGS = {
    "MMAudioSampler": MMAudioSampler,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "MMAudioSampler": "MMAudio Sampler",
}
