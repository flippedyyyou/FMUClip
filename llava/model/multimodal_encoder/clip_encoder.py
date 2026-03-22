import os

import torch
import torch.nn as nn

from transformers import CLIPVisionModel, CLIPImageProcessor, CLIPVisionConfig


class CLIPVisionTower(nn.Module):
    def __init__(self, vision_tower, args, delay_load=False):
        super().__init__()

        self.is_loaded = False

        self.vision_tower_name = vision_tower
        self.select_layer = args.mm_vision_select_layer
        self.select_feature = getattr(args, 'mm_vision_select_feature', 'patch')

        if not delay_load:
            self.load_model()
        elif getattr(args, 'unfreeze_mm_vision_tower', False):
            self.load_model()
        else:
            self.cfg_only = CLIPVisionConfig.from_pretrained(self.vision_tower_name)

    def load_model(self, our_vision_encoder=False, name=None):
        # TODO: act here to load a different visual encoder model
        if our_vision_encoder:
            print("Loading self-defined clip vision encoder...")
            self.image_processor = CLIPImageProcessor.from_pretrained(
                self.vision_tower_name
            )
            self.vision_tower = self._load_custom_vision_tower(name)

        else:
            self.image_processor = CLIPImageProcessor.from_pretrained(
                self.vision_tower_name
            )
            self.vision_tower = CLIPVisionModel.from_pretrained(self.vision_tower_name)
        
        self.vision_tower.requires_grad_(False)
        self.is_loaded = True

    def _load_custom_vision_tower(self, checkpoint_name):
        base_model = CLIPVisionModel.from_pretrained(self.vision_tower_name)
        ckpt_path = self._resolve_custom_checkpoint_path(checkpoint_name)
        raw_state = torch.load(ckpt_path, map_location="cpu")
        vision_state = self._extract_vision_state_dict(raw_state)

        if not vision_state:
            raise ValueError(
                f"No visual encoder weights found in checkpoint: {ckpt_path}"
            )

        if any(k.startswith("vision_model.") for k in vision_state):
            hf_state = vision_state
        else:
            hf_state = self._convert_openclip_vision_state_dict(vision_state)

        missing, unexpected = base_model.load_state_dict(hf_state, strict=False)
        if missing:
            print(f"Missing custom vision keys ({len(missing)}): {missing[:8]}")
        if unexpected:
            print(f"Unexpected custom vision keys ({len(unexpected)}): {unexpected[:8]}")

        return base_model

    def _resolve_custom_checkpoint_path(self, checkpoint_name):
        if checkpoint_name is None:
            raise ValueError("`ve_name` must be provided when `our_vision_encoder=True`.")

        checkpoint_name = os.path.expanduser(checkpoint_name)
        if os.path.isfile(checkpoint_name):
            return checkpoint_name

        if not os.path.isdir(checkpoint_name):
            raise FileNotFoundError(f"Custom vision checkpoint path not found: {checkpoint_name}")

        candidates = [
            "clip_unlearned.pt",
            "pytorch_model.bin",
            "pytorch_model.pt",
            "model.pt",
            "checkpoint.pt",
            "best.pt",
        ]
        for filename in candidates:
            path = os.path.join(checkpoint_name, filename)
            if os.path.isfile(path):
                return path

        pt_files = sorted(
            os.path.join(checkpoint_name, filename)
            for filename in os.listdir(checkpoint_name)
            if filename.endswith((".pt", ".bin"))
        )
        if len(pt_files) == 1:
            return pt_files[0]

        raise FileNotFoundError(
            f"Cannot locate a custom vision checkpoint under: {checkpoint_name}"
        )

    def _extract_vision_state_dict(self, raw_state):
        if isinstance(raw_state, dict):
            state = raw_state.get("model", raw_state.get("state_dict", raw_state))
        else:
            state = raw_state

        if not isinstance(state, dict):
            raise TypeError(f"Unsupported checkpoint format: {type(state)}")

        prefixes = (
            "visual.",
            "model.visual.",
            "module.visual.",
            "clip_model.visual.",
            "model.clip_model.visual.",
            "module.clip_model.visual.",
        )

        extracted = {}
        for key, value in state.items():
            clean_key = key
            if clean_key.startswith("module."):
                clean_key = clean_key[len("module."):]

            if clean_key.startswith("vision_model."):
                extracted[clean_key] = value
                continue

            for prefix in prefixes:
                if clean_key.startswith(prefix):
                    extracted["visual." + clean_key[len(prefix):]] = value
                    break

        return extracted

    def _convert_openclip_vision_state_dict(self, vision_state):
        hf_state = {}

        direct_map = {
            "visual.class_embedding": "vision_model.embeddings.class_embedding",
            "visual.conv1.weight": "vision_model.embeddings.patch_embedding.weight",
            "visual.positional_embedding": "vision_model.embeddings.position_embedding.weight",
            "visual.ln_pre.weight": "vision_model.pre_layrnorm.weight",
            "visual.ln_pre.bias": "vision_model.pre_layrnorm.bias",
            "visual.ln_post.weight": "vision_model.post_layernorm.weight",
            "visual.ln_post.bias": "vision_model.post_layernorm.bias",
        }

        for src_key, dst_key in direct_map.items():
            if src_key in vision_state:
                hf_state[dst_key] = vision_state[src_key]

        block_prefix = "visual.transformer.resblocks."
        for key, value in vision_state.items():
            if not key.startswith(block_prefix):
                continue

            suffix = key[len(block_prefix):]
            block_id, rest = suffix.split(".", 1)
            hf_prefix = f"vision_model.encoder.layers.{block_id}."

            if rest == "ln_1.weight":
                hf_state[hf_prefix + "layer_norm1.weight"] = value
            elif rest == "ln_1.bias":
                hf_state[hf_prefix + "layer_norm1.bias"] = value
            elif rest == "ln_2.weight":
                hf_state[hf_prefix + "layer_norm2.weight"] = value
            elif rest == "ln_2.bias":
                hf_state[hf_prefix + "layer_norm2.bias"] = value
            elif rest == "attn.out_proj.weight":
                hf_state[hf_prefix + "self_attn.out_proj.weight"] = value
            elif rest == "attn.out_proj.bias":
                hf_state[hf_prefix + "self_attn.out_proj.bias"] = value
            elif rest == "mlp.c_fc.weight":
                hf_state[hf_prefix + "mlp.fc1.weight"] = value
            elif rest == "mlp.c_fc.bias":
                hf_state[hf_prefix + "mlp.fc1.bias"] = value
            elif rest == "mlp.c_proj.weight":
                hf_state[hf_prefix + "mlp.fc2.weight"] = value
            elif rest == "mlp.c_proj.bias":
                hf_state[hf_prefix + "mlp.fc2.bias"] = value
            elif rest == "attn.in_proj_weight":
                q, k, v = value.chunk(3, dim=0)
                hf_state[hf_prefix + "self_attn.q_proj.weight"] = q
                hf_state[hf_prefix + "self_attn.k_proj.weight"] = k
                hf_state[hf_prefix + "self_attn.v_proj.weight"] = v
            elif rest == "attn.in_proj_bias":
                q, k, v = value.chunk(3, dim=0)
                hf_state[hf_prefix + "self_attn.q_proj.bias"] = q
                hf_state[hf_prefix + "self_attn.k_proj.bias"] = k
                hf_state[hf_prefix + "self_attn.v_proj.bias"] = v

        return hf_state

    def feature_select(self, image_forward_outs):
        image_features = image_forward_outs.hidden_states[self.select_layer]
        if self.select_feature == 'patch':
            image_features = image_features[:, 1:]
        elif self.select_feature == 'cls_patch':
            image_features = image_features
        else:
            raise ValueError(f'Unexpected select feature: {self.select_feature}')
        return image_features

    @torch.no_grad()
    def forward(self, images):
        if type(images) is list:
            image_features = []
            for image in images:
                image_forward_out = self.vision_tower(image.to(device=self.device, dtype=self.dtype).unsqueeze(0), output_hidden_states=True)
                image_feature = self.feature_select(image_forward_out).to(image.dtype)
                image_features.append(image_feature)
        else:
            image_forward_outs = self.vision_tower(images.to(device=self.device, dtype=self.dtype), output_hidden_states=True)
            image_features = self.feature_select(image_forward_outs).to(images.dtype)

        return image_features

    @property
    def dummy_feature(self):
        return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)

    @property
    def dtype(self):
        return self.vision_tower.dtype

    @property
    def device(self):
        return self.vision_tower.device

    @property
    def config(self):
        if self.is_loaded:
            return self.vision_tower.config
        else:
            return self.cfg_only

    @property
    def hidden_size(self):
        return self.config.hidden_size

    @property
    def num_patches_per_side(self):
        return self.config.image_size // self.config.patch_size

    @property
    def num_patches(self):
        return (self.config.image_size // self.config.patch_size) ** 2



class CLIPVisionTowerS2(CLIPVisionTower):
    def __init__(self, vision_tower, args, delay_load=False):
        super().__init__(vision_tower, args, delay_load)

        self.s2_scales = getattr(args, 's2_scales', '336,672,1008')
        self.s2_scales = list(map(int, self.s2_scales.split(',')))
        self.s2_scales.sort()
        self.s2_split_size = self.s2_scales[0]
        self.s2_image_size = self.s2_scales[-1]

        try:
            from s2wrapper import forward as multiscale_forward
        except ImportError:
            raise ImportError('Package s2wrapper not found! Please install by running: \npip install git+https://github.com/bfshi/scaling_on_scales.git')
        self.multiscale_forward = multiscale_forward

        # change resize/crop size in preprocessing to the largest image size in s2_scale
        if not delay_load or getattr(args, 'unfreeze_mm_vision_tower', False):
            self.image_processor.size['shortest_edge'] = self.s2_image_size
            self.image_processor.crop_size['height'] = self.image_processor.crop_size['width'] = self.s2_image_size

    def load_model(self, device_map=None):
        if self.is_loaded:
            print('{} is already loaded, `load_model` called again, skipping.'.format(self.vision_tower_name))
            return

        self.image_processor = CLIPImageProcessor.from_pretrained(self.vision_tower_name)
        self.vision_tower = CLIPVisionModel.from_pretrained(self.vision_tower_name, device_map=device_map)
        self.vision_tower.requires_grad_(False)

        self.image_processor.size['shortest_edge'] = self.s2_image_size
        self.image_processor.crop_size['height'] = self.image_processor.crop_size['width'] = self.s2_image_size

        self.is_loaded = True

    @torch.no_grad()
    def forward_feature(self, images):
        image_forward_outs = self.vision_tower(images.to(device=self.device, dtype=self.dtype), output_hidden_states=True)
        image_features = self.feature_select(image_forward_outs).to(images.dtype)
        return image_features

    @torch.no_grad()
    def forward(self, images):
        if type(images) is list:
            image_features = []
            for image in images:
                image_feature = self.multiscale_forward(self.forward_feature, image.unsqueeze(0), img_sizes=self.s2_scales, max_split_size=self.s2_split_size)
                image_features.append(image_feature)
        else:
            image_features = self.multiscale_forward(self.forward_feature, images, img_sizes=self.s2_scales, max_split_size=self.s2_split_size)

        return image_features

    @property
    def hidden_size(self):
        return self.config.hidden_size * len(self.s2_scales)
