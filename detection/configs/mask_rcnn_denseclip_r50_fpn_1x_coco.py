_base_ = [
    "../configs/_base_/models/mask_rcnn_r50_fpn.py",
    "../configs/_base_/datasets/coco_instance_clip.py",
    # '../configs/_base_/schedules/schedule_1x.py',
    "../configs/_base_/default_runtime.py",
]

model = {
    "type": "DenseCLIP_MaskRCNN",
    "pretrained": "pretrained/RN50.pt",
    "context_length": 5,
    "seg_loss": True,
    "clip_head": False,
    "backbone": {
        "type": "CLIPResNetWithAttention",
        "layers": [3, 4, 6, 3],
        "output_dim": 1024,
        "input_resolution": 1344,
        "style": "pytorch",
    },
    "text_encoder": {
        "type": "CLIPTextContextEncoder",
        "context_length": 13,
        "embed_dim": 1024,
        "transformer_width": 512,
        "transformer_heads": 8,
        "transformer_layers": 12,
        "style": "pytorch",
    },
    "context_decoder": {
        "type": "ContextDecoder",
        "transformer_width": 256,
        "transformer_heads": 4,
        "transformer_layers": 3,
        "visual_dim": 1024,
        "dropout": 0.1,
        "outdim": 1024,
        "style": "pytorch",
    },
    "neck": {
        "type": "FPN",
        "in_channels": [256, 512, 1024, 2048 + 80],
        "out_channels": 256,
        "num_outs": 5,
    },
}
# optimizer
optimizer = {
    "type": "AdamW",
    "lr": 0.0002,
    "weight_decay": 0.0001,
    "paramwise_cfg": {
        "custom_keys": {
            "backbone": {"lr_mult": 0.1},
            "text_encoder": {"lr_mult": 0.0},
            "norm": {"decay_mult": 0.0},
        },
    },
}
optimizer_config = {"grad_clip": {"max_norm": 0.1, "norm_type": 2}}
# learning policy
lr_config = {
    "policy": "step",
    "warmup": "linear",
    "warmup_iters": 500,
    "warmup_ratio": 0.001,
    "step": [8, 11],
}
total_epochs = 12
