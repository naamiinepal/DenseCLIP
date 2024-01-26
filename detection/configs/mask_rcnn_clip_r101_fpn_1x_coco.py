_base_ = [
    "../configs/_base_/models/mask_rcnn_r50_fpn.py",
    "../configs/_base_/datasets/coco_instance_clip.py",
    # '../configs/_base_/schedules/schedule_1x.py',
    "../configs/_base_/default_runtime.py",
]

model = {
    "pretrained": "pretrained/RN101.pt",
    "backbone": {
        "type": "CLIPResNet",
        "layers": [3, 4, 23, 3],
        "output_dim": 1024,
        "input_resolution": 1344,
        "style": "pytorch",
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
