# optimizer
optimizer = {"type": "SGD", "lr": 0.02, "weight_decay": 0.0001}
optimizer_config = {"grad_clip": None}
# learning policy
lr_config = {
    "policy": "step",
    "warmup": "linear",
    "warmup_iters": 500,
    "warmup_ratio": 0.001,
    "step": [8, 11],
}
runner = {"type": "EpochBasedRunner", "max_epochs": 12}
