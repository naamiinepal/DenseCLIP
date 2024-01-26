# model settings
norm_cfg = {"type": "SyncBN", "requires_grad": True}
model = {
    "type": "EncoderDecoder",
    "pretrained": "open-mmlab://resnet50_v1c",
    "backbone": {
        "type": "ResNetV1c",
        "depth": 50,
        "num_stages": 4,
        "out_indices": (0, 1, 2, 3),
        "dilations": (1, 1, 1, 1),
        "strides": (1, 2, 2, 2),
        "norm_cfg": norm_cfg,
        "norm_eval": False,
        "style": "pytorch",
        "contract_dilation": True,
    },
    "neck": {
        "type": "FPN",
        "in_channels": [256, 512, 1024, 2048],
        "out_channels": 256,
        "num_outs": 4,
    },
    "decode_head": {
        "type": "FPNHead",
        "in_channels": [256, 256, 256, 256],
        "in_index": [0, 1, 2, 3],
        "feature_strides": [4, 8, 16, 32],
        "channels": 128,
        "dropout_ratio": 0.1,
        "num_classes": 19,
        "norm_cfg": norm_cfg,
        "align_corners": False,
        "loss_decode": {
            "type": "CrossEntropyLoss",
            "use_sigmoid": False,
            "loss_weight": 1.0,
        },
    },
    # model training and testing settings
    "train_cfg": {},
    "test_cfg": {"mode": "slide", "crop_size": (512, 512), "stride": (341, 341)},
}
