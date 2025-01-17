# model settings
model = {
    "type": "RetinaNet",
    "pretrained": "torchvision://resnet50",
    "backbone": {
        "type": "ResNet",
        "depth": 50,
        "num_stages": 4,
        "out_indices": (0, 1, 2, 3),
        "frozen_stages": 1,
        "norm_cfg": {"type": "BN", "requires_grad": True},
        "norm_eval": True,
        "style": "pytorch",
    },
    "neck": {
        "type": "FPN",
        "in_channels": [256, 512, 1024, 2048],
        "out_channels": 256,
        "start_level": 1,
        "add_extra_convs": "on_input",
        "num_outs": 5,
    },
    "bbox_head": {
        "type": "RetinaHead",
        "num_classes": 80,
        "in_channels": 256,
        "stacked_convs": 4,
        "feat_channels": 256,
        "anchor_generator": {
            "type": "AnchorGenerator",
            "octave_base_scale": 4,
            "scales_per_octave": 3,
            "ratios": [0.5, 1.0, 2.0],
            "strides": [8, 16, 32, 64, 128],
        },
        "bbox_coder": {
            "type": "DeltaXYWHBBoxCoder",
            "target_means": [0.0, 0.0, 0.0, 0.0],
            "target_stds": [1.0, 1.0, 1.0, 1.0],
        },
        "loss_cls": {
            "type": "FocalLoss",
            "use_sigmoid": True,
            "gamma": 2.0,
            "alpha": 0.25,
            "loss_weight": 1.0,
        },
        "loss_bbox": {"type": "L1Loss", "loss_weight": 1.0},
    },
}
# training and testing settings
train_cfg = {
    "assigner": {
        "type": "MaxIoUAssigner",
        "pos_iou_thr": 0.5,
        "neg_iou_thr": 0.4,
        "min_pos_iou": 0,
        "ignore_iof_thr": -1,
    },
    "allowed_border": -1,
    "pos_weight": -1,
    "debug": False,
}
test_cfg = {
    "nms_pre": 1000,
    "min_bbox_size": 0,
    "score_thr": 0.05,
    "nms": {"type": "nms", "iou_threshold": 0.5},
    "max_per_img": 100,
}
