dataset_type = "CocoDataset"
data_root = "data/coco/"
IMAGE_MEAN = [v * 255 for v in (0.48145466, 0.4578275, 0.40821073)]
IMAGE_VAR = [v * 255 for v in (0.26862954, 0.26130258, 0.27577711)]
img_norm_cfg = {"mean": IMAGE_MEAN, "std": IMAGE_VAR, "to_rgb": True}
train_pipeline = [
    {"type": "LoadImageFromFile"},
    {"type": "LoadAnnotations", "with_bbox": True},
    {"type": "Resize", "img_scale": (1333, 800), "keep_ratio": True},
    {"type": "RandomFlip", "flip_ratio": 0.5},
    dict(type="Normalize", **img_norm_cfg),
    {"type": "Pad", "size_divisor": 32},
    {"type": "DefaultFormatBundle"},
    {"type": "Collect", "keys": ["img", "gt_bboxes", "gt_labels"]},
]
test_pipeline = [
    {"type": "LoadImageFromFile"},
    {
        "type": "MultiScaleFlipAug",
        "img_scale": (1333, 800),
        "flip": False,
        "transforms": [
            {"type": "Resize", "keep_ratio": True},
            {"type": "RandomFlip"},
            dict(type="Normalize", **img_norm_cfg),
            {"type": "Pad", "size_divisor": 32},
            {"type": "ImageToTensor", "keys": ["img"]},
            {"type": "Collect", "keys": ["img"]},
        ],
    },
]
data = {
    "samples_per_gpu": 2,
    "workers_per_gpu": 6,
    "train": {
        "type": dataset_type,
        "ann_file": data_root + "annotations/instances_train2017.json",
        "img_prefix": data_root + "train2017/",
        "pipeline": train_pipeline,
    },
    "val": {
        "type": dataset_type,
        "ann_file": data_root + "annotations/instances_val2017.json",
        "img_prefix": data_root + "val2017/",
        "pipeline": test_pipeline,
    },
    "test": {
        "type": dataset_type,
        "ann_file": data_root + "annotations/instances_val2017.json",
        "img_prefix": data_root + "val2017/",
        "pipeline": test_pipeline,
    },
}
evaluation = {"interval": 1, "metric": "bbox"}
