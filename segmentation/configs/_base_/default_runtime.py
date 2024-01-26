# yapf:disable
log_config = {
    "interval": 50,
    "hooks": [
        {"type": "TextLoggerHook", "by_epoch": False},
        # dict(type='TensorboardLoggerHook')
    ]}
# yapf:enable
dist_params = {"backend": "nccl"}
log_level = "INFO"
load_from = None
resume_from = None
workflow = [("train", 1)]
cudnn_benchmark = True
find_unused_parameters = True
