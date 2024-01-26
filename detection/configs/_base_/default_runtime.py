checkpoint_config = {"interval": 1}
# yapf:disable
log_config = {
    "interval": 50,
    "hooks": [
        {"type": "TextLoggerHook"},
        # dict(type='TensorboardLoggerHook')
    ]}
# yapf:enable
dist_params = {"backend": "nccl"}
log_level = "INFO"
load_from = None
resume_from = None
workflow = [("train", 1)]
# fp16 = dict(loss_scale=512.)
find_unused_parameters = True
