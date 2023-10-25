_base_ = "../mmdetection/configs/yolo/yolov3_d53_mstrain-416_273e_coco.py"

# use caffe img_norm
img_norm_cfg = dict(mean=[0, 0, 0], std=[255., 255., 255.], to_rgb=True)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='Expand',
        mean=img_norm_cfg['mean'],
        to_rgb=img_norm_cfg['to_rgb'],
        ratio_range=(1, 2)),
    dict(type='Resize', img_scale=[(320, 320), (416, 416)], keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.0),
    dict(type='PhotoMetricDistortion'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(416, 416),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip', flip_ratio=0.0),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ])
]

#--------------------------------------DEEPLAKE INPUTS------------------------------------------------------------#
TOKEN = "INSERT_YOUR_DEEPLAKE_TOKEN"

data = dict(
    # samples_per_gpu=4, # Is used instead of batch_size if deeplake_dataloader is not specified below
    # workers_per_gpu=8, # Is used instead of num_workers if deeplake_dataloader is not specified below
    train=dict(
        pipeline=train_pipeline,

        # Credentials for authentication. See documendataion for deeplake.load() for details
        deeplake_path="hub://activeloop/coco-train",
        deeplake_credentials={
            "username": None,
            "password": None,
            "token": TOKEN,
            "creds": None,
        },
        #OPTIONAL - Checkout teh specified commit_id before training
        deeplake_commit_id="",
        #OPTIONAL - Loads a dataset view for training based on view_id
        deeplake_view_id="",

        # OPTIONAL - {"mmdet_key": "deep_lake_tensor",...} - Maps Deep Lake tensors to MMDET dictionary keys. 
        # If not specified, Deep Lake will auto-infer the mapping, but it might make mistakes if datasets have many tensors
        deeplake_tensors = {"img": "images", "gt_bboxes": "boxes", "gt_labels": "categories"},
        
        # OPTIONAL - Parameters to use for the Deep Lake dataloader. If unspecified, the integration uses
        # the parameters in other parts of the cfg file such as samples_per_gpu, and others.
        deeplake_dataloader = {"shuffle": True, "batch_size": 4, 'num_workers': 8}
    ),

    # Parameters as the same as for train
    val=dict(
        pipeline=test_pipeline,
        deeplake_path="hub://activeloop/coco-val",
        deeplake_credentials={
            "username": None,
            "password": None,
            "token": TOKEN,
            "creds": None,
        },
        deeplake_tensors = {"img": "images", "gt_bboxes": "boxes", "gt_labels": "categories"},
        deeplake_dataloader = {"shuffle": False, "batch_size": 1, 'num_workers': 8}
    ),
)

# Which dataloader to use
deeplake_dataloader_type = "c++"  # "c++" is available to enterprise users. Otherwise use "python"

# Which metrics to use for evaulation. In MMDET (without Deeplake), this is inferred from the dataset type.
# In the Deep Lake integration, since the format is standardized, a variety of metrics can be used for a given dataset.
deeplake_metrics_format = "COCO"

#----------------------------------END DEEPLAKE INPUTS------------------------------------------------------------#

evaluation = dict(metric=["bbox"], interval=10)

load_from = "checkpoints/yolov3_d53_mstrain-416_273e_coco-2b60fcd9.pth"

work_dir = "./mmdet_outputs"

log_config = dict(interval=10)

checkpoint_config = dict(interval=12)

seed = None
gpu_ids = range(1)

device = "cuda"

runner = dict(type='EpochBasedRunner', max_epochs=10)