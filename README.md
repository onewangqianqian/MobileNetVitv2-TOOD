## Running

### Install
We implement SSA-Co-DETR using [MMDetection V2.15.0](https://github.com/open-mmlab/mmdetection/releases/tag/v2.15.0) and [MMCV V1.5.0](https://github.com/open-mmlab/mmcv/releases/tag/v1.5.0).
The source code of MMdetection has been included in this repo and you only need to build MMCV following [official instructions](https://github.com/open-mmlab/mmcv/tree/v1.5.0#installation).
We test our models under ```python=3.7.11,pytorch=1.8.1```. Other versions may not be compatible. 

### Data
Solar Radio Burst Dataset: https://drive.google.com/file/d/1v3zkDgHatldortmoB8QE6Xw2NcyPOaU3/view?usp=drive_link. We are addressing the long-tail issue of the dataset in our next steps, and the dataset will be made publicly available upon completion (original data source: https://www.e-callisto.org/index.html).

The COCO dataset should be organized as:
```

── annotations
    ├── instances_train2017.json
    │      └── instances_val2017.json
    │── train2017
    └── val2017
      
```

### Training
Train:
```shell
python tools/train.py configs/tood/tood_mvit_v2_cefpn_1x_coco_anchorbase_1504.py
```
test:
```shell
python tools/testtrain.py configs/tood/tood_mvit_v2_cefpn_1x_coco_anchorbase_1504.py
```

