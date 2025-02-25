import os
import sys
import mmcv
import numpy as np
import argparse
import matplotlib.pyplot as plt

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from mmcv import Config
from mmdet.datasets import build_dataset


def plot_pr_curve(config_file, result_file, out_pic, metric="bbox"):
    """plot precison-recall curve based on testing results of pkl file.

        Args:
            config_file (list[list | tuple]): config file path.
            result_file (str): pkl file of testing results path.
            metric (str): Metrics to be evaluated. Options are
                'bbox', 'segm'.
    """

    cfg = Config.fromfile(config_file)
    # turn on test mode of dataset
    if isinstance(cfg.data.test, dict):
        cfg.data.test.test_mode = True
    elif isinstance(cfg.data.test, list):
        for ds_cfg in cfg.data.test:
            ds_cfg.test_mode = True

    # build dataset
    dataset = build_dataset(cfg.data.test)
    # load result file in pkl format
    pkl_results = mmcv.load(result_file)
    # convert pkl file (list[list | tuple | ndarray]) to json
    json_results, _ = dataset.format_results(pkl_results)
    # initialize COCO instance
    coco = COCO(annotation_file=cfg.data.test.ann_file)
    coco_gt = coco
    coco_dt = coco_gt.loadRes(json_results[metric])
    # initialize COCOeval instance
    coco_eval = COCOeval(coco_gt, coco_dt, metric)
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    # extract eval data
    precisions = coco_eval.eval["precision"]
    '''
    precisions[T, R, K, A, M]
    T: iou thresholds [0.5 : 0.05 : 0.95], idx from 0 to 9
    R: recall thresholds [0 : 0.01 : 1], idx from 0 to 100
    K: category, idx from 0 to ...
    A: area range, (all, small, medium, large), idx from 0 to 3
    M: max dets, (1, 10, 100), idx from 0 to 2
    '''
    pr_array1 = precisions[0, :, 0, 0, 2]
    pr_array2 = precisions[1, :, 0, 0, 2]
    pr_array3 = precisions[2, :, 0, 0, 2]
    pr_array4 = precisions[3, :, 0, 0, 2]
    pr_array5 = precisions[4, :, 0, 0, 2]
    pr_array6 = precisions[5, :, 0, 0, 2]
    pr_array7 = precisions[6, :, 0, 0, 2]
    pr_array8 = precisions[7, :, 0, 0, 2]
    pr_array9 = precisions[8, :, 0, 0, 2]
    pr_array10 = precisions[9, :, 0, 0, 2]

    x = np.arange(0.0, 1.01, 0.01)
    # plot PR curve
    plt.plot(x, pr_array1, label="iou=0.5")
    plt.plot(x, pr_array2, label="iou=0.55")
    plt.plot(x, pr_array3, label="iou=0.6")
    plt.plot(x, pr_array4, label="iou=0.65")
    plt.plot(x, pr_array5, label="iou=0.7")
    plt.plot(x, pr_array6, label="iou=0.75")
    plt.plot(x, pr_array7, label="iou=0.8")
    plt.plot(x, pr_array8, label="iou=0.85")
    plt.plot(x, pr_array9, label="iou=0.9")
    plt.plot(x, pr_array10, label="iou=0.95")

    plt.xlabel("recall")
    plt.ylabel("precison")
    plt.xlim(0, 1.0)
    plt.ylim(0, 1.01)
    plt.grid(True)
    plt.legend(loc="lower left")
    plt.savefig(out_pic)

# python plot_pr_curve.py configs/my_coco_config/my_coco_config.py faster_rcnn_fpn_coco.pkl
# python tools/analyze_logs.py plot_curve work_dirs/faster_rcnn_r50_fpn_1x/20200305_162713.log.json
# work_dirs/faster_rcnn_r50_fpn_1x/20200306_175509.log.json --keys acc --legend run1 run2

# 画图1：画map、loss曲线
# 对于同一文件，不同指标
# python tools/analysis_tools/analyze_logs.py plot_curve worksp
# ace/task2/anchor_based_ls/20240220_173003_all.log.json --keys loss_cls loss_bbox --legend loss_cls loss_bbox --out workspace/task2/anchor_based_ls/loss_cls_bbox.png
# 对于不同文件，同一指标
# python tools/analysis_tools/analyze_logs.py plot_curve jsonfile1 jsonfile2 --keys loss_cls --legend loss_cls1 loss_cls2 --out workspace/task2/anchor_based_ls/loss_cls_bbox.png
# python tools/analysis_tools/analyze_logs.py plot_curve workspace/task2/anchor_based_ls/20240220_173003_all.log.json workspace/task2/anchor_free_l6/20240311_152907_all.log.json --legend anchor_based anchor_free --out workspace/task2/anchor_based_ls/mAP_free6_base5.png
# 画图2：画pr曲线

# 画图3：画可视化图

# 画图4：log文件可以读取各个类别的变化。
# 画图5：画混淆矩阵
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument('--config', default='/mnt/d/deeplearning/no_bur/mmdetection/configs/tood/tood_mvit_v2_cefpn_1x_coco_anchorbase_1504.py', help='config file path')
    # parser.add_argument('--pkl_result_file', default='/mnt/d/deeplearning/no_bur/mmdetection/workspace/task2/anchor_based_ls/best_bbox_mAP_epoch_223.pkl', help='pkl result file path')
    # parser.add_argument('--out', default='/mnt/d/deeplearning/no_bur/mmdetection/workspace/task2/anchor_based_ls/pr_curve_4.png')

    parser.add_argument('--config', default='/mnt/d/deeplearning/no_bur/mmdetection/configs/ssd/ssdlite_mobilevit_ciouloss.py', help='config file path')
    parser.add_argument('--pkl_result_file', default='/mnt/d/deeplearning/no_bur/mmdetection/workspace/keshihua/mvit_ciouloss/mvit_ciouloss.pkl', help='pkl result file path')
    parser.add_argument('--out', default='/mnt/d/deeplearning/no_bur/mmdetection/workspace/keshihua/ke2/pr_curve_mvit2.png')

    parser.add_argument('--eval', default='bbox')
    cfg = parser.parse_args()

    plot_pr_curve(config_file=cfg.config, result_file=cfg.pkl_result_file, out_pic=cfg.out, metric=cfg.eval)
