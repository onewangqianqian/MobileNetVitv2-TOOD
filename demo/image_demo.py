# Copyright (c) OpenMMLab. All rights reserved.
import asyncio
import os
from argparse import ArgumentParser

from mmdet.apis import (async_inference_detector, inference_detector,
                        init_detector, show_result_pyplot)


def parse_args():
    parser = ArgumentParser()
    # parser.add_argument('--img', default='/mnt/d/deeplearning/pili_up/deep-learning-for-image-processing/pytorch_object_detection/ssd/cocolike_bur4_COCO/images/val2017/', help='Image file')
    parser.add_argument('--img', default='/mnt/d/deeplearning/voclike_bur2/VOCdevkit/VOC2012/JPEGImages/', help='Image file')
    parser.add_argument('--config', default='/mnt/d/deeplearning/no_bur/mmdetection/configs/ssd/ssdlite_mobilevit_ciouloss.py', help='Config file')
    parser.add_argument('--checkpoint', default='/mnt/d/deeplearning/no_bur/mmdetection/workspace/keshihua/mvit_ciouloss/best_bbox_mAP_epoch_107_ciou.pth', help='Checkpoint file')
    parser.add_argument('--out-file', default='/mnt/d/deeplearning/no_bur/mmdetection/workspace/keshihua/mvit_ciouloss/demo_result2/', help='Path to output file')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--palette',
        default='coco',
        choices=['coco', 'voc', 'citys', 'random'],
        help='Color palette used for visualization')
    parser.add_argument(
        '--score-thr', type=float, default=0.3, help='bbox score threshold')
    parser.add_argument(
        '--async-test',
        action='store_true',
        help='whether to set async options for async inference.')
    args = parser.parse_args()
    return args


# def main(args):
#     # build the model from a config file and a checkpoint file
#     model = init_detector(args.config, args.checkpoint, device=args.device)
#     # test a single image
#     result = inference_detector(model, args.img)
#     # show the results
#     show_result_pyplot(
#         model,
#         args.img,
#         result,
#         palette=args.palette,
#         score_thr=args.score_thr,
#         out_file=args.out_file)

def main(args):
    model = init_detector(args.config, args.checkpoint, device=args.device)
    for img in os.listdir(args.img):
        img_path = os.path.join(args.img, img)
        out_file = os.path.join(args.out_file, img)
        result = inference_detector(model, img_path)

        show_result_pyplot(
            model,
            img_path,
            result,
            palette=args.palette,
            score_thr=args.score_thr,
            out_file=out_file)

async def async_main(args):
    # build the model from a config file and a checkpoint file
    model = init_detector(args.config, args.checkpoint, device=args.device)
    # test a single image
    tasks = asyncio.create_task(async_inference_detector(model, args.img))
    result = await asyncio.gather(tasks)
    # show the results
    show_result_pyplot(
        model,
        args.img,
        result[0],
        palette=args.palette,
        score_thr=args.score_thr,
        out_file=args.out_file)


if __name__ == '__main__':
    # data_root = '/mnt/d/deeplearning/pili_up/deep-learning-for-image-processing/pytorch_object_detection/ssd/cocolike_bur4_COCO/images/val2017/'
    # data_testimg = data_root + 'images/val2017/'
    # data_testimg = '/mnt/d/deeplearning/voclike_bur2/VOCdevkit/VOC2012/JPEGImages/'
    # for img_path in os.listdir(data_testimg):
    #     img = os.path.join(data_testimg, img_path)
    args = parse_args()
    if args.async_test:
        asyncio.run(async_main(args))
    else:
        main(args)
