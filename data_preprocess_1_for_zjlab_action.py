import os
import cv2
import time
import argparse
import torch
import numpy as np
from distutils.util import strtobool
import xml.etree.ElementTree as ET
from tqdm import tqdm
import json

from detector import build_detector
from deep_sort import build_tracker
from utils.draw import draw_boxes
from utils.parser import get_config


class Tracker(object):
    def __init__(self, cfg, args):
        self.cfg = cfg
        self.args = args

        # self.deepsort = build_tracker(cfg, use_cuda=use_cuda)
        use_cuda = self.args.use_cuda and torch.cuda.is_available()
        if not use_cuda:
            raise UserWarning("Running in cpu mode!")
        self.detector = build_detector(cfg, use_cuda=use_cuda)

        self.class_names = self.detector.class_names

        self.snippet_id = 0

    def run(self, video_path):
        #
        video = cv2.VideoCapture()
        video.open(video_path)
        fps = int(video.get(cv2.CAP_PROP_FPS))
        fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
        height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        resolution = (width, height)

        #
        bbox_dict = dict()

        save_path_video = video_path.replace('.avi', '_bbox_.avi')
        videoWriter = cv2.VideoWriter(
            save_path_video, fourcc, fps, resolution)

        frame_id = -1
        while video.grab():
            frame_id += 1
            # print('frame_id: ',frame_id)

            _, img_bgr = video.retrieve()
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

            bboxes_xywh, cls_confs, cls_ids = self.detector(img_rgb)
            if bboxes_xywh is not None:

                mask = (cls_ids == 0)
                bboxes_xywh_person = bboxes_xywh[mask]
                cls_conf_person = cls_confs[mask]

                bboxes_xyxy_person = list()
                img_rgb
                for bbox_xywh in bboxes_xywh_person:
                    xmin = int(bbox_xywh[0] - 0.6 * bbox_xywh[2])
                    xmin = 0 if xmin < 0 else xmin
                    xmin = width if xmin > width else xmin

                    ymin = int(bbox_xywh[1] - 0.6 * bbox_xywh[3])
                    ymin = 0 if ymin < 0 else ymin
                    ymin = height if ymin > height else ymin

                    xmax = int(bbox_xywh[0] + 0.6 * bbox_xywh[2])
                    xmax = 0 if xmax < 0 else xmax
                    xmax = width if xmax > width else xmax

                    ymax = int(bbox_xywh[1] + 0.6 * bbox_xywh[3])
                    ymax = 0 if ymax < 0 else ymax
                    ymax = height if ymax > height else ymax

                    bboxes_xyxy_person.append([xmin, ymin, xmax, ymax])

                #
                bbox_dict[frame_id] = np.array(bboxes_xyxy_person).tolist()

                #
                for bbox_xyxy_person in bboxes_xyxy_person:
                    cv2.rectangle(
                        img_bgr, (bbox_xyxy_person[0], bbox_xyxy_person[1]),
                        (bbox_xyxy_person[2], bbox_xyxy_person[3]), (0, 255, 0), 2)
            else:
                bbox_dict[frame_id] = list()

            videoWriter.write(img_bgr)

        #
        videoWriter.release()

        # print('stack_track done')
        save_path_bbox = video_path.replace('.avi', '_bbox.json')
        with open(save_path_bbox, 'w') as json_writer:
            json.dump(bbox_dict, json_writer)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_detection", type=str,
                        default="./configs/yolov3.yaml")
    parser.add_argument("--config_deepsort", type=str,
                        default="./configs/deep_sort.yaml")
    parser.add_argument("--dataset_dir", type=str,
                        default='/data1/zhumh/zjlab_action_segments')
    parser.add_argument("--work_dir", type=str,
                        default="/data1/zhumh/yolo_test/work_dir")
    parser.add_argument("--cpu", dest="use_cuda",
                        action="store_false", default=True)
    return parser.parse_args()


def listdir(dir, list_name):
    for file in os.listdir(dir):
        file_path = os.path.join(dir, file)
        if os.path.isdir(file_path):
            listdir(file_path, list_name)
        elif os.path.splitext(file_path)[1] == '.avi' and file_path.find('_.') == -1:
            list_name.append(file_path)


if __name__ == "__main__":
    args = parse_args()

    video_paths = []
    listdir(args.dataset_dir, video_paths)
    cfg = get_config()
    cfg.merge_from_file(args.config_detection)
    cfg.merge_from_file(args.config_deepsort)
    
    trk = Tracker(cfg, args)
    for video_path in tqdm(video_paths):
            trk.run(video_path)
