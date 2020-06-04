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
    def __init__(self, cfg, args, video_path):
        self.cfg = cfg
        self.args = args
        self.video_path = video_path
        use_cuda = args.use_cuda and torch.cuda.is_available()
        if not use_cuda:
            raise UserWarning("Running in cpu mode!")
       
        self.video = cv2.VideoCapture()
        self.detector = build_detector(cfg, use_cuda=use_cuda)
        self.deepsort = build_tracker(cfg, use_cuda=use_cuda)
        self.class_names = self.detector.class_names

        self.snippet_id = 0


    def __enter__(self):
        assert os.path.isfile(self.video_path), "Error: path error"
        
        self.video.open(self.video_path)
        self.fps = int(self.video.get(cv2.CAP_PROP_FPS))
        self.fourcc = cv2.VideoWriter_fourcc('M','J','P','G')  

        assert self.video.isOpened()
        return self

    
    def __exit__(self, exc_type, exc_value, exc_traceback):
        if exc_type:
            print(exc_type, exc_value, exc_traceback)
    

    def stack_track(self, frame_id, track_ids, bboxes_xyxy, img_bgr, track_dict):
        for track_id, bbox_xyxy in zip(track_ids, bboxes_xyxy):
            if track_id not in list(track_dict.keys()):
                track_dict[track_id] = list()

            xmin, ymin, xmax, ymax = [int(coord) for coord in bbox_xyxy]
            img_bbox = img_bgr[ymin:ymax, xmin:xmax]
            track_dict[track_id].append((img_bbox, bbox_xyxy, frame_id))

    def parse_annotation(self):
        annotation_path = self.video_path.replace('avi','xml')
        annotation_tree = ET.ElementTree(file = annotation_path)

        annotation = dict()
        for track in annotation_tree.iter(tag='track'):
            for box in track:
                frame_id = int(box.attrib['frame'])
                xtl = int(float(box.attrib['xtl']))
                ytl = int(float(box.attrib['ytl']))
                xbr = int(float(box.attrib['xbr']))
                ybr = int(float(box.attrib['ybr']))
                bbox_center = ((xtl + xbr) * 0.5, (ytl + ybr) * 0.5)

                if frame_id not in list(annotation.keys()):
                    annotation[frame_id] = list()
                
                annotation[frame_id].append(bbox_center)

        return annotation
    
    def find_max_size(self, infoes):
        max_width = 0
        max_height = 0
        for info in infoes:
            xmin, ymin, xmax, ymax = info[1]
            if (xmax - xmin + 1) > max_width:
                max_width = (xmax - xmin + 1)
            if (ymax - ymin + 1) > max_height:
                max_height = (ymax - ymin + 1)

        return max_width, max_height
    
    def uniform_img_bbox(self, max_width, max_height, img_bbox):
        img = np.zeros((max_height, max_width, img_bbox.shape[2]), np.uint8)
        img[0:img_bbox.shape[0], 0:img_bbox.shape[1]] = img_bbox
        return img

    def is_closed(self, center, bbox_center, closed_thres):
        diff = (abs(center[0] - bbox_center[0]), abs(center[1] - bbox_center[1]))
        if diff[0] < closed_thres[0] and diff[1] < closed_thres[1]:
            return True
        return False
    
    def find_label(self, annotation, bbox_xyxy, frame_id):
        if frame_id in list(annotation.keys()):
            bbox_centers = annotation[frame_id]
            center = ((bbox_xyxy[0] + bbox_xyxy[2]) * 0.5, (bbox_xyxy[1] + bbox_xyxy[3]) * 0.5)
            closed_thres = ((bbox_xyxy[2] - bbox_xyxy[0]) * 0.3, (bbox_xyxy[3] - bbox_xyxy[1]) * 0.3)
            for bbox_center in bbox_centers:
                if self.is_closed(center, bbox_center, closed_thres):
                    return 1 # smoking

        return 0 # not smoking

    def save_snippets(self, snippets, resolution):
        dir_name_id = self.video_path.split('/')[-2]
        dir_name_video = self.video_path.split('/')[-1].split('.')[0]
        dir_video = os.path.join(self.args.work_dir, dir_name_id, dir_name_video)
        is_exist = os.path.exists(dir_video)
        if not is_exist:                   
            os.makedirs(dir_video)
          
        for key, snippet in snippets.items():
            if len(snippet) >= 10:
                save_path_video = os.path.join(dir_video, str(self.snippet_id) + '.avi')
                save_path_label = os.path.join(dir_video, str(self.snippet_id) + '.json')
                
                videoWriter = cv2.VideoWriter(save_path_video, self.fourcc, self.fps, resolution)
                labels = list()
                
                for frame, bbox_xyxy, label in snippet:
                    videoWriter.write(frame)
                    labels.append((bbox_xyxy.tolist(), label))

                videoWriter.release()
                with open(save_path_label, 'w') as json_writer:
                    json.dump(labels, json_writer)

                self.snippet_id += 1

    def generate_snippet(self, track_dict):
        annotation = self.parse_annotation()
        for track_id, infoes in track_dict.items():
            max_width, max_height = self.find_max_size(infoes)
            snippets = dict()
            key = -1
            for i, info in enumerate(infoes):
                img_bbox = info[0]
                bbox_xyxy = info[1]
                frame_id = info[2]

                if 0 == i:
                    frame_id_previous = frame_id

                if frame_id - frame_id_previous != 1:
                    key += 1
                    snippets[key] = []
                
                frame = self.uniform_img_bbox(max_width, max_height, img_bbox)
                label = self.find_label(annotation, bbox_xyxy, frame_id)

                snippets[key].append((frame, bbox_xyxy, label))

                frame_id_previous = frame_id
            
            self.save_snippets(snippets, (max_width, max_height))

    def run(self):
        track_dict = dict()
        
        frame_id = -1
        while self.video.grab():
            frame_id += 1
            # print('frame_id: ',frame_id)

            _, img_bgr = self.video.retrieve()
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

            bboxes_xywh, cls_confs, cls_ids = self.detector(img_rgb)
            if bboxes_xywh is not None:
                mask = (cls_ids == 0)
                bbox_xywh_person = bboxes_xywh[mask]
                cls_conf_person = cls_confs[mask]
                
                track_outputs = self.deepsort.update(bbox_xywh_person, cls_conf_person, img_rgb)
                if len(track_outputs) > 0:
                    bboxes_xyxy = track_outputs[:,:4]
                    track_ids = track_outputs[:,-1]

                    self.stack_track(frame_id, track_ids, bboxes_xyxy, img_bgr, track_dict)
            
        # print('stack_track done')
        self.generate_snippet(track_dict)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_detection", type=str, default="./configs/yolov3.yaml")
    parser.add_argument("--config_deepsort", type=str, default="./configs/deep_sort.yaml")
    parser.add_argument("--dataset_dir", type=str, default = '/data1/dataset/smoking/dataset_annotation')
    parser.add_argument("--work_dir", type=str, default="/data1/dataset/smoking/dataset_annotation_work_dir")
    parser.add_argument("--cpu", dest="use_cuda", action="store_false", default=True)
    return parser.parse_args()

def listdir(dir, list_name):
    for file in os.listdir(dir):
        file_path = os.path.join(dir, file)
        if os.path.isdir(file_path):
            listdir(file_path, list_name)
        elif os.path.splitext(file_path)[1]=='.avi':
            list_name.append(file_path)

if __name__=="__main__":
    args = parse_args()

    video_paths = []
    listdir(args.dataset_dir, video_paths)

    for video_path in tqdm(video_paths):
        cfg = get_config()
        cfg.merge_from_file(args.config_detection)
        cfg.merge_from_file(args.config_deepsort)

        with Tracker(cfg, args, video_path) as trk:
            trk.run()