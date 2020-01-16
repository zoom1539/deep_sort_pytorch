import os
import cv2
import time
import argparse
import torch
import numpy as np
from distutils.util import strtobool

from detector import build_detector
from deep_sort import build_tracker
from utils.draw import draw_boxes
from utils.parser import get_config
from tqdm import tqdm


class Tracker(object):
    def __init__(self, cfg, args):
        self.cfg = cfg
        self.args = args
        use_cuda = args.use_cuda and torch.cuda.is_available()
        if not use_cuda:
            raise UserWarning("Running in cpu mode!")

        if args.display:
            cv2.namedWindow("test", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("test", args.display_width, args.display_height)

        self.video = cv2.VideoCapture()
        self.detector = build_detector(cfg, use_cuda=use_cuda)
        self.deepsort = build_tracker(cfg, use_cuda=use_cuda)
        self.class_names = self.detector.class_names


    def __enter__(self):
        assert os.path.isfile(self.args.video_path), "Error: path error"
        self.video.open(self.args.video_path)
        self.im_width = int(self.video.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.im_height = int(self.video.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if self.args.save_path:
            fourcc =  cv2.VideoWriter_fourcc(*'MJPG')
            self.writer = cv2.VideoWriter(self.args.save_path, fourcc, 20, (self.im_width,self.im_height))

        assert self.video.isOpened()
        return self

    
    def __exit__(self, exc_type, exc_value, exc_traceback):
        if exc_type:
            print(exc_type, exc_value, exc_traceback)
    
    def stack_identities(self, img, bboxes_xyxy, identities, identity_stack):     
        for i, id in enumerate(identities):
            if id >0 and id < 50:
                img_single_person = np.zeros(img.shape[:3], np.uint8)
                x1,y1,x2,y2 = [int(coord) for coord in bboxes_xyxy[i]]
                img_single_person[y1:y2, x1:x2] = img[y1:y2, x1:x2]
                identity_stack[id].append(img_single_person)

    def generate_video(self, frames, fps, resolution, save_path):
        fourcc = cv2.VideoWriter_fourcc('M','J','P','G')  
        videoWriter = cv2.VideoWriter(save_path, fourcc, fps, resolution)  
        for frame in frames:
            videoWriter.write(frame)
        videoWriter.release()

    def generate_snippet(self, identity_stack):
        fps = 25
        snippet_frame_num = 100
        for key, frames in identity_stack.items():
            snippet_num = len(frames) // snippet_frame_num
            for i in range(snippet_num):
                if i % 5 == 0:
                    file_name = self.args.video_path.split('.')[0].split('/')[-1]
                    path = self.args.save_dir + file_name + "_" + str(key) + "_" + str(i) + ".avi"
                    self.generate_video(frames[i * snippet_frame_num: (i + 1) * snippet_frame_num],
                                        fps,
                                        frames[0].shape[1::-1],
                                        path)

    def run_snippet(self):
        max_identities = 50
        identity_stack = dict()
        for id in range(1, max_identities):
            identity_stack[id] = []

        idx_frame = 0
        while self.video.grab(): 
            idx_frame += 1
            if idx_frame % self.args.frame_interval:
                continue

            start = time.time()
            _, ori_im = self.video.retrieve()
            im = cv2.cvtColor(ori_im, cv2.COLOR_BGR2RGB)

            # do detection
            bbox_xywh, cls_conf, cls_ids = self.detector(im)
            if bbox_xywh is not None:
                # select person class
                mask = cls_ids==0

                bbox_xywh = bbox_xywh[mask]
                bbox_xywh[:,3:] *= 1.2 # bbox dilation just in case bbox too small
                cls_conf = cls_conf[mask]

                # do tracking
                outputs = self.deepsort.update(bbox_xywh, cls_conf, im)

                # draw boxes for visualization
                if len(outputs) > 0:
                    bbox_xyxy = outputs[:,:4]
                    identities = outputs[:,-1]

                    # zmh add
                    self.stack_identities(ori_im, bbox_xyxy, identities, identity_stack)

                    ori_im = draw_boxes(ori_im, bbox_xyxy, identities)
                    

            end = time.time()
            # print("time: {:.03f}s, fps: {:.03f}".format(end-start, 1/(end-start)))

            if self.args.display:
                cv2.imshow("test", ori_im)
                cv2.waitKey(1)

            # if self.args.save_path:
            #     self.writer.write(ori_im)
        
        # zmh add
        self.generate_snippet(identity_stack)

    def run(self):

        idx_frame = 0
        while self.video.grab(): 
            idx_frame += 1
            if idx_frame % self.args.frame_interval:
                continue

            start = time.time()
            _, ori_im = self.video.retrieve()
            im = cv2.cvtColor(ori_im, cv2.COLOR_BGR2RGB)

            # do detection
            bbox_xywh, cls_conf, cls_ids = self.detector(im)
            if bbox_xywh is not None:
                # select person class
                mask = cls_ids==0

                bbox_xywh = bbox_xywh[mask]
                bbox_xywh[:,3:] *= 1.2 # bbox dilation just in case bbox too small
                cls_conf = cls_conf[mask]

                # do tracking
                outputs = self.deepsort.update(bbox_xywh, cls_conf, im)

                # draw boxes for visualization
                if len(outputs) > 0:
                    bbox_xyxy = outputs[:,:4]
                    identities = outputs[:,-1]

                    ori_im = draw_boxes(ori_im, bbox_xyxy, identities)

            end = time.time()
            print("time: {:.03f}s, fps: {:.03f}".format(end-start, 1/(end-start)))

            if self.args.display:
                cv2.imshow("test", ori_im)
                cv2.waitKey(1)

            if self.args.save_path:
                self.writer.write(ori_im)
        

def parse_args(video_path):
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_path", type=str, default = video_path)
    parser.add_argument("--config_detection", type=str, default="./configs/yolov3.yaml")
    parser.add_argument("--config_deepsort", type=str, default="./configs/deep_sort.yaml")
    parser.add_argument("--ignore_display", dest="display", action="store_false", default=True)
    parser.add_argument("--frame_interval", type=int, default=1)
    parser.add_argument("--display_width", type=int, default=800)
    parser.add_argument("--display_height", type=int, default=600)
    parser.add_argument("--save_path", type=str, default="./demo/demo.avi")
    parser.add_argument("--save_dir", type=str, default="/media/ubuntu/share/smoking_dataset/dataset_no_smoking_snippet/")
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
    work_dir = '/media/ubuntu/share/smoking_dataset/dataset_annotation/5/'
    list_path = []
    listdir(work_dir, list_path)

    for path in tqdm(list_path):
        args = parse_args(path)
        cfg = get_config()
        cfg.merge_from_file(args.config_detection)
        cfg.merge_from_file(args.config_deepsort)

        with Tracker(cfg, args) as trk:
            trk.run_snippet()
