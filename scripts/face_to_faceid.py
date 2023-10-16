from deep_sort.parser import get_config
# from deep_sort.draw import draw_boxes
from deep_sort import build_tracker
from PIL import Image 
import argparse
import os
import pandas as pd
import time
import numpy as np
import warnings
import cv2
from deep_sort.sort.track import Track
import torch
import torch.backends.cudnn as cudnn
from retinaface import RetinaFace
from deepface import DeepFace
from deepface.commons import functions
from face_search import io
# face detector
#from facenet_pytorch import MTCNN

import sys

currentUrl = os.path.dirname(__file__)
sys.path.append(os.path.abspath(os.path.join(currentUrl, 'yolov5')))


# cudnn.benchmark = True


def calc_iou(xywh1, xywh2):
    x1 = xywh1[:,0][:,np.newaxis]
    y1 = xywh1[:,1][:,np.newaxis]
    w1 = xywh1[:,2][:,np.newaxis]
    h1 = xywh1[:,3][:,np.newaxis]

    x2 = xywh2[:,0][np.newaxis,:]
    y2 = xywh2[:,1][np.newaxis,:]
    w2 = xywh2[:,2][np.newaxis,:] 
    h2 = xywh2[:,3][np.newaxis,:]

    xA = np.maximum(x1,x2)
    yA = np.maximum(y1,y2)
    xB = np.minimum(x1+w1, x2+w2)
    yB = np.minimum(y1+h1, y2+h2)
    intersection = np.abs(np.maximum(0, xB-xA) * np.maximum(0,yB - yA))
    A1 = w1 * h1 
    A2 = w2 * h2 
    iou = intersection / (A1 + A2 - intersection)
    return iou



class VideoTracker(object):
    def __init__(self, args,video_fname):
        # ***************** Initialize ******************************************************
        self.args = args
        self.margin_ratio = args.margin_ratio
        self.root_dir = os.path.splitext(video_fname)[0] + '.pipeline'
        self.db_fname = os.path.join(self.root_dir, 'faces.csv')
        

        self.db = pd.read_csv(self.db_fname)           # 0.2

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.device = torch.device('cpu')
        self.half = self.device.type != 'cpu'  # half precision only supported on CUDA

        # ***************************** initialize DeepSORT **********************************
        cfg = get_config()
        cfg.merge_from_file(args.config_deepsort)

        # use_cuda = self.device.type != 'cpu' and torch.cuda.is_available()
        use_cuda = False
        self.deepsort = build_tracker(cfg, use_cuda=use_cuda)

        if self.device == 'cpu':
            logging.warn("Running in cpu mode which maybe very slow!", UserWarning)

    def __enter__(self):
        # ************************* Load video from camera *************************
        self.frames = list(set(self.db['frame_num'].values))
        self.cur_frame = 0
        frame_num = self.frames[0]
        img_dir = os.path.join(self.root_dir, f'frame_{frame_num:04d}.png')
        img = Image.open(img_dir)
        self.im_width = img.size[0]
        self.im_height = img.size[1]
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        if exc_type:
            logging.warn(exc_type, exc_value, exc_traceback)

    def run(self):
        t_start = time.time()
        avg_fps = list()
        idx_frame = 0
        last_out = None
        face_ids = list()
        for frame_num in self.frames:
            t0 = time.time()
            img_dir = os.path.join(self.root_dir, f'frame_{frame_num:04d}.png')
            t0 = time.time()
            img0 = Image.open(img_dir)
            # Inference *********************************************************************
            db_frame = self.db[self.db.frame_num==frame_num]
            outputs = self.image_track(img0, db_frame)        # (#ID, 5) x1,y1,x2,y2,id
            t1 = time.time()
            logging.info(f'Frame {frame_num} Done in {t1-t0:.3f}secs')
            avg_fps.append(t1 - t0)

            if len(outputs) > 0:                
                bbox_xyxy = outputs[:, :4]
                bbox_xywh = xyxy2xywh(bbox_xyxy)
                b1 = db_frame[['x','y','w','h']].values 
                iou = calc_iou(b1, bbox_xywh)
                face_ids = iou.argmax(axis=1)
                new_face_ids = (0 * face_ids)-1
                for ix in range(face_ids.size):
                    tid = face_ids[ix]
                    if iou[ix,tid] > 0.6 and ix == iou[:,tid].argmax():
                        new_face_ids[ix] = outputs[tid,-1]
                # outputs[:,-1] = new_face_ids
                db_frame['face_id'] = new_face_ids
                self.db.loc[db_frame.index,'face_id'] = new_face_ids

                db_tmp = db_frame[db_frame['face_id'] >=0]
                for ix in range(len(db_tmp)):
                    x0,y0,w,h = db_tmp[['x','y','w','h']].iloc[0]
                    # (x0,y0,x1,y1) = outputs[ix,:4]
                    x1 = x0 + w
                    y1 = y0 + h
                    x0 = max(0,x0 - int(w/2))
                    y0 = max(0,y0 - int(h/2))
                    x1 = min(img0.size[0],x0+int(2*w))
                    y1 = min(img0.size[1],y0+int(2*h))
                    face_img = img0.crop((x0,y0,x1,y1))
                    face_id = db_tmp.iloc[0].face_id
                    # face_id = outputs[ix,-1]
                    frame_num = db_tmp.iloc[0].frame_num
                    # face_num = fd.id2num.get(face_id, 0)
                    # fd.id2num[face_id] = face_num + 1
                    fname = f'faceid_{face_id:04d}_{frame_num:04d}.png'
                    fname = os.path.join(self.root_dir,fname)
                    # img = Image.fromarray(face_img)
                    face_img.save(fname)
                    self.db.loc[db_tmp.index[ix],['x0','y0','x1','y1']] = (x0,y0,x1,y1)



                    
            idx_frame += 1                                                            
        t_end = time.time()
        dt = t_end - t_start
        logging.info(f'Total time {dt:.3f}s')
        io.save_table(self.root_dir, self.db, "faces")
        fname = os.path.join(self.root_dir,'face_ids.csv')
        logging.info(f'saving updated faces in {fname}')
        self.db.to_csv(fname)

    def image_track(self, im0, db_frame):
        """
        :param im0: original image, BGR format cv2
        :return:
        """
        # db_frame = db_frame.sortby()
        boxes = db_frame[['x','y','w','h']].values
        confs = db_frame['confidence'].values
        bbox_xywh = boxes.copy() # xyxy2xywh(boxes)    # (#obj, 4)     xc,yc,w,h
        bbox_xywh[:, 2:] = bbox_xywh[:, 2:] * (1 + self.margin_ratio)
        im0_arr = np.array(im0)
        outputs = self.deepsort.update(bbox_xywh, confs, im0_arr)
        return outputs


def xyxy2xywh(x):
    # Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h] where xy1=top-left, xy2=bottom-right
    y = torch.zeros_like(x) if isinstance(x, torch.Tensor) else np.zeros_like(x)
    y[:, 0] = (x[:, 0] + x[:, 2]) / 2  # x center
    y[:, 1] = (x[:, 1] + x[:, 3]) / 2  # y center
    y[:, 2] = x[:, 2] - x[:, 0]  # width
    y[:, 3] = x[:, 3] - x[:, 1]  # height
    return y

def faces2faceids(video_files,args):
    for video_fname in video_files:   
        video_root = os.path.splitext(video_fname)[0] + '.pipeline'     
        face_tbl = io.load_table(video_root, "faces")
        if 'name' in face_tbl:
            names = sorted(list(set(face_tbl.name)))
            name2id = dict(zip(names, np.arange(len(names))))
            face_tbl['face_id'] = face_tbl.name.apply(lambda x:name2id[x])
            io.save_table(video_root, face_tbl, 'face_id')
            continue

        with VideoTracker(args, video_fname) as vdo_trk:
            vdo_trk.run()


from face_search.fs_logger import logger_init
from face_search.utils import get_files2process, is_video
import logging
if __name__ == "__main__":
    logger_init()
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_directory', help="The root directory containing video frames")
    parser.add_argument("--config_deepsort", type=str, default="./configs/deep_sort.yaml")
        # face detecot parameters
    parser.add_argument("--scale", type=int, default=2)
    parser.add_argument("--margin_ratio", type=int, default=0.2)

    args = parser.parse_args()

    logging.info(f'input_directory={args.input_directory}')

    from face_search import utils
    video_files = utils.get_video_files(args)

    logging.info(f'detecting faces in {len(video_files)} videos')
    t0 = time.time()
    faces2faceids(video_files, args)
    t1 = time.time()
    logging.info(f'process took {t1-t0}secs')

