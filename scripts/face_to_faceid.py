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

from face_search.utils import xyxy2xywh, xcycwh2xywh, get_video_files, get_output_dir

import sys
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

currentUrl = os.path.dirname(__file__)
sys.path.append(os.path.abspath(os.path.join(currentUrl, 'yolov5')))

# cudnn.benchmark = True
IOU_THRESH = 0.6  # IoU threshold between tracker bounding box and detection bounding box


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
        self.process_dir = get_output_dir(video_fname, args.output_directory, 'face_ids')
        self.frames_dir = get_output_dir(video_fname, args.output_directory, 'frames', create=False)
        self.embeddings_dir = get_output_dir(video_fname, args.output_directory, 'embeddings', create=False)
        # self.faces_dir = get_output_dir(video_fname, args.output_directory, 'face_detections', create=False)
        self.db_fname = os.path.join(self.embeddings_dir, 'faces.csv')
        e_fname = os.path.join(self.embeddings_dir, 'embeddings.pth')
        self.embeddings = None
        if os.path.isfile(e_fname):             
            self.embeddings = torch.load(e_fname)

        self.db = pd.read_csv(self.db_fname)  # TODO anna: change to io function and if needed used face_detections/faces.csv)

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
        self.frames = list(set(self.db['frame_num'].values))  # list of frame nums
        self.cur_frame = 0
        frame_num = self.frames[0]
        img_path = os.path.join(self.frames_dir, f'frame_{frame_num:04d}.png')
        img = Image.open(img_path)
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
            img_path = os.path.join(self.frames_dir, f'frame_{frame_num:04d}.png')
            t0 = time.time()
            img0 = Image.open(img_path)
            # Inference *********************************************************************
            db_frame = self.db[self.db.frame_num==frame_num]  # db rows of faces for current frame
            features = None
            if self.embeddings is not None:
                features = self.embeddings[db_frame.index]
            outputs = self.image_track(img0, db_frame, features)        # (#ID, 5) x1,y1,x2,y2,id
            t1 = time.time()
            logging.info(f'Frame {frame_num} Done in {t1-t0:.3f}secs')
            avg_fps.append(t1 - t0)

            if len(outputs) > 0:                
                bbox_xyxy = outputs[:, :4]  # output box coordinates [x1, y1, x2, y2]
                bbox_xywh = xyxy2xywh(bbox_xyxy)  # convert to [x1, y1, w, h]
                b1_xywh = db_frame[['x','y','w','h']].values  # [x1, y1, w, h]
                # iou between boxes in frame and boxes from tracker 
                iou = calc_iou(b1_xywh, bbox_xywh)
                face_ids = iou.argmax(axis=1)
                new_face_ids = (0 * face_ids)-1
                for ix in range(face_ids.size):
                    tid = face_ids[ix]
                    if iou[ix,tid] > IOU_THRESH and ix == iou[:,tid].argmax():
                        new_face_ids[ix] = outputs[tid,-1]
                # outputs[:,-1] = new_face_ids
                db_frame['face_id'] = new_face_ids
                self.db.loc[db_frame.index,'face_id'] = new_face_ids

                # Save face crop (with margins) with face_id
                db_tmp = db_frame[db_frame['face_id']>=0]
                for ix in range(len(db_tmp)):
                    x0,y0,w,h = db_tmp.iloc[ix][['x','y','w','h']]
                    # (x0,y0,x1,y1) = outputs[ix,:4]
                    x1 = x0 + w
                    y1 = y0 + h
                    x0 = max(0,x0 - int(w/2))
                    y0 = max(0,y0 - int(h/2))
                    x1 = min(img0.size[0],x0+int(2*w))
                    y1 = min(img0.size[1],y0+int(2*h))
                    face_img = img0.crop((x0,y0,x1,y1))
                    face_id = db_tmp.iloc[ix].face_id
                    # face_id = outputs[ix,-1]
                    frame_num = db_tmp.iloc[ix].frame_num
                    # face_num = fd.id2num.get(face_id, 0)
                    # fd.id2num[face_id] = face_num + 1
                    self.db.loc[db_tmp.index[ix],['x0','y0','x1','y1']] = (x0,y0,x1,y1)

                    if self.args.save_intermediate:
                        fname = f'faceid_{face_id:04d}_{frame_num:04d}.png'
                        fname = os.path.join(self.process_dir, fname)
                        # img = Image.fromarray(face_img)
                        face_img.save(fname)
                    
            idx_frame += 1            

        t_end = time.time()
        dt = t_end - t_start
        io.save_table(self.process_dir, self.db, "faces")
        logging.info(f'Total time {dt:.3f}s')


    def image_track(self, im0, db_frame, features = None):
        """
        :param im0: original image, BGR format cv2
        :return:
        """
        # db_frame = db_frame.sortby()
        boxes = db_frame[['x','y','w','h']].values
        confs = db_frame['confidence'].values
        bbox_xywh = boxes.copy() # xyxy2xywh(boxes)    # (#obj, 4)     xc,yc,w,h
        bbox_xywh[:, 2:] = bbox_xywh[:, 2:] * (1 + self.margin_ratio)  # add height/width margins
        im0_arr = np.array(im0)
        outputs = self.deepsort.update(bbox_xywh, confs, im0_arr, features)
        return outputs


def faces2faceids(video_files, args):
    for video_fname in video_files: 
        process_dir = get_output_dir(video_fname, output_directory, 'face_ids')
        faces_dir = get_output_dir(video_fname, output_directory, 'face_detections', create=False)  
        embeddings_dir = get_output_dir(video_fname, output_directory, 'embeddings', create=False)  
        # video_root = os.path.splitext(video_fname)[0] + '.pipeline'     
        logger_init(os.path.join(process_dir,'faces2faceids.log'))     
        face_tbl = io.load_table(embeddings_dir, "faces")
        if face_tbl is None:
            # search for table in the face detection output dir
            face_tbl = io.load_table(faces_dir, "faces")

        if 'person_id' in face_tbl:
            names = sorted(list(set(face_tbl['person_id'])))
            name2id = dict(zip(names, np.arange(len(names))))
            face_tbl['face_id'] = face_tbl['person_id'].apply(lambda x:name2id[x])
            io.save_table(process_dir, face_tbl, 'faces')
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
    parser.add_argument('--output_directory', help="Directory where output is saved, default is input_directory")
    parser.add_argument("--config_deepsort", type=str, default="./configs/deep_sort.yaml")
    parser.add_argument('--save_intermediate', action='store_true', default=False)

        # face detecot parameters
    parser.add_argument("--scale", type=int, default=2)
    parser.add_argument("--margin_ratio", type=int, default=0.2)

    args = parser.parse_args()

    input_directory = args.input_directory    
    output_directory = args.output_directory
    if output_directory is None:
        output_directory = input_directory

    logging.info(f'input_directory={input_directory}')

    video_files = get_video_files(input_directory)

    logging.info(f'detecting faces in {len(video_files)} videos')
    t0 = time.time()
    faces2faceids(video_files, args)
    t1 = time.time()
    logging.info(f'process took {t1-t0}secs')

