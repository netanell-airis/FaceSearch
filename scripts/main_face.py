from utils_ds.parser import get_config
from utils_ds.draw import draw_boxes
from deep_sort import build_tracker
from PIL import Image 
import argparse
import os
import pandas as pd
import time
import numpy as np
import warnings
import cv2
import torch
import torch.backends.cudnn as cudnn
from retinaface import RetinaFace
from deepface import DeepFace
from deepface.commons import functions
# face detector
#from facenet_pytorch import MTCNN
DeepFace.verify
class FaceDetector:
    def __init__(self, device, output_dir):
        self.model = RetinaFace.build_model()
        self.device = device
        self.frame_num = 0
        self.frame_results = list()
        self.faces = list()
        self.output_dir = output_dir
        self.id2num = dict()
    
    def __call__(self, frame):
        frame_num = self.frame_num
        res = RetinaFace.detect_faces(frame, model=self.model)
        frame_results, faces = self.process_retina_output(res, frame, frame_num)            
        self.frame_results +=frame_results
        self.faces = faces
        return frame_results, faces


    def detect(self, frame):
        frame_results, faces =  self.__call__(frame)
        xyxy = [(x['x0'],x['y0'],x['x1'],x['y1']) for x in frame_results]
        boxes = np.array(xyxy)
        conf = np.array([x['score'] for x in frame_results])
        self.frame_num += 1
        return boxes, conf

    
    @staticmethod
    def process_retina_output(res, frame, frame_num=0):
        frame_results = list()
        faces = list()
        if not isinstance(res, dict):
            res = dict()
        for k in res.keys():
            face = res[k]
            landmarks = face['landmarks']
            facial_area = face['facial_area']
            landmarks = face['landmarks']
            facial_area = face['facial_area']
            res_dict = dict( 
                face_id = int(k.split('_')[1]),
                frame_num = frame_num,
                score = face['score'],
                x0 = facial_area[0],
                y0 = facial_area[1],
                x1 = facial_area[2],
                y1 = facial_area[3],            
                right_eye_x = landmarks['right_eye'][0],
                right_eye_y = landmarks['right_eye'][1],
                left_eye_x = landmarks['left_eye'][0],
                left_eye_y = landmarks['left_eye'][1],
                mouth_right_x = landmarks['mouth_right'][0],
                mouth_right_y = landmarks['mouth_right'][1],
                mouth_left_x = landmarks['mouth_left'][0],
                mouth_left_y = landmarks['mouth_left'][1],
                nose_x = landmarks['nose'][0],
                nose_y = landmarks['nose'][1]
            )
            x0 = res_dict['x0']
            y0 = res_dict['y0']
            x1 = res_dict['x1']
            y1 = res_dict['y1']
            w0 = x1 - x0 
            h0 = y1 - y0         
            x0 = max(0, x0- int(w0/2))
            y0 = max(0, y0- int(h0/2))
            x1 = min(frame.shape[1], x0 + 2*w0)
            y1 = min(frame.shape[0], y0 + 2*h0)
            face_crop = frame[y0:y1,x0:x1]
            faces.append(face_crop)
            # face_gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
            # corners = cv2.goodFeaturesToTrack(face_gray, 10,0.01,5)
            frame_results.append(res_dict)
        return frame_results, faces

import sys

currentUrl = os.path.dirname(__file__)
sys.path.append(os.path.abspath(os.path.join(currentUrl, 'yolov5')))


cudnn.benchmark = True


class VideoTracker(object):
    def __init__(self, args):
        print('Initialize DeepSORT & YOLO-V5')
        # ***************** Initialize ******************************************************
        self.args = args
        self.scale = args.scale                         # 2
        self.margin_ratio = args.margin_ratio           # 0.2
        self.frame_interval = args.frame_interval       # frequency

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.half = self.device.type != 'cpu'  # half precision only supported on CUDA

        # create video capture ****************
        if args.display:
            cv2.namedWindow("test", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("test", args.display_width, args.display_height)

        if args.cam != -1:
            print("Using webcam " + str(args.cam))
            self.vdo = cv2.VideoCapture(args.cam)
        else:
            self.vdo = cv2.VideoCapture()

        # ***************************** initialize DeepSORT **********************************
        cfg = get_config()
        cfg.merge_from_file(args.config_deepsort)

        use_cuda = self.device.type != 'cpu' and torch.cuda.is_available()
        self.deepsort = build_tracker(cfg, use_cuda=use_cuda)

        # ***************************** initialize Face Det **********************************
        # self.face_detector = MTCNN(keep_all=True, device=self.device)
        self.face_detector = FaceDetector(device=self.device,
                                          output_dir = args.save_path)

        print('Done..')
        if self.device == 'cpu':
            warnings.warn("Running in cpu mode which maybe very slow!", UserWarning)

    def __enter__(self):
        # ************************* Load video from camera *************************
        if self.args.cam != -1:
            print('Camera ...')
            ret, frame = self.vdo.read()
            assert ret, "Error: Camera error"
            self.im_width = int(self.vdo.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.im_height = int(self.vdo.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # ************************* Load video from file *************************
        else:
            assert os.path.isfile(self.args.input_path), "Path error"
            self.vdo.open(self.args.input_path)
            self.im_width = int(self.vdo.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.im_height = int(self.vdo.get(cv2.CAP_PROP_FRAME_HEIGHT))
            assert self.vdo.isOpened()
            print('Done. Load video file ', self.args.input_path)

        # ************************* create output *************************
        if self.args.save_path:
            os.makedirs(self.args.save_path, exist_ok=True)
            # path of saved video and results
            self.save_video_path = os.path.join(self.args.save_path, "results.mp4")

            # create video writer
            fourcc = cv2.VideoWriter_fourcc(*self.args.fourcc)
            self.writer = cv2.VideoWriter(self.save_video_path, fourcc,
                                          self.vdo.get(cv2.CAP_PROP_FPS), (self.im_width, self.im_height))
            print('Done. Create output file ', self.save_video_path)

        if self.args.save_txt:
            os.makedirs(self.args.save_txt, exist_ok=True)

        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.vdo.release()
        self.writer.release()
        if exc_type:
            print(exc_type, exc_value, exc_traceback)

    def run(self):
        yolo_time, sort_time, avg_fps = [], [], []
        t_start = time.time()

        idx_frame = 0
        last_out = None
        while self.vdo.grab():
            # Inference *********************************************************************
            t0 = time.time()
            _, img0 = self.vdo.retrieve()
            save_faces = False
            if idx_frame % self.args.frame_interval == 0:
                outputs, yt, st = self.image_track(img0)        # (#ID, 5) x1,y1,x2,y2,id
                last_out = outputs
                yolo_time.append(yt)
                sort_time.append(st)
                save_faces = True
                print('Frame %d Done. Det-time:(%.3fs) SORT-time:(%.3fs)' % (idx_frame, yt, st))
            else:
                outputs = last_out  # directly use prediction in last frames
            t1 = time.time()
            avg_fps.append(t1 - t0)

            # post-processing ***************************************************************
            # visualize bbox  ********************************
            if len(outputs) > 0:
                if save_faces:
                    tracks = self.deepsort.tracker.tracks
                    bbox_xyxy = outputs[:, :4]

                    identities = outputs[:, -1]
                    fd = self.face_detector
                    for ix in range(outputs.shape[0]):
                        (x0,y0,x1,y1) = outputs[ix,:4]
                        w = x1 - x0
                        h = y1 - y0
                        x0 = max(0,x0 - int(w/2))
                        y0 = max(0,y0 - int(h/2))
                        x1 = min(img0.shape[1],x0+int(2*w))
                        y1 = min(img0.shape[0],y0+int(2*h))
                        face = img0[y0:y1,x0:x1]
                        face_id = outputs[ix,-1]
                        face_num = idx_frame
                        # face_num = fd.id2num.get(face_id, 0)
                        # fd.id2num[face_id] = face_num + 1
                        fname = f'face_{face_id:04d}_{face_num:04d}.png'
                        fname = os.path.join(fd.output_dir,fname)
                        img = Image.fromarray(face)
                        img.save(fname)

                    
                img0 = draw_boxes(img0, bbox_xyxy, identities)  # BGR
                
                # add FPS information on output video
                text_scale = max(1, img0.shape[1] // 1600)
                cv2.putText(img0, 'frame: %d fps: %.2f ' % (idx_frame, len(avg_fps) / sum(avg_fps)),
                        (20, 20 + text_scale), cv2.FONT_HERSHEY_PLAIN, text_scale, (0, 0, 255), thickness=2)

            # display on window ******************************
            if self.args.display:
                cv2.imshow("test", img0)
                if cv2.waitKey(1) == ord('q'):  # q to quit
                    cv2.destroyAllWindows()
                    break

            # save to video file *****************************
            if self.args.save_path:
                self.writer.write(img0)

            if self.args.save_txt:
                with open(self.args.save_txt + str(idx_frame).zfill(4) + '.txt', 'a') as f:
                    for i in range(len(outputs)):
                        x1, y1, x2, y2, idx = outputs[i]
                        f.write('{}\t{}\t{}\t{}\t{}\n'.format(x1, y1, x2, y2, idx))

            idx_frame += 1

        print('Avg Det time (%.3fs), Sort time (%.3fs) per frame' % (sum(yolo_time) / len(yolo_time),
                                                            sum(sort_time)/len(sort_time)))
        t_end = time.time()
        print('Total time (%.3fs), Total Frame: %d' % (t_end - t_start, idx_frame))

    def image_track(self, im0):
        """
        :param im0: original image, BGR format cv2
        :return:
        """
        # preprocess ************************************************************
        h, w, _ = im0.shape
        img = cv2.resize(im0, (w // self.scale, h // self.scale))         # down sample to speed up
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)      #

        # Detection time *********************************************************
        # Inference
        t1 = time.time()
        with torch.no_grad():
            boxes, confs = self.face_detector.detect(img)
            # boxes: (#obj, 4) x1,y1,x2,y2      in img scale !
            # confs: ()

        t2 = time.time()

        # get all obj ************************************************************

        if boxes is not None and len(boxes):
            boxes = boxes * self.scale      # x1,y1,x2,y2  go back to original image

            bbox_xywh = xyxy2xywh(boxes)    # (#obj, 4)     xc,yc,w,h

            # add margin here. only need to revise width and height
            bbox_xywh[:, 2:] = bbox_xywh[:, 2:] * (1 + self.margin_ratio)

            # ****************************** deepsort ****************************
            outputs = self.deepsort.update(bbox_xywh, confs, im0)
            # (#ID, 5) x1,y1,x2,y2,track_ID
        else:
            outputs = torch.zeros((0, 5))

        t3 = time.time()
        return outputs, t2-t1, t3-t2


def xyxy2xywh(x):
    # Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h] where xy1=top-left, xy2=bottom-right
    y = torch.zeros_like(x) if isinstance(x, torch.Tensor) else np.zeros_like(x)
    y[:, 0] = (x[:, 0] + x[:, 2]) / 2  # x center
    y[:, 1] = (x[:, 1] + x[:, 3]) / 2  # y center
    y[:, 2] = x[:, 2] - x[:, 0]  # width
    y[:, 3] = x[:, 3] - x[:, 1]  # height
    return y


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # input and output
    parser.add_argument('--input_path', type=str, default='video.mp4', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--save_path', type=str, default='output/', help='output folder')  # output folder
    parser.add_argument("--frame_interval", type=int, default=1)
    parser.add_argument('--fourcc', type=str, default='mp4v', help='output video codec (verify ffmpeg support)')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--save_txt', default='output/predict/', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')

    # camera only
    parser.add_argument("--display", action="store_true")
    parser.add_argument("--display_width", type=int, default=800)
    parser.add_argument("--display_height", type=int, default=600)
    parser.add_argument("--camera", action="store", dest="cam", type=int, default="-1")

    # face detecot parameters
    parser.add_argument("--scale", type=int, default=2)
    parser.add_argument("--margin_ratio", type=int, default=0.2)

    # deepsort parameters
    parser.add_argument("--config_deepsort", type=str, default="./configs/deep_sort.yaml")

    args = parser.parse_args()
    print(args)
    # args.config_deepsort = '/Users/eranborenstein/pc/airis/DeepSORT_Face/configs/deep_sort.yaml'
    # args.input_path = '/Users/eranborenstein/pc/airis/DeepSORT_Face/video.mp4'
    # args.scale = int(1)
    #python main_face.py --config_deepsort=deep_sort.yaml --input_path=video.mp4 --save_path=/tmp --scale=1
    in_dir = args.input_path
    out_dir = args.save_path
    video_fnames = list(filter(lambda x:os.path.splitext(x)[-1] == '.mp4', 
                               os.listdir(in_dir)))
    video_fnames = list(map(lambda x:os.path.join(in_dir,x), video_fnames))
    for video_fname in video_fnames:
        args.input_path = video_fname 
        output_path = os.path.splitext(video_fname)[0] + '.faces'
        output_path = os.path.join(out_dir, os.path.split(output_path)[1])
        args.save_path = output_path 
        with VideoTracker(args) as vdo_trk:
            vdo_trk.run()

