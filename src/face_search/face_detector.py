from deepface import DeepFace
from deepface.commons import functions, distance as dst
import numpy as np
import pandas as pd
from face_search.utils import get_gallery_templates
from face_search.utils import cosine_distance
import time
import cv2


class FaceDetector:
    def __init__(self, device, output_dir):
        self.device = device
        self.frame_num = 0
        self.frame_results = list()
        self.faces = list()
        self.face_ids = list()
        self.output_dir = output_dir
        self.id2num = dict()
        self.detector_backend = 'retinaface'
        self.model_name = 'ArcFace'
        self.align_faces = False #True
        self.target_size = DeepFace.functions.find_target_size(model_name="ArcFace")
        self.detector_outputs = list()
        #self.model = RetinaFace.build_model()
        self.embedding_model = DeepFace.build_model("ArcFace")
        self.embedding_sim_th = dst.findThreshold("ArcFace","cosine")
        self.embedding_dim = 512
        self.embeddings = np.zeros((0,self.embedding_dim))
    
    def __call__(self, frame):
        frame_num = self.frame_num
        size = frame.shape[:2]
        size = (112, 112)
        res = DeepFace.extract_faces(frame, target_size=size,
                                   detector_backend='retinaface',
                                   enforce_detection=False,
                                   align=False)
        batch = [functions.normalize_input(img=x['face'],
                                            normalization='ArcFace') for x in res]
        batch = [np.expand_dims(img, axis=0) for img in batch]
        batch = np.concatenate(batch, axis=0)
        # represent
        embedding = self.embedding_model.predict(batch, verbose=0)
        faces = [x['face'] for x in res]
        scores = [x['confidence'] for x in res]
        pos = [x['facial_area'] for x in res]
        frame_res = pd.DataFrame(pos)
        frame_res['score'] = scores 
        frame_res['frame_num'] = self.frame_num
        

        # res1 = RetinaFace.detect_faces(frame, model=self.model)
        # frame_results, faces = self.process_retina_output(res, frame, frame_num)            
        self.frame_results.append(frame_res)
        self.embeddings.append(embedding)
        self.faces = faces
        return frame_res, faces
    
    def detect_and_track(self, frame, idx=0,scale=1):
        """
        returns detector_output list of dictionary with:
        one item for each detected face:
        'face' -- self.target_size window with face 
        'confidence'
        'facial_area'       

        """
        # extract faces
        t0 = time.time()
        scaled_frame = frame.copy()
        if scale != 1:
            h, w, _ = frame.shape
            scaled_frame = cv2.resize(frame, (w // scale, h // scale))         # down sample to speed up
            #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)      #

        detector_output = DeepFace.extract_faces(scaled_frame,
                                      target_size=self.target_size,
                                   detector_backend=self.detector_backend,
                                   enforce_detection=False,
                                   align=self.align_faces)
        # print(f'Extract_faces time={time.time()-t0}')
        if len(detector_output) == 0:
            return list()
        t0 = time.time()
        scores = [x['confidence'] for x in detector_output]
        bboxes = [x['facial_area'] for x in detector_output]
        frame_res = pd.DataFrame(bboxes)
        frame_res['score'] = scores 
        frame_res['frame_num'] = self.frame_num
        frame_res[['x','y','w','h']] = frame_res[['x','y','w','h']] * scale
        db = pd.DataFrame(frame_res)
        faces = self.extract_bboxes(frame, db)
        revised_output = list()
        for ix, row in enumerate(detector_output): 
            new_row = db.iloc[ix].to_dict()
            new_row['face'] = faces[ix]
            revised_output.append(new_row)
        detector_output = revised_output
            


        
        # extract embeddings

        batch = [functions.normalize_input(x[0].copy(), normalization='ArcFace') for x in faces]
        batch = [np.expand_dims(img, axis=0) for img in batch]
        batch = np.concatenate(batch, axis=0)
        embedding = self.embedding_model.predict(batch, verbose=0)

        # print(f'embedding time = {time.time()-t0}')

        t0 = time.time()
        # track face_ids
        th = self.embedding_sim_th
        th = 0.1
        templates = get_gallery_templates(self.face_ids, self.embeddings)
        id0 = max(self.face_ids + [-1]) + 1
        cos_dst = cosine_distance(templates, embedding)
        mx = cos_dst.min(axis=0)
        amx = cos_dst.argmin(axis=0)
        mx2 = cos_dst.min(axis=1)
        face_ids = list()
        for face_idx in range(cos_dst.shape[1]):
            matched_template_idx = amx[face_idx]
            match_score = mx[face_idx]
            if match_score < th and match_score == mx2[matched_template_idx]:
                face_ids.append(matched_template_idx)
            else:
                face_ids.append(id0)
                id0 +=1

        self.face_ids += face_ids
        [x.update(dict(frame_num=idx)) for x in detector_output]
        # embedding_normed = embedding / np.linalg.norm(embedding, axis=1)[:,np.newaxis]
        self.embeddings = np.concatenate((self.embeddings, embedding), axis=0)
        self.detector_outputs += detector_output
        # print(f'track_face_ids time={time.time()-t0}')
        return detector_output

    def extract_bboxes(self, frame, db):
        """
        face_objs: list of tuples:

        """
        target_size = self.target_size
        bbox = db[['x','y','w','h']].values 
        extracted_faces = list()
        for ix in range(bbox.shape[0]):
            x,y,w,h = bbox[ix]
            current_img = frame[y:(y+h),x:(x+w)]
            if current_img.shape[0] > 0 and current_img.shape[1] > 0:
                # resize and padding
                if current_img.shape[0] > 0 and current_img.shape[1] > 0:
                    factor_0 = target_size[0] / current_img.shape[0]
                    factor_1 = target_size[1] / current_img.shape[1]
                    factor = min(factor_0, factor_1)

                    dsize = (int(current_img.shape[1] * factor), int(current_img.shape[0] * factor))
                    current_img = cv2.resize(current_img, dsize)

                    diff_0 = target_size[0] - current_img.shape[0]
                    diff_1 = target_size[1] - current_img.shape[1]
                    current_img = np.pad(
                        current_img,
                        (
                            (diff_0 // 2, diff_0 - diff_0 // 2),
                            (diff_1 // 2, diff_1 - diff_1 // 2),
                            (0, 0),
                        ),
                        "constant",
                    )

                    # double check: if target image is not still the same size with target.
                    if current_img.shape[0:2] != target_size:
                        current_img = cv2.resize(current_img, target_size)

                    # normalizing the image pixels
                    from deepface.commons.functions import image
                    img_pixels = image.img_to_array(current_img)  # what this line doing? must?
                    img_pixels = np.expand_dims(img_pixels, axis=0)
                    img_pixels /= 255  # normalize input in [0, 1]


                    extracted_faces.append(img_pixels)

        return extracted_faces


    def detect(self, frame):
        frame_results, faces_ =  self.__call__(frame)
        p = frame_results[['x','y','w','h']].values
        p[:,2:] += p[:,:2] 
        xyxy = p.astype(np.int32)
        # xyxy = [(x['x0'],x['y0'],x['x1'],x['y1']) for x in frame_results]
        conf = frame_results.score.values
        boxes = np.array(xyxy)
        # conf = np.array([x['score'] for x in frame_results])
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

def detect_in_batches(video_fname):
    from facenet_pytorch import MTCNN
    import cv2
    from PIL import Image
    import numpy as np
    from matplotlib import pyplot as plt
    from tqdm import tqdm
    import torch 
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    # v_cap = cv2.VideoCapture('20231007_072338_hamza20300_159830.mp4')
    v_cap = cv2.VideoCapture(video_fname)
    v_len = int(v_cap.get(cv2.CAP_PROP_FRAME_COUNT))

    mtcnn = MTCNN(margin=20, keep_all=True, post_process=False, device=device)

    # Loop through video
    batch_size = 4
    frames = []
    boxes = []
    landmarks = []
    frame_nums = []
    scores = []
    frame_offset = 0
    
    for i in tqdm(range(v_len)):
        # Load frame
        success, frame = v_cap.read()
        if not success:
            continue

        # Add to batch
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(Image.fromarray(frame))        

        # When batch is full, detect faces and reset batch list
        if len(frames) >= batch_size:
            batch_boxes, det_scores, batch_landmarks = mtcnn.detect(frames, landmarks=True)
            boxes.extend(batch_boxes)
            landmarks.extend(batch_landmarks)
            scores.extend(det_scores)
            frame_nums.extend([np.ones(len(det_scores[i]))*i+frame_offset for i in range(len(det_scores))])
            frame_offset += len(det_scores)
            frames = list()

    if len(frames):
        batch_boxes, det_scores, batch_landmarks = mtcnn.detect(frames, landmarks=True)
        boxes.extend(batch_boxes)
        landmarks.extend(batch_landmarks)
        scores.extend(det_scores)        
        frame_nums.extend([np.ones(len(det_scores[i]))*i+frame_offset for i in range(len(det_scores))])

    
    #remove all no-detections from list
    n0 = len(boxes)
    boxes = [boxes[i] for i in range(n0) if scores[i][0] is not None]
    landmarks = [landmarks[i] for i in range(n0) if scores[i][0] is not None]
    frame_nums = [frame_nums[i] for i in range(n0) if scores[i][0] is not None]
    fscores = [scores[i] for i in range(n0) if scores[i][0] is not None]

    boxes = np.concatenate(boxes,axis=0).astype(np.int32)
    boxes_wh = boxes[:,2:] - boxes[:,:2]
    landmarks = np.concatenate(landmarks,axis=0).astype(np.int32)
    landmarks = (landmarks.reshape((-1,10))).astype(np.int32)
    scores = np.concatenate(fscores,axis=0)[:,np.newaxis]
    frame_nums = np.concatenate(frame_nums, axis=0)[:,np.newaxis].astype(np.int32)
    A = np.concatenate((frame_nums,boxes[:,:2],boxes_wh, scores, landmarks), axis = 1)
    columns = ['frame_num'] + list('xywh') +['scores']
    for i in range(5):
        columns.append(f'lx{i}')
        columns.append(f'ly{i}')
    face_tbl = pd.DataFrame(A, columns=columns)
    return face_tbl


if __name__ == '__main__':
    video_fname = '/Users/eranborenstein/data/debug2/VID-20231008-WA0033.mp4'
    detect_in_batches(video_fname)
