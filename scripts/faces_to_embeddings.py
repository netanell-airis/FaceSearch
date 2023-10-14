import os
import logging
import time
from tqdm import tqdm
import re
import pandas as pd
import numpy as np
import torch
import PIL.Image
from PIL import Image
import cv2
import argparse
from face_search.face_alignment import mtcnn
from face_search.fs_logger import logger_init
from face_search.utils import is_video_frame, get_video_process_dir
from face_search.utils import is_video, get_files2process, is_video_face_roi
from face_search.face_alignment.align import get_aligned_face
from face_search.adanet import build_model

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'


adaface_models = {
    'ir_50':os.path.join(os.environ['HOME'],'models','adaface_ir50_ms1mv2.ckpt')
}

def load_pretrained_model(architecture='ir_50'):
    # load model and pretrained statedict
    assert architecture in adaface_models.keys()
    model = build_model(architecture)
    statedict = torch.load(adaface_models[architecture], map_location=torch.device(device))['state_dict']
    model_statedict = {key[6:]:val for key, val in statedict.items() if key.startswith('model.')}
    model.load_state_dict(model_statedict)
    model.eval()
    return model

def to_input(pil_rgb_image):
    np_img = np.array(pil_rgb_image)
    brg_img = ((np_img[:,:,::-1] / 255.) - 0.5) / 0.5
    tensor = torch.tensor([brg_img.transpose(2,0,1)]).float()
    return tensor

# def add_padding(pil_img, top, right, bottom, left, color=(0,0,0)):
#     width, height = pil_img.size
#     new_width = width + right + left
#     new_height = height + top + bottom
#     result = Image.new(pil_img.mode, (new_width, new_height), color)
#     result.paste(pil_img, (left, top))
#     return result

# def get_aligned_face(mtcnn_model, image_path, rgb_pil_image=None):
#     if rgb_pil_image is None:
#         img = Image.open(image_path).convert('RGB')
#     else:
#         assert isinstance(rgb_pil_image, Image.Image), 'Face alignment module requires PIL image or path to the image'
#         img = rgb_pil_image
#     # find face
#     try:
#         bboxes, faces = mtcnn_model.align_multi(img, limit=1)
#         face = faces[0]
#     except Exception as e:
#         print('Face detection Failed due to error.')
#         print(e)
#         face = None

#     return face

def faces2embeddings(video_files):
    # Initialize the MTCNN detector
    #detector = MTCNN(steps_threshold=[0.6,0.7,0.9])
    # mtcnn_model = mtcnn.MTCNN(device='cpu', crop_size=(112, 112))
    #device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    mtcnn_model = mtcnn.MTCNN(device='cpu', crop_size=(112, 112))

    model = load_pretrained_model('ir_50')
    invalid_input = torch.zeros((1,3,112,112))
    

    # Iterate over each video file
    for video_file in video_files:
        t0 = time.time()
        process_dir = get_video_process_dir(video_file)
        db = pd.read_csv(os.path.join(process_dir, 'faces.csv'))
        face_list = db[['frame_num','idx']].values.tolist()
        face_fnames = [f'frame_{fr:04d}_{idx:04d}.png' for fr, idx in face_list]
        face_fnames = [os.path.join(process_dir,x) for x in face_fnames]
        logging.info(f'working on video {video_file} with {len(face_list)} faces')
        face_list = list()
        e_list = list()
        det_list = list()
        aligned_faces_db = list()
        for ix,row in tqdm(db.iterrows()):
            frame_num = int(row.frame_num)
            idx = int(row.idx)
            fname = os.path.join(process_dir,
                                 f'face_{frame_num:05d}_{idx:04d}.png')
            # frame_num = int(is_video_frame(input_image_path))
            # Load the input image
            aligned_face = get_aligned_face(mtcnn_model,fname)
            det_list.append(aligned_face is not None)
            if det_list[-1] is False:
                aligned_face = Image.new(mode="RGB", size=(112, 112), color=(128,128,128))
                # aligned_face.fill(128)
            
            bgr_tensor_input = to_input(aligned_face)
            aligned_faces_db.append(bgr_tensor_input)
        
        i0 = 0
        bs = 16 
        n0 = len(aligned_faces_db)
        elist = list()
        tbar = tqdm(desc='extract emb',total=n0)
        while i0 < n0:
            i1 = min(i0+bs,n0)
            x = torch.concatenate(aligned_faces_db[i0:i1],dim=0)
            with torch.no_grad():
                fe, fnorm = model(x)
            elist.append((fe, fnorm))
            tbar.update(i1-i0)
            i0 = i1
            
        
        e = torch.concat([x[0] for x in elist])
        enorm = torch.concat([x[1] for x in elist])
        db['aligned'] = det_list 
        db['enorm'] = enorm
        db.to_csv(os.path.join(process_dir,'faces.csv'))
        fname = os.path.join(process_dir, 'embeddings.pth')
        torch.save(e, fname)
        t1 = time.time()
        logging.info(f'finished detecting {len(db)} faces in {t1-t0}secs')





if __name__ == "__main__":
    logger_init()
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_directory', help="The root directory containing video frames")
    args = parser.parse_args()

    video_files = get_files2process(args.input_directory, flt=lambda x:is_video(x))

    logging.info(f'detecting faces in {len(video_files)} videos')
    t0 = time.time()
    faces2embeddings(video_files)
    t1 = time.time()
    logging.info(f'process took {t1-t0}secs')








