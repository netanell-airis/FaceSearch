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
#from face_search.face_alignment import mtcnn
from face_search.fs_logger import logger_init
from face_search.utils import is_video_frame, get_video_process_dir
from face_search.utils import is_video, get_files2process, is_video_face_roi
#from face_search.face_alignment.align import get_aligned_face
from face_search.adanet import build_model
from FaceCoresetNet import head

from FaceCoresetNet import config as facecoreset_config
from FaceCoresetNet.train_val_template import FaceCoresetNet
from FaceCoresetNet.utils import dotdict
from FaceCoresetNet.face_align_utils import prepare_face_for_recognition

#from mtcnn_pytorch.src.align_trans import get_reference_facial_points, warp_and_crop_face


device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

# adaface_models = {
#     'ir_50': os.path.join(os.environ['HOME'], 'models', 'adaface_ir50_ms1mv2.ckpt')
# }



# -*- coding: utf-8 -*-
"""
Created on Mon Apr 24 15:43:29 2017
@author: zhaoy
"""
import numpy as np
import cv2

# from scipy.linalg import lstsq



# -*- coding: utf-8 -*-
"""
Created on Tue Jul 11 06:54:28 2017

@author: zhaoyafei
"""


def load_embedding_model():
    path_to_facecoresetnet_checkpoint = os.path.join(os.environ['HOME'],'models',
                                                      'FaceCoresetNet.ckpt')
    args = facecoreset_config.get_args()
    hparams = dotdict(vars(args))
    model = FaceCoresetNet(**hparams)
    checkpoint = torch.load(path_to_facecoresetnet_checkpoint,
                            map_location=torch.device(device))
    model.load_state_dict(checkpoint['state_dict'])
    #model.aggregate_model.gamma = torch.nn.Parameter(torch.tensor(0.0))
    model.eval()
    return model

def f_embeddings2t_embeddings(video_files):

    model = load_embedding_model()

    # Iterate over each video file
    for video_file in video_files:
        t0 = time.time()
        process_dir = get_video_process_dir(video_file)
        db = pd.read_csv(os.path.join(process_dir, 'faces.csv',))

        faces_group_by_id = db.groupby(['idx'])
        templates = []
        fname = os.path.join(process_dir, 'embeddings.pth')
        face_emb = torch.load(fname)

        template_emb_list = []
        for key, group in faces_group_by_id:
            indexes = [i for i in group['Unnamed: 0'].iloc]
            template = [face_emb[i].unsqueeze(0) for i in indexes]
            norms = torch.tensor([i for i in group['enorm']]).unsqueeze(0)
            template_emb = torch.cat(template, dim=0 )

            with torch.no_grad():
                template_emb = template_emb.unsqueeze(0)
                template_emb, template_emb_norm = model(templates=template, compute_feature=True, embeddings=template_emb, norms=norms, mode='f_emb_to_t_emb')
                unnorm_template_emb = template_emb * template_emb_norm
                template_emb_list.append(unnorm_template_emb)

        template_emb_list = torch.cat(template_emb_list, dim=0)
        template_emb_fname = os.path.join(process_dir, 'template_embeddings.pth')
        torch.save(template_emb_list, template_emb_fname)

        t1 = time.time()
        logging.info(f'finished computing template embedding {len(db)} faces in {t1 - t0}secs')


if __name__ == "__main__":
    logger_init()
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_directory', help="The root directory containing video frames")
    args = parser.parse_args()

    video_files = get_files2process(args.input_directory, flt=lambda x: is_video(x))

    logging.info(f'detecting faces in {len(video_files)} videos')
    t0 = time.time()

    f_embeddings2t_embeddings(video_files)
    t1 = time.time()
    logging.info(f'process took {t1 - t0}secs')








