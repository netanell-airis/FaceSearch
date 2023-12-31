import os
import re
import numpy as np
import scipy.sparse as sp
from deepface import DeepFace
from tqdm import tqdm
import torch

def get_gallery_templates(face_ids, embeddings, model=None ):
    import pandas as pd

    face_ids = np.uint32(face_ids)
    embedding_dim = embeddings.shape[1]
    t = pd.DataFrame(face_ids, columns=['face_id'])
    tid_to_faceid = list()
    templates = list()
    for face_id, g in t.groupby(by='face_id'):
        if face_id >=0:
            T = embeddings[g.index]
            if model == None:
                T = T.mean(axis=0)[np.newaxis,:]
            else:
                with torch.no_grad():
                    T = model(embeddings=T, mode='f_emb_to_t_emb')

            templates.append(T)
            tid_to_faceid.append(face_id)
    T = np.concatenate(templates, axis=0)
    return T, tid_to_faceid


    # if len(face_ids) == 0:
    #     return np.zeros((1,embedding_dim))
    # num_ids = int(max(face_ids))+1
    # n = embeddings.shape[0]
    # data = np.ones(len(face_ids))
    # row = np.arange(n)
    # col = face_ids.astype(row.dtype)
    # W = sp.csr_matrix((data, (row, col)), (n,num_ids),dtype = np.float32)
    # Z = W.sum(axis=0)
    # Z[Z==0] = 1.0
    # T = W.transpose() @ embeddings
    # T = T / Z.transpose()
    # return T

def copy_partial_dict(src_dict, ignore_keys):
    dst_dict = dict([(k,v) for k,v in src_dict.items() if k not in ignore_keys])
    return dst_dict

def cosine_distance(x,y):
    """
    calculate cos_dist (nxm) between x (nxd) and y (m_d)
    """
    s = x @ y.transpose()
    z = np.linalg.norm(x,axis=1)[:,np.newaxis] @ np.linalg.norm(y,axis=1)[np.newaxis,:]
    z[z==0] = 1.0
    s = s/z 
    s = np.array(s)
    return 1 - s

def get_files2process(in_dir, flt=lambda x:'.pth' in x):
    """
    Walk on a directory and returns a list
    of all file names passing the filter flt
    """
    imgs2proc = list()
    for dirpath, dirnames, filenames in os.walk(in_dir):
        imgs = list(filter(lambda x:flt(x), filenames))
        imgs = list(map(lambda x:os.path.join(dirpath,x), imgs))
        if len(imgs):
            imgs2proc += imgs
    return imgs2proc

def extract_signatures(fnames, detector_backend = 'retinaface',target_size=(112,112)):
    signatures = list()
    for img_path in tqdm(fnames,desc='extract_signatures'):
        try: 
            sig = DeepFace.represent(img_path,
                                        model_name='ArcFace',
                                        detector_backend=detector_backend)
        except:
            sig = DeepFace.represent(img_path,
                                        model_name='ArcFace',
                                        detector_backend='skip')

        signatures.append(sig[0])
    return signatures


def get_video_files(args):
    if os.path.splitext(args.input_directory)[-1] == '.pipeline':
        video_files = [args.input_directory]
    else:
        video_files = get_files2process(args.input_directory, flt=lambda x:is_video(x))
    return video_files


def is_video(fname):
    return os.path.splitext(fname)[-1].lower() == '.mp4'

def is_video_frame(fname):
    r0 = re.compile('frame_(\d+).png')
    g = r0.search(fname)
    frame_num = g.groups()[0] if g else None
    return frame_num 

def is_video_face_roi(fname):
    r0 = re.compile('face_(\d+)_(\d).png')
    g = r0.match(fname)
    frame_num = g.groups()[0] if g else None
    return frame_num
    

def img_filter(x):
    return os.path.splitext(x)[-1].lower() in ['.jpeg','jpg','png']

def is_img_fname(fname):
    ext = os.path.splitext(fname)[-1]
    ext = ext.lower()
    return ext in ['.jpeg','.jpg','.png','tif']



def get_video_process_dir(video_path):
    process_dir = os.path.splitext(video_path)[0] + '.pipeline'
    os.makedirs(process_dir, exist_ok=True)
    return process_dir
