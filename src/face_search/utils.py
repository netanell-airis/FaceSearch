import os
import numpy as np
import scipy.sparse as sp
from deepface import DeepFace
from tqdm import tqdm 

def get_gallery_templates(face_ids, embeddings ):
    embedding_dim = embeddings.shape[1]
    if len(face_ids) == 0:
        return np.zeros((1,embedding_dim))
    num_ids = max(face_ids)+1
    n = embeddings.shape[0]
    data = np.ones(len(face_ids))
    row = np.arange(n)
    col = face_ids
    W = sp.csr_matrix((data, (row, col)), (n,num_ids),dtype = np.float32)
    Z = W.sum(axis=0)
    T = W.transpose() @ embeddings
    T = T / Z.transpose()
    return T

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
