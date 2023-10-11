import numpy as np
import scipy.sparse as sp

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
