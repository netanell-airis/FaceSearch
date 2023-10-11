import numpy as np
import pandas as pd
from face_search.utils import cosine_distance, get_gallery_templates
import torch 
from face_search.utils import copy_partial_dict
from PIL import Image

class SearchIndex:
    def __init__(self, dim=512):
        self.index_files = list()
        self.embedding = np.zeros((0, dim))
        self.templates = np.zeros((0, dim))
        self.db = pd.DataFrame()
        self.face_list = list()

    @classmethod 
    def from_index_files(cls,index_files):
        emb_list = list()
        last_face_id = 0
        template_list = list()
        db_list = list()
        face_list = list()
        for ix, fname in enumerate(index_files):
            d = torch.load(fname)
            detector_out = d['detector_outputs']
            embeddings = d['embeddings']
            face_ids = d['face_ids']
            templates = get_gallery_templates(face_ids, embeddings)
            d_list = [copy_partial_dict(x,'face') for x in detector_out]
            faces = [x['face'] for x in detector_out]
            db = pd.DataFrame(d_list)
            db['face_id'] = [id_ + last_face_id for id_ in face_ids]
            db['video_id'] = ix 
            last_face_id = db['face_id'].values.max()
            db_list.append(db)
            emb_list.append(embeddings)
            template_list.append(templates)
            face_list += faces

        index = cls()
        index.index_files = index_files 
        index.embedding = np.concatenate(emb_list, axis=0)
        index.templates = np.concatenate(template_list,axis=0)
        index.db = pd.concat(db_list, axis=0)
        index.face_list = face_list
        return index

    def search_gallery(self, query_templates):
        bs = 1024
        i0 = 0
        templates = self.embedding
        num_samples = templates.shape[0]
        scores = np.zeros((query_templates.shape[0], num_samples))
        while i0 < num_samples:
            i1 = min(i0 + bs, num_samples)
            batch = templates[i0:i1]
            d = cosine_distance(query_templates, batch) 
            scores[:,i0:i1] = d
            i0 = i1
        return scores


    # def create_query_gallery(self, root):
        



