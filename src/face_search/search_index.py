import numpy as np
import pandas as pd
from face_search.utils import cosine_distance, get_gallery_templates
import torch 
from face_search.utils import copy_partial_dict, extract_signatures
from PIL import Image
from tqdm import tqdm
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
        for ix, fname in tqdm(enumerate(index_files),desc='generate-index'):
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

    def query_2_corpus_distance(self, query_templates):
        """
        computes the dist between mxdim query_templates/embedding
        and nxdim corpus (index) signatures
        """
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

    def search_by_query_images(self, query_images):
        # query_images = get_files2process(args.query, flt=img_filter)
        query = extract_signatures(query_images, 
                                detector_backend='retinaface',
                                target_size=(112,112))
        index = self 
        emb = np.array([x['embedding'] for x in query])
        dst = self.query_2_corpus_distance(emb)
        K = 5
        query_res = list()
        for query_id in range(dst.shape[0]):
            qfname = query_images[query_id]
            query_img = Image.open(qfname)
            # query_img.save(f'/tmp/query_{query_id}.jpeg')
            query_info = dict(img=query_img,fname=qfname)
            cur_res = list()
            for k in range(K):
                bix = dst.argmin(axis=1)[query_id]
                score = dst.min(axis=1)[query_id]
                dst[query_id,bix] = 1000 + score
                row = self.db.iloc[bix]
                video_id = int(row.video_id)
                fname = index.index_files[video_id]
                print(f'match score = {score}, video={fname}, frame={row.frame_num}')
                matched_index_img = Image.fromarray((index.face_list[bix][0]*255).astype(np.uint8))
                cur_res.append((dict(img=matched_index_img,
                                       score= score,
                                        fname= fname,
                                         bix= bix )))
                # matched_img = Image.open(fname)
                #matched_index_img.save(f'/tmp/matched_{query_id}_{k:02d}.jpg')
            query_res.append((query_info, cur_res))
        return query_res




