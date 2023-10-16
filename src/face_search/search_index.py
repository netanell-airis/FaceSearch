import logging
import os
import h5py
import numpy as np
import pandas as pd
from face_search.utils import cosine_distance, get_gallery_templates
import torch 
from face_search.utils import copy_partial_dict, extract_signatures
from PIL import Image
from tqdm import tqdm
from face_search import io
class SearchIndex:
    def __init__(self, dim=512):
        self.corpus_dir = None
        self.sigs = None
        self.tensor_db = None
        self.vid2fname = list()
        self.index_files = list()
        self.embedding = np.zeros((0, dim))
        self.templates = np.zeros((0, dim))
        self.face_list = list()
        self.mode = 'r'
        self.face_tbl = None 
        self.video_tbl = None 
        self.t_tbl = None 

    def __enter__(self):
        fnames = self.get_db_fnames(self.corpus_dir)
        if os.path.isfile(fnames['sigs_store_fname']):
            self.sigs = h5py.File(fnames['sigs_store_fname'],self.mode)
        if os.path.isfile(fnames['t_tbl_fname']):
            self.t_tbl = pd.read_csv(fnames['t_tbl_fname'])
        self.face_tbl = pd.read_csv(fnames['face_tbl_fname'])
        self.video_tbl = pd.read_csv(fnames['video_tbl_fname'])
        return self.sigs
    
    def __exit__(self, exc_type, exc_value, traceback):
        if self.sigs is not None:
            self.sigs.close()


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
    
    @staticmethod
    def from_pipeline_outputs(pipeline_dirs, corpus_dir, dim=512):
        """
        this function collects all data from 
        processed pipelines to create an index
        
        """
        dim = int(dim)
        os.makedirs(corpus_dir, exist_ok=True)       
        fnames = SearchIndex.get_db_fnames(corpus_dir)
        h5 = h5py.File(fnames['sigs_store_fname'], 'w')
        h5.create_dataset('embeddings',(4096, dim),maxshape=(None,dim),dtype=np.float32)        
        h5.create_dataset('templates',(4096, dim),maxshape=(None,dim),dtype=np.float32)        
        face_tbl_list = list()
        i0 = 0
        for vid, root_dir in enumerate(pipeline_dirs):
            face_tbl = io.load_table(root_dir,'face_id')
            if face_tbl is None:
                logging.warn(f'could not find face_id.csv in {root_dir}')
                continue
            # db_fname = os.path.join(root_dir, 'face_ids.csv')
            e_fname = os.path.join(root_dir, 'embeddings.pth')
            # face_tbl = pd.read_csv(db_fname)
            E = torch.load(e_fname)
            i1 = i0 + E.shape[0]
            if i1 > h5['embeddings'].shape[0]:
                h5['embeddings'].resize((i1*2, dim))
            h5['embeddings'][i0:i1,:] = E
            i0 = i1
            face_tbl['video_id'] = vid
            face_tbl_list.append(face_tbl)

        
        face_tbl = pd.concat(face_tbl_list,axis=0)
        face_tbl.to_csv(fnames['face_tbl_fname'])
        db_videos = pd.DataFrame(pipeline_dirs, columns=['video_fname'])
        db_videos.to_csv(fnames['video_tbl_fname'])

        h5.close()
        corpus = SearchIndex(dim=dim)
        corpus.corpus_dir = corpus_dir 
        return corpus 

    @staticmethod
    def get_db_fnames(corpus_dir):
        return dict(
            face_tbl_fname = os.path.join(corpus_dir,'faces_db.csv'),
            video_tbl_fname = os.path.join(corpus_dir, 'videos.csv'),
            t_tbl_fname = os.path.join(corpus_dir, 'templates.csv'),
            sigs_store_fname = os.path.join(corpus_dir, 'sigs.pth'),
        )



            

            

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




