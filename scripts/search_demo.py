import argparse
import datetime

import os
import sys
from PIL import Image
import cv2 
import numpy as np
import torch 
import pandas as pd
from tqdm import tqdm 
import torch 
from face_search.utils import get_gallery_templates
from face_search.search_index import SearchIndex
from face_search.utils import copy_partial_dict
from face_search.utils import extract_signatures, get_files2process
from face_search.viz import render_query_res, serve_app
import logging
from face_search.fs_logger import logger_init


def filter_index_files(x):
    return  os.path.split(x)[-1] == 'video_summary.pth'


def run(args):
    imgs2proc = get_files2process(args.input_path)    
    # batch_transform(imgs2proc, transform=lambda x:rotate_90(x,clockwise=False))
    # return

    sigs = extract_signatures(imgs2proc)
    sigs = [np.array(x['signature']) for x in sigs]
    torch.save(dict(sigs= sigs, fnames=imgs2proc))

def img_path2info(fname):
    video_id = os.path.split(os.path.split(fname)[0])[1]
    video_id = video_id.split('.')[0]
    face_info = os.path.split(fname)[-1].split('_')
    face_id = int(face_info[1])
    frame_num = int(face_info[2].split('.')[0])
    return video_id, face_id, frame_num

def create_index(args):
    fnames = get_files2process(args.input_path)
    sigs = extract_signatures(fnames)
    # d = torch.load(os.path.join(args.input_path, 'sigs.pth'))
    # sigs = d['signatures']
    # fnames = d['fnames']
    embeddings = [np.array(x['embedding'])[np.newaxis,:] for x in sigs]
    embeddings = np.concatenate(embeddings, axis=0)
    face_info = [(x,)+img_path2info(x) for x in fnames]
    db = pd.DataFrame(face_info, columns = ['img_path','video_id','face_id','frame_num'])
    index_fname = args.save_path
    print(f'saving index into {index_fname}')
    torch.save(dict(embeddings=embeddings, db=db),index_fname)

def extract_query_sig(args):
    fnames = [args.query_img_path]
    sig = extract_signatures(fnames,detector_backend='retinaface')
    query = [np.array(x['embedding'])[np.newaxis,:] for x in sig]
    query = np.concatenate(query, axis=0)
    d = torch.load(args.input_path)
    embeddings= d['embeddings']
    db = d['db']
    num_face_id = 0
    for g1, db0 in db.groupby('video_id'):
        num_face_id += len(list(set(db0.face_id)))
    query = query / np.linalg.norm(query, axis=1)[:,np.newaxis]
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1)[:,np.newaxis]
    cdist = embeddings @ query.transpose()
    ix = np.argsort(-cdist, axis=0)[:,0]
    ix = ix[:10]
    db0 = db[ix]
    

def search_missing(missing_db_root, index_root, K=10):
    corpus = SearchIndex(512)
    corpus.corpus_dir = index_root 
    query = SearchIndex(512)
    query.corpus_dir = missing_db_root 

    with corpus as sig_tbl:
        enorm = corpus.t_tbl.enorm.values
        ns = enorm.size 
        sigs = sig_tbl['templates'][:ns]
        nsigs = sigs / enorm[:,np.newaxis]        
        file_name_list = list()
        with query as qtbl:
            n0 = len(query.face_tbl)
            cos_max = np.zeros((n0,K))
            cos_amax = np.zeros((n0,K),dtype=np.int32)
            qnorm = query.face_tbl.enorm.values
            for i, g in query.face_tbl.groupby(by='face_id'):
                ix = np.array(g.index)
                m0 = len(g)                
                qnsigs = qtbl['embeddings'][ix]
                # qnsigs = qsigs / qnorm[ix][:,np.newaxis]
                
                cos_sim = nsigs @ qnsigs.transpose()
                for k in range(K):
                    cos_max[ix,k] = cos_sim.max(axis=0)
                    cos_amax[ix,k] = cos_sim.argmax(axis=0)
                    cos_sim[cos_amax[ix,k],np.arange(m0)] = 0
                score = cos_max[ix,0].max()
                fid = int(g.iloc[0].face_id)
                logging.info(f'face_id={fid},num_faces={m0},max={score:.3f}')
        cos_max_org = cos_max.copy()
        for i in range(5):
            score = cos_max.max()
            # qi points to matching query
            qi, qj = np.where(cos_max==score)
            cos_max[qi,:] = 0
            match_query = query.face_tbl.loc[qi]
            # ti is pointing to matching template ()
            ti = cos_amax[qi,qj]
            video_id, face_id = corpus.t_tbl.loc[ti,['video_id','tid2fid']].values.tolist()[0]
            face_id = int(face_id)
            video_id = int(video_id)
            video_fname = corpus.video_tbl.loc[video_id].video_fname    
            t = corpus.face_tbl[(corpus.face_tbl.video_id==video_id) & (corpus.face_tbl.face_id == face_id)]
            frame_nums = t.frame_num.tolist()
            fnames = [f'faceid_{face_id:04d}_{frame_num:04d}.png' for frame_num in frame_nums]
            fnames = [os.path.join(video_fname,x) for x in fnames]
            name = match_query.person_id.tolist()[0] 
            res_name = f'{name}.{score}.txt'
            with open(os.path.join(video_fname,res_name), 'w') as f:
                f.writelines('\n'.join(fnames))
            logging.info(f'matching {name} to {face_id}={score:4f}')
            logging.info(f'cosine_sim={score}')
        
    return cos_max_org, cos_amax

def search_missing_usig_embeddings(missing_db_root, index_root, K=10):
    corpus = SearchIndex(512)
    corpus.corpus_dir = index_root 
    query = SearchIndex(512)
    query.corpus_dir = missing_db_root 

    with corpus as sig_tbl:
        enorm = corpus.face_tbl.enorm.values
        ns = enorm.size 
        nsigs = sig_tbl['embeddings'][:ns]
        # nsigs = sigs / enorm[:,np.newaxis]        
        file_name_list = list()
        with query as qtbl:
            n0 = len(query.face_tbl)
            cos_max = np.zeros((n0,K))
            cos_amax = np.zeros((n0,K),dtype=np.int32)
            qnorm = query.face_tbl.enorm.values
            for i, g in query.face_tbl.groupby(by='face_id'):
                ix = np.array(g.index)
                m0 = len(g)                
                qnsigs = qtbl['embeddings'][ix]
                # qnsigs = qsigs / qnorm[ix][:,np.newaxis]
                
                cos_sim = nsigs @ qnsigs.transpose()
                for k in range(K):
                    cos_max[ix,k] = cos_sim.max(axis=0)
                    cos_amax[ix,k] = cos_sim.argmax(axis=0)
                    cos_sim[cos_amax[ix,k],np.arange(m0)] = 0
                score = cos_max[ix,0].max()
                fid = int(g.iloc[0].face_id)
                logging.info(f'face_id={fid},num_faces={m0},max={score:.3f}')
        cos_max_org = cos_max.copy()
        for i in range(10):
            score = cos_max.max()
            # qi points to matching query
            qi, qj = np.where(cos_max==score)
            cos_max[qi,:] = 0
            match_query = query.face_tbl.loc[qi]
            # ti is pointing to matching template ()
            ti = cos_amax[qi,qj]
            video_id, face_id, frame_num, idx = (corpus.face_tbl.loc[ti,['video_id','face_id','frame_num','idx']].values).astype(np.int32).tolist()[0]
            video_fname = corpus.video_tbl.loc[video_id].video_fname
            tvid = corpus.face_tbl[corpus.face_tbl.video_id==video_id]
            if face_id >=0:
                t = tvid[tvid.face_id == face_id]
            else:
                t = tvid[(tvid.frame_num == frame_num) &(tvid.idx == idx)]
            for ix, r in t.iterrows():
                frame_num = int(r.frame_num)
                idx = int(r.idx)
                fname = f'face_{frame_num:05d}_{idx:04d}.png'
                fname = os.path.join(video_fname, fname)
                file_name_list.append(fname)
                name = match_query.person_id #.tolist()[0]
                logging.info(f'missing {name}')
                logging.info(f'{fname}')
                logging.info(f'cosine_sim={score}')
                if face_id < 0:
                    break
        with open('/tmp/files.txt','w') as fh:
            fh.writelines('\n'.join(file_name_list))
    return cos_max_org, cos_amax



def debug_sigs(args):
    from deepface import DeepFace
    from deepface.DeepFace import functions

    fnames = [args.query_img_path]
    d = torch.load(args.input_path)
    db = d['db']
    db0 = db[db.video_id.str.contains('33')]
    img_paths = db0.img_path.to_list()

    db_path = os.path.split(img_paths[-1])[0]
    db_path = os.path.split(db_path)[0]
    #args.query_img_path = img_paths[20]
    a = DeepFace.find(args.query_img_path, db_path, 
                      model_name='ArcFace', detector_backend='retinaface',
                      enforce_detection=False)


    sigs = extract_signatures(img_paths, detector_backend='retina_face')
    a = DeepFace.verify(img1_path = img_paths[0], img2_path=img_paths[1],
                    model_name='ArcFace',detector_backend='skip')

    sig = extract_signatures(fnames,detector_backend='retinaface')
    query = [np.array(x['embedding'])[np.newaxis,:] for x in sig]
    query = np.concatenate(query, axis=0)
    






    
def img_filter(x):
    return os.path.splitext(x)[-1].lower() in ['.jpeg','jpg','png']

def display_results():
    if 0:
        queries = [os.path.join('/tmp/query',x) for x in os.listdir('/tmp/query/')]
        q_list = list()
        for qname in queries:
            try:
                q = torch.load(qname)
            except:
                continue
            q_list += q
        scores = [x[1][0]['score'] for x in q_list]
        scores = np.array(scores)
        ix = np.argsort(scores)
        top_k_queries = [q_list[ix[i]] for i in range(10)]
    else:
        top_k_queries = torch.load('/tmp/best_queries.pth')

    layout = render_query_res(top_k_queries[:10])
    serve_app(layout)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # input and output
    parser.add_argument('--corpus_dataset', type=str, default='video.mp4', help='root of index')  
    parser.add_argument('--query_dataset', type=str, default='output/', help='query') 
    args = parser.parse_args()
    #display_results()    
    logger_init()
    #import logging
    logger = logging.getLogger()

    root = os.environ['HOME']
    query_dataset = args.query_dataset 
    corpus_dataset = args.corpus_dataset

    # search_missing_usig_embeddings(query_dataset, corpus_dataset)
    search_missing(query_dataset, corpus_dataset)

    search_missing('')
    logger.info(f'command line={sys.argv}')
    logging.info('hello')
    index_files = get_files2process(args.db_root, flt=filter_index_files) #[:4]
    query_images = get_files2process(args.query, flt=img_filter)
    logger.info(f'working on {len(index_files)} index files and {len(query_images)} queries')
    logger.info(f'index_files={index_files}')
    logger.info(f'query_images={query_images}')
    logger.info("Generating index")
    corpus_fname = f'/tmp/search_corpus.{len(index_files)}.pth'
    if os.path.isfile(corpus_fname):
        corpus = torch.load(corpus_fname)
    else:
        corpus = SearchIndex.from_index_files(index_files)
        #torch.save(corpus, corpus_fname)
    num_faces = len(corpus.db)
    num_queries = len(query_images)
    logger.info(f'looking for {num_queries} people in {num_faces} embeddings')
    all_queries = list()
    for ix,query_fname in tqdm(enumerate(query_images), desc='extract-sig'):
        try:
            cimg = Image.open(query_fname)
        except:
            continue
        logger.info(f'working on image {query_fname}, size={cimg.size}')
        query_res = corpus.search_by_query_images([query_fname])
        torch.save(query_res,f'/tmp/query_{ix:03d}_results.pth')
        all_queries += query_res
        if ix % 5 == 0:
            torch.save(all_queries,f'/tmp/all_queries.pth')


    # layout = render_query_res(query_res)
    # serve_app(layout)
