from face_search.fs_logger import logger_init
from face_search.utils import get_files2process, is_video
import logging


from utils_ds.parser import get_config
from utils_ds.draw import draw_boxes
from deep_sort import build_tracker
from PIL import Image 
import argparse
import os
import pandas as pd
import time
import numpy as np
from deep_sort.sort.track import Track
from tqdm import tqdm
from face_search.search_index import SearchIndex
import sys

def build_dataset(video_list, corpus_dir):
    logging.info(f'building a corpus from {len(video_list)}')
    logging.info(f'saving the dataset in {corpus_dir}')
    pipeline_dirs = [os.path.splitext(x)[0]+'.pipeline' for x in video_list]
    pipeline_dirs = list(filter(lambda x:os.path.isdir(x), pipeline_dirs))
    corpus = SearchIndex.from_pipeline_outputs(pipeline_dirs,corpus_dir, 512)


def build_template_db(corpus_dir):
    from face_search.utils import get_gallery_templates
    corpus = SearchIndex(512)
    corpus.mode = 'a'
    corpus.corpus_dir = corpus_dir 
    with corpus as sig_tbl:
        E = sig_tbl['embeddings']
        sig_tbl['templates'].resize(E.shape)
        face_tbl = corpus.face_dbl 
        video_tbl = corpus.video_tbl 
        ftbl = face_tbl[face_tbl.face_id.isna() == False]
        ftbl = ftbl[ftbl.face_id >= 0]        
        i0 = 0 
        t_tbl_lst = list()
        for vix, vg in ftbl.groupby(by='video_id'):
            ix = np.array(vg.index)
            E_vid = sig_tbl['embeddings'][ix]
            norms = vg.enorm.values 
            E_vid = E_vid * norms[:,np.newaxis]
            fid = vg.face_id.values
            T = get_gallery_templates(fid, E_vid)
            tid2fid = np.arange(int(fid.max()+1))
            tn = np.linalg.norm(T,axis=1)
            tid2fid = tid2fid[tn >0]
            T = T[tn > 0]
            i1 = i0 + T.shape[0]
            sig_tbl['templates'][i0:i1] = T
            enorm = np.linalg.norm(T, axis=1)
            i0 = i1
            et = np.concatenate((enorm[:,np.newaxis], tid2fid[:,np.newaxis]),axis=1)
            ttbl = pd.DataFrame(et, columns=['enorm','tid2fid'])
            ttbl['video_id'] = vix
            t_tbl_lst.append(ttbl)
        t_tbl = pd.concat(t_tbl_lst, axis=0)
        fnames = corpus.get_db_fnames(corpus.corpus_dir)
        t_tbl.to_csv(fnames['t_tbl_fname'])


def create_virtual_video(frames, video_dir):
    import re 
    rname = re.compile(r'([A-Za-z0-9]+)_([A-Za-z0-9]+).')    
    frame_tbl = list()
    os.makedirs(video_dir, exist_ok=True)
    for frame_num,img_dir in tqdm(enumerate(sorted(frames)),desc='virt-video'):
        img = Image.open(img_dir)
        fid = os.path.splitext(os.path.split(img_dir)[-1])[0]
        r0 = rname.findall(fid)
        if r0 is None:
            logging.warning(f'could not parse {img_dir}')
            continue
        first,last = r0[0]
        fid = f'{first}.{last}'
        new_fname = os.path.join(video_dir,f'frame_{frame_num:04d}.png')

        first, last, _ = new_fname.split('_')
        frame_tbl.append((frame_num, img_dir,fid))
        img.save(new_fname)
    face_tbl = pd.DataFrame(frame_tbl, columns=['frame_num','fname','name'])
    face_tbl.to_csv(os.path.join(video_dir, 'frames.csv'))
    logging.info(f'saved {len(face_tbl)} faces into {video_dir}')

    
        

from face_search.utils import is_img_fname



if __name__ == "__main__":
    logger_init()
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_directory', help="The root directory containing video frames")
    parser.add_argument('--corpus_directory', help="The root directory to save the dataset")
    parser.add_argument('--images_root', help="The root for all images we want to use for virtual video.")
    args = parser.parse_args()

    if os.path.splitext(args.input_directory)[-1] == '.pipeline':
        video_files = [args.input_directory]
        args.corpus_directory = os.path.splitext(video_files[0])[0] + '.index'
    else:
        video_files = get_files2process(args.input_directory, flt=lambda x:is_video(x))
        missing_faces = get_files2process(args.images_root, 
                                      flt=is_img_fname)
        video_dir = args.images_root + '.pipeline'
    t0 = time.time()
    
    # create_virtual_video(missing_faces, video_dir)
    # build_template_db(args.corpus_directory)
    build_dataset(video_files, args.corpus_directory)
    t1 = time.time()
    logging.info(f'process took {t1-t0:.2f}secs')

