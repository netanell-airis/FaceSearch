import argparse
import logging
import os
import sys
import pandas as pd
import time
import numpy as np
from PIL import Image 
from tqdm import tqdm
from face_search.fs_logger import logger_init
import face_search.utils as utils
from face_search.search_index import SearchIndex
import re 

def build_index(args):
    """
    needs video_list, corpus_dir
    """
    video_files = utils.get_video_files(args)
    corpus_dir = args.output_directory
    logging.info(f'building a corpus from {len(video_files)}')
    logging.info(f'saving the dataset in {corpus_dir}')
    pipeline_dirs = [os.path.splitext(x)[0]+'.pipeline' for x in video_files]
    pipeline_dirs = list(filter(lambda x:os.path.isdir(x), pipeline_dirs))
    corpus = SearchIndex.from_pipeline_outputs(pipeline_dirs,corpus_dir, 512)
    build_template_db(args, corpus)


def build_template_db(args, corpus=None):
    """
    requires corpus_dir
    """
    if corpus is None:
        corpus_dir = args.output_directory
        corpus = SearchIndex(512)
        corpus.corpus_dir = corpus_dir 
    corpus.mode = 'a'
    
    with corpus as sig_tbl:
        E = sig_tbl['embeddings']
        sig_tbl['templates'].resize(E.shape)
        face_tbl = corpus.face_tbl 
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
            T = utils.get_gallery_templates(fid, E_vid)
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


def create_virtual_video(args):
    video_dir = args.output_directory
    frames = utils.get_files2process(args.input_directory,
                                     flt=lambda x:utils.is_img_fname(x))
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


def update_index(args):
    logging.info('not implmented')
    return


if __name__ == "__main__":
    logger_init()
    parser = argparse.ArgumentParser()
    parser.add_argument('--action',
        help="action=['create_virtual_video','build_index','update_index']")
    parser.add_argument('--input_directory',
        help="The root directory containing video frames")
    parser.add_argument('--output_directory',
        help="The root directory to save the dataset")
    args = parser.parse_args()

    if 'Users' in os.environ['HOME']:
        args.input_directory = '/Users/eranborenstein/data/missing_faces.pipeline'
        args.output_directory = '/Users/eranborenstein/data/missing_faces.dataset'
        args.action = 'build_index'


    action_fn = {"create_virtual_video": create_virtual_video,
              "build_index": build_index, 
              "update_index": update_index,
              "build_templates": build_template_db}
    if args.action not in action_fn:
        logging.error(f'action={args.action} not implemented.')
        sys.exit(-1)

    logging.info(f'Running {args.action}')
    logging.info(f'args={sys.argv}')

    t0 = time.time()
    action_fn[args.action](args)    
    t1 = time.time()
    logging.info(f'process took {t1-t0:.2f}secs')

