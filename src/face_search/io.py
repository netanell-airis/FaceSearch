import os 
import pandas as pd 
import logging 

def get_face_table(video_root):
    return os.path.join(video_root,'faces.csv')
def get_faceid_table(video_root):
    return os.path.join(video_root,'face_ids.csv')
def get_templates_table(video_root):
    return os.path.join(video_root,'templates.csv')
def get_video_table(video_root):
    return os.path.join(video_root,'videos.csv')
def get_frames_table(video_root):
    return os.path.join(video_root,'frames.csv')

name2fn = {"frames": get_frames_table,
            "faces": get_face_table,
            "face": get_face_table,
            "face_id": get_faceid_table,
            "template": get_templates_table,
            "video": get_video_table,
            }

def load_table(video_root, tbl_name='frames'):
    fname = name2fn[tbl_name](video_root)
    if os.path.isfile(fname):
        logging.info(f'loading {tbl_name} table from {fname}')
        return pd.read_csv(fname)
    logging.warn(f'could not find {fname} for {tbl_name}')
    return None

def save_table(video_root, tbl, tbl_name='frames'):
    fname = name2fn[tbl_name](video_root)
    logging.info(f'saving {tbl_name} to {fname}')
    tbl.to_csv(fname, index=False)
