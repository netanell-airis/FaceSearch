import argparse
import os
from PIL import Image
from deepface import DeepFace
from deepface.DeepFace import functions
import cv2 
import numpy as np
import torch 
import pandas as pd
from tqdm import tqdm 


def rotate_90(img, clockwise=True):
    code = cv2.ROTATE_90_CLOCKWISE if clockwise else cv2.ROTATE_90_COUNTERCLOCKWISE
    img = cv2.rotate(img, code)
    return img



def filter_imgs_by_ext(fname):
    ext = os.path.splitext(fname)[-1]
    res = ext in ['.png']
    return res

def batch_transform(fnames, transform):
    for img_path in tqdm(fnames,desc='transform'):
        img = cv2.imread(img_path)
        img1 = transform(img)
        cv2.imwrite(img_path, img1)


def batch_bgr2rgb(fnames):
    for img_path in tqdm(fnames,desc='bgr2rgb'):
        img = cv2.imread(img_path)
        img1 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        cv2.imwrite(img_path, img1)

def extract_signatures(fnames, detector_backend = 'skip',target_size=(112,112)):
    signatures = list()
    model = DeepFace.build_model('ArcFace')
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

def get_imgs2process(in_dir):
    imgs2proc = list()
    for dirpath, dirnames, filenames in os.walk(in_dir):
        imgs = list(filter(lambda x:filter_imgs_by_ext(x), filenames))
        imgs = list(map(lambda x:os.path.join(dirpath,x), imgs))
        if len(imgs):
            imgs2proc += imgs
    return imgs2proc

def run(args):
    imgs2proc = get_imgs2process(args.input_path)    
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
    fnames = get_imgs2process(args.input_path)
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

    dbg = 1


def debug_sigs(args):
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
    

    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # input and output
    parser.add_argument('--input_path', type=str, default='video.mp4', help='source')  
    parser.add_argument('--save_path', type=str, default='output/', help='output folder') 
    args = parser.parse_args()
    args.input_path = "/Users/eranborenstein/data/airis/face_tracks/"
    args.save_path = "/Users/eranborenstein/data/airis/face_tracks/index.pth"
    args.input_path = args.save_path
    args.query_img_path = "/Users/eranborenstein/data/airis/face_tracks/noa_query_2.jpg"
    #create_index(args)
    debug_sigs(args)
    #extract_query_sig(args)
    #run(args)
    



    dbg = 1


    #run(args)
