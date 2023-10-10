import argparse
import os
from PIL import Image
from deepface import DeepFace
from deepface.DeepFace import functions
import cv2 
import numpy as np
from tqdm import tqdm 

def filter_imgs_by_ext(fname):
    ext = os.path.splitext(fname)[-1]
    res = ext in ['.png']
    return res

def batch_bgr2rgb(fnames):
    for img_path in fnames:
        img = cv2.imread(img_path)
        img1 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        cv2.imwrite(img_path, img1)
        img1 = Image.open(img_path)

def extract_signatures(fnames, target_size=(112,112)):
    signatures = list()
    for img_path in tqdm(fnames,desc='extract_signatures'):
        try: 
            sig = DeepFace.represent(img_path,
                                        model_name='ArcFace',
                                        detector_backend='retinaface')
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
    extract_signatures(imgs2proc)
            #sigs, sig_db = extract_signatures(imgs)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # input and output
    parser.add_argument('--input_path', type=str, default='video.mp4', help='source')  
    parser.add_argument('--save_path', type=str, default='output/', help='output folder') 
    args = parser.parse_args()
    print(args)

    run(args)
