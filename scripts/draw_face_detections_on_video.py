import os
import argparse
from PIL import Image, ImageDraw, ImageFont
import cv2
import pandas as pd
import numpy as np

from face_search import io
from face_search import utils
from face_search.utils import xcycwh2xyxy, xywh2xyxy

DEFAULT_FACE_ID = -1
DEFAULT_COLOR = 'darkviolet'

TEXT_BOX_MARGIN = 5
TEXT_BOX_TOP = 15

np.random.seed(5)

def get_random_colors_for_face_ids(unique_face_ids):
    colors = {}
    for id in unique_face_ids:
        colors[id] = tuple(np.random.choice(range(256), size=3))
    
    colors[DEFAULT_FACE_ID] = DEFAULT_COLOR

    return colors


def get_face_ids(frame_df):
    if 'face_id' in frame_df.columns:
        face_ids = frame_df['face_id'].values
    else:
        face_ids = [DEFAULT_FACE_ID for _ in range(frame_df.shape[0])]
    
    # convert nans to default value
    face_ids = [int(id) if not np.isnan(id) else DEFAULT_FACE_ID for id in face_ids]

    return face_ids   


def draw_faces_on_image(img, frame_df, colors):
    draw = ImageDraw.Draw(img)

    boxes = frame_df[['x','y','w','h']].values
    # boxes = xcycwh2xyxy(boxes)
    boxes = xywh2xyxy(boxes)
    face_ids = get_face_ids(frame_df)
    
    for box, face_id in zip(boxes, face_ids):
        # draw detected face bounding box
        if np.isnan(face_id) or face_id == -1:
            id_color = DEFAULT_COLOR
        else:
            id_color = colors[face_id]
        draw.rectangle(tuple(box), outline=id_color, width=3)

        if not (np.isnan(face_id) or face_id == -1):
            # add text
            text = f'face_id = {face_id}'
            tbpos = (box[0], box[1] - TEXT_BOX_TOP)
            # tbleft, tbtop, tbright, tbbottom = draw.textbbox((tbpos[0], tbpos[1]), text)
            # draw.rectangle((tbleft-TEXT_BOX_MARGIN, tbtop+TEXT_BOX_MARGIN, 
            #                 tbright+TEXT_BOX_MARGIN, tbbottom+TEXT_BOX_MARGIN), fill=id_color)
            draw.text(tbpos, text, fill=id_color) 


def draw_faces_on_frames(video_files, output_path=None):
    for video_fname in video_files:   
        video_root = os.path.splitext(video_fname)[0] + '.pipeline'     
        faces_df = io.load_table(video_root, "faces")
        frame_nums = list(set(faces_df['frame_num'].values))  # list of frame nums

        frame0_path = os.path.join(video_root, f'frame_{0:04d}.png')
        frame0 = Image.open(frame0_path)
        width, height = frame0.size[:2]
        video_name = 'detections_video.mp4'
        vid_path = os.path.join(video_root, '..', video_name)
        video = cv2.VideoWriter(vid_path, cv2.VideoWriter_fourcc(*'DIVX'), 5, (width, height))
    
        # get colors for all face_ids:
        # TODO Anna: see what happens when no face_ids at all
        unique_face_ids = faces_df.face_id.unique() if 'face_id' in faces_df.columns else []
        colors = get_random_colors_for_face_ids(unique_face_ids)

        for frame_num in frame_nums:
            img_path = os.path.join(video_root, f'frame_{frame_num:04d}.png')
            img = Image.open(img_path)
            frame_df = faces_df[faces_df['frame_num']==frame_num] 

            draw_faces_on_image(img, frame_df, colors)

            # for debug - save frame:
            fname = f'frame_{frame_num:04d}_face_detections.png'
            fname = os.path.join(video_root,fname)
            img.save(fname)

            # add frame to video:
            cv_img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
            video.write(cv_img)

            # if frame_num > 22:
            #     break

        video.release()
        break
        
    

    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_directory', help="The root directory containing video frames")
    # parser.add_argument('faces_csv', help='Path to faces csv file')
    parser.add_argument('--output_path', help='Path to save output. Default will be same as videos_path')

    args = parser.parse_args()
    video_files = utils.get_video_files(args)
    # faces_df = pd.read_csv(args.csv_path)
    output_path = args.output_path

    draw_faces_on_frames(video_files, output_path)
