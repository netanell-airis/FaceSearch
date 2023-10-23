import os
import argparse
from PIL import Image, ImageDraw, ImageFont
import cv2
import pandas as pd
import numpy as np

from face_search import io
from face_search import utils
from face_search.utils import xcycwh2xyxy, xywh2xyxy, get_output_dir, get_video_process_dir

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


def draw_faces_on_frames(video_files, pipeline_dir, output_path=None, save_frames=False):
    for video_fname in video_files:   
        vid_name = os.path.splitext(os.path.basename(video_fname))[0]

        if pipeline_dir is None:
            videos_dir = os.path.dirname(video_fname)
            pipeline_dir = get_video_process_dir(video_fname, videos_dir, create=False)
        face_ids_root = get_output_dir(video_fname, pipeline_dir, 'face_ids', create=False)  # os.path.splitext(video_fname)[0] + '.pipeline'     
        frames_root = get_output_dir(video_fname, pipeline_dir, 'frames', create=False)  # os.path.splitext(video_fname)[0] + '.pipeline'     
        
        faces_df = io.load_table(face_ids_root, "faces")
        if faces_df is None:
            faces_root = get_output_dir(video_fname, pipeline_dir, 'face_detections', create=False)
            faces_df = io.load_table(faces_root, "faces")
        frame_nums = list(set(faces_df['frame_num'].values))  # list of frame nums

        frame0_path = os.path.join(frames_root, f'frame_{0:04d}.png')
        frame0 = Image.open(frame0_path)
        width, height = frame0.size[:2]
        output_video_name = f'{vid_name}_detections_video.mp4'
        
        if output_path is None:
            output_path = pipeline_dir
        out_vid_path = os.path.join(output_path, output_video_name)
        video = cv2.VideoWriter(out_vid_path, cv2.VideoWriter_fourcc(*'DIVX'), 5, (width, height))
    
        # get colors for all face_ids:
        # TODO Anna: see what happens when no face_ids at all
        unique_face_ids = faces_df.face_id.unique() if 'face_id' in faces_df.columns else []
        colors = get_random_colors_for_face_ids(unique_face_ids)

        for frame_num in frame_nums:
            img_path = os.path.join(frames_root, f'frame_{frame_num:04d}.png')
            img = Image.open(img_path)
            frame_df = faces_df[faces_df['frame_num']==frame_num] 

            draw_faces_on_image(img, frame_df, colors)

            # for debug - save frame:
            if save_frames:
                fname = f'frame_{frame_num:04d}_face_detections.png'
                fname = os.path.join(face_ids_root,fname)
                img.save(fname)

            # add frame to video:
            cv_img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
            video.write(cv_img)

        video.release()        
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--videos_directory', help="The root directory containing video files")
    parser.add_argument('--pipeline_directory', help="The root directory containing pipeline files")
    # parser.add_argument('faces_csv', help='Path to faces csv file')
    parser.add_argument('--output_path', help='Path to save output video. Default will be same as outputs_directory') 
    parser.add_argument('--save_intermediate', action='store_true', default=False)

    
    # TODO anna: fix this mess
    args = parser.parse_args()
    videos_directory = args.videos_directory
    pipeline_directory = args.pipeline_directory
    # pipeline_directory = pipeline_directory if pipeline_directory is not None else videos_directory
    output_path = args.output_path
    save_frames = args.save_intermediate

    video_files = utils.get_video_files(args.videos_directory)
    # faces_df = pd.read_csv(args.csv_path)

    draw_faces_on_frames(video_files, pipeline_directory, output_path, save_frames)
