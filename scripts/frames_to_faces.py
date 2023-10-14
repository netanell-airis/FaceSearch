import os
import logging
import time
import re
import pandas as pd
import PIL.Image
from mtcnn import MTCNN
import cv2
import argparse
from face_search.fs_logger import logger_init
from face_search.utils import is_video_frame, get_video_process_dir
from face_search.utils import is_video, get_files2process

def detect_faces(video_files):
    # Initialize the MTCNN detector
    detector = MTCNN(steps_threshold=[0.6,0.7,0.9])

    # Iterate over each video file
    for video_file in video_files:
        process_dir = get_video_process_dir(video_file)
        frame_list = sorted(get_files2process(process_dir, flt=lambda x:is_video_frame(x) is not None))
        logging.info(f'working on video {video_file} with {len(frame_list)}')
        face_list = list()
        for input_image_path in frame_list: 
            frame_num = int(is_video_frame(input_image_path))
            # Load the input image
            image = cv2.cvtColor(cv2.imread(input_image_path), cv2.COLOR_BGR2RGB)
            #image = Image.open(input_image_path)
            #pixels = image.convert('RGB')
            #pixels = pixels.resize((224, 224))

            # Detect faces in the image
            faces = detector.detect_faces(image)
            image = PIL.Image.fromarray(image)
            # Save the detector output to a file
            for i, face in enumerate(faces):
                x, y, width, height = face['box']
                confidence = face['confidence']
                output_filename = os.path.join(process_dir, f'face_{frame_num:05d}_{i:04d}.png')
                # Crop the face from the original image
                face_image = image.crop((x, y, x + width, y + height))
                face_image.save(output_filename, 'PNG')
                face_list.append((frame_num, i, confidence, x, y, width, height))
        
        db = pd.DataFrame(face_list, columns = ['frame_num','idx','confidence', 'x','y','w','h'])
        db.to_csv(os.path.join(process_dir,'faces.csv'))
        logging.info(f'finished detecting {len(db)} faces for video')





if __name__ == "__main__":
    logger_init()
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_directory', help="The root directory containing video frames")
    args = parser.parse_args()

    args.input_directory = '/Users/eranborenstein/pc/FaceSearch/configs'   
    video_files = get_files2process(args.input_directory, flt=lambda x:is_video(x))

    logging.info(f'detecting faces in {len(video_files)} videos')
    t0 = time.time()
    detect_faces(video_files)
    t1 = time.time()
    logging.info(f'process took {t1-t0}secs')








