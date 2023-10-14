

#input: input directory containing video files.
#output: output directgory containing each video frames in a different directory
import logging
import time

import cv2
import os
import argparse
from face_search.fs_logger import logger_init
from face_search.utils import get_video_process_dir
from face_search.utils import is_video, get_files2process

def extract_frames(video_files):
    # List all video files in the input directory

    # Iterate over each video file
    for video_file in video_files:
        process_dir = get_video_process_dir(video_file)
        logging.info(f'working on {video_file}')


        # Create a subdirectory with the same name as the video file
        # output_subdir = os.path.join(output_dir, video_name)
        logging.info(f'saving results into {process_dir}')
        # Open the video file
        cap = cv2.VideoCapture(video_file)

        # Initialize a frame counter
        frame_count = 0
        t0 = time.time()

        while True:
            # Read a frame from the video
            ret, frame = cap.read()

            if not ret:
                break

            # Construct the output file name
            frame_filename = os.path.join(process_dir, 
                                          f'frame_{frame_count:04d}.png')

            # Save the frame as an image file
            cv2.imwrite(frame_filename, frame)

            frame_count += 1

        # Release the video capture object
        cap.release()
        t1 = time.time()
        logging.info(f'Extracted {frame_count} frames from {video_file}.')
        logging.info(f'processing time={t1-t0}')
        

if __name__ == "__main__":
    logger_init()
    parser = argparse.ArgumentParser()

    # Define the two arguments
    parser.add_argument('--input_directory', help="The directory containing videos")

    # Parse the command-line arguments
    args = parser.parse_args()
    args.input_directory = '/Users/eranborenstein/pc/FaceSearch/configs/'
    
    # Access the values of the arguments
    input_directory = args.input_directory    
    video_files = get_files2process(input_directory, flt=lambda x:is_video(x))

    extract_frames(video_files)
