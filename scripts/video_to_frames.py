

#input: input directory containing video files.
#output: output directgory containing each video frames in a different directory
import logging
import time
import pandas as pd
import cv2
import os
import argparse
from face_search.fs_logger import logger_init
from face_search.utils import get_video_process_dir, get_output_dir, is_video, get_files2process
from face_search import io
from face_search.face_detector import detect_in_batches

   
def extract_frames(video_files, output_directory):
    # List all video files in the input directory

    # Iterate over each video file
    for video_file in video_files:
        # process_dir = get_video_process_dir(video_file)
        process_dir = get_output_dir(video_file, output_directory, 'frames')
        logger_init(os.path.join(process_dir,'extract_frames.log'))
        logging.info(f'working on {video_file}')

        # Create a subdirectory with the same name as the video file
        # output_subdir = os.path.join(output_dir, video_name)
        logging.info(f'saving results into {process_dir}')
        # Open the video file
        cap = cv2.VideoCapture(video_file)

        # Initialize a frame counter
        frame_count = 0
        t0 = time.time()
        frame_list = list()

        while True:
            # Read a frame from the video
            ret, frame = cap.read()
            if not ret:
                break

            # Construct the output file name
            base_name = f'frame_{frame_count:04d}.png'
            frame_filename = os.path.join(process_dir, base_name) 
            frame_list.append((frame_count, base_name))
            # Save the frame as an image file
            cv2.imwrite(frame_filename, frame)

            frame_count += 1

        # Release the video capture object
        cap.release()
        t1 = time.time()
        logging.info(f'Extracted {frame_count} frames from {video_file}.')
        logging.info(f'processing time={t1-t0}')
        tbl = pd.DataFrame(frame_list, columns=['frame_num','file_name'])
        io.save_table(process_dir, tbl, "frames")
        

if __name__ == "__main__":
    logger_init()
    parser = argparse.ArgumentParser()

    # Define the two arguments
    parser.add_argument('--input_directory', help="The directory containing videos")
    parser.add_argument('--output_directory', help="Directory where output is saved, default is input_directory")

    # Parse the command-line arguments
    args = parser.parse_args()
   
    # Access the values of the arguments
    from face_search.utils import get_video_files
    input_directory = args.input_directory    
    output_directory = args.output_directory
    if output_directory is None:
        output_directory = input_directory

    video_files = get_video_files(args)

    extract_frames(video_files, output_directory)
    # extract_faces_from_videos(video_files)
