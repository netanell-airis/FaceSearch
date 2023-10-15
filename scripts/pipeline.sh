#!/bin/bash
#root_dir=${PWD}../configs/ 
root_dir=${HOME}/data/missing_faces.pipeline

python video_to_frames.py --input_directory ${root_dir}
python frames_to_faces.py --input_directory ${root_dir}
python faces_to_embeddings.py --input_directory ${root_dir}
python face_to_faceid.py --input_directory ${root_dir} --config --config_deepsort ../configs/deep_sort.yaml 
