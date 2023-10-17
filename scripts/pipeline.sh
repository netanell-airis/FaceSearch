#!/bin/bash
#root_dir=${PWD}../configs/ 
root_dir=${HOME}/data/missing_faces.pipeline
root_dir=${HOME}/data/VID-20231008-WA0033.mp4
root_dir=${HOME}/data/debug_0

python video_to_frames.py --input_directory ${root_dir}
python frames_to_faces.py --input_directory ${root_dir} --mp4
# python faces_to_embeddings.py --input_directory ${root_dir}
# python face_to_faceid.py --input_directory ${root_dir} --config_deepsort ../configs/deep_sort.yaml 
#python index_builder.py --input_directory ${root_dir} --output_directory ${root_dir}.dataset --action build_index

