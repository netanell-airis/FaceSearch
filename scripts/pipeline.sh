#!/bin/bash
#root_dir=${PWD}../configs/ 
root_dir=/home/anna-airis/work/data/FaceSearch/test_video

python scripts/video_to_frames.py --input_directory ${root_dir}
#python frames_to_faces.py --input_directory ${root_dir}
#python faces_to_embeddings.py --input_directory ${root_dir}
#python face_to_faceid.py --input_directory ${root_dir} --config_deepsort ../configs/deep_sort.yaml 
#python index_builder --input_directory ${root_dir} --output_directory ${root_dir}.dataset --action build_index
