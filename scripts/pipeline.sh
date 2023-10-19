#!/bin/bash
#root_dir=${PWD}../configs/ 
root_dir=${HOME}/data/missing_faces.pipeline
root_dir=${HOME}/data/VID-20231008-WA0033.mp4
root_dir=${HOME}/data/debug_0
root_dir=${HOME}/data/debug1
root_dir=${HOME}/data/sample_data

python video_to_frames.py --input_directory ${root_dir}
python frames_to_faces.py --input_directory ${root_dir} --mp4
python faces_to_embeddings.py --input_directory ${root_dir}
python face_to_faceid.py --input_directory ${root_dir} --config_deepsort ../configs/deep_sort.yaml 
python index_builder.py --input_directory ${root_dir} --output_directory ${root_dir}.dataset.coreset --action build_index --aggregation FaceCoresetNet
python index_builder.py --input_directory ${root_dir} --output_directory ${root_dir}.dataset.avgpool --action build_index --aggregation avg_pool

