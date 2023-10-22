#!/bin/bash
#root_dir=${PWD}../configs/ 
images_dir=${HOME}/data/missing_faces
pipeline_dir=${images_dir}.pipeline 
dataset_dir=${images_dir}.dataset 

python index_builder.py --input_directory ${images_dir} --output_directory ${pipeline_dir} --action create_virtual_video
python frames_to_faces.py --input_directory ${pipeline_dir}
python faces_to_embeddings.py --input_directory ${pipeline_dir}
python face_to_faceid.py --input_directory ${pipeline_dir} --config_deepsort ../configs/deep_sort.yaml 
python index_builder.py --input_directory ${pipeline_dir} --output_directory ${dataset_dir}.coreset --action build_index --aggregation FaceCoresetNet
python index_builder.py --input_directory ${pipeline_dir} --output_directory ${dataset_dir}.avgpool --action build_index --aggregation avg_pool


