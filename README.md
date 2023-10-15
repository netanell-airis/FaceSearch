# FaceSearch

A pipeline for large scale face recognition in videos and images. 

Given a list of videos to process: 
video_list:
    video_1.mp4
    video_2.mp4
    ...
    video_n.mp4
we create a pipeline with 4 steps:

1) video_to_frames.py: given a video_k.mp4 or a directory of images, opens the video/images, creates a pipeline output directory, e.g. video_1.pipeline 
and extracts frame_{frame_num:04d}.png 
It also creates a frame table with frame_nums: frame_tbl.csv

2) frames_to_faces.py:
given a video_k.pipeline directory:
for each frame in frame_tbl.csv: 
    extract faces with roi, score

create a face table faces.csv 

3) faces_to_embeddings.py 
given a video_k.pipeline directory:
for each face in faces.csv extract embeddings. 
Save embeddings into embeddings.pth 
add the embedding norm as enorm to faces.csv 

4) face_to_faceid.py 
given a video_k.pipeline directory, assign to 
each face in faces.csv a face_id. 

5) 

A library and scripts to extract faces and their signatures from a list of videos. 

install:
python setup.py (develop)

see scripts directory for examples. 

