# FaceSearch

A pipeline for large scale face recognition in videos and images. 

## Usage
The input to the pipeline script (pipeline.sh) is a directory root_dir with a list of mp4 videos in it:

Directory root_dir:

* clip_1.mp4
* clip_2.mp4
* clip_3.mp4
...
* clip_\<n>.mp4

For each clip_\<k>.mp4 it will create a pipeline directory clip_\<k>.pipeline in root_dir:

* clip_1.pipeline
* clip_2.pipeline
* clip_3.pipeline
...
* clip_\<n>.pipeline


In each pipeline directory the following happens:

```
python video_to_frames.py --input_directory ${root_dir}
```
Will convert the clips to a list of frames: frame_\<frame_num>.png
and generate a frames.csv table 

```
python frames_to_faces.py --input_directory ${root_dir}
```

Will extract faces (png) from each frame and create a faces.csv table 

```
python faces_to_embeddings.py --input_directory ${root_dir}
```

Extract signatures (embeddings) for each face in faces.csv and store them in an embeddings.pth file.  


```
python face_to_faceid.py --input_directory ${root_dir} --deepsort = deep_sort.yaml
```
Assigns a face_id to each face in faces.csv 



we create a pipeline with 4 steps:

```
video_to_frames.py
```
Given a video_k.mp4 or a directory of images, opens the video/images, creates a pipeline output directory, e.g. video_1.pipeline 
and extracts frame_{frame_num:04d}.png 
It also creates a frame table with frame_nums: frame_tbl.csv

```
frames_to_faces.py
```
Given a video_k.pipeline directory:
for each frame in frame_tbl.csv: 
    extract faces with roi, score

create a face table faces.csv 

```
faces_to_embeddings.py
```
Given a video_k.pipeline directory:
for each face in faces.csv extract embeddings. 
Save embeddings into embeddings.pth 
add the embedding norm as enorm to faces.csv 

```
face_to_faceid.py 
```
Given a video_k.pipeline directory, assign to each face in faces.csv a face_id. 



A library and scripts to extract faces and their signatures from a list of videos. 

## Installation
```
pip install -r requirements
python setup.py (develop/install)
```


