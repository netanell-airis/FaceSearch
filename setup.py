import os
from setuptools import setup, find_packages

scripts=['video_to_frames.py',
           'frames_to_faces.py',
           'face_to_faceid.py',
           'faces_to_embeddings.py',
           'faceid_to_template.py']

scripts = [os.path.join('scripts',x) for x in scripts]

setup(
	name='face_search',
	version='1.0',
    packages=find_packages(
        where='src',
        include=['pkg*'],  # alternatively: `exclude=['additional*']`
    ),
    package_dir={"": "src"},
    data_files=[('configs',['configs/video.mp4','configs/deep_sort.yaml']),
                ('scripts',scripts)], 
	include_package_data=True,    
)
