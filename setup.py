from setuptools import setup, find_packages
setup(
	name='face_search',
	version='1.0',
	package_dir={"": "src"},
	packages=find_packages("src"),
	scripts=['video_to_frames.py', 'frames_to_faces.py','face_to_faceid.py','faces_to_embeddings.py','faceid_to_template.py'],
	include_package_data=True,
)
