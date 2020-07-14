import cv2
import numpy as np
from glob import glob
from os import path

def images_from_folder(folder_path, output_size=None, output_type=np.float32, bgr2rgb=False, interpolation=cv2.INTER_LINEAR):
	list_files = glob(path.join(folder_path, '*'))
	return images_from_list_files(list_files, output_size, output_type, bgr2rgb, interpolation), list_files
	
def images_from_list_files(list_files, output_size=None, output_type=np.float32, bgr2rgb=False, bgr2gray=False, interpolation=cv2.INTER_LINEAR):
	assert(output_size is None or len(output_size) == 2)
	assert(not (bgr2rgb and bgr2gray))

	n_files = len(list_files)

	im = cv2.imread(list_files[0], cv2.IMREAD_ANYCOLOR)
	is_gray = (im.ndim == 2)
	output_size = im.shape[:2] if output_size is None else output_size
	image_tensor = np.empty(shape=(n_files, *output_size[::-1], 1 if is_gray or bgr2gray else im.shape[2]), dtype=output_type)

	for i, file in enumerate(list_files):
		print('{} of {}'.format(i+1, n_files), end='\r')
		im = cv2.imread(file, cv2.IMREAD_ANYCOLOR)
		assert(im is not None), file

		if (not is_gray) and bgr2rgb:
			im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

		if (not is_gray) and bgr2gray:
			code = cv2.COLOR_RGB2GRAY if bgr2rgb else cv2.COLOR_BGR2GRAY
			im = cv2.cvtColor(im, code)

		im = cv2.resize(im, output_size, interpolation=interpolation).astype(output_type)
		image_tensor[i] = np.expand_dims(im, axis=2) if im.ndim == 2 else im
	print()

	return image_tensor