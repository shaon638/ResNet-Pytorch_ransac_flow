import cv2
from glob import glob
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms
from torchvision.datasets.folder import find_classes
import queue
import sys

# Image formats supported by the image processing library
IMG_EXTENSIONS = ("jpg", "jpeg", "png", "ppm", "bmp", "pgm", "tif", "tiff", "webp")

# The delimiter is not the same between different platforms
if sys.platform == "win32":
    delimiter = "\\"
else:
    delimiter = "/"

class ImageDataset(Dataset):
	def __init__(self, image_dir:"/ssd_scratch/cvit/shaon/Data/train", image_size: 28, transform=None, target_transform=None):
		super(ImageDataset, self).__init__()
		#iterate over all image paths
		self.image_file_paths = glob(f"{image_dir}/*/*")
		_,self.class_to_idx = find_classes(image_dir)
		self.image_size = image_size
		self.delimeter = delimiter
		self.transform = transform
		self.target_transform = target_transform

	def __len__(self):
		return len(self.image_files_paths)

	def __getitem__(self, )





