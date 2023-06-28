import os
from dataset import Dataset
from helper_functions import *
from tqdm import tqdm

################################################################################
# Pfade abgreifen
################################################################################
DATA_DIR = './data'

x_train_dir = os.path.join(DATA_DIR, 'image/train')
y_train_dir = os.path.join(DATA_DIR, 'seg13mod/train')

x_test_dir = os.path.join(DATA_DIR, 'image/test')
y_test_dir = os.path.join(DATA_DIR, 'seg13mod/test')

################################################################################
# Dataset-Objekt erstellen
################################################################################
CLASSES = ['bed', 'books', 'ceiling', 'chair', 'floor', 'furniture', 'objects',
            'picture', 'sofa', 'table' ,'tv', 'wall', 'window', 'unlabeled']

dataset = Dataset(x_train_dir, y_train_dir, classes=CLASSES)

################################################################################
# Dataset visualisieren
################################################################################
image, mask = dataset[3]
visualize_img_mask(image=image, gt_mask=mask, pr_mask=mask, filename='test.png')