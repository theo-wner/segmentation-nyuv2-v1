import os
from dataset import Dataset
from helper_functions import *
from tqdm import tqdm
import segmentation_models_pytorch as smp

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
            'picture', 'sofa', 'table' ,'tv', 'wall', 'window', 'unlabeled', 'no information']

dataset = Dataset(x_train_dir, y_train_dir, classes=CLASSES, 
                  augmentation=get_training_augmentation())

################################################################################
# Dataset visualisieren
################################################################################
for i in tqdm(range(len(dataset))):
    image, mask = dataset[i]
    filename = 'test_' + str(i) + '.png'
    visualize_img_mask(image, mask, mask, filename=filename)