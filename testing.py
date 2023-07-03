import os
import segmentation_models_pytorch as smp
import segmentation_models_pytorch.utils as smp_utils
import torch
from torch.utils.data import DataLoader
from dataset import Dataset
from helper_functions import *

################################################################################
# GPU-Abfrage
################################################################################
gpu_id = input('GPU ID: ')

################################################################################
# Pfade abgreifen
################################################################################
DATA_DIR = './data'

x_train_dir = os.path.join(DATA_DIR, 'image/train')
y_train_dir = os.path.join(DATA_DIR, 'seg13mod/train')

x_test_dir = os.path.join(DATA_DIR, 'image/test')
y_test_dir = os.path.join(DATA_DIR, 'seg13mod/test')

################################################################################
# Netz spezifizieren
################################################################################
ENCODER = 'se_resnext50_32x4d'
ENCODER_WEIGHTS = 'imagenet'
CLASSES = ['bed', 'books', 'ceiling', 'chair', 'floor', 'furniture', 'objects',
            'picture', 'sofa', 'table' ,'tv', 'wall', 'window', 'unlabeled', 'no information']
ACTIVATION = 'softmax2d'
DEVICE = 'cuda:' + gpu_id

preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)

loss = smp_utils.losses.CrossEntropyLoss()
#metrics = [smp_utils.metrics.IoU()]

################################################################################
# Bestes trainiertes Modell laden
################################################################################
best_model = torch.load('./best_model_multiclass.pth')

################################################################################
# Test-Datensatz
################################################################################
test_dataset = Dataset(
    x_test_dir, 
    y_test_dir, 
    augmentation=get_validation_augmentation(), 
    preprocessing=get_preprocessing(preprocessing_fn),
    classes=CLASSES,
)

test_dataloader = DataLoader(test_dataset)

################################################################################
# Predictions berechnen und visualisieren
################################################################################
test_dataset_vis = Dataset(x_test_dir, y_test_dir, classes=CLASSES)

for i in range(5):
    n = np.random.choice(len(test_dataset))
    
    image_vis = test_dataset_vis[n][0].astype('uint8')
    image, gt_mask = test_dataset[n]
    
    gt_mask = gt_mask.squeeze()
    
    x_tensor = torch.from_numpy(image).to(DEVICE).unsqueeze(0)
    pr_mask = best_model.predict(x_tensor)
    pr_mask = (pr_mask.squeeze().cpu().numpy().round())

    print(image_vis.shape)
    print(gt_mask.shape)
    print(pr_mask.shape)

    visualize_img_mask(image_vis, gt_mask, pr_mask, filename='pred_multiclass_' + str(i) + '.png')