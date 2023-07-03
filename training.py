import os
import segmentation_models_pytorch as smp
import ssl
from dataset import Dataset
from helper_functions import *
from torch.utils.data import DataLoader
# Utils muss nochmal separat importiert werden, weil es sonst nicht erkannt wird
import segmentation_models_pytorch.utils as smp_utils
import torch

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
# Modell erstellen
################################################################################
ENCODER = 'se_resnext50_32x4d'
ENCODER_WEIGHTS = 'imagenet'
CLASSES = ['bed', 'books', 'ceiling', 'chair', 'floor', 'furniture', 'objects',
            'picture', 'sofa', 'table' ,'tv', 'wall', 'window', 'unlabeled', 'no information']
ACTIVATION = 'softmax2d'
DEVICE = 'cuda:' + gpu_id

# Diese Zeile musste hinzugefügt werden, weil sonst ein ssl-Fehler auftritt
ssl._create_default_https_context = ssl._create_unverified_context

model = smp.FPN(
    encoder_name=ENCODER, 
    encoder_weights=ENCODER_WEIGHTS, 
    classes=len(CLASSES), 
    activation=ACTIVATION,
)

preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)

################################################################################
# Datensatz und DataLoader erstellen
################################################################################
train_dataset = Dataset(
    x_train_dir, 
    y_train_dir, 
    augmentation=get_training_augmentation(), 
    preprocessing=get_preprocessing(preprocessing_fn),
    classes=CLASSES,
)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=12)

################################################################################
# Trainieren
################################################################################
# Loss-Funktion und Metrik festlegen
#loss = smp_utils.losses.CrossEntropyLoss(ignore_index=[13, 14])
metrics = [smp_utils.metrics.IoU(ignore_channels=[13, 14])]

# Optimizer Festlegen
optimizer = torch.optim.Adam([dict(params=model.parameters(), lr=0.0001)])

# TrainEpoch-Objekt erstellen, vereinfacht das Trainieren
train_epoch = smp_utils.train.TrainEpoch(model, loss=loss, metrics=[], optimizer=optimizer,device=DEVICE,verbose=True)

# Trainieren
max_score = 0
for i in range(0, 40):
    print('\nEpoch: {}'.format(i))
    train_logs = train_epoch.run(train_loader)
    
    # Immer das Modell mit dem höchsten iou speichern
    # if max_score < train_logs['iou_score']:
    #     max_score = train_logs['iou_score']
    #     torch.save(model, './best_model_multiclass.pth')
    #     print('Model saved!')
    torch.save(model, './best_model_multiclass.pth')
    print('Model saved!')

    # Nach gewisser Iterationszahl Learning Rate verkleinern    
    if i == 25:
        optimizer.param_groups[0]['lr'] = 1e-5
        print('Decrease decoder learning rate to 1e-5!')