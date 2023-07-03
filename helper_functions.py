import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
import cv2
import os
import albumentations as albu
from matplotlib import patches as mpatches

# ################################################################################
# # Hilfsfunktion, die ein Tripel aus Bild, GT Maske und Segm. Maske darstellt
# ################################################################################
def visualize_img_mask(image, gt_mask, pr_mask, filename='test.png'):
    # Bild
    plt.figure(figsize=(16, 5))

    # Subplots mittig platzieren
    plt.subplots_adjust(left=0.02,
                        bottom=0.02,
                        right=0.90,
                        top=0.98,
                        wspace=0.05,
                        hspace=0.05)
    
    plt.subplot(1, 3, 1)
    plt.xticks([])
    plt.yticks([])
    plt.title('Bild')
    plt.imshow(image)

    # Labels und dazugehörige Farben als dictionary definieren
    labels_and_colors = {'bed' : 'lightblue',
                         'books' : 'brown',
                         'ceiling' : 'lightyellow',
                         'chair' : 'orange',
                         'floor' : 'magenta',
                         'furniture' : 'blue',
                         'objects' : 'green',
                         'picture' : 'red',
                         'sofa' : 'purple',
                         'table' : 'goldenrod',
                         'tv' : 'lightgreen',
                         'wall' : 'gray',
                         'window' : 'lightgray',
                         'unlabeled' : 'white',
                         'no information' : 'black'}

    # Eigene Colormap erstellen
    cmap = mcolors.ListedColormap(list(labels_and_colors.values()))

    # Ground Truth Maske
    plt.subplot(1, 3, 2)
    plt.xticks([])
    plt.yticks([])
    plt.title('Ground Truth')
    plt.imshow(gt_mask, cmap=cmap, vmin=0, vmax=14)

    # Vorhersage Maske
    plt.subplot(1, 3, 3)
    plt.xticks([])
    plt.yticks([])
    plt.title('Vorhersage')
    plt.imshow(pr_mask, cmap=cmap, vmin=0, vmax=14)

    # Legende
    legend_elements = [mpatches.Patch(facecolor=labels_and_colors[label],
                             edgecolor='black',
                             label=label) for label in labels_and_colors]
    plt.legend(handles=legend_elements,
               loc='center left',
               bbox_to_anchor=(1, 0.5))

    # Bild in Unterverzeichnis Abbildungen speichern
    directory = './Abbildungen/'
    if not os.path.exists(directory):
        os.makedirs(directory)
    plt.savefig(os.path.join(directory, filename))
    plt.close()

################################################################################
# Funktion, die eine Pipeline für die training augmentation bereitstellt
# (Mehrere zufällige Transformationen)
################################################################################
def get_training_augmentation():
    train_transform = [
        albu.HorizontalFlip(p=0.5),
        # Wert der Bereiche ohne Information mit border_mode konstant und mit mask_value auf einen bestimmten Wert setzen
        albu.ShiftScaleRotate(scale_limit=0.5, rotate_limit=0, shift_limit=0.1, p=1, border_mode=cv2.BORDER_CONSTANT, mask_value=14),
        albu.PadIfNeeded(min_height=320, min_width=320, always_apply=True, border_mode=cv2.BORDER_CONSTANT, mask_value=14),
        albu.RandomCrop(height=320, width=320, always_apply=True),
        albu.GaussNoise(p=0.2),
        albu.Perspective(p=0.5),
        albu.OneOf([albu.CLAHE(p=1), albu.RandomBrightnessContrast(p=1), albu.RandomGamma(p=1)], p=0.9),
        albu.OneOf([albu.Sharpen(p=1), albu.Blur(blur_limit=3, p=1), albu.MotionBlur(blur_limit=3, p=1)], p=0.9),
        albu.OneOf([albu.RandomBrightnessContrast(p=1), albu.HueSaturationValue(p=1)],p=0.9)
    ]
    return albu.Compose(train_transform)

################################################################################
# Funktion, die dasselbe für den Validierungdatensatz macht, aber hier sind die Trafos nicht 
# so wichtig. Es sollen nur alle Validierungsbilder auf dieselbe, durch 32 teilbare Größe gebracht werden
################################################################################
def get_validation_augmentation():
    test_transform = [
        albu.PadIfNeeded(384, 480)
    ]
    return albu.Compose(test_transform)

################################################################################
# Für einige Encoder müssen die Bilder vorverarbeitet werden, hierfür wird eine zum Encoder passende
# preprocesing function benötigt.
################################################################################
def get_preprocessing(preprocessing_fn):
    _transform = [
        albu.Lambda(image=preprocessing_fn),
        albu.Lambda(image=image_to_tensor, mask=mask_to_tensor),
    ]
    return albu.Compose(_transform)

################################################################################
# Bild in eine Form bringen, die vom Netz verarbeitet werden kann
# Transpose ändert die Reihenfolge der Kanäle (row, col, rgb) --> (rgb, row, col)
################################################################################
def image_to_tensor(x, **kwargs):
    return x.transpose(2, 0, 1).astype('float32')

################################################################################
# Maske in eine Form bringen, die vom Netz verarbeitet werden kann
# Erst die dritte Dimension dazuerfinden, dann Transpose
# Transpose ändert die Reihenfolge der Kanäle (row, col, rgb) --> (rgb, row, col)
################################################################################
def mask_to_tensor(x, **kwargs):
    x = np.expand_dims(x, axis=2)
    return x.transpose(2, 0, 1).astype('float32')