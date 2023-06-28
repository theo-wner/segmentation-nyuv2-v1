import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
import cv2
import os
import albumentations as albu

################################################################################
# Hilfsfunktion, die ein Tripel aus Bild, GT Maske und Segm. Maske darstellt
################################################################################
def visualize_img_mask(image, gt_mask, pr_mask, filename='test'):
    # Bild
    plt.figure(figsize=(16, 5))
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
                         'floor' : 'black',
                         'furniture' : 'blue',
                         'objects' : 'green',
                         'picture' : 'red',
                         'sofa' : 'purple',
                         'table' : 'goldenrod',
                         'tv' : 'lightgreen',
                         'wall' : 'gray',
                         'window' : 'lightgray',
                         'unlabeled' : 'white'}
    
    # Eigene Colormap erstellen
    cmap = mcolors.ListedColormap(list(labels_and_colors.values()))

    # Ground Truth Maske
    plt.subplot(1, 3, 2)
    plt.xticks([])
    plt.yticks([])
    plt.title('Ground Truth')
    plt.imshow(gt_mask, cmap=cmap)

    # Prediction Maske
    plt.subplot(1, 3, 3)
    plt.xticks([])
    plt.yticks([])
    plt.title('Segmentierungsergebnis')
    plt.imshow(pr_mask, cmap=cmap)

    # Legende erstellen
    legend_patches = [mpatches.Patch(color=color, label=label) for label, color in labels_and_colors.items()]
    plt.legend(handles=legend_patches, title='Klassen', loc='upper left', bbox_to_anchor=(1.02, 1))

    # Plots richtig platzieren
    plt.subplots_adjust(left=0.01,
                    bottom=0.1, 
                    right=0.9, 
                    top=0.9, 
                    wspace=0.2, 
                    hspace=0.35)

    # Bild in Unterverzeichnis Abbildungen speichern
    directory = './Abbildungen/'
    if not os.path.exists(directory):
        os.makedirs(directory)
    plt.savefig(os.path.join(directory, filename))