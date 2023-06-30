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
    plt.imshow(gt_mask, cmap=cmap, vmin=0, vmax=13)

    # Vorhersage Maske
    plt.subplot(1, 3, 3)
    plt.xticks([])
    plt.yticks([])
    plt.title('Vorhersage')
    plt.imshow(pr_mask, cmap=cmap, vmin=0, vmax=13)

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
