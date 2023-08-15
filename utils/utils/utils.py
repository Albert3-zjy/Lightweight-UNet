import matplotlib.pyplot as plt
import os
import torch
import numpy as np


def plot_img_and_mask(img, mask):
    classes = mask.max() + 1
    fig, ax = plt.subplots(1, classes + 1)
    ax[0].set_title('Input image')
    ax[0].imshow(img)
    for i in range(classes):
        ax[i + 1].set_title(f'Mask (class {i + 1})')
        ax[i + 1].imshow(mask == i)
    plt.xticks([]), plt.yticks([])
    plt.show()

def draw_plt(x_label, y_label, num_epoch, data, fold, plt_path):
    save_path = f"{plt_path}/fold_{fold}"
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    # epochs = torch.arange(1, num_epoch+1)
    epochs = [i for i in range(1, num_epoch + 1)]

    fig = plt.figure()
    plt.plot(torch.tensor(epochs).cpu().numpy(), torch.tensor(data).cpu().numpy())
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend()
    plt.savefig(f"{save_path}/{y_label}.png")

    return