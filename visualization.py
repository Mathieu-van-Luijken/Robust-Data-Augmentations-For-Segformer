import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import wandb
import numpy as np
import PIL.Image
import  utils
import os
import torch 

from pathlib import Path
from datetime import datetime
import matplotlib.image as mpimg


# from dataloader import CityscapesDataLoader
import arguments

def visualize_label_tensor(tensor):
    im = PIL.Image.new(mode="RGB", size=(2048,1024))
    im = np.array(im)
    
    tensor=tensor#.cpu()

    for label in utils.LABELS:
        im[tensor == label.trainId] = label.color

    fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(5, 10))
    axs.imshow(im)

    cwd = os.getcwd()
    current_time = datetime.now().strftime("%H%M%S")
    path = os.path.join(cwd, f"results/visualizations/{current_time}.png")
    fig.savefig(fname=path)


def visualize_eval_tensor(tensor):
    im = PIL.Image.new(mode="RGB", size=(512,512))
    im = np.array(im)
    # im = np.transpose(im, (2,0,1))

    tensor=tensor.cpu()

    for label in utils.LABELS:
        im[tensor == label.trainId] = label.color

    fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(5, 10))
    axs.imshow(im)

    cwd = os.getcwd()
    current_time = datetime.now().strftime("%H%M%S")
    path = os.path.join(cwd, f"results/visualizations/{current_time}.png")
    fig.savefig(fname=path)

def visualize_image(tensor: torch.Tensor):
    im = tensor.permute(1,2,0)
    im = im.cpu()
    
    fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(5, 10))
    axs.imshow(im)

    cwd = os.getcwd()
    current_time = datetime.now().strftime("%H%M%S")
    path = os.path.join(cwd, f"results/visualizations/{current_time}.png")
    fig.savefig(fname=path)

def visualize_pil(image: PIL.Image):
    fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(5, 10))
    axs.imshow(image)

    cwd = os.getcwd()
    current_time = datetime.now().strftime("%H%M%S")
    path = os.path.join(cwd, f"results/visualizations/{current_time}.png")
    fig.savefig(fname=path)

def visualize_dataset():
    directory_path = Path(".\eval_data")

    # Get a list of folders in the directory
    folders = [folder for folder in os.listdir(directory_path) if os.path.isdir(os.path.join(directory_path, folder))]

    # Create a figure with 2 rows and 4 columns
    fig, axes = plt.subplots(2, 4, figsize=(12, 6))

    # Flatten the axes array for easier iteration
    axes = axes.flatten()

    # Loop through folders and display images in each folder
    for i, folder in enumerate(folders):
        # Get a list of image files in the folder
        image_files = [file for file in os.listdir(os.path.join(directory_path, f"{folder}/leftImg8bit/train/aachen")) if file.endswith('.jpg') or file.endswith('.png')]
        # Display up to 4 images from each folder
        if image_files:
            img = mpimg.imread(os.path.join(directory_path, f"{folder}/leftImg8bit/train/aachen/{image_files[0]}"))
            axes[i].imshow(img)
            axes[i].axis('off')
            axes[i].set_title(folder)

    plt.tight_layout()
    plt.show()
    fig.savefig(Path(f"./results/visualizations/all_datasets"))

visualize_dataset()