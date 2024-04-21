# Project

This repository contains the code for the Robust Data Augmentations for Segformer paper for the TU/e course on Neural Networks for Computer vision.
The code makes use of different data augmentations and upscaling methods to increase robustness in a pretrained segformer model. 
This model can be tested on datasets containing different weather circumstances for different evaluation scores. 

The evaluation data can be obtained by following the code and instructions in:
https://github.com/vnmusat/multi-weather-city?tab=readme-ov-file

## Getting Started

### Dependencies

- argparse
- torch==2.1.0 
- torchvision==0.16.0
- torchmetrics
- transformers
- numpy
- pathlib
- skimage
- PIL
- datetime
- wandb
- json

  
### File Descriptions

Here's a brief overview of the files you'll find in this repository:

- **run_container.sh:** Contains the script for running the container. In this file you have the option to enter your wandb keys if you have them and additional arguments if you have implemented them in the train.py file.
  
- **run_main:** Includes the code for building the Docker container. In this file, you only need to change the settings SBATCH (the time your job will run on the server) and ones you need to put your username at the specified location.
  
- **model.py:** Defines the neural network architecture.
  
- **train.py:** Contains the code for training the neural network.

- **arguments.py:** Contains all the arguments necessary for the code

- **dataloader.py:** Contains code building a dataloader with different data augmentations.

- **eval.py:**: Using 8 different weather datasets and a model, test the model for DICE and mean IoU.

- **process_data.py:** Process the data in a way that it can be submitted to the codalab challange.

- **utils.py:** Contains any functions that support other code files.

- **visualization.py:**: Visualizes images depending on different data type inputs. 

- 
### Authors

- Mathieu van Luijken 
- m.f.a.c.v.luijken@student.tue.nl
  
