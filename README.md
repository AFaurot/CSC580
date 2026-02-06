# GAN on CIFAR-10 

This readme is for the portfolio project. Critical thinking projects are not currently covered by this document

The portfolio project implements a Generative Adversarial Network (GAN) using TensorFlow 2 / Keras to generate images from the CIFAR-10 dataset.  
The generator and discriminator architectures follow the assignment template provided by Colorado State University Global Campus, 
while the training process uses a modern TensorFlow `GradientTape` workflow for improved stability and performance.

### Features
- DCGAN-style generator and discriminator matching the provided architecture  
- Training with `tf.GradientTape()` instead of the legacy combined-model approach  
- `tf.data` pipeline for efficient batching and shuffling  
- Periodic image grid saving for visual evaluation  
- Loss curve plotting  
- Tested on TensorFlow 2.19 and Python 3.12.3 with NVIDIA RTX 4070 CUDA support. (Windows GPU support requires running Python intrpreter in WSL. Highly recommended as CPU training will take a very long time)

---

## Installation

Create a virtual environment (recommended):

```bash
python -m venv venv
source venv/bin/activate      # Linux / WSL
# or
venv\Scripts\activate         # Windows PowerShell

Install required libraries:

pip install "tensorflow[and-cuda]==2.19.0"
pip install numpy
pip install matplotlib
```


**Note on TensorFlow GPU Support**

For GPU acceleration, follow the official TensorFlow installation guide:

https://www.tensorflow.org/install

On Windows systems, GPU support is recommended through WSL2.

https://www.tensorflow.org/install/pip#windows-wsl2

Also refer to NVIDIA documentation here:

https://docs.nvidia.com/cuda/wsl-user-guide/index.html


**NVIDIA CUDA Toolkit**

For GPU Support the CUDA toolkit needs installed

https://docs.nvidia.com/cuda/index.html

## Running the project

Simply run the following but note that for images to show you need to be running in a GUI supported env. Otherwise the script will still run and images will be saved to the output folder:

```python portfolio_project.py```

The script will:

 1. Load CIFAR-10 and select class 8 (ships)
 2. Display real sample images
 3. Train for 15,000 steps
 4. Save generated image grids at regular intervals
 5. Save a loss curve plot

Outputs are written to:

```gan_outputs/
 ├── step_0_grid.png
 ├── step_2499_grid.png
 ├── ...
 ├── step_14999_grid.png
 └── loss_curves.png
```

## Project Structure

build_generator() – DCGAN-style upsampling network

build_discriminator() – convolutional classifier

train_step() – adversarial training with GradientTape

tf.data pipeline – efficient input handling

save_image_grid() – visualization utility

## Notes

Images are normalized to [-1, 1]

Label smoothing/noise follows the assignment template

The GradientTape approach avoids issues with trainable flag toggling present in older examples

CIFAR-10 images are low detail (32 x 32 ) even in the dataset so generated images will match this pattern. See below example:

<img width="600" height="600" alt="Sample_cifar_images" src="https://github.com/user-attachments/assets/a21e036b-586d-4481-aa3a-f54c7aae8562" />

## References

Tensorflow Installation
https://www.tensorflow.org/install

TensorFlow DCGAN Tutorial
https://www.tensorflow.org/tutorials/generative/dcgan

TensorFlow Dataset API
https://www.tensorflow.org/api_docs/python/tf/data/Dataset

WSL Installation Guide
https://docs.microsoft.com/windows/wsl/install

NVIDIA Cuda Documentation
https://docs.nvidia.com/cuda/index.html
