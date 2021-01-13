# Neural Image Style Transfer

A pre-trained Convolutional Neural Network (CNN) for transferring style from patterns and apparels images onto fashion apparels. 
### Motivation
In the past, style transfers have been widely used to impose an artistic style on arbitrary images. Here, we try to use style transfer with the help of Convolutional Neural Networks (CNN) to impose the artistic style of fashion wears, giving us a neural network algorithm to create artistic fashion images. 

### Frameworks
- Tensorflow - Open source library used for deeplearning 
- PIL - Image processing library
- OpenCV - Image Processing Library
- Matplotlib - Library used for visualizations

### Installation
Using python's package manager pip to install the dependencies
``` sh
pip install tensorflow==1.14
pip install PIL
pip install opencv-python
pip install matplotlib
pip install yacs
```

### How to run 
Execute the following command on the terminal
``` sh
!python3 NeuralStyleTransfer.py --content image_path --style style_path --output output_path
```
```--content: ```Path to the original image

```--style: ``` Path to the image which contains the style to apply

```--output: ```Path to save your stylized image

```--help: ``` Look at other commands

The results images are compressed as the **img.zip** file

**NOTE :** The python script can also be executed on Google Colab using **StyleTransfer.ipynb** 
