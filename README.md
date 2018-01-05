
***
# Traffic Sign Classifier  
---
## Overview 
---
***
We are in the era of self-driving cars recognizing and classifying traffic signs is of utmost importance. The model analyzes the properties of traffic sign images and recognizing the traffic signs out of them. Identifying the traffic signs correctly and taking actions is crucial to the operation of autonomous vehicles this process of classification of traffic sign would help safe driving and help preventing the accidents.  
This algorithm has to aspects:

*	Extracting the features from the traffic sign images like size color etc.
*	Image classification classifying images to the corresponding class based on the features


A car doesn’t have eyes but in self-driving cars we use cameras and other sensors to achieve a similar function. For Example taking the following image as input the algorithm has to identify whether or not it’s a no left turn sign:

<figure>
 <img src="examples/No Turn Sign.png" width="580" alt="Combined Image" />
 <figcaption>
 <p></p> 
 <p style="text-align: center;"> This is a sample image to detect road lines on. </p> 
</figcaption>

This project has several application areas: 
*	Self-driving Cars 
*	Driver assistance systems 
*	Urban scene understanding sign monitoring for maintenance 
---
***
## Approach 
---
***
With advancement towards deep learning model. The major breakthrough was in 2012 when a model called Alex net was trained on millions of images and classified real world images and won the world’s biggest image classifying contest. 

I have extended this popular architecture and made it more suitable for traffic sign image recognition and classification. Alexnet is a neural inspired model which is one of the popular architecture for convolutional neural networks. It’s quite easy to understand and easy to learn. I have used RELU activation and Adam Optimizer for optimization and loss function. Than used softmax to view the top 5 guess for a given traffic sign. I have used [German Traffic Sign Recognition Benchmark](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset), which is one of the most reliable datasets for training testing and validating traffic sign recognition and classification algorithm.

The dataset has more than 50000 images with 43 classes.  The refined dataset was provided by Udacity. Which contained 32x32 RGB colored images. For this classifier training set contained 34799, validation set has 4410 images whereas test set contained 12630 images. 

	X_train		(34799, 32, 32, 3)
			uint8
	Y_train		(34799,)
			uint8
	X_Valid		(4410, 32, 32, 3)
			uint8
	Y_Valid		(4410,)
			uint8
	X_test		(12630, 32, 32, 3)
			uint8
	Y_test		(12630,)
			uint8

I have designed this model for 48x48 RGB images. Therefore I resized all of the images to my desired size to work with AlexNet. The original [ALexnet model](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf) was designed for 224x224x3 images. But here in Traffic Sign Classifier case I modified the network for 48x48x3 images. The number of filters and other techniques applied are inspiration from [ZFNet](https://cs.nyu.edu/~fergus/papers/zeilerECCV2014.pdf) winner of 2013 imagenet classification contest. This architecture was a more of a fine tuning of Alexnet and developed very key ideas on improving the performance of the model. Here the figure shows the architecture of the modified model in terms of filters and dimension. 

<figure>
 <img src="examples/Modified ALexnet.jpg" width="580" alt="Combined Image" />
 <figcaption>
 <p></p> 
 <p style="examples/text-align: center;"> </p> 
 </figcaption> 

Below Configurations show how the model is implemented with its output.

	Layer 1 Convolution (23, 23, 32)
	Layer 1 MaxPooling (11, 11, 32)
	Layer 2 Convolution (11, 11, 128)
	Layer 2 MaxPooling (5, 5, 128)
	Layer 3 Convolution (5, 5, 256)
	Layer 4 Convolution (5, 5, 256)
	Layer 5 Convolution (5, 5, 128)
	Layer 5 MaxPooling (2, 2, 128)
	layer 6 flattened shape: (512)
	FullConnectionLayer 6 (512)
	Drop Out Shape: (512)
	FullConnectionLayer 7 (512)
	Drop Out Shape: (512)
	FullConnectionLayer 8 (43)

Note: These are calculations are considering 48x48x3 input image dimensions. But input image is resized to 51x51x3 because tensorflow operations reduced 3 layers so input was increased to overcome [this effect.](https://stackoverflow.com/questions/38167455/tensorflow-output-from-stride)

	x = tf.placeholder(tf.float32, (None, 51, 51, 3))
---
***
## Data Preprocessing and Augmentation 
---
***
As computer takes a number of pixels as an image. Data Augmentation techniques let us make our dataset even larger than its original size. I have used some techniques to generate fake data.
*	Image Rotation
*	Image Translation
*	Brightness
*	Constrast
*	Image Scaling 
*	Image Sharpness
*	Histogram Equalization
I have generated 5 different images per single image. 
<figure>
 <img src="examples/Augmented Image.png" width="580" alt="Combined Image" />
 <figcaption>
 <p></p> 
 <p style="examples/text-align: center;"> </p> 
 </figcaption>
Befre feeding the data to a neural network obviously it needs to be preprocessed. I have normalised the data so that the neurons do not shoot during the training phase. To help the Optimizer for training the model, Some of the techniques I applied are: 
*	Random Shuffling 
*	Image Centering 
*	Image translation 
*	Contrast and Brightness Enhancement 
*	Normalization
---
***
## Training the Model  
---
***
There are some hyper parameters that needs to be tuned to get some good results from the architecture.
*	Epochs 
*	Batch Size 
*	Mu
*	Sigma 
*	Keep_Prob
*	Training Rate

	EPOCH 91 ...
	Validation Accuracy = 0.934
	EPOCH 92 ...
	Validation Accuracy = 0.949
	EPOCH 93 ...
	Validation Accuracy = 0.925
	EPOCH 94 ...
	Validation Accuracy = 0.951
	EPOCH 95 ...
	Validation Accuracy = 0.953
	Model saved


---
***
## Testing the Model  
---
***
Once model is trained with training and validation set. I have used unseen test data to check the accuracy. 
Test set was consisted of 12630 images 
It predict the classes of each image in test set and compare against the actual class for the image. 
The average accuracy of the model is 94%.

More on training and testing the network in terms of code and visualizations is included in Jupyter Notebook.
	
	Test Set Accuracy = 0.940
	
