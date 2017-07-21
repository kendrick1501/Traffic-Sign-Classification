# Traffic-Sign-Classification

# UDACITY Self-Driving Car Nanodegree Program

## Project # 2: Deep Learning - Build a Traffic Sign Recognition Classifier

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./figures/viz_dataset.png "Dataset Visualization"
[image2]: ./figures/pre_processing_input.png "Training Data set Images"
[image3]: ./figures/pre_processing_output.png "Pre-processing Output"
[image4]: ./figures/viz_aug_dataset.png "Augmented Dataset Visualization"
[image5]: ./figures/augmented_training_set.png "Augmented Training Set"
[image6]: ./figures/new_dataset_examples.png "New Dataset example images"
[image7]: ./figures/pres_recall_test.png "Precision and Recall Test Set"
[image8]: ./figures/pres_recall_newdata.png "Precision and Recall New Dataset"
[image9]: ./figures/softmax_img1.png "Traffic Sign 1"
[image10]: ./figures/softmax_img2.png "Traffic Sign 2"
[image11]: ./figures/softmax_img3.png "Traffic Sign 3"
[image12]: ./figures/softmax_img4.png "Traffic Sign 4"
[image13]: ./figures/softmax_img5.png "Traffic Sign 5"
[image14]: ./figures/viz_conv1_img2.png "First Convolutional layer Traffic Sign 2"
[image15]: ./figures/viz_conv1_img3.png "First Convolutional layer Traffic Sign 3"
[image16]: ./figures/viz_conv1_img5.png "First Convolutional layer Traffic Sign 5"

---
###Data Set Summary & Exploration

####1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

To obtain the basic information of the data set, I followed the approach we've been using in previous lessons, i.e., numpy method *shape*. 

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is 32x32x3
* The number of unique classes/labels in the data set is 43

####2. Include an exploratory visualization of the dataset.

The following figure depicts how the different classes are distributed in the training, validation, and test sets. Clearly, the sets have approximately the same data distribution with labels two, one, and thirteen having the highest frequency.

![alt text][image1]

In view of the data distribution, I believe the convolutional neural network lacks information to guarantee the same level of prediction for all classes.

###Design and Test a Model Architecture

####1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

The images provided in the data set are color images in RGB format. For the current project, such color information is not required, and so the images were converted to grayscale. This pre-processing step lessens the computational burden by a factor of 3.

The grayscale images are then normalized to prevent losing information due to a big difference in pixels values. The normalized data is guaranteed to have zero mean and equal variance.

The outcome of the pre-processing stage is illustrated in the following figure.

![alt text][image2]
![alt text][image3]

Given the uneven distribution of traffic signs examples in the data set, the training set was augmented in an attempt to provide more information on those classes with less representation.

The additional data were generated randomly in such a way that labels with less representation would have more probability of being augmented. This new data is composed of images of the training set randomly rotated with added random noise. Below, the new distribution of the training set is depicted along with an example of the additional images.

![alt text][image4]
![alt text][image5]

The new size of the training set is 69598.

####2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

The convolutional neural network architecture is described in the following table:

| Layer         		|     Description	        					 | 
|:-----------------------:|:--------------------------------------------------------------:| 
| Input         		| 32x32x1 Grayscale image   				| 
| Convolution 5x5  	| 1x1 stride, same padding, outputs 32x32x16	|
| RELU			|										|
| Max pooling	      	| 2x2 stride, outputs 16x16x16 				|
| Convolution 3x3	| 1x1 stride, valid padding, outputs 14x14x80      |
| RELU             	|        								        |
| Max pooling	      	| 1x1 stride, outputs 14x14x80      			|
 Convolution 3x3  	| 1x1 stride, valid padding, outputs 12x12x30 	|
| RELU			|										|
| Max pooling	      	| 2x2 stride, outputs 6x6x30				        | 
|Fully connected	| outputs 750 							|
| Dropout                 |                                                                               |
|Fully connected	| outputs 350  							|
| Dropout                 |                                                                               |
|Fully connected	| outputs 43 							|

####3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used the augmented training set introduced before. As for the optimizer, I followed the suggestion from the LeNet-Lab and used ADAM. Due to computing limitations, I set the batch size to 128 with five epochs. The learning rate is set to 0.001. The initialization of weights was performed with a normal distribution function with zero mean and 0.05 of standard deviation. During the training process, the keep probability of the dropout layers was chosen as 0.35.

####4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:

* Training set accuracy of 99.21%
* Validation set accuracy of 97.57%
* Test set accuracy of 94.52%

The model architecture is based on LeNet convolutional neural network provided in LeNet-Lab. The LeNet neural network is a relatively simple yet powerful model architecture for image classification. Therefore, it was chosen as a baseline for my model (MKNet).

In MKNet, a third convolution layer is added to enable the convolutional neural network to recognize the more complex features presented in traffic signs as compared to the handwritten numbers analyzed in the LeNet-lab.

The size of the layers in the architecture was chosen considering the performance during the training and validation processes. In the iterative procedure, it was noticed that the depth of the second convolutional layer needs to be as high as possible to achieve a decent performance. This requirement may arise due to a higher level of abstraction the second convolutional layer has to perform so as to characterize different traffic signs.

The depth of the third layer barely improves the performance of the network for values greater than 25. As for the first layer, a fixed value of 16 filters was chosen with good results. 

The activation functions of the fully connected layers are chosen as a dropout to avoid eventual overfitting of the data set. This choice turned out to be relevant to guarantee good prediction performance during the test process, given that the keep probability parameter played a major role in improving the training and validation accuracy. Low values of keep probability (<0.5) increase the prediction accuracy of the convolutional neural network throughout all data set, despite the strong distortion in the training set due to the augmented images.

###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

A new data set composed of 30 traffic signs images downloaded from the web is utilized to further evaluate the performance of MKNet. Five examples of the new data set are shown next.

![alt text][image6] 

In this example of images, it can be clearly observed that three traffic signs are occluded, being the first and fourth images the most difficult examples to be predicted given the confusing and/or incomplete set of features that can be detected by the convolutional neural network. 

For instance, the first image might be confused with Keep Left traffic sign (As it happened during the iterative training of the MKNet) due to the arrow like shape at the bottom-left of the number two in the image. In the case of the *Children Crossing* traffic sign, about 50% of the sign is occluded diminishing the information available to perform the classification. As for the second sign, the features surrounding the traffic sign might also mislead the convolutional neural network.

In the whole set, the images present an important level of noise which may further complicate the classification process.

####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Despite the confusing or incomplete features presented in the images of the new dataset, the MKNet performed satisfactorily with an overall prediction accuracy of 86.67%. Such a result compares favorably with the test accuracy and evidences the capability of the model architecture for traffic sign classification. 

Considering the precision and recall of MKNet on the test set and the new dataset illustrated below, it can be noticed that labels 25, 27, and 33 exhibit similar precision and recall relation, which seems to validate the performance of the MKNet on different datasets. 

Also, the low precision on class 27 (*Pedestrians* traffic sign) on both cases may evidence the necessity of augmenting the examples of this traffic sign in the training set.

![alt text][image7] 
![alt text][image8] 

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

To evaluate the model confidence in classifying traffic signs, the above five images of the new dataset were used.

For the first image, the model is only 57.4% sure about the traffic sign. This result is expected given the noise and confusing features in the classified image.

![alt text][image9] 

In the second case, the MKNet is quite confident about the prediction despite the confusing and incomplete features presented in the image 

![alt text][image10] 

The third image is classified almost with total confidence, even though the model relates the image to traffic signs with similar characteristic features.

![alt text][image11]

 For the fourth traffic sign, the MKNet convolutional neural network surpassed the expected performance. The *Children Crossing* traffic sign is predicted with 94.47% of certainty despite the lack of information in the corresponding image.
 
![alt text][image12]

In the last example, the model predicts the traffic sign with very high confidence. However, it relates the image to traffic signs with similar features.
 
![alt text][image13] 

####1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?

The following figures illustrate the feature maps of the first convolutional layer for the second, third, and fifth images of the new dataset example.

![alt text][image14] 
![alt text][image15] 
![alt text][image16] 

The convolutional neural network seems to utilized the edges and the content of the traffic sign to classified the image. For instance, notice how the feature map 1 of image 3, corresponding to the *Road Work* traffic sign, seems to has a representation of the "worker on the road." 

In the feature maps of the fifth image, MKNet clearly detects the edges of the traffic sign and seems to represent the vehicle in it by a black area within the triangle.

In the case of image 2, the prediction is based on the circular edges and some representation of the numbers in the traffic sign.
