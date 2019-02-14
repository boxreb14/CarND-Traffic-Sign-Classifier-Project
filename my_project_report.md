# **Traffic Sign Recognition** 

## Writeup

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

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

[image1]: ../CarND-Traffic-Sign-Classifier-Project/test_folder/bar_chart.jpg "Exploratory Visualization"
[image2]: ../CarND-Traffic-Sign-Classifier-Project/test_folder/greyscale_image.jpg "Grayscaling"
[image3]: ../CarND-Traffic-Sign-Classifier-Project/test_folder/translated.jpg "Translated Image"
[image4]: ../CarND-Traffic-Sign-Classifier-Project/sermanet.jpg "Model Architecture"
[image5]: ../CarND-Traffic-Sign-Classifier-Project/test_folder/test_picture1.png "Test Image 1"
[image6]: ../CarND-Traffic-Sign-Classifier-Project/test_folder/test_picture2.png "Test Image 2"
[image7]: ../CarND-Traffic-Sign-Classifier-Project/test_folder/test_picture3.png "Test Image 3"
[image8]: ../CarND-Traffic-Sign-Classifier-Project/test_folder/test_picture4.png "Test Image 4"
[image9]: ../CarND-Traffic-Sign-Classifier-Project/test_folder/test_picture5.png "Test Image 5"
[image10]: ../CarND-Traffic-Sign-Classifier-Project/test_folder/test_picture6.png "Test Image 6"
[image11]: ../CarND-Traffic-Sign-Classifier-Project/test_folder/test_picture7.png "Test Image 7"
[image12]: ../CarND-Traffic-Sign-Classifier-Project/test_folder/test_picture8.png "Test Image 8"




## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the numpy library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 37184
* The size of the validation set is 9296
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how the data differs for each unique class. For some of the classes the data samples are very low so I have added the data samples for those classes by some data augmenting techniques like translation, scaling, warping etc. 

![alt text][image1]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I decided to convert the images to grayscale because it helps in reducing the training time and it was suggested to do so.

Here is an example of a traffic sign image before and after grayscaling.

![alt text][image2]

As a last step, I normalized the image data because the mean of the dataset was not close to zero and doing that made it close to zero. Also it helps making the distribution of features less wide and helps us to train using a particular learning rate.

I decided to generate additional data because the distribution of samples for each unique class varied a lot.

To add more data to the the data set, I used the following techniques because they were easy to perform and I had worked with them before.
Techniques used are translation, scaling, warping and adjusting brightness.

Here is an example of an original image and an augmented image:

![alt text][image3]

The difference between the original data set and the augmented data set is the following that the augmented data set has better and less fluctuating distribution of the samples to be trained for each unique class. Also the augmented data set is normalized.


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 3x3     	| 1x1 stride, same padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6 					|
| Convolution 3x3	    | 1x1 stride, same padding, outputs 10x10x16	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x16 					|
| Convolution 3x3	    | 1x1 stride, same padding, outputs 1x1x400 	|
| RELU					|												|
| Flatten of 2nd MaxPool| Output 400									|
| Flatten of 3rd MaxPool| Output 400									|
| Concat both Flatten   | Output 800									|
| Dropout				|												|
| Fully connected		| Output 43 									|
|						|												|
|						|												|
 

#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used the Adam optimizer because it is suggested as the default optimization algorithm in stanford course CS231n and also it was well explained in the paper [An overview of gradient descent optimization algorithms](https://arxiv.org/pdf/1609.04747.pdf).

Here are the rest of the parameters used:
-> Batch Size = 64
-> Epochs = 15
-> learning rate = 0.0009
-> mu = 0
-> sigma = 0.1
-> dropout = 0.5

I tried tuning the batch size and epoch from 32 and 10 and increased it gradually till 30 and 1024. The best performance was given around 15 epochs and a batch size of 64.  I started with the learning rate of 0.0001 and increased it gradually to check if I am overshooting. I found the best results at 0.0009. Mu and sigma were chosen 0 amd 0.1 because I want the mean of the weights to be zero and standard deviation to be 0.1. Dropout was used and its keep probability was used 0.5 which was suggested in the dropout paper. 



#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* validation set accuracy of 99.2% 
* test set accuracy of 94%

I started with the model suggested in the course for this problem i.e. LeNet. I took the model architecture having 2 convolution layers with 2 dense layers. The results were good in this model and the validation accuracy was about 71%. To improve these results I was reading and then found the published baseline model for this problem which was suggested in the notebook also i.e. Sermanet. I chose this model because it was already published baseline model which works well for this problem. To implement this I took three convolution layers with activation relu and max pool and then added the cancatenation of max pooled part of flattened 2nd and 3rd layers to it. And lastly added the dropout and dense layer to get the logits. This time around the  results were far better and the model attained 87% validation accuracy. Now I just had to tune the hyperparameters. I kept the Mu & sigma untouched and started with increasing batch size from 32 to 1024. The results were better at batch size of 64. Also simultaneously I tried increasing the epoch from 10 to 30. But the validation accuracy was increasing till around 15th epoch and was more or less stagnant after that. For tuning learning rate I started from 0.0001 and increased it till 0.001, the performance was mostly same in each apart from 0.001 where there was around 1.5% drop in the validation accuracy. So I went with 0.0009 as the learning rate.

Here is the image of the architecture followed and the [link to the published paper](http://yann.lecun.com/exdb/publis/pdf/sermanet-ijcnn-11.pdf). 

![alt text][image4]


### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are the German traffic signs that I found on the web:

![alt text][image5] ![alt text][image6] ![alt text][image7] 
![alt text][image8] ![alt text][image9] ![alt text][image10]
![alt text][image11] ![alt text][image12]  

All the images I added are quite clear and does not seem difficult to classify, only the last image is little bit blurry so the probablity will be distributed among multiple classes.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Right-of-way      	| Right-of-way   								| 
| Speed limit (60km/h)	| Speed limit (60km/h)      					|
| Keep right			| Keep right									|
| Priority road 		| Priority road 				 				|
| Turn left ahead    	| Turn left ahead	 							|
| General caution    	| General caution    							|
| Speed limit (30km/h)  | Speed limit (30km/h)							|
| Road Work			 	| Road Work										|

The model was able to correctly guess 6 of the 8 traffic signs, which gives an accuracy of 87.5%. This compares favorably to the accuracy on the test set of 93.9%. The last two images were also correctly guesses taking into account the top softmax probability but the probablity was distributed for these 2 images.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 73rd cell of the Ipython notebook.

For the first image, the model shows 100% probability that this is a Right-of-way at the next intersection sign and the image does contain a Right-of-way at the next intersection sign. The top five soft max probabilities were (100, 0, 0, 0, 0).

For the Second image, the model shows 99.9% probability that this is a Speed limit (60km/h) sign and the image does contain a Speed limit (60km/h) sign. The top five soft max probabilities were (99.938, 0.042, 0.019, 0.001, 0).

For the third image, the model shows 100% probability that this is a Keep right sign and the image does contain a Keep right sign. The top five soft max probabilities were (100, 0, 0, 0, 0).

For the fourth image, the model shows 100% probability that this is a Priority road sign and the image does contain a Priority road sign. The top five soft max probabilities were (100, 0, 0, 0, 0).

For the fifth image, the model shows 100% probability that this is a Turn left ahead sign and the image does contain a Turn left ahead sign. The top five soft max probabilities were (100, 0, 0, 0, 0).

For the sixth image, the model shows 100% probability that this is a General caution sign and the image does contain a General caution sign. The top five soft max probabilities were (100, 0, 0, 0, 0).
 
For the seventh image, the model shows 100% probability that this is a Speed limit (30km/h) sign and the image does contain a Speed limit (30km/h) sign. The top five soft max probabilities were (100, 0, 0, 0, 0).

For the eigth image, the model shows 93.9% probability that this is a Road Work sign and the image does contain a Road Work sign. The top five soft max probabilities were (93.983, 3.319, 2.690, 0.006, 0.002).

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?