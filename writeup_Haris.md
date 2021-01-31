# Behavioral Cloning


**Behavioral Cloning Project**

The goals/steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track 1 without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/placeholder.png "Model Visualization"
[image2]: ./examples/placeholder.png "Grayscaling"
[image3]: ./examples/placeholder_small.png "Recovery Image"
[image4]: ./examples/placeholder_small.png "Recovery Image"
[image5]: ./examples/placeholder_small.png "Recovery Image"
[image6]: ./examples/placeholder_small.png "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"

---
### Required Files and Quality of Code

My project includes the following files:
* `model.py` containing the script to create and train the model
* `drive.py` for driving the car in autonomous mode
* `model.h5` containing a trained convolution neural network
* `run1.mp4` video of autonomous driving
* `writeup_Haris.md` summarizing the results

#### Submission includes functional code
Using the Udacity provided simulator and my `drive.py` file, the car can be driven autonomously around the track by executing 
```
python drive.py model.h5
```
The `model.py` file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of convolutional layers with 5x5 and 3x3 filter sizes and depths between 24 and 64 (`model.py lines 35-39`).

The model includes RELU activations (within each convolutional layer) to introduce nonlinearity, and the data is normalized using a Keras lambda layer (`line 33`).

#### 2. Attempts to reduce overfitting of the model

I have used the split of 75% training data and 25% validation data of the full data set. To avoid overfitting, more data was collected and augmented by carrying out the following driving methods.
* 3 laps of center lane driving
* 1 lap of recovery driving from the sides
* 1 lap of driving smoothly around the curves

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (`model.py line 46`).

#### 4. Appropriate training data

I used a combination of center lane driving, recovery driving from both sides and smooth driving around the curves to collect enough useful training data.

### Architecture and Training Documentation

#### 1. Solution Design

My first step was to use a single convolutional layer neural network to test my `model.py` script and simulation. I had only collected 1 lap of center lane driving data. Then I simply implemented the following.
* 1 convolution layer
* Flatten layer
* 1 fully-connected layer

I split the datat set into 75% training and 25% validation data sets. The simulation worked fine in the autonomous mode (as intended, although the car went off road right as the simulation started) but the results had high mean squared error (MSE) on both the training and validation sets. This implied underfitting. In order to avoid that, I had to incorporate more convolutional layers and train on more data.

Hence, I implemented a convolution neural network model similar to the NVIDIA model for self-driving cars. I thought this model might be more appropriate because it incorporates similar data and challenges as the task at hand.

I trained on the same data set as before and I found that my new model had a low MSE on the training set but a high MSE on the validation set. This implied that the model was overfitting. Running the simulation, there were improvements than the previous iteration but the car would go off road at the second curve.

To combat this challenge, I first decided to collect more data by driving more.
* 3 laps of center lane driving
* 1 lap of recovery driving from the sides
* 1 lap of driving smoothly around the curves

With this bigger dataset, my model gave out low MSE on both the training and validation sets.

Once more, there came the turn of running the simulator yet again to see if the car would drive the complete lap without going off the road. And it did! You can check out the video `run1.mp4`.

#### 2. Model Architecture

The final model architecture (model.py lines 18-24) consisted of a convolution neural network with the following layers and layer sizes ...

Here is a visualization of the architecture (note: visualizing the architecture is optional according to the project rubric)

![alt text][image1]

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to .... These images show what a recovery looks like starting from ... :

![alt text][image3]
![alt text][image4]
![alt text][image5]

Then I repeated this process on track two in order to get more data points.

To augment the data sat, I also flipped images and angles thinking that this would ... For example, here is an image that has then been flipped:

![alt text][image6]
![alt text][image7]

Etc ....

After the collection process, I had X number of data points. I then preprocessed this data by ...


I finally randomly shuffled the data set and put Y% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was Z as evidenced by ... I used an adam optimizer so that manually training the learning rate wasn't necessary.
