## Vehicle Detection Project

The goals / steps of this project are the following:

* Using labeled training data of vehicles and non-vehicles, perform feature extraction after preprocessing with color space transformations, spatial binning, image channel histograms, and computed Histogram of Oriented Gradients (HOG) for each training image
* Training a Linear SVM classifier on the preprocessed and normalized training data mentioned above
* Randomly split training data into a training and testing set (no validation set was used here).
* Implement a HOG subsampling window search technique to use the trained classifier to search for vehicles in input images from a vehicle's hood-mounted camera feed.
* Run pipeline on a video stream and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./output_images/vehicle.png
[image2]: ./examples/non_vehicle.png
[image3]: ./examples/sliding_windows.jpg
[image4]: ./examples/sliding_window.jpg
[image5]: ./examples/bboxes_and_heat.png
[image6]: ./examples/labels_map.png
[image7]: ./examples/output_bboxes.png
[video1]: ./project_video.mp4

###Feature Extracting Using Color Space Modifications, Spatial Binning, Color Channel Histograms, and Histogram of Oriented Gradients (HOG)

####1. Loading the Dataset 

I started by reading in all the "vehicle" and "non vehicle" images.  Below is an example of each of these classes respectively. 

![vehicle][image1]
![non-vehicle][image2]

Refer to the code in `./model.py` in the `pipeline_v1` function, which calls the function `extract_features` in the `./features.py` with the filenames of all vehicle and non-vehicle images to be extracted.

####2. Transformation from RGB to YCrCb Color Space

Input images were transformed from RGB to YCrCb color space. 

This was selected after trying the below pipeline with many different color spaces, including HSV, YUV, HLS, and RGB. The YCrCb color space resulted in the highest Test Set Accuracy score of 99.63%, whereas the second best score from 'HSV' came as high to 98.7%.

####2. Spatial Binning

Spatial Binning was performed to resize each 64x64 image to a smaller size. This has the effect of reducing the available pixels to train from, which helps prevent overfitting to highly detailed training examples. the spatial size selected was (24, 24) after trial and error to maximize the Test Set Accuracy.

Note that the computed spatial features are concatenated with others and fed into the SVM classifier.

Refer to the code in `./model.py` in the `pipeline_v1` function, which calls the function `bin_spatial` in the `./features.py`.

####3. Color Histograms

Color histograms were computed for each input image and passed in as features

![alt text][image2]

####2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of parameters and...

####3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a linear SVM using...

###Sliding Window Search

####1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I decided to search random window positions at random scales all over the image and came up with this (ok just kidding I didn't actually ;):

![alt text][image3]

####2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on two scales using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  Here are some example images:

![alt text][image4]
---

### Video Implementation

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./project_video.mp4)


####2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

### Here are six frames and their corresponding heatmaps:

![alt text][image5]

### Here is the output of `scipy.ndimage.measurements.label()` on the integrated heatmap from all six frames:
![alt text][image6]

### Here the resulting bounding boxes are drawn onto the last frame in the series:
![alt text][image7]



---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  

