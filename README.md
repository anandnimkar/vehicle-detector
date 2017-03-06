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
[image3]: ./examples/hog_visualization_vehicle.png
[image4]: ./examples/hog_visualization_non_vehicle.png
[image5]: ./examples/detection1.png
[image6]: ./examples/detection2.png
[image7]: ./examples/bboxes_and_heatmaps_1.png
[image8]: ./examples/bboxes_and_heatmaps_2.png
[video1]: ./output_project_video.mp4

###Feature Extracting Using Color Space Modifications, Spatial Binning, Color Channel Histograms, and Histogram of Oriented Gradients (HOG)

####1. Loading the Dataset 

I started by reading in all the "vehicle" and "non vehicle" images.  Below is an example of each of these classes respectively. 

![vehicle][image1]
![non-vehicle][image2]

Refer to the code in `./model.py` in the `pipeline_v1` function, which calls the function `extract_features` in the `./features.py` with the filenames of all vehicle and non-vehicle images to be extracted.

####2. Color space conversion from RGB to YCrCb

Input images were transformed from RGB to YCrCb color space. 

This was selected after trying the below pipeline with many different color spaces, including HSV, YUV, HLS, and RGB. The YCrCb color space resulted in the highest Test Set Accuracy score of 99.63%, whereas the second best score from 'HSV' came as high to 98.7%.

Refer to the `convert_cspace` function in `./features.py`, which is called by `extract_features` in `./model.py` for training and the `find_cars` function in `./pipeline.ipynb` for prediction.

####3. Spatial Binning

Spatial Binning was performed to resize each 64x64 image to a smaller size. This had the effect of removing detail from the available pixels by downscaling each image, which would help prevent overfitting to more detailed training examples. the spatial size was selected after trial and error to maximize the Test Set Accuracy.

Note that the computed spatial features are concatenated with others and fed into the SVM classifier.

Refer to the function `bin_spatial` in `./features.py`, which is called by `extract_features` in `./model.py` for training and the `find_cars` function in `./pipeline.ipynb` for prediction.

####4. Color Channel Histograms

Histograms were computed for each channel of the YCrCb input images and concatenated as features of the training and testing set.

Refer to the function `color_hist` in `./features.py`, which is called by `extract_features` in `./model.py` for training and the `find_cars` function in `./pipeline.ipynb` for prediction.

####5. Histogram of Oriented Gradients (HOG) and its parameters

HOG was computed for each channel of input YCrCb images. Parameters were selected using trial and error to maximize the Test Set Accuracy. The parameters selected were as follows:

```python
params = {
    'cspace': 'YCrCb',
    'cell_per_block': 2,
    'hog_channel': 'ALL',
    'orient': 9,
    'pix_per_cell': 8,
}
```

A visualization of the HOG features for each channel is presented below for both a vehicle and non-vehicle input image:

![vehicle][image3]
![non-vehicle][image4]

Refer to the function `get_hog_features` in `./features.py`, which is called by `extract_features` in `./model.py` for training and the `find_cars` function in `./pipeline.ipynb` for prediction.


####6. Training a classifier using your selected features

The following approach was taken to train the classifier:
*  parameter for all feature extraction methods described were defined
*  features were extracted for both vehicles and non-vehicles. Both had a flattened feature array consisting of the Spatial, Color, and HOG features.
*  The `StandardScaler` was fit to the data to ensure that all features were normalized and certain features were not orders of magnitude larger than others.
*  A labels vector was defined in line with the number of vehicle and non-vehicle samples
*  A test set was randomly sampled from the training set using a 20% sampling rate and the `train-test-split` method of `scikit-learn`. As noted in the section below on possible improvement opportunities, this sampling methodology can be improved because several training examples from the GTI vehicle image database represent frames from a video feed and thus are very similar to each other. See the last section below for a discussion on this.
*  A Linear SVC classifer was fit to the training set. The training time took about 90 seconds for ~15,000 training samples.
*  The trained model was test against the Test Set and achieved an accuracy of 99.63% as specified above.
*  The SVC classifier was saved using the pickle module along with its datasets and feature transformation parameters

Refer to the function `pipeline_v1` which is well documented in `./model.py`.

###Sliding Window Search

####1. Sliding window search, scales to search, and overlap of windows.

Sliding window search was implemented using a HOG Sub-sampling Window Search approach. This is slightly more efficient because HOG features are extracted only once.

Scales searched using this approach were 0.8, 1.25, and 1.75 between the pixel ranges of 380 to 444, 380 to 508, and 380 to 656. 

As you can see, there is a high degree of overlap amongst the scales. Empirically, this resulted in more stable and accurate detections. The higher overlap did decrease performance because the same area is searched with multiple sized windows.

Also, the difference in the scales was kept smaller to avoid high variability in the size of the detected bounding boxes.

Refer to the `findcars` function and the `VehicleDetector` class implementation in `./pipeline.ipynb` for details on the techniques used.

####2. Optimizing the performance of the classifier and Examples

Searching was performed on three scales using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.

The current performance of this window search algorithm is ~2 frames per second.

Example images of the detected bounding boxes are included below.

![example detection 1][image5]
![example detection 2][image6]


---

### Video Implementation

####1. Link to project video output.

Here's a [link to the video result](./output_project_video.mp4)


####2. Filter for false positives and some method for combining overlapping bounding boxes

The following steps were taken to filter for false positives and for combining overlapping bounding boxes:
* Positions of positive detectiosn were recorded and a heatmap was created for each frame
* The heatmap for each frame was thresholded and stored in an instance queue of prior thresholded heatmaps
* The heatmaps were averaged over a number of frames (specified in the constructur argument `keep` to the `VehicleDetector` class in `./pipeline.ipynb`); the averaged heatmap also had to meet the threshold value. For example, with a threshold of 1 and a keep value of 5, that means the heatmap pixel must have maintained at least 1 detection for the past 5 frames.
*  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the averaged heatmap.
*  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.
*  Additionally, the function `non_max_suppression_fast` in `./pipeline.ipynb` that used to merge overlapping bounding boxes that would sometimes appear when two vehicles were next to each other.

### Here are six frames with initial overlapping bounding boxes and their corresponding heatmaps:

![First 3][image7]
![Last 3][image8]

### Here the is an example with the full bounding box drawn following all steps in the pipeline.
![example detection 2][image6]


---

###Discussion

####1. Problems / issues, where pipeline will likely, and what could be done to make it more robust

* The training images from the GTI contain time-series images from video feeds. Selecting a random test set from this might end up creating a number of test examples that are too similar to training examples. Some approaches to tackle this would be getting meta data on which images are part of each set in one time-series and then randomly sampling those sets instead. Additionally, randomly jittering the training data in brightness, contrast, translation, rotation, and other methods would create many additional samples for training that would look very different than the cleaner test examples if selected from the same set.
* The time it takes to search the input camera feeds is quite high. Even if it does process ~2 frames per second, this is too slow for real life, where the camera feeds might generate between 24 to 120 frames per second. This might be solvable by finding points at which the overall feature extraction algorithm is not running in parallel, and building a parallelizable solution. We are fundamentally limited by the Global Interpretor Lock in python here, but C++ might be better for this pipeline.
* The approach of using fixed scale rectangular window does not accurately represent the shape of vehicles and thus the boundaries around vehicles are not effectively captured. The data produced by the algorithms in this pipeline are not at the sufficient level to be useful to a self-driving vehicle.
* The camera used requires calibration, which might improve the accuracy of detection given that the shapes are distorted around the camera edges. Since this pipeline uses the same videos as the Advanced Lane Line Detector previously created, this step is left as a simple TODO to add in.
* The current pipeline is quite jittery in the detected height and width of vehicles. Some type of smoothing for previously detected vehicle positons and speed is needed to ensure that the shapes do not fluctate frame to frame as they do now.

