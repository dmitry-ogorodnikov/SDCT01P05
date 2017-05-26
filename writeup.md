##Writeup Template
###You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./output_images/car.png "Car image"
[image2]: ./output_images/noncar.png "Non car image"
[image3]: ./output_images/hog.png "Hog image"
[image4]: ./output_images/all_windows.png "All windows"
[image5]: ./output_images/hot_windows.png "Hot windows"
[image6]: ./output_images/additional_search.png "Additional search"
[image7]: ./output_images/detection_example0.png "Detection example 0"
[image8]: ./output_images/detection_example1.png "Detection example 1"
[image9]: ./output_images/detection_example2.png "Detection example 2"
[image10]: ./output_images/detection_example3.png "Detection example 3"
[image11]: ./output_images/detection_example4.png "Detection example 4"
[image12]: ./output_images/detection_example5.png "Detection example 5"
[image13]: ./output_images/frame0.png "Frame 0"
[image14]: ./output_images/heatmap0.png "Heatmap 0"
[image15]: ./output_images/frame1.png "Frame 1"
[image16]: ./output_images/heatmap1.png "Heatmap 1"
[image17]: ./output_images/frame2.png "Frame 2"
[image18]: ./output_images/heatmap2.png "Heatmap 2"
[image19]: ./output_images/frame3.png "Frame 3"
[image20]: ./output_images/heatmap3.png "Heatmap 3"
[image21]: ./output_images/frame4.png "Frame 4"
[image22]: ./output_images/heatmap4.png "Heatmap 4"
[image23]: ./output_images/frame5.png "Frame 5"
[image24]: ./output_images/heatmap5.png "Heatmap 5"
[image25]: ./output_images/heatmaps5.png "Result heatmap"
[image26]: ./output_images/bounding_boxes.png "Bounding boxes"
[video1]: ./output_project_video.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
###Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  


###Histogram of Oriented Gradients (HOG)

####1. Explain how (and identify where in your code) you extracted HOG features from the training images.

Functions for extracting HOG features from an image are contained in lines 7-36 of the file `extract_features.py`. `skimage.hog` function is used for visualization, but for finally computations I used `cv2.HOGDescriptor`, because it works faster. 

I started by reading in all the `vehicle` and `non-vehicle` images (lines 6-27 in `data_exploration.py`). Amount of `vehicle` and `non-vehicle` images is 8792 and 8968 respectively.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![Car image][image1]
![Non car image][image2]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using `Y` channel of the `YCrCb` color space and HOG parameters of `orientations=9`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:

![Car image][image1]
![Hog image][image3]

####2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of parameters and check results by using function 'score' of a linear SVM.  I was trying to find a trade-off between speed (amount of features) and accuracy. To reduce the number of features I resize images to 32x32 pixels. The final HOG parameters are `orientations=9`, `pixels_per_cell=(4, 4)`, `cells_per_block=(2, 2)`, `hog_channels=ALL`. Also I used `YCrCb` color space. All parameters are contained in lines 3-14 of the file `parameters.py`.

####3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a linear SVM using HOG features, color histogram features (lines 49-57 in `extract_features.py`, parameter `hist_bins=16`) and binned color features (lines 41-45 in `extract_features.py`, parameter `spatial_size=(16, 16)`). `extract_features` function is contained in lines 61-112 in `extract_features.py`. To reduce the number of features I resize images to 32x32 pixels. The total number of features is 6108.
For training a linear SVM classifier I used 80% of all images. The remaining 20% were used for testing. After training (lines 11-40 in `training.py`), accuracy of SVM was 99% on test set.

###Sliding Window Search

####1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I realized multiscale sliding window search algorithm (lines 137-151 in `searching.py`) based on `slide_window` (lines 87-122 in `searching.py`) and `search_windows` (lines 66-84 in `searching.py`) functions, which were described in the lessons. I considered only part of the image from 400 to 600 pixels in y-axis. Experimenting with various combinations of windows size, I noticed that small windows with 75% overlapping show a good result, but take a long time. So I decided to resize the initial images from 1280x720 pixels to 640x360 pixels and use 32x32 and 64x64 windows with 75% overlapping.

![All windows][image4]
![Hot windows][image5]

In addition, for more accurate detection of vehicles, I implemented another function `additional_search` (lines 154-166 in `searching.py`) that performs sliding window search (with window size (32,32) and 85% overlapping) in areas where the vehicle may be located.

![Additional search][image6]

####2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on two scales with different overlapping using HOG features from all 3 channels in YCrCb space plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  Here are some example images:

![Detection example 0][image7]
![Detection example 1][image8]
![Detection example 2][image9]
![Detection example 3][image10]
![Detection example 4][image11]
![Detection example 5][image12]

### Video Implementation

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./output_project_video.mp4)


####2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

Pipeline of vehicle detection algorithm (lines 19-41 in 'file.py'):
1. On each frame I apply the function `multiscale_search` to extract hot windows where vehicles may be located.
2. I create heatmap using found hot windows and apply `cv2.GaussianBlur` for better extraction of the object.
3. I use `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap. Then I extract the candidates by using `get_candidates` function (lines 46-61 in `searching.py`).
4. For the candidates I apply `additional_search` function to increase the number of hot windows for vehicles.
5. I supplement the heatmap with new hot windows.
6. I apply threshold (value is equal to 5) to the heatmap to reduce false positives.
7. I add the heatmap to deque (max length is 10).
8. I summarize the heatmaps which are contained in deque, and apply threshold (value is equal to 50) to the result to reduce false positives.
9. I apply `cv2.GaussianBlur` for better extraction of the object and use `scipy.ndimage.measurements.label()` to identify individual blobs in the result heatmap.
10. I draw bounding boxes to cover the area of each blob detected.

  
Here's an example result showing the heatmap from a series of frames of video, the result heatmap from deque on last frame of video and the bounding boxes then overlaid on the last frame of video:

### Here are six frames and their corresponding heatmaps:

![Frame 0][image13]
![Heatmap 0][image14]

![Frame 1][image15]
![Heatmap 1][image16]

![Frame 2][image17]
![Heatmap 2][image18]

![Frame 3][image19]
![Heatmap 3][image20]

![Frame 4][image21]
![Heatmap 4][image22]

![Frame 5][image23]
![Heatmap 5][image24]


### Here is the result heatmap from deque on last frame in the series:
![Result heatmap][image25]

### Here the resulting bounding boxes are drawn onto the last frame in the series:
![Bounding boxes][image26]

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Problems:
1. The algorithm works quite slowly.
2. Likely, the algorithm will work bad on difficult illumination.
3. I think a linear SVM is not a very reliable classifier.

Possible improvements:
1. Rewrite code on C++ and parallelize algorithms for acceleration.
2. Use deep network.
