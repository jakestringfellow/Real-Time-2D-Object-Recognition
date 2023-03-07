# Real-Time-2D-Object-Recognition

# Youtube Demo Link: 
* https://www.youtube.com/watch?v=wILkbFamZw0
* Quick, <1 minute Demonstration of the Real-Time 2D Object Recognition Project on some of the recognizable objects

<img align="right" src="https://user-images.githubusercontent.com/98133775/223347231-5374a2f2-cc68-4be8-bbb1-3018c8cfcf04.png" width="280px" alt="picture">
<img align="right" src="https://user-images.githubusercontent.com/98133775/223347383-1ce5eaf6-c510-447f-9eb2-de6bb3957655.png" width="280px" alt="picture">

## Project Description: 

*  The Real-Time 2D Object Recognition project created includes functionality to identify and display the name of the object presented in front of the camera given proper conditions. These conditions include a lightly colored (ideally white) background and that the object be one of those featured in the database provided in the same directory, “DatabaseImages”. The object should be the only object in the frame but the project will run decently well as long as the object is the largest “region” in the frame. Functionality to identify multiple objects will be added later in development. The project does this by first thresholding an image into black and white values by making every dark pixel white and every light pixel dark, cleaning up the thresholded image with morphological filtering, segmenting the cleaned up image into regions using connected components analysis, and computing features for each major region before comparing these features to the database images. The database image names are then sorted in ascending order based on difference size, and thus the first image in the vector represents the most similar object. The name of this object is displayed to the user as what the program identifies the user’s object as.
* Video was given a threshold by first converting the image to grayscale, then blurring and then changing any pixel brighter than the value 100 to black (to indicate background) and any pixel darker to white (to indicate foreground). 
<img align="right" src="https://user-images.githubusercontent.com/98133775/207860960-6d09561f-93ae-4663-aa24-b004edfebdf5.png" width="360px" alt="picture">
* Morphological filtering was then used to clean up the binary images. The function uses erosion followed by dilation to remove noise in the image, then dilation followed by erosion to remove holes caused by reflections.
<img align="right" src="https://user-images.githubusercontent.com/98133775/207860960-6d09561f-93ae-4663-aa24-b004edfebdf5.png" width="360px" alt="picture">
* The segmentation function uses cv::connectedComponenetsWithStats to identify regions in the image. Colors the background black, then colors each other region in the image a random color.
<img align="right" src="https://user-images.githubusercontent.com/98133775/207860960-6d09561f-93ae-4663-aa24-b004edfebdf5.png" width="360px" alt="picture">
* Calculates moments with cv::moments() and cv::HuMoments(). The moments() function returns spatial, central, and normalized central moments, the HuMoments() function uses these and returns 7 moments that are invariant to translation, rotation, and scale, the seventh moment covers reflection/mirroring of the object. A log transform is used on the Hu Moments to bring them into a similar range of each other.
The bounding box coordinates were obtained from the cv::connectedComponentsWithStats() function. Within the cv::Mat stats parameter the top and left coordinates to the bounding box are given, as well as the height and width of the object. The coordinates were obtained as follows, top-left: (x = left-coord, y= top-coord), top-right: (x = left-coord + width, y = top-coord), bottom-left: (x = left-coord, y = top-coord + height), bottom-right: (x = left-coord + width, y = top-coord + height). These coordinates had a line drawn from one to the next using cv::line().
<img align="right" src="https://user-images.githubusercontent.com/98133775/207860960-6d09561f-93ae-4663-aa24-b004edfebdf5.png" width="360px" alt="picture">


## Training Explanation
* Each object in the system has a feature vector, this feature vector consists of the 7 moments calculated in the feature computation step. Within the given database 10 images are stored, these images are jpg 3 color-channel images with no alterations. Each picture represents one object we can recognize. From the database the program computes a vector of  ImageData structs, with the ID set as the image name without the “.pdf” and the feature vector being the HuMoments vector created after each image is thresholded, cleaned up, segmented, and has it’s features (moments) computed. These feature vectors are used as the training data. A standard deviation is calculated for each Hu moment stored in the feature vectors, to be used in the calculation of distances while comparing live video frames to the database images.
* When calculating which ID best suits the current image’s object in real-time, the feature vectors created from the live video frames are compared to the database image feature vectors at every index (comparing each Hu Moment respectively). This is done by, at each index of the Hu Moments vectors, subtracting the Hu Moment database from the current database image being compared from the Hu Moment of the current video frame, then dividing by the standard deviation calculated before the live video began. A sum of differences is calculated for each database image based on their collective difference across all 7 Hu Moment values, this value sum is then divided by the size of the Hu Moment vector (7), and is added to the databaseImageData vector for that particular database image’s ID.
* The databaseImageData vector is then sorted in ascending order of distance from the current frame, making the most similar object lie in position 0. The ID of this object is displayed to the user through the live video output.





## Environment Setup: 
* For ideal performance objects should be presented as the largest object in camera view with a light background (ideally white).
* Environment used in demo: Dark objects attached to white paper as background.
                  
## Technologies used:  
* OpenCV - Open Source Computer Vision LIBRARY

## How to install and run: 
* 	Have the following files in the current directory: 
*		- makefile (update opencv libraries location on your computer)
*		- main.cpp
*		- preprocessing.cpp
*   - preprocessing.hpp
*		- DatabaseImages (Given database, can technically be replace with images as long as they feature the same objects and file names)


* Enter "make objectRecognition" to the command-line
* Run the executable with: "./objectRecognition 
*	- There are 10 different identifiable objects:
*                - "pen" in particular a sharpie pen
*                - "arrow" a drawn arrow shape
*                - "tree" a drawn christmas tree shape
*                - "chip" a poker chip, most circles will do, however.
*                - "leaf" a dried out leaf (probably difficult to recreate)
*                - "comb" a fine tooth comb
*                - "flash" a dark usb flash drive
*                - "heart" a drawn heart shape
*                - "eraser" a rectangular eraser
*                - "marker" a large sharpie marker


## Going Further: 
* Adding more objects to be recognized, as well as new methods for detecting objects in more complex setups.

## Acknowledgements: 
* Code for reading into databases was heavily based on the code provided from Professor Maxwell.
