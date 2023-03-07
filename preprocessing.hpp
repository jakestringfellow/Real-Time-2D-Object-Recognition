/* Jake Stringfellow
   CS 5330 Spring 2023  
   Project 3: Real-time 2D Object Recognition
   preprocessing.hpp
   Header file for live video threshold preprocessing
*/

#ifndef PREPROCESSING_HPP
#define PREPROCESSING_HPP

#include <iostream>
#include <cstdio>
#include <vector>
#include <opencv2/opencv.hpp>
#include <cstdlib>

    /*  Function for computing preprocessing filters on live video frames
        Parameters: src, a mat object that will be transformed into a black and white thresholded image
        Returns: a black and white Mat image where the black pixels indicate background and the white pixels indicate foreground.
    */
    cv::Mat preprocess_threshold(cv::Mat &src);

    /* Function for cleaning up the binary image, uses morphological filtering to first shrink any unexpected noise,
       then grows back to clean up holes in the image. Uses erosion followed by dilation to remove noise, then 
       dilation followed by erosion to remove the holes caused by the reflections.
       Parameters: src, a binary mat object that will be cleaned up
       Returns: A cleaned up binary Mat image 
    */
    cv::Mat cleanup(cv::Mat &src);

    std::vector<float> feature_computation(cv::Mat &src, cv::Mat &src_regions, cv::Mat &dst, cv::Mat stats, int nLabels);

    cv::Mat analysis(cv::Mat &src, cv::Mat labels, cv::Mat stats, cv::Mat centroids, int nLabels);

    // Flattens Mat object into feature vector
    std::vector<float> flattenMat(cv::Mat &src);

#endif
