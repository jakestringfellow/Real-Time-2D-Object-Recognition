/* Jake Stringfellow
   CS 5330 Spring 2023
   Project 3: Real-time 2D Object Recognition
   preprocessing.cpp
   Implements functionality necessary for thresholding live video
*/

#include <cstdio>
#include <vector>
#include <opencv2/opencv.hpp>
#include <cstdlib>
#include "preprocessing.hpp"
#include <cmath>
#include "math.h"
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

/* Function for computing preprocessing filters on live video frames
   Parameters: src, a mat object that will be transformed into a black and white thresholded image
   Returns: a black and white Mat image where the black pixels indicate background and the white pixels indicate foreground.
*/
cv::Mat preprocess_threshold(cv::Mat &src) {
    // Initialize variables, result = frame after pre-processing
    cv::Mat result  = cv::Mat::zeros( src.size(), CV_8UC1 );
    
    // Apply grayscale filter
    cv::Mat gray = cv::Mat::zeros( src.size(), CV_8UC1 );
    cv::cvtColor(src, gray, cv::COLOR_BGR2GRAY);

    // Apply blur filter
    cv::Mat blur_gray = cv::Mat::zeros ( src.size(), CV_8UC1);
    cv::Size kSize = cv::Size(5,5);
    cv::GaussianBlur(gray, blur_gray, kSize, 0);
//    std::cout << blur_gray.shape() << std::endl;

    // Access individual pixels determine foreground and background
    // Pixels with an intensity over 100 will be determined background
    // All other pixels will be determined foreground
    for(int i=0; i<blur_gray.rows; i++) {

        // Create a pointer to the pixel
        uchar *rptr = blur_gray.ptr<uchar>(i);

        for (int j=0; j< blur_gray.cols; j++) {
	    // For each color channel
                // Set the processed pixel to 0 for all background pixels, 255 for all foreground
		if (rptr[j] >= 100) {
                    result.at<uchar>(i,j) = 0;
		}
		else {
		    result.at<uchar>(i,j) = 255;
		}
            
        }
    }
    return result;
}

/* Function for cleaning up the binary image, uses morphological filtering to first shrink any unexpected noise,
   then grows back to clean up holes in the image. Uses erosion followed by dilation to remove noise, then 
   dilation followed by erosion to remove the holes caused by the reflections.
   Parameters: src, a binary mat object that will be cleaned up
   Results: A cleaned up binary Mat image 
*/
cv::Mat cleanup(cv::Mat &src) {
    // Initialize variables, result = thresholded frame after clean up
    cv::Mat result = cv::Mat::zeros( src.size(), CV_8UC1 );

    // Use openCV's opening to create erosion followed by dilation
    cv::Size okSize = cv::Size(3,3);
    cv::Mat openElem = cv::getStructuringElement(0, okSize, cv::Point(0,0));
    cv::morphologyEx(src, result, cv::MORPH_OPEN, openElem); 

    cv::Size ckSize = cv::Size(6,6);
    cv::Mat closeElem = cv::getStructuringElement(0, ckSize, cv::Point(0,0));
    // Use openCV's closing to close up reflections (dilation followed by erosion)
    cv::morphologyEx(result, result, cv::MORPH_CLOSE, closeElem);

    return result;
}

/*Function for connected component analysis

*/
cv::Mat analysis(cv::Mat &src, cv::Mat labels, cv::Mat stats, cv::Mat centroids, int nLabels) {

    cv::Mat result = cv::Mat::zeros( src.size(), CV_8UC3 );

    
    //Use openCV's connectedComponentsWithStats
    // returns the a 4 tuple of the total number of unique labels
    //         a mask named labels that has the same spacial dimensions as our input image
    //         stats: statistics on each connected component, including bound box coords and area
    //	       centroids: x,y coords of each connected component
    nLabels = cv::connectedComponentsWithStats(src, labels, stats, centroids);

    // minimum size to be considered a region
    int min_size = 400;

    std::vector<cv::Vec3b> colors(nLabels);
    
    // PRINTING BOUND BOX COORDS
    //for(int label = 1; label < nLabels; ++label){
    //    if (stats.at<int>(label, cv::CC_STAT_AREA) >= min_size) {
    //        std::cout << stats.at<int>(label, cv::CC_STAT_LEFT) << std::endl;
    //        std::cout << stats.at<int>(label, cv::CC_STAT_TOP) << std::endl;
	//    std::cout << stats.at<int>(label, cv::CC_STAT_WIDTH) << std::endl;
	//    std::cout << stats.at<int>(label, cv::CC_STAT_HEIGHT) << std::endl;
        //}
    //}

    

    // The first value is for the background, color black
    colors[0] = cv::Vec3b(0, 0, 0);
    // For all other values, designate a random color value
    for(int label = 1; label < nLabels; ++label){
	if (stats.at<int>(label, cv::CC_STAT_AREA) >= min_size) {
            colors[label] = cv::Vec3b( (rand()&255), (rand()&255), (rand()&255) );
        }
    }
    

    //cv::Mat dst(src.size(), CV_8UC3);
    // Color the entire region the same color
    for(int r = 0; r < result.rows; ++r){
        for(int c = 0; c < result.cols; ++c){
            int label = labels.at<int>(r, c);
            cv::Vec3b &pixel = result.at<cv::Vec3b>(r, c);
            pixel = colors[label];
         }
     }
    return result;
}

/* Computes huMoment feature vector, computes and displays oriented bounding box

*/
std::vector<float> feature_computation(cv::Mat &src, cv::Mat &src_regions, cv::Mat &dst, cv::Mat stats, int nLabels) {

    // Necessary to get nLabels and stats, will refactor out later
    cv::Mat labels;
    cv::Mat centroids;
    
    // Output of frame with bounding box
    dst = cv::Mat::zeros( src.size(), CV_8UC3 );
    dst = src_regions;

    int min_size = 400;


    // Calculate moments
    cv::Moments moments = cv::moments(src, false);
    // Calculate Hu Moments
    double huMoments[7];
    cv::HuMoments(moments, huMoments);
    // Resulting hu moments have a HUGE range, use a log transform to bring them to same range
    for(int i=0; i<7; i++) {
        huMoments[i] = -1 * copysign(1.0,huMoments[i]) * log10(abs(huMoments[i]));
    }


    //Use openCV's connectedComponentsWithStats
    // returns the a 4 tuple of the total number of unique labels
    //         a mask named labels that has the same spacial dimensions as our input image
    //         stats: statistics on each connected component, including bound box coords and area
    //         centroids: x,y coords of each connected component
    nLabels = cv::connectedComponentsWithStats(src, labels, stats, centroids);

    // Obtain bound box coords for each object
    // Draw bound box for each object
    //std::cout << nLabels << std::endl;
    for(int label = 1; label < nLabels; label++){
        if (stats.at<int>(label, cv::CC_STAT_AREA) >= min_size) {
	    std::vector<cv::Point> points;
	    // Store the point for the top left vertex
	    cv::Point top_left = cv::Point(stats.at<int>(label, cv::CC_STAT_LEFT),stats.at<int>(label, cv::CC_STAT_TOP));
            points.push_back(top_left);
	    // Store the point for the top right vertex
	    cv::Point top_right = cv::Point(stats.at<int>(label, cv::CC_STAT_WIDTH)+stats.at<int>(label, cv::CC_STAT_LEFT),stats.at<int>(label, cv::CC_STAT_TOP));
	    points.push_back(top_right);
	    // Store the point for the bottom left vertex
	    cv::Point bottom_left = cv::Point(stats.at<int>(label, cv::CC_STAT_LEFT),stats.at<int>(label, cv::CC_STAT_TOP) +stats.at<int>(label, cv::CC_STAT_HEIGHT));
	    points.push_back(bottom_left);
	    // Store the point for the bottom right vertex
	    cv::Point bottom_right = cv::Point(stats.at<int>(label, cv::CC_STAT_WIDTH)+stats.at<int>(label, cv::CC_STAT_LEFT),stats.at<int>(label, cv::CC_STAT_HEIGHT)+stats.at<int>(label, cv::CC_STAT_TOP));
	    //points.push_back(bottom_right);
	    cv::RotatedRect box = cv::minAreaRect(cv::Mat(points));

	    // Draw the bound box
	    cv::Point2f vertices[4];
	    box.points(vertices);
	    for (int i=0; i<4; ++i) {
		cv::line(dst, vertices[i], vertices[(i+1) %4], cv::Scalar(0, 255, 0), 1, cv::LINE_AA);
	    }


	    // Display first region HuMoment for each region underneath the bounding box
            std::string firstHu = std::to_string(huMoments[0]);
	    std::string text = "hu[0] =" + firstHu;
            cv::Point textCoords = (cv::Point(stats.at<int>(label, cv::CC_STAT_LEFT),stats.at<int>(label, cv::CC_STAT_TOP) +stats.at<int>(label, cv::CC_STAT_HEIGHT) + 40));
            //cv::Font font = cv::FONT_HERSHEY_SIMPLEX;
            int fontScale = 1;
            cv::Vec3b textColor = cv::Vec3b(255,0,255);
            int thickness = 2;

            cv::putText(dst, text, textCoords, cv::FONT_HERSHEY_SIMPLEX, fontScale, textColor, thickness, cv::LINE_AA);
        }
    }
    //cv::Point p1 = cv::Point(0,0);
    //cv::Point p2 = cv::Point(200,200);
    //cv::line(dst, p1, p2, cv::Scalar(0,255,0),1, cv::LINE_AA);

    // Calculate moments
    //cv::Moments moments = cv::moments(src, false);
    // Calculate Hu Moments
    //double huMoments[7];
    //cv::HuMoments(moments, huMoments);
    // Resulting hu moments have a HUGE range, use a log transform to bring them to same range
    //for(int i=0; i<7; i++) {
    //	huMoments[i] = -1 * copysign(1.0,huMoments[i]) * log10(abs(huMoments[i]));
    //}
    
    // Display first region HuMoment for each region underneath the bounding box
    //std::string text = "hu[0] =" + huMoments[0];
    //cv::Point textCoords = (cv::Point(stats.at<int>(label, cv::CC_STAT_LEFT),stats.at<int>(label, cv::CC_STAT_TOP) +stats.at<int>(label, cv::CC_STAT_HEIGHT) - 20))
    //cv::Font font = cv::FONT_HERSHEY_SIMPLEX;
    //cv::fontScale fontScale = 1;
    //cv::Vec3b textColor = cv::Vec3b(255,0,255);
    //int thickness = 2;

    //dst = cv::putText(dst, text, textCoords, font, fontScale, textColor, thickness, cv::LINE_AA);



    int n = sizeof(huMoments) / sizeof(huMoments[0]);
    std::vector<float> huVector(huMoments, huMoments + n);
    return huVector;
}

// Flattens Mat object into feature vector
std::vector<float> flattenMat(cv::Mat &src) {
    std::vector<float> flatMat;

    // Use openCV's reshape to turn the 3D Mat into one row with many columns
    cv::Mat result = src.reshape(0,1);

    // Creates the capacity to contain the same number as elements as there are cols
    flatMat.reserve(result.cols);

    // Copy the data into the vector
    result.copyTo(flatMat);

    // Return the vector
    return flatMat;
}


