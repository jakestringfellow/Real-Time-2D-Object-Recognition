/* Jake Stringfellow
   Real-time 2D Object Recognition
   preprocessing.cpp
   Implements functionality necessary for manipulating Mat
   images and Mat image data for 2D object recognition
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
		    // Set light pixels to black
                    result.at<uchar>(i,j) = 0;
		}
		else {
		    // Set dark pixels to white
		    result.at<uchar>(i,j) = 255;
		}
            
        }
    }
    // Return binary image
    return result;
}

/* Function for cleaning up the binary image, uses morphological filtering to first shrink any unexpected noise,
   then grows back to clean up holes in the image. Uses erosion followed by dilation to remove noise, then 
   dilation followed by erosion to remove the holes caused by the reflections.
   Parameters: src, a binary mat object that will be cleaned up
   Returns: A cleaned up binary Mat image 
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

    // Return cleaned up binary image
    return result;
}

/*Function for connected component analysis, creates segmented, region-colored version of the src image
  Parameters: a src image to be sampled from, then Mat data types for labels, stats, and centroid calculation.
  Returns: a segmented, region colored version of the src image
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
    

    // The first value is for the background, color black
    colors[0] = cv::Vec3b(0, 0, 0);
    // For all other values, designate a random color value
    for(int label = 1; label < nLabels; ++label){
	if (stats.at<int>(label, cv::CC_STAT_AREA) >= min_size) {
            colors[label] = cv::Vec3b( (rand()&255), (rand()&255), (rand()&255) );
        }
    }
    
    
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
   Paramters: Mat src: binary image to be sampled from
	      Mat src_regions: colored_region image to sample for dst image
	      Mat dst: destination image, segmented, colored regions, bound box and first Hu Moment display
	      Mat stats: stats for bound box coordinates
	      int nLabels, number of regions in given frame
   Returns:   Hu momoment feature vector of floating point numbers
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
	
	    // Create the box based on the coordinates
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

	    // Draw the text to the dest image
            cv::putText(dst, text, textCoords, cv::FONT_HERSHEY_SIMPLEX, fontScale, textColor, thickness, cv::LINE_AA);
        }
    }

    // Convert the Hu Moment array to a vector for ease of use
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


