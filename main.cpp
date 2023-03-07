/* Jake Stringfellow
   Spring 2023
   Real-time 2D Object Recognition
   Main file
*/  

#include <stdio.h>
#include <string>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/videoio.hpp>
#include <cstdio>
#include <cstring>
#include <vector>
#include <dirent.h>
#include <iostream>
#include "preprocessing.hpp"

// Data type to store image ID and distance from target image together
struct ImageData {
    // the image filename
    std::string ID;
    // the distance from the target image data
    std::vector<float> featureVector;
    // distance from frame
    float distance;
};


// Create a program to cycle through different mat mask filters based on user key inputs
int main(int argc, char *argv[]) {
  

    // Store ID and feature vectors for database images
    std::vector<ImageData> databaseImageData;

    std::vector<std::string> databaseImageNames; 
 
    // Name of database
    char dirname[256] = "DatabaseImages";
    std::string dbName = "DatabaseImages";    

    char buffer[256];
    FILE *fp;
    DIR *dirp;
    struct dirent *dp;
    int i;

    // open the directory
    dirp = opendir( dirname );
    
    // loop over all the files in the image file listing
    while( (dp = readdir(dirp)) != NULL ) {
	 
 	// check if the file is an image
        if( strstr(dp->d_name, ".jpg") ||
            strstr(dp->d_name, ".png") ||
            strstr(dp->d_name, ".ppm") ||
            strstr(dp->d_name, ".tif") ) {


            // build the overall filename
            strcpy(buffer, dirname);
            strcpy(buffer, "/");
            strcpy(buffer, dp->d_name);

            std::string filename = dp->d_name;

            databaseImageNames.push_back(filename);

        }
    }


    // For every database image
    for(int i=0; i<databaseImageNames.size(); i++) {
	
	// Read image i
	std::string fullPathName = dbName + "/" + databaseImageNames[i];
	cv::Mat image = cv::imread( fullPathName );

	// Prime the image names to be IDs	
	std::string imageName = databaseImageNames[i].erase(databaseImageNames[i].length() - 4);

        // Store connectcomponents() parameters 
	cv::Mat labels;
        cv::Mat stats;
        cv::Mat centroids;
	int image_nLabels;

	// Convert the database image to a binary image with foreground white and background black
	cv::Mat thresh_image = preprocess_threshold(image);
        
	// Clean up the binary image with morphological filtering
	cv::Mat image_CU = cleanup(thresh_image);

	// Segment the image into regions, color regions, calculate bounding box coords
	cv::Mat imageRegions = analysis(image_CU, labels, stats, centroids, image_nLabels);
	cv::Mat image_boundBox; 

	// Calculate the feature vector for that image, display bounding box and first Hu Moment value
	std::vector<float> huMoments = feature_computation(image_CU, imageRegions, image_boundBox, stats, image_nLabels);
	
	// Store the ID and feature vector of the image
 	ImageData currentImage;
	
	// Store the huMoments vector as the database image's feature vector
	currentImage.featureVector = huMoments;
	
	
	// Based on the name of the file, assign it 1 of n available IDs
	// If the name of the file is pen...
	if (imageName == "pen") {
		//std::cout << "Marked Pen" << std::endl;
		currentImage.ID = imageName;
	}
	// If the name of the file is arrow...
	else if (imageName == "arrow") {
                //std::cout << "Marked arrow" << std::endl;
                currentImage.ID = imageName;
        }
	// If the name of the file is tree...
	else if (imageName == "tree") {
                //std::cout << "Marked tree" << std::endl;
                currentImage.ID = imageName;
        }
	// If the name of the file is chip...
	else if (imageName == "chip") {
                //std::cout << "Marked chip" << std::endl;
                currentImage.ID = imageName;
        }
	// If the name of the file is leaf...
	else if (imageName == "leaf") {
                //std::cout << "Marked leaf" << std::endl;
                currentImage.ID = imageName;
        }
	// If the name of the file is comb...
	else if (imageName == "comb") {
                //std::cout << "Marked comb" << std::endl;
                currentImage.ID = imageName;
        }
	// If the name of the file is flash...
	else if (imageName == "flash") {
                //std::cout << "Marked flash" << std::endl;
                currentImage.ID = imageName;
        }
	// If the name of the file is heart...
	else if (imageName == "heart") {
                //std::cout << "Marked heart" << std::endl;
                currentImage.ID = imageName;
        }
	// If the name of the file is eraser...
	else if (imageName == "eraser") {
                //std::cout << "Marked eraser" << std::endl;
                currentImage.ID = imageName;
        }
	else if (imageName == "marker") {
                //std::cout << "Marked marker" << std::endl;
                currentImage.ID = imageName;
        }
	// If the name of the file is not one of the expected options, display it as "Null"
	else {
		currentImage.ID = "Null";
	}
	
	// Push back the current Image's data onto the databaseImageData vector
	databaseImageData.push_back(currentImage);
	

    }

    
    // Vector storing our standard deviation for each Hu Moment
    std::vector<float> HuStdDev;

    // Calculate std dev of each feature 
    for (int i=0; i<databaseImageData.at(0).featureVector.size(); ++i) {
	float HuSum = 0;
	
	// Add up each of the database image values for that index of Hu Moment vector
	for (int j=0; j<databaseImageData.size(); j++) {
	    HuSum+= databaseImageData.at(j).featureVector.at(i);
	}
	// Calculate the mean
	float HuMean = HuSum/databaseImageData.size();
	std::vector<float> squared_differences;
	// For each database image value for that index of Hu Moment vector
	for (int j=0; j<databaseImageData.size(); j++) {
	    // Subtract the mean from the value
	    float temp = databaseImageData.at(j).featureVector.at(i) - HuMean;
	    // Square the result
	    float result = temp * temp;
	    squared_differences.push_back(result);
	}
	float sum_differences = 0;
	
	// Sum up the squared differences for that Hu Moment index
	for (int x=0; x<squared_differences.size(); x++) {
	    sum_differences += squared_differences.at(x);
	}
	// Calculate the mean of those values
	float mean_differences = sum_differences/squared_differences.size();
	// Take the square root of that mean to get the standard deviation
	float std_dev = std::sqrt(mean_differences);
	
	HuStdDev.push_back(std_dev);
    }
    
    std::vector<float> distances;   
        
  
    cv::VideoCapture *capdev;
    // open the user video, set pointer to it
    capdev = new cv::VideoCapture(0);
    // If a video device is not being accessed alert the user
    if( !capdev->isOpened() ) {
        printf("Unable to open video device\n");
        return(0);
    }
    
    // Identify windows
    cv::namedWindow("Live Video", 1);
    cv::namedWindow("Bound Box", 1);
    cv::moveWindow("Bound Box", 640, 0);
    // Optional windows
    //cv::namedWindow("Threshold Video", 1);
    //cv::moveWindow("Threshold Video", 640, 0);    
    //cv::namedWindow("Regions", 1);
    //cv::moveWindow("Regions", 640, 0);

    // Initialize frame, will be streamed from the camera
    cv::Mat frame;

    // For each frame    
    for(;;) {
        *capdev >> frame; // get a new frame from the camera, treat as a stream
        
	// Convert the live video frame to a binary image with foreground white and background black 
	cv::Mat threshold = preprocess_threshold(frame);

	// Clean up the binary image with morphological filtering
	cv::Mat threshold_CU = cleanup(threshold);
	
	// Store connectcomponents() parameters 
	cv::Mat frame_labels;
        cv::Mat frame_stats;
        cv::Mat frame_centroids;
	int frame_nLabels;

	// Segment the image into regions, color regions, calculate bounding box coords
        cv::Mat frame_colored_regions = analysis(threshold_CU, frame_labels, frame_stats, frame_centroids, frame_nLabels);
	cv::Mat frame_boundBox;

        // Calculate the feature vector for that frame
        std::vector<float> frame_huMoments = feature_computation(threshold_CU, frame_colored_regions, frame_boundBox, frame_stats, frame_nLabels);	
	
	// Compare frame to DB objects
	for (int i=0; i<databaseImageData.size(); i++) {
	    float sum_distances = 0;
	    for (int j=0; j<frame_huMoments.size();j++) {
		// Difference = abs(x1-x2)/Std Dev
		float difference = abs((abs(frame_huMoments.at(j)) - abs(databaseImageData.at(i).featureVector.at(j)))/HuStdDev.at(j));
		// Add the differences up for that particular Hu Moment
		sum_distances += difference;
	    }
	    // Compute the average distance between the frame and that database image
	    float avgDistance = sum_distances/frame_huMoments.size();
	    // Assign that distance to the imageData for that image
	    databaseImageData.at(i).distance = avgDistance;
	}

	// Sort the database objects in ascending order based on distance value
	std::sort(databaseImageData.begin(), databaseImageData.end(), [](const ImageData &i, const ImageData &j) {
            return i.distance < j.distance;
        });

	// The most similar image (lowest distance) is now in position one
	// Label this frame the same as the most likely object based on our calculations	
	std::string label = databaseImageData.at(0).ID;

	// Display the label of the object in the frame
        std::string text = "Object: " + label;
        cv::Point textCoords = cv::Point(400, 700);
        int fontScale = 3;
	// White text (due to black background)
        cv::Vec3b textColor = cv::Vec3b(255,255,255);
        int thickness = 3;
	// Pink/Purple text for the live video feed
	cv::Vec3b liveColor = cv::Vec3b(200,0,200);

	// Display the text in the frames
        cv::putText(frame_boundBox, text, textCoords, cv::FONT_HERSHEY_SIMPLEX, fontScale, textColor, thickness, cv::LINE_AA);
	cv::putText(frame, text, textCoords, cv::FONT_HERSHEY_SIMPLEX, fontScale, liveColor, thickness, cv::LINE_AA);

        // If the frame is empty, alert the user, break the loop
        if( frame.empty() ) {
            printf("frame is empty\n");
            break;
        }
	
        else {
            cv::imshow("Live Video", frame);
	    //cv::imshow("Threshold Video", threshold_CU);
	    //cv::imshow("Regions", frame_colored_regions);
            cv::imshow("Bound Box", frame_boundBox);
	}
    
        // see if there is a waiting keystroke
        char key = cv::waitKey(1);
        
        // If that keystroke is s, take a screenshot and show processing at each stage
        if( key == 's') {
            cv::imwrite("screenshot.jpg", frame);
	    cv::imwrite("threshold.jpg", threshold);
	    cv::imwrite("cu_threshold.jpg", threshold_CU);
            cv::imwrite("regions.jpg", frame_colored_regions);
	    cv::imwrite("frame_boundBox.jpg", frame_boundBox);
	}
        // If that keystroke is q, quit the program
        else if( key == 'q') {
            break;
        }
    }
    
    // Delete the live video object
    delete capdev;
    return(0);
}

