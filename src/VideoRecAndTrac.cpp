#include <opencv2/opencv.hpp>
#include "opencv2/core/core.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/nonfree/nonfree.hpp"


#include <iostream>
#include <stdio.h>

using namespace cv;
using namespace std;

/*
void drawRectOnFrame(Mat img_scene, std::vector<Point2f> scene_corners);

char key = 'a';
int framecount = 0;

SurfFeatureDetector detector( 500 );
SurfDescriptorExtractor extractor;
FlannBasedMatcher matcher;

Mat frame, des_object, image;
Mat des_image, img_matches, H;

std::vector<KeyPoint> kp_object;
std::vector<Point2f> obj_corners(4);
std::vector<KeyPoint> kp_image;
std::vector<vector<DMatch > > matches;
std::vector<DMatch > good_matches;
std::vector<Point2f> obj;
std::vector<Point2f> scene;
std::vector<Point2f> scene_corners(4);

int main()
{
                //reference image
    Mat object = imread( "notebook.jpg", CV_LOAD_IMAGE_GRAYSCALE );

    if( !object.data )
    {
        std::cout<< "Error reading object " << std::endl;
        return -1;
    }

                //compute detectors and descriptors of reference image
    detector.detect( object, kp_object );
    extractor.compute( object, kp_object, des_object );

    //create video capture object
    VideoCapture cap(0);

    //Get the corners from the object
    obj_corners[0] = cvPoint(0,0);
    obj_corners[1] = cvPoint( object.cols, 0 );
    obj_corners[2] = cvPoint( object.cols, object.rows );
    obj_corners[3] = cvPoint( 0, object.rows );

    //wile loop for real time detection
    while (true)
    {
    	//capture one frame from video and store it into image object name 'frame'
    	cap >> frame;

    	if (framecount < 5)
    	{
    		framecount++;
    		drawRectOnFrame(frame, scene_corners);
    		try
    		{
    			imshow( "Good Matches", frame );
    		}
    		catch (Exception& e)
    		{
    			const char* err_msg = e.what();
    			std::cout << "exception caught: imshow:\n" << err_msg << std::endl;
    		}
    		continue;
    	}

    	//converting captured frame into gray scale
    	cvtColor(frame, image, CV_RGB2GRAY);

    	//extract detectors and descriptors of captured frame
    	detector.detect( image, kp_image );
    	extractor.compute( image, kp_image, des_image );

    	//find matching descriptors of reference and captured image
    	matcher.knnMatch(des_object, des_image, matches, 1);

    	//finding matching keypoints with Euclidean distance 0.6 times the distance of next keypoint
    	//used to find right matches
    	for(int i = 0; i < min(des_image.rows-1,(int) matches.size()); i++)
    	{
    		if((matches[i][0].distance < 0.6*(matches[i][1].distance)) && ((int) matches[i].size()<=2 && (int) matches[i].size()>0))
    		{
    			good_matches.push_back(matches[i][0]);
    		}
    	}

    	//Draw only "good" matches
    	//drawMatches( object, kp_object, frame, kp_image, good_matches, img_matches,
    	//		Scalar::all(-1), Scalar::all(-1), vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );

    	//3 good matches are enough to describe an object as a right match.
    	if (good_matches.size() >= 3)
    	{

    		for( int i = 0; i < good_matches.size(); i++ )
    		{
    			//Get the keypoints from the good matches
    			obj.push_back( kp_object[ good_matches[i].queryIdx ].pt );
    			scene.push_back( kp_image[ good_matches[i].trainIdx ].pt );
    		}
    		try
    		{
    			H = findHomography( obj, scene, CV_RANSAC );
    		}
    		catch(Exception e){}

    		perspectiveTransform( obj_corners, scene_corners, H);

    		//Draw lines between the corners (the mapped object in the scene image )
    		drawRectOnFrame(frame, scene_corners);
    	}

    	//Show detected matches

    	try
    	{
    		imshow( "Good Matches", frame );
    	}
    	catch (Exception& e)
    	{
    	    const char* err_msg = e.what();
    	    std::cout << "exception caught: imshow:\n" << err_msg << std::endl;
    	}

    	//clear array
    	good_matches.clear();

    	key = waitKey(30);
    	if( key >= 0) break;
    }
    return 0;
}

void drawRectOnFrame( Mat img_matches, std::vector<Point2f> scene_corners){
	if (!scene_corners.empty()){
		line( img_matches, scene_corners[0], scene_corners[1], Scalar(0, 255, 0), 4 );
		line( img_matches, scene_corners[1], scene_corners[2], Scalar( 0, 255, 0), 4 );
		line( img_matches, scene_corners[2], scene_corners[3], Scalar( 0, 255, 0), 4 );
		line( img_matches, scene_corners[3], scene_corners[0], Scalar( 0, 255, 0), 4 );
	}
}
*/

void readme();
void drawRectOnFrame( Mat img_scene, std::vector<Mat> img_objects);
bool isNmeric(const std::string& str);

bool isNumeric(const std::string& str) {
    char *end;
    long val = std::strtol(str.c_str(), &end, 10);
    if ((errno == ERANGE && (val == LONG_MAX || val == LONG_MIN)) || (errno != 0 && val == 0)) {
        //  if the converted value would fall out of the range of the result type.
        return false;
    }
    if (end == str) {
       // No digits were found.
       return false;
    }
    // check if the string was fully processed.
    return *end == '\0';
}

int main(int argc, char** argv)
{
	if( argc < 3 )
	{	readme(); return -1; }

	//VideoCapture cap(0); // open the default camera

	VideoCapture cap;
	if (isNumeric(argv[1]))
		cap.open(atoi(argv[1]));
	else
		cap.open(argv[1]);


	if(!cap.isOpened())  // check if we succeeded
		return -1;

	std::vector<Mat> objects;

	for (int i = 2; i < argc; i++){
		Mat object;
		try{
			object = imread(argv[i], CV_LOAD_IMAGE_COLOR);
		}catch(Exception& e){
			printf("Fail loading image '%s'", argv[i]);
		}
		objects.push_back(object);
	}


	/*
	VideoWriter outputVideo;                                        // Open the output

	outputVideo.open("scene2_2.mp4", static_cast<int>(cap.get(CV_CAP_PROP_FOURCC)), cap.get(CV_CAP_PROP_FPS), Size((int) cap.get(CV_CAP_PROP_FRAME_WIDTH),    // Acquire input size
            (int) cap.get(CV_CAP_PROP_FRAME_HEIGHT)), true);

	if (!outputVideo.isOpened())
	{
		cout  << "Could not open the output video for write." << endl;
		return -1;
	}
	*/


    Mat edges;
    cv::namedWindow("edges",1);
    cv::CascadeClassifier();

    for(;;)
    {
        Mat frame;
        Mat processed_frame;

        cap >> frame; // get a new frame from camera

        drawRectOnFrame(frame, objects);

        //cv::cvtColor(frame, edges, CV_BGR2GRAY);
        //cv::GaussianBlur(edges, edges, Size(7,7), 1.5, 1.5);
        //cv::Canny(edges, edges, 0, 30, 3);


        imshow("edges", frame);
        //outputVideo << frame;
        if(waitKey(30) >= 0) break;
    }
    // the camera will be deinitialized automatically in VideoCapture destructor
    return 0;

}


void drawRectOnFrame( Mat img_scene, std::vector<Mat> img_objects){
  Mat img_object_gray;
  Mat img_scene_gray;

  cv::cvtColor(img_scene, img_scene_gray, CV_BGR2GRAY);

  for(int i = 0; i < img_objects.size(); i++){

	  cv::cvtColor(img_objects[i], img_object_gray, CV_BGR2GRAY);


	  //-- Step 1: Detect the keypoints using SURF Detector
	  int minHessian = 400;

	  SurfFeatureDetector detector( minHessian );

	  std::vector<KeyPoint> keypoints_object, keypoints_scene;

	  detector.detect( img_object_gray, keypoints_object );
	  detector.detect( img_scene_gray, keypoints_scene );

	  //-- Step 2: Calculate descriptors (feature vectors)
	  SurfDescriptorExtractor extractor;

	  Mat descriptors_object, descriptors_scene;

	  extractor.compute( img_object_gray, keypoints_object, descriptors_object );
	  extractor.compute( img_scene_gray, keypoints_scene, descriptors_scene );

	  //-- Step 3: Matching descriptor vectors using FLANN matcher
	  FlannBasedMatcher matcher;
	  //std::vector< vector<DMatch> > all_matches;
	  std::vector<DMatch> match;

	  //matcher.knnMatch( descriptors_object, descriptors_scene , all_matches, 2);
	  matcher.match( descriptors_object, descriptors_scene , match);
	  //all_matches.push_back(match);

	  //all_matches.push_back(match);
	  //std::vector<vector<DMatch>> all_good_matches;

	  //for (int j = 0; j < number_of_matches; j++){
	  double max_dist = 0; double min_dist = 100;

	  //-- Quick calculation of max and min distances between keypoints
	  for( int i = 0; i < descriptors_object.rows; i++ )
	  { double dist = match[i].distance;
	  if( dist < min_dist ) min_dist = dist;
	  if( dist > max_dist ) max_dist = dist;
	  }

	  //printf("-- Max dist : %f \n", max_dist );
	  //printf("-- Min dist : %f \n", min_dist );

	  //-- Draw only "good" matches (i.e. whose distance is less than 3*min_dist )
	  std::vector< DMatch > good_matches;

	  float nndr_ratio = 3;
	  for( int i = 0; i < descriptors_object.rows; i++ )
	  { if( match[i].distance < max(nndr_ratio * min_dist, 0.02) )
	  { good_matches.push_back( match[i]); }
	  }

	  //all_good_matches.push_back(good_matches);


	  /*
		  drawMatches( img_object_gray, keypoints_object, img_scene_gray, keypoints_scene,
						good_matches, img_matches, Scalar::all(-1), Scalar::all(-1),
					   vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );
	   */



	  //-- Localize the object from img_1 in img_2
	  std::vector<Point2f> obj;
	  std::vector<Point2f> scene;

	  for( size_t i = 0; i < good_matches.size(); i++ )
	  {
		  //-- Get the keypoints from the good matches
		  obj.push_back( keypoints_object[ good_matches[i].queryIdx ].pt );
		  scene.push_back( keypoints_scene[ good_matches[i].trainIdx ].pt );
	  }

	  Mat H;

	  try
	  {

		  H = findHomography( obj, scene, CV_RANSAC );


		  //-- Get the corners from the image_1 ( the object to be "detected" )
		  std::vector<Point2f> obj_corners(4);
		  obj_corners[0] = Point(0,0);
		  obj_corners[1] = Point( img_object_gray.cols, 0 );
		  obj_corners[2] = Point( img_object_gray.cols, img_object_gray.rows );
		  obj_corners[3] = Point( 0, img_object_gray.rows );
		  std::vector<Point2f> scene_corners(4);

		  perspectiveTransform( obj_corners, scene_corners, H);
		  /*
		  for (int i = 0; i <= 3; i++)
			  std::cout << obj_corners[i] <<"\n";

		  for (int i = 0; i <= 3; i++)
			  std::cout << scene_corners[i] << "\n";
		   */

		  //-- Draw lines between the corners (the mapped object in the scene - image_2 )
		  //Point2f offset( (float)img_object_gray.cols, 0);
		  Point2f offset( 0, 0);
		  line( img_scene, scene_corners[0] + offset, scene_corners[1] + offset, Scalar(0, 255, 0), 4 );
		  line( img_scene, scene_corners[1] + offset, scene_corners[2] + offset, Scalar( 0, 255, 0), 4 );
		  line( img_scene, scene_corners[2] + offset, scene_corners[3] + offset, Scalar( 0, 255, 0), 4 );
		  line( img_scene, scene_corners[3] + offset, scene_corners[0] + offset, Scalar( 0, 255, 0), 4 );
	  }
	  catch(Exception e){}
	}
}


  /*
  //-- Show detected matches
  //cv::namedWindow("Good Matches & Object detection");
  Mat resized_img_matches = img_matches;
  //cv::resize(img_matches, resized_img_matches, cv::Size(), 0.1, 0.1, cv::INTER_LANCZOS4);

  imshow( "Good Matches & Object detection", resized_img_matches );
  */

  //cv::waitKey(0);
//}



void readme()
{ printf(" Usage: ./SURF_Homography <img1>\n"); }






