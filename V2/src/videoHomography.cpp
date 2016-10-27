/*
* video_homography.cpp
*
*  Created on: Oct 18, 2010
*      Author: erublee
*/

#include "videoHomography.h"

using namespace cv::xfeatures2d;

VideoHomography::VideoHomography(cv::VideoCapture cap, std::vector<cv::Mat> obj){
	capture = cap;
	objects = obj;

    std::cout << "Controls:" << std::endl;
    std::cout << "t : grabs a reference frame to match against" << std::endl;
    std::cout << "l : makes the reference frame new every frame" << std::endl;
    std::cout << "q or escape: quit" << std::endl;
    
    K = determineK();
    colours.push_back(Scalar(0, 255, 0));
    names.push_back("objA");
    near.push_back(UNKNOWN);
    count.push_back(0);
    colours.push_back(Scalar(0, 0, 255));
    names.push_back("objB");
    near.push_back(UNKNOWN);
    count.push_back(0);
    colours.push_back(Scalar(125, 125, 0));
    names.push_back("objC");
    near.push_back(UNKNOWN);
    count.push_back(0);
    colours.push_back(Scalar(255, 0, 0));
    names.push_back("objD");
    near.push_back(UNKNOWN);
    count.push_back(0);
    colours.push_back(Scalar(0, 125, 125));
    names.push_back("objE");
    near.push_back(UNKNOWN);
    count.push_back(0);
    colours.push_back(Scalar(125, 0, 125));
    names.push_back("objF");
    near.push_back(UNKNOWN);
    count.push_back(0);
    colours.push_back(Scalar(0, 0, 0));
    names.push_back("objG");
    near.push_back(UNKNOWN);
    count.push_back(0);
    colours.push_back(Scalar(255, 255, 255));
    names.push_back("objH");
    near.push_back(UNKNOWN);
    count.push_back(0);


} 

int VideoHomography::run(){ 

	Ptr<BriefDescriptorExtractor> brief = BriefDescriptorExtractor::create(32);

    cv::Mat frame;

    std::vector<cv::DMatch> matches;

    cv::BFMatcher desc_matcher(cv::NORM_HAMMING);

    std::vector<cv::Point2f> train_pts, query_pts;
    std::vector<cv::KeyPoint> train_kpts, query_kpts;
    std::vector<unsigned char> match_mask;

    cv::Mat gray;

    bool ref_live = true;

    cv::Mat train_desc, query_desc;
    //const int DESIRED_FTRS = 500;
    //cv::GridAdaptedFeatureDetector detector(new cv::FastFeatureDetector(10, true), DESIRED_FTRS, 4, 4);
    Ptr<FastFeatureDetector> detector = FastFeatureDetector::create(10, true);

    cv::Mat H_prev = cv::Mat::eye(3, 3, CV_32FC1);

    cv::namedWindow("frame", cv::WINDOW_AUTOSIZE);

    frame_number = 0;
    for (;;)
    {
        capture >> frame;
        if (frame.empty())
            break;
        frame_number++;

        cvtColor(frame, gray, CV_RGB2GRAY);

        drawRectOnFrame(frame, objects);
        putText(frame, std::to_string(frame_number), Point(frame.cols - 60, 20), FONT_HERSHEY_COMPLEX_SMALL, 0.8, Scalar(0, 0, 255), 1, CV_AA);

        bool showHomo = false;
        if (showHomo){
			detector->detect(gray, query_kpts); //Find interest points

			brief->compute(gray, query_kpts, query_desc); //Compute brief descriptors at each keypoint location

			if (!train_kpts.empty())
			{

				std::vector<cv::KeyPoint> test_kpts;
				Mat invertedH;
				invert(H_prev,invertedH,DECOMP_LU);
				warpKeypoints(invertedH, query_kpts, test_kpts);

				cv::Mat mask = windowedMatchingMask(test_kpts, train_kpts, 25, 25);
				desc_matcher.match(query_desc, train_desc, matches, mask);
				drawKeypoints(frame, test_kpts, frame, cv::Scalar(255, 0, 0), cv::DrawMatchesFlags::DRAW_OVER_OUTIMG);

				matches2points(train_kpts, query_kpts, matches, train_pts, query_pts);

				if (matches.size() > 5)
				{
					cv::Mat H = findHomography(train_pts, query_pts, cv::RANSAC, 4, match_mask);
					if (countNonZero(cv::Mat(match_mask)) > 15)
					{
						//TODO
						poseFromH(H);
						H_prev = H;
					}
					else
						resetH(H_prev);
					drawMatchesRelative(train_kpts, query_kpts, matches, frame, match_mask);
				}
				else
					resetH(H_prev);

			}
			else
			{
				H_prev = cv::Mat::eye(3, 3, CV_32FC1);
				cv::Mat out;
				drawKeypoints(gray, query_kpts, out);
				frame = out;
			}

		   // drawRectOnFrame(frame, objects);
        }

        imshow("frame", frame);

        if (ref_live)
        {
            train_kpts = query_kpts;
            query_desc.copyTo(train_desc);
        }
        char key = (char)cv::waitKey(2);
        switch (key)
        {
        case 'l':
            ref_live = true;
            resetH(H_prev);
            break;
        case 't':
            ref_live = false;
            train_kpts = query_kpts;
            query_desc.copyTo(train_desc);
            resetH(H_prev);
            break;
        case 27:
        case 'q':
            return 0;
            break;
        }

    }

    for (vector<vector<int> >::iterator record = records.begin(); record!=records.end(); ++record){
    	copy((*record).begin(),(*record).end(),ostream_iterator<int>(cout,"\t"));
    	cout << "\n";
    }
    return 0;
}


void VideoHomography::help(){
    std::cout << "usage: programName <video device number>\n" << std::endl;
    std::cout << "Controls:" << std::endl;
    std::cout << "  t : grabs a reference frame to match against" << std::endl;
    std::cout << "  l : makes the reference frame new every frame" << std::endl;
    std::cout << "  q or escape: quit" << std::endl;
}

void VideoHomography::drawMatchesRelative(const std::vector<cv::KeyPoint>& train, 
		const std::vector<cv::KeyPoint>& query,
		std::vector<cv::DMatch>& matches, cv::Mat& img,
		const std::vector<unsigned char>& mask){
    for (int i = 0; i < (int)matches.size(); i++){
        if (mask.empty() || mask[i])
        {
            cv::Point2f pt_new = query[matches[i].queryIdx].pt;
            cv::Point2f pt_old = train[matches[i].trainIdx].pt;

            cv::line(img, pt_new, pt_old, cv::Scalar(125, 255, 125), 1);
            cv::circle(img, pt_new, 2, cv::Scalar(255, 0, 125), 1);

        }
    }
}

//Takes a descriptor and turns it into an xy point
void VideoHomography::keypoints2points(const std::vector<cv::KeyPoint>& in, std::vector<cv::Point2f>& out){
    out.clear();
    out.reserve(in.size());
    for (size_t i = 0; i < in.size(); ++i)
    {
        out.push_back(in[i].pt);
    }
}

//Takes an xy point and appends that to a keypoint structure
void VideoHomography::points2keypoints(const std::vector<cv::Point2f>& in, std::vector<cv::KeyPoint>& out){
    out.clear();
    out.reserve(in.size());
    for (size_t i = 0; i < in.size(); ++i)
    {
        out.push_back(cv::KeyPoint(in[i], 1));
    }
}


//Uses computed homography H to warp original input points to new planar position
void VideoHomography::warpKeypoints(const cv::Mat& H, const std::vector<cv::KeyPoint>& in, std::vector<cv::KeyPoint>& out){
    std::vector<cv::Point2f> pts;
    keypoints2points(in, pts);
    std::vector<cv::Point2f> pts_w(pts.size());
    cv::Mat m_pts_w(pts_w);
    perspectiveTransform(cv::Mat(pts), m_pts_w, H);
    points2keypoints(pts_w, out);
}

//Converts matching indices to xy points
void VideoHomography::matches2points(const std::vector<cv::KeyPoint>& train, const std::vector<cv::KeyPoint>& query,
    const std::vector<cv::DMatch>& matches, std::vector<cv::Point2f>& pts_train,
    std::vector<cv::Point2f>& pts_query){

    pts_train.clear();
    pts_query.clear();
    pts_train.reserve(matches.size());
    pts_query.reserve(matches.size());

    size_t i = 0;

    for (; i < matches.size(); i++)
    {

        const cv::DMatch & dmatch = matches[i];

        pts_query.push_back(query[dmatch.queryIdx].pt);
        pts_train.push_back(train[dmatch.trainIdx].pt);

    }

}

void VideoHomography::resetH(cv::Mat& H)
{
    H = cv::Mat::eye(3, 3, CV_32FC1);
}


void VideoHomography::poseFromH(cv::Mat H){
    
    std::vector<cv::Mat> Rs;
    std::vector<cv::Mat> ts;
    std::vector<cv::Mat> ns;

    cv::decomposeHomographyMat(H, K, Rs, ts, ns);  

    if(Rs.size() != ts.size() || ts.size() != ns.size()){
        std::cout << "Error decomposition spat out mismatched number of arrays" << std::endl;
    } else {
        std::cout << "Found" << Rs.size() << " possible solutions" << std::endl;
        for(int i = 0; i < Rs.size(); ++i){
            printPose(Rs[i], ts[i], ns[i]);    
        }
    }

    //std::cin.ignore();

}

void VideoHomography::printPose(cv::Mat R, cv::Mat t, cv::Mat n){
    //cv::Matx33d R; //rotation matrix
    //cv::Vec3d t; //translation
    //cv::Vec3d n; //normal

    std::cout << std::setprecision(3) << "Rotation:\n" << R << std::endl; 
    std::cout << std::setprecision(3) << "Translation:\n" << t << std::endl; 
    std::cout << std::setprecision(3) << "Normal:\n" << n << std::endl; 
}


cv::Mat VideoHomography::determineK(){
    //TODO replace this guess with proper K matrix
    cv::Mat K = (cv::Mat_<double>(3, 3) << 4.0027643658231323e+03, 0.0, 1.5585561040942616e+03, 0,
            4.0027643658231323e+03, 1.3883433136360570e+03, 0, 0, 1);

    return K;
}

cv::Mat VideoHomography::windowedMatchingMask( const vector<cv::KeyPoint>& keypoints1, const vector<cv::KeyPoint>& keypoints2,
                          float maxDeltaX, float maxDeltaY )
{
  if( keypoints1.empty() || keypoints2.empty() )
    return cv::Mat();

  int n1 = (int)keypoints1.size(), n2 = (int)keypoints2.size();
  cv::Mat mask( n1, n2, CV_8UC1 );
  for( int i = 0; i < n1; i++ )
    {
      for( int j = 0; j < n2; j++ )
        {
          cv::Point2f diff = keypoints2[j].pt - keypoints1[i].pt;
          mask.at<uchar>(i, j) = std::abs(diff.x) < maxDeltaX && std::abs(diff.y) < maxDeltaY;
        }
    }
  return mask;
}

bool VideoHomography::isNumeric(const std::string& str) {
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

void VideoHomography::drawRectOnFrame( Mat img_scene, std::vector<Mat> img_objects){
	Mat img_object_gray, img_scene_gray;
	std::vector<KeyPoint> keypoints_object, keypoints_scene;
	Mat descriptors_object, descriptors_scene;

	cv::cvtColor(img_scene, img_scene_gray, CV_BGR2GRAY);

	//-- Step 1: Detect the keypoints using SURF Detector
	int minHessian = 400;

	//SurfFeatureDetector detector( minHessian );

	Ptr<cv::xfeatures2d::SURF> detector=cv::xfeatures2d::SURF::create(minHessian);
	detector->detect( img_scene_gray, keypoints_scene );

	//-- Step 2: Calculate descriptors (feature vectors)
	//SurfDescriptorExtractor extractor;
	Ptr<cv::xfeatures2d::SURF> extractor = cv::xfeatures2d::SURF::create();
	extractor->compute( img_scene_gray, keypoints_scene, descriptors_scene );
	//-- Step 3: Matching descriptor vectors using FLANN matcher
	FlannBasedMatcher matcher;

	vector<int> record;
	record.push_back(frame_number);
	for(int i = 0; i < img_objects.size(); i++){

		cv::cvtColor(img_objects[i], img_object_gray, CV_BGR2GRAY);

		detector->detect( img_object_gray, keypoints_object );

		extractor->compute( img_object_gray, keypoints_object, descriptors_object );


		//std::vector< vector<DMatch> > all_matches;

		/*
		  std::vector<DMatch> match;
		  //matcher.knnMatch( descriptors_object, descriptors_scene , all_matches, 2);
		  matcher.match( descriptors_object, descriptors_scene , match);


		  //all_matches.push_back(match);
		  //std::vector<vector<DMatch>> all_good_matches;

		  //for (int j = 0; j < number_of_matches; j++){
		  double max_dist = 0; double min_dist = 100;

		  //-- Quick calculation of max and min distances between keypoints
		  for( int i = 0; i < descriptors_object.rows; i++ )
		  {
			  double dist = match[i].distance;
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
		 */

		std::vector<vector<DMatch > > matches;
		matcher.knnMatch(descriptors_object, descriptors_scene, matches, 2);
		std::vector< DMatch > good_matches;
		float nndr_ratio = 0.7f;
		for(int i = 0; i < min(descriptors_object.rows-1,(int) matches.size()); i++) //THIS LOOP IS SENSITIVE TO SEGFAULTS
		{
			if((matches[i][0].distance < nndr_ratio*(matches[i][1].distance)) && ((int) matches[i].size()<=2 && (int) matches[i].size()>0))
			{
				good_matches.push_back(matches[i][0]);
			}
		}


		if (good_matches.size() < 10){
			record.push_back(0);
			continue;
		}else{
			record.push_back(1);
		}



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
		 /* bool allZero = true;
		  for(int i = 0; i < 4; ++i){
			  if(scene_corners[i].x != 0 || scene_corners[i].y != 0){
				  allZero = false;
			  }
		  }*/

		  //determine object position
		  float x = (scene_corners[0].x + scene_corners[1].x + scene_corners[2].x + scene_corners[3].x)/4;
		  float y = (scene_corners[0].y + scene_corners[1].y + scene_corners[2].y + scene_corners[3].y)/4;
		  if(x < LOW_EDGE){
			  near[i] = LEFT;
			  count[i] = 0;
		  } else if (x > (img_scene.cols - LOW_EDGE)){
			  near[i] = RIGHT;
			  count[i] = 0;

		  } else if (y < LOW_EDGE){
			  near[i] = TOP;
			  count[i] = 0;

		  } else if (y > (img_scene.rows - LOW_EDGE)){
			  near[i] = BOTTOM;
			  count[i] = 0;

		  } else {
			  count[i] += 1;
			  //if(count[i] > 3){
				  near[i] = UNKNOWN;
			  //}
		  }

		  if(near[i] != UNKNOWN/*H.empty()*/){
			  //std::cout << "Can't find" << i << " its " << near[i] << std::endl;
			  if(near[i] == LEFT){
				  cv::Point2f A = Point2f(img_scene.cols/4.0, img_scene.rows/2.0+20*i-100);
				  cv::Point2f B = Point2f(0, img_scene.rows/2.0+20*i-100);
				  arrowedLine(img_scene, A, B, colours[i], 5);
				  putText(img_scene, (names[i]+"LEFT").c_str(), Point(10, 20+20*i), FONT_HERSHEY_COMPLEX_SMALL, 0.8, colours[i], 1, CV_AA);
			  } else if (near[i] == RIGHT){
				  cv::Point2f A = Point2f(img_scene.cols*3/4.0, img_scene.rows/2.0+20*i-100);
				  cv::Point2f B = Point2f(img_scene.cols-1, img_scene.rows/2.0+20*i-100);
				  arrowedLine(img_scene, A, B, colours[i], 5);
				  putText(img_scene, (names[i]+"RIGHT").c_str(), Point(10, 20+20*i), FONT_HERSHEY_COMPLEX_SMALL, 0.8, colours[i], 1, CV_AA);

			  } else if (near[i] == TOP){


			  } else if (near[i] == BOTTOM){

			  }
		  } else {


			  //-- Draw lines between the corners (the mapped object in the scene - image_2 )
			  //Point2f offset( (float)img_object_gray.cols, 0);
			  Point2f offset( 0, 0);
			  line( img_scene, scene_corners[0] + offset, scene_corners[1] + offset, colours[i], 4 );
			  line( img_scene, scene_corners[1] + offset, scene_corners[2] + offset, colours[i], 4 );
			  line( img_scene, scene_corners[2] + offset, scene_corners[3] + offset, colours[i], 4 );
			  line( img_scene, scene_corners[3] + offset, scene_corners[0] + offset, colours[i], 4 );
			  putText(img_scene, names[i].c_str(), Point(10, 20+20*i), FONT_HERSHEY_COMPLEX_SMALL, 0.8, colours[i], 1, CV_AA);
		  }

	  }
	  catch(Exception& e){

	  }
	}
	records.push_back(record);
}
