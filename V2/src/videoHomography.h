#ifndef VID_HOM_H
#define VID_HOM_H

#include <opencv2/opencv.hpp>
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/xfeatures2d.hpp"

#include "opencv2/core/core.hpp"



#include <iostream>
#include <iomanip>
#include <list>
#include <vector>
#include <stdio.h>
#include <stdlib.h>

#define TOP 0
#define BOTTOM 1
#define LEFT 2
#define RIGHT 3
#define UNKNOWN 4

#define LOW_EDGE 60

using namespace cv::xfeatures2d;
using namespace cv;
using namespace std;

class VideoHomography{
    public:
        VideoHomography(cv::VideoCapture cap, std::vector<cv::Mat> obj);

        int run();

        cv::Mat windowedMatchingMask( const vector<cv::KeyPoint>& keypoints1, const vector<cv::KeyPoint>& keypoints2,
                                         float maxDeltaX, float maxDeltaY );

        bool isNumeric(const std::string& str);

        std::vector<vector<int> > records;

        int frame_number;
    private:
        cv::Mat K;

        std::vector<cv::Scalar> colours;
        std::vector<std::string> names;
        std::vector<int> near;
        std::vector<int> count;



        cv::VideoCapture capture;
        std::vector<cv::Mat> objects;

        cv::Mat determineK();


        void help();

        void drawRectOnFrame(cv::Mat img_scene, std::vector<cv::Mat> img_objects);

        void drawMatchesRelative(const std::vector<cv::KeyPoint>& train, 
            const std::vector<cv::KeyPoint>& query, 
            std::vector<cv::DMatch>& matches, cv::Mat& img, 
            const std::vector<unsigned char>& mask = std::vector<unsigned char>());

        void keypoints2points(const std::vector<cv::KeyPoint>& in, 
            std::vector<cv::Point2f>& out);

        void points2keypoints(const std::vector<cv::Point2f>& in, 
            std::vector<cv::KeyPoint>& out);

        void warpKeypoints(const cv::Mat& H, const std::vector<cv::KeyPoint>& in, 
            std::vector<cv::KeyPoint>& out);

        void matches2points(const std::vector<cv::KeyPoint>& train, 
            const std::vector<cv::KeyPoint>& query, const std::vector<cv::DMatch>& matches, 
            std::vector<cv::Point2f>& pts_train, std::vector<cv::Point2f>& pts_query);

        void resetH(cv::Mat& H);

        void poseFromH(cv::Mat H);

        void printPose(cv::Mat R, cv::Mat t, cv::Mat n);
};
#endif


