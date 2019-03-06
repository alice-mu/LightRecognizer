#include "pch.h"
#include <iostream>
#include <opencv2/opencv.hpp>  
#include <opencv/cv.h>  
#include <opencv2/core/core.hpp>  
#include <opencv2/highgui/highgui.hpp>  
#include <opencv2/ml/ml.hpp>  

using namespace std;
using namespace cv;
using namespace cv::ml;

int main()
{
	Mat img = imread("normal.jpg");
	Mat img2;
	img.copyTo(img2);

	cvtColor(img2, img2, COLOR_RGB2HSV);
	int minhueG = 40; int maxhueG = 75;
	int minsatG = 60; int maxsatG = 255;
	int minvalG = 80; int maxvalG = 255;

	Mat colourFound;
	inRange(img2, Scalar(minhueG, minsatG, minvalG), Scalar(maxhueG, maxsatG, maxvalG), colourFound);
	Mat dilateElement = getStructuringElement(MORPH_ELLIPSE, Size(2, 2));
	dilate(colourFound, colourFound, dilateElement, Point(-1, -1), 3);

	Mat colourFound2;
	bitwise_and(img2, img2, colourFound2, colourFound);

	vector<Mat> contours;
	findContours(colourFound, contours, RETR_EXTERNAL, CHAIN_APPROX_NONE);

	vector<Rect> boundRects(contours.size());
	for (int i = 0; i < contours.size(); i++) {
		boundRects[i] = boundingRect(contours.at(i));
	}


	vector<Rect>::iterator currRect = boundRects.begin();
	while (currRect != boundRects.end()) {
		int w = currRect->width;
		int h = currRect->height;

		bool notSquare = abs(w - h) > 0.25 * min(w, h);
		bool tooSmall = w * h < img.cols*img.rows / 20000;
		bool notInRoi = currRect->tl().y > img.rows * (0.80);
		bool touchingEdge = currRect->tl().x < 2 || currRect->tl().y < 2 ||
			currRect->br().x > img.cols - 2 || currRect->br().y > img.rows - 2;

		if ( notSquare || tooSmall || notInRoi || touchingEdge)
			currRect = boundRects.erase(currRect);
		else
			currRect++;
	}


	Mat final;
	img.copyTo(final);
	Mat colourFound3;
	colourFound2.copyTo(colourFound3);
	for (int i = 0; i < boundRects.size(); i++)
	{
		rectangle(final, boundRects[i].tl(), boundRects[i].br(), Scalar(0, 255, 255), 2, 8, 0);
	}
	
	imshow("", final);
	waitKey();

	return 0;
}
