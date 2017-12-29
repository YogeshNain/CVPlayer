// Segmentation.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include <opencv2\opencv.hpp>
#include "VideoPlayer.h"
#include "ImageRW.h"
using namespace OpenCVPlayer;

int Video()
{
	std::string path;
	
	std::cout << "\nEnter video path : ";
	std::cin >> path;

	
	CVideoPlayer player;
	int res = player.setStreamPath(path);
	std::cout << "\nFPS : " << player.FPS;
	double delay = 1/(double)player.FPS;
	delay = delay * 1000;
	if (res !=-1)
		while (1)
		{
			cv::Mat frame = player.getRGBFrame();
			if (frame.size().width > 1280)
				cv::resize(frame, frame, cv::Size(1280, 720));


			cv::imshow("Video", frame);
			char key = cv::waitKey((int)delay);
			if (key == 'p')
				key = cv::waitKey();
			if (key == 'q')
				break;

		}

	
    return 0;
}

int ImageProcessing()
{
	std::cout << "\nEnter Path : ";
	std::string path;
	std::cin >> path;

	CImageRW imreader;
	imreader.readPath(path);
	int i = 0;
	do
	{
		cv::Mat image = imreader.getRGBImage();
		cv::imshow("Picture", image);
		char key = cv::waitKey();
		if (key == 'q')
			break;
		++i;
	} while (i < imreader.totalFiles);
	
	return 0;
}

int main()
{
	cv::setNumThreads(1);
	Video();
}
