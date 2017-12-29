#pragma once

#include <opencv2\opencv.hpp>

namespace OpenCVPlayer
{

	class CVideoPlayer
	{
	public:
		CVideoPlayer();
		~CVideoPlayer();
		cv::Mat getRGBFrame();
		cv::Mat getGrayFrame();
		cv::Mat getHSVFrame();
		cv::Mat getYurFrame();
		int setStreamPath(std::string streamPath);
		int FPS = 0;
	private:
		cv::VideoCapture videoPlayer;

	};

}