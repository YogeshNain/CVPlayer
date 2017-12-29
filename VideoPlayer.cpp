#include "stdafx.h"
#include "VideoPlayer.h"

namespace OpenCVPlayer
{

	CVideoPlayer::CVideoPlayer()
	{
		
	}


	CVideoPlayer::~CVideoPlayer()
	{
	}

	cv::Mat CVideoPlayer::getRGBFrame()
	{
		cv::Mat frame;
		videoPlayer.read(frame);
		return frame;
	}

	cv::Mat CVideoPlayer::getGrayFrame()
	{
		cv::Mat frame;
		videoPlayer.read(frame);
		if (frame.channels() > 2)
			cv::cvtColor(frame, frame, CV_BGR2GRAY);
		return frame;
	}

	cv::Mat CVideoPlayer::getHSVFrame()
	{
		cv::Mat frame;
		videoPlayer.read(frame);
		if (frame.channels() > 2)
			cv::cvtColor(frame, frame, CV_BGR2HSV);
		return frame;
	}

	cv::Mat CVideoPlayer::getYurFrame()
	{
		cv::Mat frame;
		videoPlayer.read(frame);
		if (frame.channels() > 2)
			cv::cvtColor(frame, frame, CV_BGR2YCrCb);
		return frame;
	}

	int CVideoPlayer::setStreamPath(std::string streamPath)
	{
		videoPlayer.open(streamPath);
		if (!videoPlayer.isOpened())
		{
			return -1;
		}
		else
		{
			FPS = (int)videoPlayer.get(CV_CAP_PROP_FPS);
			return 0;
		}
	}

}