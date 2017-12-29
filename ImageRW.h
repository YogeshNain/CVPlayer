#pragma once
#include <opencv2\opencv.hpp>
class CImageRW
{
public:
	CImageRW();
	~CImageRW();
	void readPath(std::string);
	void writePath(std::string);
	cv::Mat getRGBImage();
	cv::Mat getGrayImage();
	int totalFiles =0;
private :
	std::vector<cv::String> jpgFiles, pngFiles, bmpFiles;
	bool bFolder = true;
	std::vector<cv::String>fileNames;
	std::string imreadPath;
	std::string imwritePath;
	int currentIndex=0;

	void readFolder(std::string);
	bool checkPath(std::string path);
	void setpath();
};

