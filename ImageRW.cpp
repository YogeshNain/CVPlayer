#include "stdafx.h"
#include "ImageRW.h"


CImageRW::CImageRW()
{
}


CImageRW::~CImageRW()
{
}

void CImageRW::readPath(std::string filePath)
{
	checkPath(filePath);
	if (!bFolder)
	{
		imreadPath = filePath;
	}
	else
	{
		readFolder(filePath);
	}
}

void CImageRW::writePath(std::string filePath)
{
}

cv::Mat CImageRW::getRGBImage()
{
	cv::Mat image;
	if (bFolder)
		setpath();
	std::cout << "\nReading Path : " << imreadPath;
	image = cv::imread(imreadPath, 1);

	return image;
}

cv::Mat CImageRW::getGrayImage()
{
	cv::Mat image;
	if (bFolder)
		setpath();
	std::cout << "\nReading Path : " << imreadPath;
	image = cv::imread(imreadPath, 0);
	return image;
}

void CImageRW::readFolder(std::string folderPath)
{
	try
	{
		cv::glob(folderPath + "\\*.png", jpgFiles);
		cv::glob(folderPath + "\\*.jpg", pngFiles);
		cv::glob(folderPath + "\\*.bmp", bmpFiles);
	}
	catch (std::exception ex)
	{
		std::cout << "\nCan not read files in Folder : " << folderPath;
	}
	std::cout << "\nReading JPG!";
	for (int i = 0; i < (int)jpgFiles.size(); ++i)
	{
		fileNames.push_back(jpgFiles[i]);
	}
	std::cout << "\nReading PNG!";
	for (int i = 0; i < (int)pngFiles.size(); ++i)
	{
		fileNames.push_back(pngFiles[i]);
	}
	std::cout << "\nReading BMP!";
	for (int i = 0; i < (int)bmpFiles.size(); ++i)
	{
		fileNames.push_back(bmpFiles[i]);
	}
	totalFiles = (int)fileNames.size();
}

bool CImageRW::checkPath(std::string filePath)
{
	size_t pos = filePath.find_last_of(".");
	std::string fName;

	if (pos != std::string::npos)
	{
		fName = filePath.substr(pos + 1);
		if (fName == "jpg" || fName == "jpeg" || fName == "png" || fName == "tiff" || fName == "bmp" || fName == "tif")
		{
			bFolder = false;
		}
		else
		{
			bFolder = true;
		}
	}

	return bFolder;
}

void CImageRW::setpath()
{
	imreadPath = fileNames[currentIndex];
	++currentIndex;
}
