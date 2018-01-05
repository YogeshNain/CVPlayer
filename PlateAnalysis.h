#pragma once
#include <opencv2\opencv.hpp>
class CPlateAnalysis
{
	/*
#define uget(x,y)    at<unsigned char>(y,x)
#define uset(x,y,v)  at<unsigned char>(y,x)=v;
#define fget(x,y)    at<float>(y,x)
#define fset(x,y,v)  at<float>(y,x)=v;
*/
	struct SymbolDetail
	{
		std::vector<cv::Point> countors;
		cv::Rect bbox;
		bool goodChar;
		int overlap;
		bool needSegment;
		cv::Rect original;
		int meanHeight;
		int medianHeight;
		int meanWidth;
		int spacing;
		cv::Mat img;
	};
public:
	CPlateAnalysis();
	~CPlateAnalysis();
	void processPlate(cv::Mat input, cv::Mat &outPut, std::vector<cv::Mat> &charImages);
	double maxCharHeight, minCharHeight;
	cv::Size plateSize;
	int blockSize = 25;
	int cellSize = 15;
private:
	void PreProcessFrame(cv::Mat & im);
	void GetRotationAngle(cv::Mat & im, double * skew);
	void RotateByAngle(cv::Mat & im, double thetaRad);
	static bool sortRectboxX(const cv::Rect & a, const cv::Rect & b);
	static bool sortboxdX(const SymbolDetail & a, const SymbolDetail & b);
	static bool sortRectboxY(const cv::Rect & a, const cv::Rect & b);
	void generateThreshold(cv::Mat gray,cv::Mat &);
	void findThreshValue(cv::Mat gray);
	void getThreshValue(int &block, int &cell);

	void getPlateLocation(cv::Mat gray, cv::Mat & pos);

	bool verifyOneLineChars(std::vector<cv::Point> charContor);
/*
	double calcLocalStats(cv::Mat & im, cv::Mat & map_m, cv::Mat & map_s, int winx, int winy);
	void NiblackSauvolaWolfJolion(cv::Mat im, cv::Mat output, int version, int winx, int winy, double k, double dR);
	int computeSW(IplImage * image);
	enum CustomeThres
	{
		NIBLACK = 0,
		SAUVOLA,
		WOLFJOLION,
	};
	*/
	

	
};

