#include "stdafx.h"
#include "PlateAnalysis.h"
#define SC 1

CPlateAnalysis::CPlateAnalysis()
{
	maxCharHeight = 0.8;
	minCharHeight = 0.27;
}


CPlateAnalysis::~CPlateAnalysis()
{
}

void CPlateAnalysis::processPlate(cv::Mat input, cv::Mat & outPut, std::vector<cv::Mat>& charImages)
{
	cv::Mat gray, angleImage;;
	input.copyTo(gray);

	if (gray.channels() > 2)
		cv::cvtColor(gray, gray, CV_BGR2GRAY);

	plateSize = gray.size();

	//cv::medianBlur(gray, gray, 3);
	gray.copyTo(angleImage);
	PreProcessFrame(angleImage);
	double angle = 0;
	GetRotationAngle(angleImage, &angle);
	if (SC)
		std::cout << "\nAngle : " << angle;
	if (angle > 20 || angle < -20)
		return;
	if (angle != 0)
		RotateByAngle(gray, angle* CV_PI / 180);
	cv::Mat pos;
	getPlateLocation(gray, pos);
	if (pos.data)
		pos.copyTo(gray);

	cv::Mat binaryImage;
	generateThreshold(gray, binaryImage);
	cv::copyMakeBorder(binaryImage, binaryImage, 2, 2, 2, 2, cv::BORDER_CONSTANT, cv::Scalar(255, 255, 255));
	cv::Mat grayRGB;
	gray.copyTo(grayRGB);
	if (grayRGB.channels() < 2)
		cv::cvtColor(grayRGB, grayRGB, cv::COLOR_GRAY2BGR);

	cv::copyMakeBorder(grayRGB, grayRGB, 2, 2, 2, 2, cv::BORDER_CONSTANT, cv::Scalar(255, 255, 255));
	cv::Mat textSymbl;
	grayRGB.copyTo(textSymbl);
	if (SC)
	{
#ifdef _IMSHOW
		cv::imshow("Gray", gray);
		cv::imshow("Thresh", binaryImage);
#endif // _IMSHOW
	}
	std::vector< std::vector< cv::Point > > characterCountors, bboxChars;
	std::vector<cv::Vec4i> herachy;
	std::vector<SymbolDetail> symbolBox;
	std::vector<int> vecHeight, vecWidth, vectyCord;
	cv::Mat ccImage;
	binaryImage.copyTo(ccImage);

	cv::findContours(ccImage, characterCountors, herachy, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE);

	cv::Mat rgbBinary;

	cv::cvtColor(binaryImage, rgbBinary, cv::COLOR_GRAY2BGR);
	cv::Mat clearMat = cv::Mat(binaryImage.size(), CV_8UC1, cv::Scalar(255, 255, 255));
	cv::Mat segmentMat = cv::Mat(binaryImage.size(), CV_8UC1, cv::Scalar(255, 255, 255));
	cv::Mat clearMatRGB = cv::Mat(binaryImage.size(), CV_8UC1, cv::Scalar(255, 255, 255));
	for (unsigned int i = 0; i < characterCountors.size(); ++i)
	{
		cv::Vec4i parentId = herachy[i];
		int pId = parentId[3];
		if (pId != -1)
		{
			if (verifyOneLineChars(characterCountors[pId]))
			{
				continue;
			}
		}
		if (verifyOneLineChars(characterCountors[i]))
		{
			SymbolDetail d;
			cv::Rect r = cv::boundingRect(characterCountors[i]);
			d.countors = characterCountors[i];
			d.bbox = r;
			d.original = r;
			symbolBox.push_back(d);

			bboxChars.push_back(characterCountors[i]);
			vecHeight.push_back(r.height);
			vecWidth.push_back(r.width);
			vectyCord.push_back(r.y);
			cv::Mat b = cv::Mat(binaryImage, r);
			b.copyTo(clearMatRGB(r));
			cv::Mat a = cv::Mat(grayRGB, r);
			cv::cvtColor(a, a, cv::COLOR_BGR2GRAY);
			cv::threshold(a, a, 200, 255, cv::THRESH_OTSU | cv::THRESH_BINARY);
			a.copyTo(clearMat(r));
			if(SC)
			cv::rectangle(rgbBinary, r, cv::Scalar(0, 0, 255));
		}
		else
		{
			cv::Rect r = cv::boundingRect(characterCountors[i]);
			if(SC)
			cv::rectangle(rgbBinary, r, cv::Scalar(255, 0, 0));
		}
	}
	if (SC)
	{
#ifdef _IMSHOW
		cv::imshow("Rect", rgbBinary);
		cv::imshow("Text", clearMat);
		cv::imshow("TextRGB", clearMatRGB);
#endif // _IMSHOW
	}
	if (symbolBox.size() <= 0)
		return;
	std::sort(symbolBox.begin(), symbolBox.end(), sortboxdX);
	cv::Scalar meanHeight, SDHeight, meanWidth, SDWidth;

	cv::meanStdDev(vecHeight, meanHeight, SDHeight);
	cv::meanStdDev(vecWidth, meanWidth, SDWidth);
	int idx = vectyCord.size() / 2;
	if (idx % 2 == 0)
		idx -= 1;
	if (idx > (int)vectyCord.size() || idx < 0)
		idx = 0;
	int medianY = vectyCord[idx];
	int medianH = symbolBox[idx].bbox.height;
	if (SC)
	{

	std::cout << "\nMean Height : " << meanHeight[0] << " SD : " << SDHeight[0];
	std::cout << "\nMean Width : " << meanWidth[0] << " SD : " << SDWidth[0];
	std::cout << "\nMedian Y : " << medianY;
	std::cout << "\nMedian H : " << medianH;
	}
	int ypt = 10000;
	int ypt2 = -100;
	cv::Mat grayLine, grayText;
	gray.copyTo(grayLine);
	gray.copyTo(grayText);
	cv::cvtColor(grayLine, grayLine, cv::COLOR_GRAY2BGR);
	//cv::resize(grayLine, grayLine, binaryImage.size());
	cv::copyMakeBorder(grayLine, grayLine, 2, 2, 2, 2, cv::BORDER_CONSTANT, cv::Scalar(255, 255, 255));
	cv::Mat segment;
	grayLine.copyTo(segment);
	cv::copyMakeBorder(grayText, grayText, 2, 2, 2, 2, cv::BORDER_CONSTANT, cv::Scalar(255, 255, 255));
	int spaceing = 0, lastpoint = 0;
	std::vector<SymbolDetail> symbols;
	for (unsigned t = 0; t < symbolBox.size(); ++t)
	{
		//cv::RotatedRect rbox = cv::minAreaRect(symbolBox[t].countors);
		int bx = symbolBox[t].bbox.x;
		int by = symbolBox[t].bbox.y;
		int height = symbolBox[t].bbox.height;//rbox.size.height;
		int width = symbolBox[t].bbox.width; //rbox.size.width;
		if (SC)
		{
			std::cout << "\nHeight : " << height;
			std::cout << "\tW : " << width;
		}
		if (t != 0)
		{
			spaceing = bx - lastpoint;
			if(SC)
			std::cout << "\tSpace : " << spaceing;
		}
		lastpoint = bx + width;
		//extract Symbols from y to median Height
		//This will crop any connected Char if considered by above filteration
		//and complete height of rectangle!
		symbolBox[t].spacing = spaceing;
		symbolBox[t].meanHeight =(int) std::ceil(meanHeight[0]);
		symbolBox[t].meanWidth =(int) std::ceil(meanWidth[0]);
		symbolBox[t].medianHeight = medianH;
		if (medianH <= (height + 5)|| t>0)
		{	 if(SC)
			std::cout << "\t Y!";
			cv::Rect ch = symbolBox[t].bbox;
			if (symbolBox[t].bbox.y >= medianY - 3)
			{
				symbolBox[t].goodChar = true;
			}
			else
			{	  if(SC)
				std::cout << "\nBad Char!";
				symbolBox[t].goodChar = false;
			}
			if (symbolBox[t].bbox.y > medianY + 5)
			{
				if (SC)
					std::cout << "\nBad Char!";
				symbolBox[t].goodChar = false;
			}
			ch.y = medianY;
			ch.height = medianH;
			symbolBox[t].bbox = ch;

			symbolBox[t].overlap = spaceing;
			if ((width) > meanWidth[0] + 10)
			{	if(SC)
				std::cout << "\nResegment!";
				symbolBox[t].needSegment = true;
			}
			int x_pos = ch.x;
			int y_pos = ch.y;
			int height = ch.height;
			int width = ch.width;

			if (x_pos < 0)
				ch.x = 0;

			if (y_pos < 0)
				ch.y = 0;

			if (y_pos + height > binaryImage.size().height)
			{
				int over_height = y_pos + height - binaryImage.size().height;
				ch.height -= over_height;
			}

			if (x_pos + width > binaryImage.size().width)
			{
				int over_width = x_pos + width - binaryImage.size().width;
				ch.width -= over_width;
			}
			cv::Mat a = cv::Mat(binaryImage, ch);
			a.copyTo(segmentMat(ch));
			a.copyTo(symbolBox[t].img);
			symbols.push_back(symbolBox[t]);
		}
		
		//cv::rectangle(grayLine, ch, cv::Scalar(255, 0, 00));
	//Draw Lines of segmentation.
		if (ypt > by)
			ypt = by;
		if (ypt2 < symbolBox[t].bbox.br().y)
			ypt2 = symbolBox[t].bbox.br().y;
		if (SC)
		{
			cv::Point sp(bx, ypt);
			cv::Point ep(bx, ypt2);
			cv::line(grayLine, sp, ep, cv::Scalar(200, 200, 90));
			cv::Point spacesp(bx + width, ypt);
			cv::Point spaceep(bx + width, ypt2);
			cv::line(grayLine, spacesp, spaceep, cv::Scalar(200, 200, 10));
		}
	}
	if(SC)
			std::cout << "\nCorrect Symbols : " << symbols.size();
	int bl =(int) std::ceil(meanWidth[0]);
	if(SC)
	std::cout << "\nUpper Line : " << ypt << " Lower Line : " << ypt2;
	
	//min max lines.
	int endpt = binaryImage.cols;
	if (SC)
	{
		cv::line(grayLine, cv::Point(0, ypt), cv::Point(endpt, ypt), cv::Scalar(45, 52, 110));
		cv::line(grayLine, cv::Point(0, ypt2), cv::Point(endpt, ypt2), cv::Scalar(45, 52, 110));
		//detection lines.					
		cv::line(grayLine, cv::Point(0, medianY), cv::Point(endpt, medianY), cv::Scalar(200, 52, 110));
		cv::line(grayLine, cv::Point(0, medianH + medianY), cv::Point(endpt, medianH + medianY), cv::Scalar(200, 52, 110));
	}
#ifdef _IMSHOW
		cv::imshow("TextLine", grayLine);
		cv::imshow("SegmentMat", segmentMat);
#endif // _IMSHOW
	


}

void CPlateAnalysis::PreProcessFrame(cv::Mat & im)
{
	cv::Mat thresh;
	thresh = im.clone();
	if (thresh.channels() > 2)
		cv::cvtColor(thresh, thresh, cv::ColorConversionCodes::COLOR_BGR2GRAY);

	adaptiveThreshold(thresh, thresh, 255, CV_ADAPTIVE_THRESH_GAUSSIAN_C, cv::THRESH_BINARY, 15, -2);

	cv::Mat ret = cv::Mat::zeros(im.size(), CV_8UC1);

	for (int x = 1; x < thresh.cols - 1; x++)
	{
		for (int y = 1; y < thresh.rows - 1; y++)
		{
			bool toprowblack = thresh.at<uchar>(y - 1, x) == 0 || thresh.at<uchar>(y - 1, x - 1) == 0 || thresh.at<uchar>(y - 1, x + 1) == 0;
			bool belowrowblack = thresh.at<uchar>(y + 1, x) == 0 || thresh.at<uchar>(y + 1, x - 1) == 0 || thresh.at<uchar>(y + 1, x + 1) == 0;


			uchar pix = thresh.at<uchar>(y, x);
			if ((!toprowblack && pix == 255 && belowrowblack))
			{
				ret.at<uchar>(y, x) = 255;
			}
		}
	}
	thresh.release();
	if (im.type() != CV_8UC1)
		im.convertTo(im, CV_8UC1);
	ret.copyTo(im);
	//cv::imshow("Hogrot", im);
}

void CPlateAnalysis::GetRotationAngle(cv::Mat & im, double * skew)
{
	if (!im.data)
	{
		skew = 0;
		return;
	}

	double max_r = sqrt(pow(.5*im.cols, 2) + pow(.5*im.rows, 2));
	int angleBins = 180;
	cv::Mat acc = cv::Mat::zeros(cv::Size(2 * (int)max_r, angleBins), CV_32SC1);
	int cenx = im.cols / 2;
	int ceny = im.rows / 2;
	for (int x = 1; x < im.cols - 1; x++)
	{
		for (int y = 1; y < im.rows - 1; y++)
		{
			if (im.at<uchar>(y, x) == 255)
			{
				for (int t = 0; t < angleBins; t++)
				{
					double r = (x - cenx)*cos((double)t / angleBins*CV_PI) + (y - ceny)*sin((double)t / angleBins*CV_PI);
					r += max_r;
					if (int(r) >= acc.cols || int(r) < 0)
					{
						continue;
					}
					acc.at<int>(t, int(r))++;
				}
			}
		}
	}
	cv::Mat thresh;
	normalize(acc, acc, 255, 0, cv::NORM_MINMAX);
	convertScaleAbs(acc, acc);
	cv::Point maxLoc;
	minMaxLoc(acc, 0, 0, 0, &maxLoc);
	//cv::imshow("Hog", acc);
	double theta = (double)maxLoc.y / angleBins*CV_PI;
	double rho = maxLoc.x - max_r;
	if (abs(sin(theta)) < 0.000001)//check vertical
	{
		double m = -cos(theta) / sin(theta);
		cv::Point2d p1 = cv::Point2d(rho + im.cols / 2, 0);
		cv::Point2d p2 = cv::Point2d(rho + im.cols / 2, im.rows);
		//line(im, p1, p2, cv::Scalar(0, 0, 255), 1);
		*skew = 90;
	}
	else
	{
		double m = -cos(theta) / sin(theta);
		double b = rho / sin(theta) + im.rows / 2. - m*im.cols / 2.;
		cv::Point2d p1 = cv::Point2d(0, b);
		cv::Point2d p2 = cv::Point2d(im.cols, im.cols*m + b);
		//line(im, p1, p2, cv::Scalar(0, 0, 255), 1);
		double skewangle;
		skewangle = p1.x - p2.x > 0 ? (atan2(p1.y - p2.y, p1.x - p2.x)*180. / CV_PI) : (atan2(p2.y - p1.y, p2.x - p1.x)*180. / CV_PI);
		*skew = skewangle;
	}


}

void CPlateAnalysis::RotateByAngle(cv::Mat & im, double thetaRad)
{
	double rskew = thetaRad* CV_PI / 180;
	double nw = abs(sin(thetaRad))*im.rows + abs(cos(thetaRad))*im.cols;
	double nh = abs(cos(thetaRad))*im.rows + abs(sin(thetaRad))*im.cols;
	cv::Mat rot_mat = cv::getRotationMatrix2D(cv::Point2d(nw*.5, nh*.5), thetaRad * 180 / CV_PI, 1);
	cv::Mat pos = cv::Mat::zeros(cv::Size(1, 3), CV_64FC1);
	pos.at<double>(0) = (nw - im.cols)*.5;
	pos.at<double>(1) = (nh - im.rows)*.5;
	cv::Mat res = rot_mat*pos;
	rot_mat.at<double>(0, 2) += res.at<double>(0);
	rot_mat.at<double>(1, 2) += res.at<double>(1);
	cv::Mat dst;
	cv::warpAffine(im, dst, rot_mat, cv::Size((int)nw, (int)nh), cv::INTER_LANCZOS4, 0, cv::Scalar(0, 0, 0));
	//cv::warpAffine(im, im, rot_mat, cv::Size(), cv::INTER_LANCZOS4, 0, cv::Scalar(255, 255, 255));
	dst.copyTo(im);

}

bool CPlateAnalysis::sortRectboxX(const cv::Rect & a, const cv::Rect & b)
{
	return a.x < b.x;
}

bool CPlateAnalysis::sortboxdX(const SymbolDetail & a, const SymbolDetail & b)
{
	return a.bbox.x < b.bbox.x;
}

bool CPlateAnalysis::sortRectboxY(const cv::Rect & a, const cv::Rect & b)
{
	return a.y < b.y;
}

void CPlateAnalysis::generateThreshold(cv::Mat gray, cv::Mat &_threshold)
{
	cv::Mat thresh;
	gray.copyTo(thresh);
	if (thresh.channels() > 2)
		cv::cvtColor(thresh, thresh, CV_BGR2GRAY);
	cv::adaptiveThreshold(thresh, thresh, 255, cv::ADAPTIVE_THRESH_GAUSSIAN_C, CV_THRESH_BINARY, 27, 5);
	cv::Mat SE = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(2, 2));
	cv::Mat morp;
	cv::morphologyEx(thresh, morp, cv::MORPH_CLOSE, SE);
	morp.copyTo(_threshold);

}

void CPlateAnalysis::findThreshValue(cv::Mat gray)
{
	cv::Scalar mean, sd;

	cv::meanStdDev(gray, mean, sd);

	int i_mean = (int)mean[0];
	int i_sd = (int)sd[0];
	int i_block = 27, i_const = 6;
	if (i_sd >= 30 && i_sd <= 45)
		i_const = 8;
	if (i_sd > 45 && i_sd <= 70)
		i_const = 13;
	if (i_sd > 70 && i_sd <= 100)
		i_const = 10;

	if (i_mean >= 80 && i_mean <= 120)
	{
		if (i_sd < 40)
			i_const = 18;
		if (i_sd < 30)
			i_const = 4;

	}
	if (i_mean >= 140 && i_sd >= 50)
	{
		i_const = 24;
	}

	if (i_mean >= 150 && i_sd >= 45)
	{
		i_const = 8;
		if (i_sd > 60)
			i_const = 31;
	}

	if (i_mean <= 80 && i_sd <= 30)
	{
		i_const = 8;
	}

	blockSize = 27;
	cellSize = i_const;

}

void CPlateAnalysis::getThreshValue(int & block, int & cell)
{
	block = blockSize;
	cell = cellSize;
}

void CPlateAnalysis::getPlateLocation(cv::Mat gray, cv::Mat & pos)
{
	cv::Mat plateLoc;
	gray.copyTo(plateLoc);

	if (plateLoc.channels() > 2)
	{
		cv::cvtColor(plateLoc, plateLoc, CV_BGR2GRAY);
	}
	cv::threshold(plateLoc, plateLoc, 180, 255, CV_THRESH_BINARY + CV_THRESH_OTSU);
	//cv::adaptiveThreshold(plateLoc, plateLoc, 255, cv::ADAPTIVE_THRESH_GAUSSIAN_C, cv::THRESH_BINARY, 11, 0);
	cv::copyMakeBorder(plateLoc, plateLoc, 2, 2, 2, 2, cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));
	cv::resize(gray, gray, cv::Size(gray.size().width + 4, gray.size().height + 4));
	std::vector<std::vector<cv::Point>> Plloc;
	cv::Mat ctrim;
	plateLoc.copyTo(ctrim);
	cv::findContours(ctrim, Plloc, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);

#ifdef _IMSHOW
	cv::Mat rgbTh;
	plateLoc.copyTo(rgbTh);
	cv::cvtColor(rgbTh, rgbTh, CV_GRAY2BGR);
#endif // _IMSHOW

	int maxheight = 0;
	int maxwidth = 0;
	cv::Rect loc;
	for (unsigned int i = 0; i < Plloc.size(); ++i)
	{
		cv::Rect r = cv::boundingRect(Plloc[i]);
		if (r.height > maxheight && r.width > 25 && r.height > (gray.size().height *0.4))
		{
			loc = r;
			maxheight = r.height;
		}
#ifdef _IMSHOW
		cv::rectangle(rgbTh, r, cv::Scalar(0, 0, 255));
#endif // _IMSHOW

	}

	int x_pos = loc.x;
	int y_pos = loc.y;
	int height = loc.height;
	int width = loc.width;

	if (x_pos < 0)
		loc.x = 0;

	if (y_pos < 0)
		loc.y = 0;

	if (y_pos + height > plateLoc.size().height)
	{
		int over_height = y_pos + height - plateLoc.size().height;
		loc.height -= over_height;
	}

	if (x_pos + width > plateLoc.size().width)
	{
		int over_width = x_pos + width - plateLoc.size().width;
		loc.width -= over_width;
	}

	cv::Mat pl = cv::Mat(gray, loc);
	pl.copyTo(pos);

#ifdef _IMSHOW
	cv::imshow("PlateL", plateLoc);
	cv::rectangle(rgbTh, loc, cv::Scalar(255, 0, 0));
	cv::imshow("PlateR", rgbTh);
#endif // _IMSHOW

}

bool CPlateAnalysis::verifyOneLineChars(std::vector<cv::Point> charContor)
{
	bool isChar = false;

	double area = cv::contourArea(charContor);
	cv::Rect br = cv::boundingRect(charContor);
	cv::RotatedRect rt = cv::minAreaRect(charContor);
	int height = br.height;
	int width = br.width;

	if (height == 0)
	{
		return isChar;
		height = 1;
	}
	if (width == 0)
	{
		return isChar;
		width = 1;
	}
	double AR = 0;

	if (rt.angle > 90)
	{
		int temp = height;
		height = width;
		width = temp;
	}

	AR = width / (double)height;

	double maxer = maxCharHeight;
	double miner = minCharHeight;
	int error = 5;
	int maxHeight = (int)(plateSize.height * maxer) + error;
	int minHeight = (int)(plateSize.height * miner) - error;
	int minWidth = 3;
	if ((area > 50) & (area < (plateSize.area() * 0.90)))
	{
		if (AR >= 0.02 && AR < 1.4)
		{
			if ((height < maxHeight) & (height > minHeight) & (width > minWidth))
			{
				isChar = true;
			}
		}
	}
	return isChar;
}
  /*
double CPlateAnalysis::calcLocalStats(cv::Mat &im, cv::Mat &map_m, cv::Mat &map_s, int winx, int winy) {
	cv::Mat im_sum, im_sum_sq;
	cv::integral(im, im_sum, im_sum_sq, CV_64F);

	double m, s, max_s, sum, sum_sq;
	int wxh = winx / 2;
	int wyh = winy / 2;
	int x_firstth = wxh;
	int y_lastth = im.rows - wyh - 1;
	int y_firstth = wyh;
	double winarea = winx*winy;

	max_s = 0;
	for (int j = y_firstth; j <= y_lastth; j++) {
		sum = sum_sq = 0;

		sum = im_sum.at<double>(j - wyh + winy, winx) - im_sum.at<double>(j - wyh, winx) - im_sum.at<double>(j - wyh + winy, 0) + im_sum.at<double>(j - wyh, 0);
		sum_sq = im_sum_sq.at<double>(j - wyh + winy, winx) - im_sum_sq.at<double>(j - wyh, winx) - im_sum_sq.at<double>(j - wyh + winy, 0) + im_sum_sq.at<double>(j - wyh, 0);

		m = sum / winarea;
		s = sqrt((sum_sq - m*sum) / winarea);
		if (s > max_s) max_s = s;

		map_m.fset(x_firstth, j, m);
		map_s.fset(x_firstth, j, s);

		// Shift the window, add and remove	new/old values to the histogram
		for (int i = 1; i <= im.cols - winx; i++) {

			// Remove the left old column and add the right new column
			sum -= im_sum.at<double>(j - wyh + winy, i) - im_sum.at<double>(j - wyh, i) - im_sum.at<double>(j - wyh + winy, i - 1) + im_sum.at<double>(j - wyh, i - 1);
			sum += im_sum.at<double>(j - wyh + winy, i + winx) - im_sum.at<double>(j - wyh, i + winx) - im_sum.at<double>(j - wyh + winy, i + winx - 1) + im_sum.at<double>(j - wyh, i + winx - 1);

			sum_sq -= im_sum_sq.at<double>(j - wyh + winy, i) - im_sum_sq.at<double>(j - wyh, i) - im_sum_sq.at<double>(j - wyh + winy, i - 1) + im_sum_sq.at<double>(j - wyh, i - 1);
			sum_sq += im_sum_sq.at<double>(j - wyh + winy, i + winx) - im_sum_sq.at<double>(j - wyh, i + winx) - im_sum_sq.at<double>(j - wyh + winy, i + winx - 1) + im_sum_sq.at<double>(j - wyh, i + winx - 1);

			m = sum / winarea;
			s = sqrt((sum_sq - m*sum) / winarea);
			if (s > max_s) max_s = s;

			map_m.fset(i + wxh, j, m);
			map_s.fset(i + wxh, j, s);
		}
	}

	return max_s;
}

void CPlateAnalysis::NiblackSauvolaWolfJolion(cv::Mat im, cv::Mat output, int version, int winx, int winy, double k, double dR)
{
	double m, s, max_s;
	double th = 0;
	double min_I, max_I;
	int wxh = winx / 2;
	int wyh = winy / 2;
	int x_firstth = wxh;
	int x_lastth = im.cols - wxh - 1;
	int y_lastth = im.rows - wyh - 1;
	int y_firstth = wyh;
	int mx, my;

	// Create local statistics and store them in a double matrices
	cv::Mat map_m = cv::Mat::zeros(im.rows, im.cols, CV_32F);
	cv::Mat map_s = cv::Mat::zeros(im.rows, im.cols, CV_32F);
	max_s = calcLocalStats(im, map_m, map_s, winx, winy);

	cv::minMaxLoc(im, &min_I, &max_I);

	cv::Mat thsurf(im.rows, im.cols, CV_32F);

	// Create the threshold surface, including border processing
	// ----------------------------------------------------


	for (int j = y_firstth; j <= y_lastth; j++) {

		// NORMAL, NON-BORDER AREA IN THE MIDDLE OF THE WINDOW:
		for (int i = 0; i <= im.cols - winx; i++) {

			m = map_m.fget(i + wxh, j);
			s = map_s.fget(i + wxh, j);

			// Calculate the threshold
			switch (version) {

			case NIBLACK:
				th = m + k*s;
				break;

			case SAUVOLA:
				th = m * (1 + k*(s / dR - 1));
				break;

			case WOLFJOLION:
				th = m + k * (s / max_s - 1) * (m - min_I);
				break;

			default:
				th = m + k*s;
			}

			thsurf.fset(i + wxh, j, th);

			if (i == 0) {
				// LEFT BORDER
				for (int i = 0; i <= x_firstth; ++i)
					thsurf.fset(i, j, th);

				// LEFT-UPPER CORNER
				if (j == y_firstth)
					for (int u = 0; u < y_firstth; ++u)
						for (int i = 0; i <= x_firstth; ++i)
							thsurf.fset(i, u, th);

				// LEFT-LOWER CORNER
				if (j == y_lastth)
					for (int u = y_lastth + 1; u < im.rows; ++u)
						for (int i = 0; i <= x_firstth; ++i)
							thsurf.fset(i, u, th);
			}

			// UPPER BORDER
			if (j == y_firstth)
				for (int u = 0; u < y_firstth; ++u)
					thsurf.fset(i + wxh, u, th);

			// LOWER BORDER
			if (j == y_lastth)
				for (int u = y_lastth + 1; u < im.rows; ++u)
					thsurf.fset(i + wxh, u, th);
		}

		// RIGHT BORDER
		for (int i = x_lastth; i < im.cols; ++i)
			thsurf.fset(i, j, th);

		// RIGHT-UPPER CORNER
		if (j == y_firstth)
			for (int u = 0; u < y_firstth; ++u)
				for (int i = x_lastth; i < im.cols; ++i)
					thsurf.fset(i, u, th);

		// RIGHT-LOWER CORNER
		if (j == y_lastth)
			for (int u = y_lastth + 1; u < im.rows; ++u)
				for (int i = x_lastth; i < im.cols; ++i)
					thsurf.fset(i, u, th);
	}

	for (int y = 0; y < im.rows; ++y)
		for (int x = 0; x < im.cols; ++x)
		{
			if (im.uget(x, y) >= thsurf.fget(x, y))
			{
				output.uset(x, y, 255);
			}
			else
			{
				output.uset(x, y, 0);
			}
		}
}

int CPlateAnalysis::computeSW(IplImage *image)
{

	IplImage *binImage = cvCreateImage(cvSize(image->width, image->height), IPL_DEPTH_8U, 1);
	int *distanceHistogram = (int*)malloc(image->height * sizeof(int));
	int distance, maxBin = -1, maxDistance;
	int beginEdge, endEdge, pixel;

	// Compute the edge pixels
	cvSobel(image, binImage, 1, 1);
	// Binarize the image
	cvThreshold(binImage, binImage, 32, 255, CV_THRESH_BINARY);
	// Initialize the histogram
	for (int bin = 0; bin < binImage->height; bin++)
		distanceHistogram[bin] = 0;
	// Compute the distance histogram (histogram of the distances between two successive edge pixels)
	for (int row = 0; row < binImage->height; row++) {
		beginEdge = -1;
		endEdge = -1;
		for (int col = 0; col < binImage->width; col++) {
			pixel = row * binImage->width + col;
			if ((unsigned char)binImage->imageData[pixel] == 255 && beginEdge == -1) {
				beginEdge = pixel;
			}
			else if ((unsigned char)binImage->imageData[pixel] == 255 && beginEdge != -1) {
				endEdge = pixel;
				distance = endEdge - beginEdge;
				distanceHistogram[distance]++;
				beginEdge = endEdge;
			}
		}
	}
	for (int bin = 2; bin < binImage->height; bin++)
		if (distanceHistogram[bin] > maxBin) {
			maxBin = distanceHistogram[bin];
			maxDistance = bin;
		}

	cvReleaseImage(&binImage);
	delete[] distanceHistogram;
	return maxDistance;
}
*/