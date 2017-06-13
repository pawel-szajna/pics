#ifndef FEATURES_HH
#define FEATURES_HH

#include <fstream>
#include <iostream>
#include <vector>

struct Features
{
	struct Point
	{
		double x, y;
		std::vector<int> features;
	};

	int feats, count;
	std::vector<Point> points;
	
	Features(std::string input);
};

#endif
