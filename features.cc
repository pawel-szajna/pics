#include "features.hh"
#include "logger.hh"

Features::Features(std::string input)
{
	double tmp;
	date();
	std::cout << "Loading file " << input << "..." << std::endl;
	std::ifstream ifs(input, std::ifstream::in);
	ifs >> feats >> count;
	points.reserve(count);
	for (int i = 0; i < count; ++i) {
		Point point;
		ifs >> point.x >> point.y;
		point.features.resize(feats);
		ifs >> tmp >> tmp >> tmp; // ignore 3 doubles
		for (int j = 0; j < feats; ++j) {
			ifs >> point.features[j];
		}
		points.push_back(point);
	}
	ifs.close();
}
