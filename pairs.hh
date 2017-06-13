#ifndef PAIRS_HH
#define PAIRS_HH

#include <algorithm>
#include <map>
#include <mutex>
#include <queue>
#include <random>
#include <set>
#include <vector>

#include "features.hh"

struct KeypointPairs
{
	using KeypointPair = std::pair<int, int>;

	std::vector<KeypointPair> pairs;
	KeypointPairs(const Features& f1, const Features& f2);

	void save(std::string name, const Features& f1, const Features& f2);

	void neighborhood(const Features& f1, const Features& f2, int threads, int points, double accept);
	void ransac(const Features& f1, const Features& f2, int iter, double maxerror, bool force);
	void ransac_p(const Features& f1, const Features& f2, int iter, double maxerror, bool force);

private:

	std::random_device rd;
	std::mt19937 engine{ rd() };

	std::mutex mutex_queue, mutex_result;
	std::queue<KeypointPair> queue;
	std::vector<KeypointPair> result;

	struct KeypointDistance
	{
		KeypointPair p;
		double dist;
	};

	double distance(double x1, double y1, double x2, double y2) const;
	void process_neighborhood(const Features& f1, const Features& f2, int points, double accept);
	std::vector<int> closest(const std::vector<double>& distances, int num) const;

	std::map<int, int> nearest(const Features& f1, const Features& f2) const;
	int distance(const Features::Point& p1, const Features::Point& p2) const;
	void filter(const std::map<int, int>& nearest1, const std::map<int, int>& nearest2);
	
	
};

#endif
