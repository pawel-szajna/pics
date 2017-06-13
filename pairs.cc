#include <cmath>
#include <future>
#include <iostream>
#include <iterator>
#include <limits>

#include <Eigen/Dense>

#include "pairs.hh"
#include "logger.hh"

using namespace Eigen;

KeypointPairs::KeypointPairs(const Features& f1, const Features& f2)
{
	date();
	rlutil::setColor(rlutil::GREEN);
	std::cout << "Keypoint pair detection started" << std::endl;
	rlutil::setColor(rlutil::GREY);

	logOut << "Computing nearest neighbors" << std::endl;
		
	auto future1 = std::async(std::launch::async, &KeypointPairs::nearest, this, f1, f2);
	auto future2 = std::async(std::launch::async, &KeypointPairs::nearest, this, f2, f1);

	auto nearest1 = future1.get();
	auto nearest2 = future2.get();

	logOut << "Nearest neighbor computation finished, filtering" << std::endl;
	
	filter(nearest1, nearest2);

	logOut << "After filtering got " << pairs.size() << " keypoint pairs" << std::endl;
}

#ifdef max
#undef max
#endif

void KeypointPairs::filter(const std::map<int, int>& nearest1, const std::map<int, int>& nearest2)
{
	for (const auto& n : nearest1) {
		if (nearest2.at(n.second) == n.first) {
			pairs.push_back({ n.first, n.second });
		}
	}
}

std::map<int, int> KeypointPairs::nearest(const Features& f1, const Features& f2) const
{
	std::map<int, int> points;

	int i1 = 0;
	for (const auto& p1 : f1.points) {
		int min = std::numeric_limits<int>::max(), i2 = 0, imin = 0;
		for (const auto& p2 : f2.points) {
			int dist = distance(p1, p2);
			if (dist < min) {
				min = dist;
				imin = i2;
			}
			++i2;
		}
		points.insert({ i1, imin });
		++i1;
	}

	return points;
}

int KeypointPairs::distance(const Features::Point& p1, const Features::Point& p2) const
{
	int dist = 0;
	for (unsigned i = 0; i < p1.features.size(); ++i) {
		dist += std::abs(p1.features[i] - p2.features[i]);
	}
	return dist;
}

void KeypointPairs::save(std::string name, const Features& f1, const Features& f2)
{
	logOut << "Saving keypoints to file " << name << std::endl;
	std::ofstream os(name, std::ofstream::out);
	os << "[";
	const char* separator = "";
	for (const auto& p : pairs) {
		const auto& p1 = f1.points.at(p.first);
		const auto& p2 = f2.points.at(p.second);
		os << separator << "[" 
			<< (int)(p1.x) << "," << (int)(p1.y) << "," 
			<< (int)(p2.x) << "," << (int)(p2.y) << "]";
		separator = ",";
	}
	os << "]";
	os.close();
}

void KeypointPairs::neighborhood(const Features& f1, const Features& f2, int threads, int points, double accept)
{
	date();
	rlutil::setColor(rlutil::GREEN);
	std::cout << "Keypoint analysis started" << std::endl;
	rlutil::setColor(rlutil::GREY);

	logOut << "Computing distances" << std::endl;

	int count1 = f1.points.size(), count2 = f2.points.size();
	std::vector<std::vector<double>> distances1(count1, std::vector<double>(count1));
	std::vector<std::vector<double>> distances2(count2, std::vector<double>(count2));

	auto future1 = std::async(std::launch::async, [&distances1, &count1, &f1, this]() {
		for (int i = 0; i < count1; ++i) {
			for (int j = 0; j < count1; ++j) {
				distances1[i][j] = distance(f1.points[i].x, f1.points[i].y, f1.points[j].x, f1.points[j].y);
			}
		}
	});

	auto future2 = std::async(std::launch::async, [&distances2, &count2, &f2, this]() {
		for (int i = 0; i < count2; ++i) {
			for (int j = 0; j < count2; ++j) {
				distances2[i][j] = distance(f2.points[i].x, f2.points[i].y, f2.points[j].x, f2.points[j].y);
			}
		}
	});

	future1.get();
	future2.get();

	logOut << "Distances computed" << std::endl;

	std::vector<KeypointPair> filtered;

	for (int i = 0; i < count1; ++i) {
		const auto kp = *std::find_if(pairs.begin(), pairs.end(), [&i](const auto& x) { return x.first == i; });
		
		auto closest1 = closest(distances1[i], points);
		auto closest2 = closest(distances2[kp.second], points);
		
		for (auto& elem : closest1) {
			elem = ( *std::find_if(pairs.begin(), pairs.end(), [&elem](const auto& x) { return x.first == elem; }) ).second;
		}

		std::vector<int> common;
		std::sort(closest1.begin(), closest1.end());
		std::sort(closest2.begin(), closest2.end());
		std::set_intersection(closest1.begin(), closest1.end(), closest2.begin(), closest2.end(), std::back_inserter(common));
		
		double score = (double)( common.size() ) / (double)( points );

		if (score >= accept) {
			filtered.push_back(kp);
		}
	}

	pairs = filtered;

	logOut << "After filtering got " << pairs.size() << " keypoint pairs" << std::endl;

}

std::vector<int> KeypointPairs::closest(const std::vector<double>& distances, int num) const
{
	//num += 1; // sam do siebie
		
	std::priority_queue<
		std::pair<double, int>, 
		std::vector<std::pair<double, int>>, 
		std::less<std::pair<double, int>>
	> queue;

	for (int i = 0, len = distances.size(); i < len; ++i) {
		if (queue.size() < num) {
			queue.push(std::pair<double, int>(distances[i], i));
		} else if(queue.top().first > distances[i]) { // element rozwazamy tylko, jesli jest wystarczajaco maly
			queue.pop();
			queue.push(std::pair<double, int>(distances[i], i));
		}
	}

	num = queue.size();
	std::vector<int> result(num);

	for (int i = 0; i < num; ++i) {
		result[i] = queue.top().second;
		queue.pop();
	}

	return result;
}

void KeypointPairs::process_neighborhood(const Features& f1, const Features& f2, int points, double accept)
{
	KeypointPair kp;

	auto comparator = [](const KeypointDistance& kd1, const KeypointDistance& kd2) {
		return false;
	};

	for (;;) {
		{
			std::lock_guard<std::mutex> lock(mutex_queue);
			if (queue.empty()) return;
			kp = queue.front();
			queue.pop();
		}

		const Features::Point point1 = f1.points.at(kp.first), point2 = f2.points.at(kp.second);
		std::priority_queue<KeypointDistance, std::vector<KeypointDistance>, decltype( comparator )> distances(comparator);

		for (const auto& p : pairs) {
			
			//double distance = distance(f1, f2, )
		}


		if(0 >= accept) {
			std::lock_guard<std::mutex> lock(mutex_result);
			result.push_back(kp);
		}
	}
}

double KeypointPairs::distance(double x1, double y1, double x2, double y2) const
{
	return std::sqrt((x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2));
}

void KeypointPairs::ransac(const Features& f1, const Features& f2, int iter, double maxerror, bool force)
{
	date();
	rlutil::setColor(rlutil::GREEN);
	std::cout << "RANSAC (a) started" << std::endl;
	rlutil::setColor(rlutil::GREY);

	std::uniform_int_distribution<int> dist(0, pairs.size() - 1);
	static const double r = 7, R = 230;

	Matrix3d best_model;
	int best_score = 0;

	for (int i = 0; i < iter; ++i) {

		double x1, y1, u1, v1, x2, y2, u2, v2, x3, y3, u3, v3;

		do {

			const KeypointPair&
				kp1 = pairs[dist(engine)],
				kp2 = pairs[dist(engine)],
				kp3 = pairs[dist(engine)];

			x1 = f1.points[kp1.first].x;
			y1 = f1.points[kp1.first].y;
			u1 = f2.points[kp1.second].x;
			v1 = f2.points[kp1.second].y;
			
			x2 = f1.points[kp2.first].x;
			y2 = f1.points[kp2.first].y;
			u2 = f2.points[kp2.second].x;
			v2 = f2.points[kp2.second].y;
			
			x3 = f1.points[kp3.first].x;
			y3 = f1.points[kp3.first].y;
			u3 = f2.points[kp3.second].x;
			v3 = f2.points[kp3.second].y;

		} while (force
			&& !(
				r * r < ( x1 - x2 )*( x1 - x2 ) + ( y1 - y2 )*( y1 - y2 ) &&
				( x1 - x2 )*( x1 - x2 ) + ( y1 - y2 )*( y1 - y2 ) < R * R &&
				r * r < ( u1 - u2 )*( u1 - u2 ) + ( v1 - v2 )*( v1 - v2 ) &&
				( u1 - u2 )*( u1 - u2 ) + ( v1 - v2 )*( v1 - v2 ) < R * R &&

				r * r < ( x1 - x3 )*( x1 - x3 ) + ( y1 - y3 )*( y1 - y3 ) &&
				( x1 - x3 )*( x1 - x3 ) + ( y1 - y3 )*( y1 - y3 ) < R * R &&
				r * r < ( u1 - u3 )*( u1 - u3 ) + ( v1 - v3 )*( v1 - v3 ) &&
				( u1 - u3 )*( u1 - u3 ) + ( v1 - v3 )*( v1 - v3 ) < R * R &&

				r * r < ( x3 - x2 )*( x3 - x2 ) + ( y3 - y2 )*( y3 - y2 ) &&
				( x3 - x2 )*( x3 - x2 ) + ( y3 - y2 )*( y3 - y2 ) < R * R &&
				r * r < ( u3 - u2 )*( u3 - u2 ) + ( v3 - v2 )*( v3 - v2 ) &&
				( u3 - u2 )*( u3 - u2 ) + ( v3 - v2 )*( v3 - v2 ) < R * R
				));

		Matrix<double, 6, 6> mat;
		mat << x1, y1, 1, 0, 0, 0,
			   x2, y2, 1, 0, 0, 0,
			   x3, y3, 1, 0, 0, 0,
		       0, 0, 0, x1, y1, 1,
			   0, 0, 0, x2, y2, 1,
			   0, 0, 0, x3, y3, 1;
		
		Matrix<double, 6, 1> vec;
		vec << u1, u2, u3, v1, v2, v3;

		const auto mvec = mat.inverse() * vec;

		Matrix3d model;
		model << mvec(0), mvec(1), mvec(2),
			     mvec(3), mvec(4), mvec(5),
			     0,       0,       1;
		
		int score = 0;

		for (const auto& kp : pairs) {
			RowVector3d src;
			src << f1.points[kp.first].x,
				   f1.points[kp.first].y,
				   1;
			
			const RowVector3d dst = src * model;

			double error = distance(f2.points[kp.second].x, f2.points[kp.second].y, dst[0], dst[1]);

			if (error < maxerror) ++score;
		}

		if (score > best_score) {
			best_model = model;
			best_score = score;
		}

	}

	logOut << "Ransac finished" << std::endl;

	std::vector<KeypointPair> filtered; 

	for (const auto& kp : pairs) {
		RowVector3d src;
		src << f1.points[kp.first].x,
			f1.points[kp.first].y,
			1;

		const RowVector3d dst = src * best_model;

		double error = distance(f2.points[kp.second].x, f2.points[kp.second].y, dst[0], dst[1]);

		if (error < maxerror) filtered.push_back(kp);
	}

	logOut << "Got " << filtered.size() << " points after filtering" << std::endl;

	pairs = filtered;

}

void KeypointPairs::ransac_p(const Features& f1, const Features& f2, int iter, double maxerror, bool force)
{
	date();
	rlutil::setColor(rlutil::GREEN);
	std::cout << "RANSAC (p) started" << std::endl;
	rlutil::setColor(rlutil::GREY);

	std::uniform_int_distribution<int> dist(0, pairs.size() - 1);
	static const double r = 7, R = 230;

	Matrix3d best_model;
	int best_score = 0;

	for (int i = 0; i < iter; ++i) {

		double x1, y1, u1, v1, x2, y2, u2, v2, x3, y3, u3, v3, x4, y4, u4, v4;

		do {

			const KeypointPair&
				kp1 = pairs[dist(engine)],
				kp2 = pairs[dist(engine)],
				kp3 = pairs[dist(engine)],
				kp4 = pairs[dist(engine)];

			x1 = f1.points[kp1.first].x;
			y1 = f1.points[kp1.first].y;
			u1 = f2.points[kp1.second].x;
			v1 = f2.points[kp1.second].y;

			x2 = f1.points[kp2.first].x;
			y2 = f1.points[kp2.first].y;
			u2 = f2.points[kp2.second].x;
			v2 = f2.points[kp2.second].y;

			x3 = f1.points[kp3.first].x;
			y3 = f1.points[kp3.first].y;
			u3 = f2.points[kp3.second].x;
			v3 = f2.points[kp3.second].y;

			x4 = f1.points[kp4.first].x;
			y4 = f1.points[kp4.first].y;
			u4 = f2.points[kp4.second].x;
			v4 = f2.points[kp4.second].y;


		} while (force
			&& !(
				r * r < ( x1 - x2 )*( x1 - x2 ) + ( y1 - y2 )*( y1 - y2 ) &&
				( x1 - x2 )*( x1 - x2 ) + ( y1 - y2 )*( y1 - y2 ) < R * R &&
				r * r < ( u1 - u2 )*( u1 - u2 ) + ( v1 - v2 )*( v1 - v2 ) &&
				( u1 - u2 )*( u1 - u2 ) + ( v1 - v2 )*( v1 - v2 ) < R * R &&

				r * r < ( x1 - x3 )*( x1 - x3 ) + ( y1 - y3 )*( y1 - y3 ) &&
				( x1 - x3 )*( x1 - x3 ) + ( y1 - y3 )*( y1 - y3 ) < R * R &&
				r * r < ( u1 - u3 )*( u1 - u3 ) + ( v1 - v3 )*( v1 - v3 ) &&
				( u1 - u3 )*( u1 - u3 ) + ( v1 - v3 )*( v1 - v3 ) < R * R &&

				r * r < ( x3 - x2 )*( x3 - x2 ) + ( y3 - y2 )*( y3 - y2 ) &&
				( x3 - x2 )*( x3 - x2 ) + ( y3 - y2 )*( y3 - y2 ) < R * R &&
				r * r < ( u3 - u2 )*( u3 - u2 ) + ( v3 - v2 )*( v3 - v2 ) &&
				( u3 - u2 )*( u3 - u2 ) + ( v3 - v2 )*( v3 - v2 ) < R * R &&

				r * r < ( x4 - x2 )*( x4 - x2 ) + ( y4 - y2 )*( y4 - y2 ) &&
				( x4 - x2 )*( x4 - x2 ) + ( y4 - y2 )*( y4 - y2 ) < R * R &&
				r * r < ( u4 - u2 )*( u4 - u2 ) + ( v4 - v2 )*( v4 - v2 ) &&
				( u4 - u2 )*( u4 - u2 ) + ( v4 - v2 )*( v4 - v2 ) < R * R &&

				r * r < ( x4 - x3 )*( x4 - x3 ) + ( y4 - y3 )*( y4 - y3 ) &&
				( x4 - x3 )*( x4 - x3 ) + ( y4 - y3 )*( y4 - y3 ) < R * R &&
				r * r < ( u4 - u3 )*( u4 - u3 ) + ( v4 - v3 )*( v4 - v3 ) &&
				( u4 - u3 )*( u4 - u3 ) + ( v4 - v3 )*( v4 - v3 ) < R * R
				));

		Matrix<double, 8, 8> mat;
		mat <<
			x1, y1, 1, 0, 0, 0, -u1 * x1, -u1 * y1,
			x2, y2, 1, 0, 0, 0, -u2 * x2, -u2 * y2,
			x3, y3, 1, 0, 0, 0, -u3 * x3, -u3 * y3,
			x4, y4, 1, 0, 0, 0, -u4 * x4, -u4 * y4,
			0, 0, 0, x1, y1, 1, -v1 * x1, -v1 * y1,
			0, 0, 0, x2, y2, 1, -v2 * x2, -v2 * y2,
			0, 0, 0, x3, y3, 1, -v3 * x3, -v3 * y3,
			0, 0, 0, x4, y4, 1, -v4 * x4, -v4 * y4;

		Matrix<double, 8, 1> vec;
		vec << u1, u2, u3, u4, v1, v2, v3, v4;

		const auto mvec = mat.inverse() * vec;

		Matrix3d model;
		model << 
			mvec(0), mvec(1), mvec(2),
			mvec(3), mvec(4), mvec(5),
			mvec(6), mvec(7),       1;

		int score = 0;

		for (const auto& kp : pairs) {
			RowVector3d src;
			src << f1.points[kp.first].x,
				f1.points[kp.first].y,
				1;

			const RowVector3d dst = src * model;

			double error = distance(f2.points[kp.second].x, f2.points[kp.second].y, dst[0], dst[1]);

			if (error < maxerror) ++score;
		}

		if (score > best_score) {
			best_model = model;
			best_score = score;
		}

	}

	logOut << "Ransac finished" << std::endl;

	std::vector<KeypointPair> filtered;

	for (const auto& kp : pairs) {
		RowVector3d src;
		src << f1.points[kp.first].x,
			f1.points[kp.first].y,
			1;

		const RowVector3d dst = src * best_model;

		double error = distance(f2.points[kp.second].x, f2.points[kp.second].y, dst[0], dst[1]);

		if (error < maxerror) filtered.push_back(kp);
	}

	logOut << "Got " << filtered.size() << " points after filtering" << std::endl;

	pairs = filtered;

}
