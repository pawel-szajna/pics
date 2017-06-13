#include <fstream>
#include <iostream>

#include "cxxopts.hh"
#include "features.hh"
#include "logger.hh"
#include "pairs.hh"

int main(int argc, char** argv)
{
	try {
		std::string method, keypoints, result;
		int points, threads, iter;
		double accept, maxerror;
		bool force, persp;

		cxxopts::Options options(argv[0], "analyses pictures similarity");
		
		options.add_options()
			( "m,method", "Computation method", cxxopts::value<std::string>(method)->default_value("sim") )
			( "k,keypoints", "Keypoints output file", cxxopts::value<std::string>(keypoints) )
			( "r,result", "Results output file", cxxopts::value<std::string>(result) )
			( "t,threads", "Thread count", cxxopts::value<int>(threads)->default_value("4") )
			( "p,points", "Points count", cxxopts::value<int>(points)->default_value("25") )
			( "a,accept", "Accepted similarity", cxxopts::value<double>(accept)->default_value("0.67") )
			( "i,iter", "RANSAC iteration count", cxxopts::value<int>(iter)->default_value("50") )
			( "e,error", "RANSAC max error", cxxopts::value<double>(maxerror)->default_value("15") )
			( "f,force", "Use random choice heuristic", cxxopts::value<bool>(force) )
			( "v,perspective", "Use perspective transform", cxxopts::value<bool>(persp) )
			( "positional",
				"Positional arguments: these are the arguments that are entered "
				"without an option", cxxopts::value<std::vector<std::string>>() )
			;
		options.positional_help("INPUT1 INPUT2");
		options.parse_positional("positional");
		options.parse(argc, argv);
		
		if (options.count("positional") != 2) {
			std::cout << options.help({ "" }) << std::endl;
			return 0;
		}

		date();
		rlutil::setColor(rlutil::YELLOW);
		std::cout << "picture similarity analysis" << std::endl;
		rlutil::setColor(rlutil::GREY);

		auto& v = options["positional"].as<std::vector<std::string>>();

		Features f1(v[0]), f2(v[1]);
		KeypointPairs k(f1, f2);

		if (options.count("keypoints")) {
			k.save(keypoints, f1, f2);
		}

		if (method == "sim") {
			k.neighborhood(f1, f2, threads, points, accept);
		} else if(method == "ransac") {
			if (persp) k.ransac_p(f1, f2, iter, maxerror, force);
			else k.ransac(f1, f2, iter, maxerror, force);
		} else {
			std::cerr << "invalid method: " << method;
			return -1;
		}

		date();
		rlutil::setColor(rlutil::LIGHTCYAN);
		std::cout << "Job done!" << std::endl;
		rlutil::setColor(rlutil::GREY);

		if (options.count("result")) {
			k.save(result, f1, f2);
		}

	} catch (const cxxopts::OptionException& e) {
		std::cerr << "error parsing options: " << e.what() << std::endl;
	}
}
