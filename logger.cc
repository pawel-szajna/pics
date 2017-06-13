#include "logger.hh"

void date()
{
	rlutil::setColor(rlutil::DARKGREY);
	auto now = std::chrono::system_clock::now();
	auto now_c = std::chrono::system_clock::to_time_t(now);
	std::cout << std::put_time(std::localtime(&now_c), "%c ");
	rlutil::setColor(rlutil::GREY);
}
