#include "Helpers.h"
#include <fstream>
#include <sstream>
#include <SDL2/SDL.h>
#include <map>

std::string readFileToString(const std::string& path)
{
	std::ifstream file(path);
	std::stringstream ss;
	ss << file.rdbuf();
	file.close();
	return ss.str();
}
