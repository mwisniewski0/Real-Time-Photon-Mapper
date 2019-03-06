#ifndef HELPERS_H_INCLUDED__
#define HELPERS_H_INCLUDED__

// This file defines common functions used throughout Photon code.

#include <string>
#include <map>

// Reads the file at the given path and returns its contents as a string
std::string readFileToString(const std::string& path);

// Checks whether the given std::map has the provided key.
template<typename K, typename V>
bool mapHasKey(const std::map<K, V>& m, K key)
{
	return m.find(key) != m.end();
}

#endif
