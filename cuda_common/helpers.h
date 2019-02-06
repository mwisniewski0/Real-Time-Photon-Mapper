#ifndef HELPERS_H_INCLUDED__
#define HELPERS_H_INCLUDED__

#include <string>
#include <map>

std::string readFileToString(const std::string& path);

void registerPeriodicalSDLMessage(int periodMs, int messageCode);

template<typename K, typename V>
bool mapHasKey(std::map<K, V> m, K key)
{
	return m.find(key) != m.end();
}

#endif
