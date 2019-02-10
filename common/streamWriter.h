#pragma once
#include <istream>
#include <ostream>
#include "../common/cutil_math.h"


template <typename T>
T readFromStream(std::istream& s)
{
	T v;
	s.read((char*)&v, sizeof(v));
	return v;
}

template <typename T>
void writeToStream(std::ostream& s, const T& v)
{
	s.write((char*)&v, sizeof(T));
}

template <>
float3 readFromStream(std::istream& s) {
	float3 result;
	result.x = readFromStream<float>(s);
	result.y = readFromStream<float>(s);
	result.z = readFromStream<float>(s);
	return result;
}

template<>
void writeToStream(std::ostream& s, const float3& v) {
	writeToStream(s, v.x);
	writeToStream(s, v.y);
	writeToStream(s, v.z);
}