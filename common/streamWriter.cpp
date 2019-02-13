#include "streamWriter.h"


template <>
float3 readFromStream<float3>(std::istream& s) {
	float3 result;
	result.x = readFromStream<float>(s);
	result.y = readFromStream<float>(s);
	result.z = readFromStream<float>(s);
	return result;
}

template<>
void writeToStream<float3>(std::ostream& s, const float3& v) {
	writeToStream(s, v.x);
	writeToStream(s, v.y);
	writeToStream(s, v.z);
}

template <>
std::string readFromStream<std::string>(std::istream& s) {
	unsigned length = readFromStream<unsigned>(s);
	char* buffer = new char[length + 1];
	s.read(buffer, length);
	std::string result(buffer, length);
	delete[] buffer;
	return result;
}

template<>
void writeToStream<std::string>(std::ostream& s, const std::string& v) {
	writeToStream<unsigned>(s, v.length());
	s.write(v.c_str(), v.length());
}
