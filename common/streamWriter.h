#pragma once

// This file provides functionality for writing and reading objects to/from binary streams. This is
// used to save various data structures used by Photon to files.

#include <istream>
#include <ostream>
#include "cutil_math.h"
#include <vector>

// Given an input stream returns an instance of type T. By default the read data will set the bytes
// of the resultant object directly. This is not enough for more complex structures - see
// specializations of this function throughout the project.
template <typename T>
T readFromStream(std::istream& s)
{
    T v;
    s.read((char*)&v, sizeof(v));
    return v;
}

// Given an output stream and an object, writes the object to the stream. By default the bytes
// of the object will be written directly. This is not enough for more complex structures - see
// specializations of this function throughout the project.
template <typename T>
void writeToStream(std::ostream& s, const T& v)
{
    s.write((char*)&v, sizeof(T));
}

// Stream reader for CUDA float3s.
template <>
float3 readFromStream<float3>(std::istream& s);

// Stream writer for CUDA float3s.
template<>
void writeToStream<float3>(std::ostream& s, const float3& v);

// Stream reader for std::strings.
template <>
std::string readFromStream<std::string>(std::istream& s);

// Stream writer for std::strings.
template<>
void writeToStream<std::string>(std::ostream& s, const std::string& v);

// Reads a vector of objects of type T from the provided input stream. Note that we do not simply
// specialize readFromStream, since partial function specializations are not allowed by the C++14
// standard.
template <typename T>
std::vector<T> readVectorFromStream(std::istream& s)
{
    std::vector<T> result;
    unsigned size = readFromStream<unsigned>(s);

    result.reserve(size);
    for (unsigned i = 0; i < size; ++i) {
        result.push_back(readFromStream<T>(s));
    }
	return result;
}

// Writes a vector of objects of type T to the provided output stream. Note that we do not simply
// specialize writeToStream, since partial function specializations are not allowed by the C++14
// standard.
template <typename T>
void writeVectorToStream(std::ostream& s, const std::vector<T>& v)
{
    writeToStream<unsigned>(s, v.size());
    for (const auto& el : v) {
        writeToStream(s, el);
    }
}
