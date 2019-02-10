#pragma once
#include <istream>
#include <ostream>
#include "cutil_math.h"


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

template <typename T>
std::vector<T> readVectorFromStream(std::istream& s)
{
    std::vector<T> result;
    unsigned size = readFromStream<unsigned>(s);

    result.reserve(size);
    for (unsigned i = 0; i < size; ++i) {
        result.push_back(readFromStream<T>(s));
    }
}

template <typename T>
void writeVectorToStream(std::ostream& s, const std::vector<T>& v)
{
    writeToStream<unsigned>(s, v.size());
    for (const auto& el : v) {
        writeToStream(s, el);
    }
}
