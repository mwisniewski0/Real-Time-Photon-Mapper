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
