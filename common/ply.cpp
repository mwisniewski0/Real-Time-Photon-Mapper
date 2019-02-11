#include "ply.h"
#include <sstream>
#include <fstream>


bool startsWith(const std::string& tested, const std::string& startswith)
{
    return (tested.substr(0, startswith.length()) == startswith);
}

std::vector<Triangle> loadTriangles(const std::string& path, Material m)
{
    std::ifstream f(path);

    std::string line;
    std::getline(f, line);;
    int vtxCount, faceCount;

    while (line != "end_header")
    {
        if (startsWith(line, "element vertex "))
        {
            std::string unused1, unused2;
            std::stringstream lineStream(line);
            lineStream >> unused1 >> unused2 >> vtxCount;
        }
        if (startsWith(line, "element face "))
        {
            std::string unused1, unused2;
            std::stringstream lineStream(line);
            lineStream >> unused1 >> unused2 >> faceCount;
        }
        std::getline(f, line);
    }

    std::vector<float3> vertices;
    for (int i = 0; i < vtxCount; ++i)
    {
        std::getline(f, line);

        float x, y, z;
        std::stringstream lineStream(line);
        lineStream >> x >> y >> z;

        vertices.push_back(make_float3(x * 5, y * 5 - 0.5, z * -5 + 0.3));
    }

    std::vector<Triangle> output;
    for (int i = 0; i < faceCount; ++i)
    {
        std::getline(f, line);

        int vtxNum, a, b, c;
        std::stringstream lineStream(line);
        lineStream >> vtxNum >> a >> b >> c;

        if (vtxNum != 3)
        {
            throw std::runtime_error("Only triangular faces are supported");
        }

        Triangle result = Triangle::from3Points(vertices[a], vertices[b], vertices[c], m);
        output.push_back(result);
    }

    return output;
}
