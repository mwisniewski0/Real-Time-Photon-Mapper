#include "scene.h"

template <>
Scene readFromStream<Scene>(std::istream& s)
{
	Scene v;
	v.triangleData = readFromStream<BVHGpuDataRaw>(s);
	v.materials = readVectorFromStream<MaterialInfo>(s);
	v.lights = readVectorFromStream<PointLightSource>(s);
	//v.spheres = readVectorFromStream<Sphere>(s);
	return v;
}

template <>
void writeToStream<Scene>(std::ostream& s, const Scene& v)
{
	writeToStream(s, v.triangleData);
	writeVectorToStream(s, v.materials);
	writeVectorToStream(s, v.lights);
	//writeVectorToStream(s, v.spheres);
}
