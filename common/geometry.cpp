#include "geometry.h"
#include <vector>

/*
 * Builds bounding box contailing all triangles.
 */
BoundingBox Triangle::boundingBoxForMany(const std::vector<Triangle>& triangles)
{
	BoundingBox result;
	for (const auto& t : triangles)
	{
		result = result.merge(t.getBoundingBox());
	}
	return result;
}

/*
 * Expands bounding box to contain other.
 */
BoundingBox BoundingBox::merge(const BoundingBox& other) const
{
	BoundingBox result;
	result.minCoords = minf3(this->minCoords, other.minCoords);
	result.maxCoords = maxf3(this->maxCoords, other.maxCoords);
	return result;
}

/*
 * Expands bounding box to contain v.
 */
BoundingBox BoundingBox::merge(const float3& v) const
{
	BoundingBox result;
	result.minCoords = minf3(this->minCoords, v);
	result.maxCoords = maxf3(this->maxCoords, v);
	return result;
}

/*
 * Returns surface area of bounding box
 */
float BoundingBox::getArea()
{
	return dot((maxCoords - minCoords), (maxCoords - minCoords));
}

/*
 * Creates a material with a png texture. 
 */
Material MaterialInfo::loadWithTexture() const
{
	Material result = material;
	if (result.useDiffuseTexture)
	{
		result.diffuseTexture = GPUTexture::fromPng(diffuseTexturePath);
	}
	return result;
}

template<>
void writeToStream<MaterialInfo>(std::ostream& s, const MaterialInfo& v)
{
	writeToStream(s, v.material.transmittance);
	writeToStream(s, v.material.specular);
	writeToStream(s, v.material.diffuse);
	writeToStream(s, v.material.useDiffuseTexture);
	writeToStream(s, v.material.refractiveIndex);
	writeToStream(s, v.material.shininess);
	writeToStream(s, v.diffuseTexturePath);
}

template<>
MaterialInfo readFromStream<MaterialInfo>(std::istream& s)
{
	MaterialInfo result;

	result.material.transmittance = readFromStream<float3>(s);
	result.material.specular = readFromStream<float3>(s);
	result.material.diffuse = readFromStream<float3>(s);
	result.material.useDiffuseTexture = readFromStream<bool>(s);
	result.material.refractiveIndex = readFromStream<float>(s);
	result.material.shininess = readFromStream<float>(s);
	result.diffuseTexturePath = readFromStream<std::string>(s);

	return result;
}

/*
 * Create a new triangle from three vertices and a material.
 */
Triangle Triangle::from3Points(float3 v1, float3 v2, float3 v3, unsigned materialIndex)
{
	Triangle result;
	result.v0 = v1;
	result.v0v1 = v2 - v1;
	result.v0v2 = v3 - v1;
	result.materialIndex = materialIndex;
	result.normal = normalize(cross(result.v0v1, result.v0v2));

	result.v0vn = result.normal;
	result.v1vn = result.normal;
	result.v2vn = result.normal;

	return result;
}

/*
 * Get bounding box containing triangle
 */
BoundingBox Triangle::getBoundingBox() const
{
	BoundingBox result;
	return result.merge(v0).merge(v0 + v0v1).merge(v0 + v0v2);
}

/*
 * Calculate triangle surface area.
 */
float Triangle::approxSurfaceArea() const
{
	return length(cross((v0v1), (v0v2))) * 0.5;
}

/*
 * return the center of the triangle
 */
float3 Triangle::center() const
{
	return (3 * v0 + v0v1 + v0v2) / 3.f;
}

template<>
Triangle readFromStream<Triangle>(std::istream &s) {
	Triangle v;
	v.v0 = readFromStream<float3>(s);
	v.v0v1 = readFromStream<float3>(s);
	v.v0v2 = readFromStream<float3>(s);
	v.v0vt = readFromStream<float3>(s);
	v.v1vt = readFromStream<float3>(s);
	v.v2vt = readFromStream<float3>(s);
	v.v0vn = readFromStream<float3>(s);
	v.v1vn = readFromStream<float3>(s);
	v.v2vn = readFromStream<float3>(s);
	v.normal = readFromStream<float3>(s);
	v.materialIndex = readFromStream<unsigned>(s);
	return v;
}

template <>
void writeToStream<PointLightSource>(std::ostream& s, const PointLightSource& v)
{
	writeToStream(s, v.position);
	writeToStream(s, v.intensity);
}

template <>
PointLightSource readFromStream<PointLightSource>(std::istream& s)
{
	PointLightSource v;
	v.position = readFromStream<float3>(s);
	v.intensity = readFromStream<float3>(s);
	return v;
}

template<>
void writeToStream<Triangle>(std::ostream &s, const Triangle &v) {
	writeToStream(s, v.v0);
	writeToStream(s, v.v0v1);
	writeToStream(s, v.v0v2);
	writeToStream(s, v.v0vt);
	writeToStream(s, v.v1vt);
	writeToStream(s, v.v2vt);
	writeToStream(s, v.v0vn);
	writeToStream(s, v.v1vn);
	writeToStream(s, v.v2vn);
	writeToStream(s, v.normal);
	writeToStream(s, v.materialIndex);
}
