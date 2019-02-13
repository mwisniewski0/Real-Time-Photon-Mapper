#include "../external/catch.hpp"
#include <sstream>
#include "../common/scene.h"
#include <iostream>
#include "../cuda_common/gpuBvh.h"
#include <fstream>
#include "../common/obj_file_parser.h"

bool operator==(const float3& a, const float3& b)
{
	return a.x == Approx(b.x) && a.y == Approx(b.y) && a.z == Approx(b.z);
}

std::ostream& operator << (std::ostream& os, const float3& value) {
	os << float3ToString(value);
	return os;
}



TEST_CASE("Empty scenes are saved and loaded") {
	Scene empty;
	std::stringstream ss;
	writeToStream(ss, empty);
	Scene result = readFromStream<Scene>(ss);

	REQUIRE(result.spheres.size() == 0);
	REQUIRE(result.lights.size() == 0);
	REQUIRE(result.materials.size() == 0);
	REQUIRE(result.triangleData.triangles.size() == 0);
	REQUIRE(result.triangleData.bvhNodes.size() == 0);
}

TEST_CASE("Lights in scenes are saved and loaded") {
	Scene scene;
	scene.lights.push_back(PointLightSource{
		{ 1.5, 1.25, 1.125 },
		{ 0.5, 0.25, 0.125 }
	});
	scene.lights.push_back(PointLightSource{
		{ 4.5, 4.25, 4.125 },
		{ 2.5, 2.25, 2.125 }
	});

	std::stringstream ss;
	writeToStream(ss, scene);
	Scene result = readFromStream<Scene>(ss);

	REQUIRE(result.lights.size() == 2);

	REQUIRE(result.lights[0].position == make_float3(1.5, 1.25, 1.125));
	REQUIRE(result.lights[0].intensity == make_float3(0.5, 0.25, 0.125));

	REQUIRE(result.lights[1].position == make_float3(4.5, 4.25, 4.125));
	REQUIRE(result.lights[1].intensity == make_float3(2.5, 2.25, 2.125));
}

TEST_CASE("Materials in scenes are saved and loaded") {
	Scene scene;
	scene.materials.push_back(MaterialInfo{
		{
			{ 0.1f, 0.2f, 0.3f },
			{ 0.4f, 0.5f, 0.6f },
			{ 0.7f, 0.8f, 0.9f },
			1.0f,
			1.1f,
			false,
			{}
		},
		"blimey.texture"
	});

	std::stringstream ss;
	writeToStream(ss, scene);
	Scene result = readFromStream<Scene>(ss);

	REQUIRE(result.materials.size() == 1);
	REQUIRE(result.materials[0].toString() == scene.materials[0].toString());
}

TEST_CASE("Geometry in scenes is saved and loaded") {
	std::vector<Triangle> triangles;
	triangles.push_back(Triangle::from3Points({ 0.1f, 0.1f, 0.1f }, { 0.2f, 0.2f, 0.2f }, { 0.0f, 0.1f, 0.2f }, 0));
	triangles.push_back(Triangle::from3Points({ 1.1f, 1.1f, 1.1f }, { 1.2f, 1.2f, 1.2f }, { 1.1f, 1.1f, 1.2f }, 1));
	triangles.push_back(Triangle::from3Points({ 2.1f, 2.1f, 2.1f }, { 2.2f, 2.2f, 2.2f }, { 2.2f, 2.1f, 2.2f }, 2));
	triangles.push_back(Triangle::from3Points({ 3.1f, 3.1f, 3.1f }, { 3.2f, 3.2f, 3.2f }, { 3.3f, 3.1f, 3.2f }, 3));
	triangles.push_back(Triangle::from3Points({ 4.1f, 4.1f, 4.1f }, { 4.2f, 4.2f, 4.2f }, { 4.4f, 4.1f, 4.2f }, 4));
	triangles.push_back(Triangle::from3Points({ 5.1f, 5.1f, 5.1f }, { 5.2f, 5.2f, 5.2f }, { 5.5f, 5.1f, 5.2f }, 5));
	triangles.push_back(Triangle::from3Points({ 6.1f, 6.1f, 6.1f }, { 6.2f, 6.2f, 6.2f }, { 6.6f, 6.1f, 6.2f }, 6));
	triangles.push_back(Triangle::from3Points({ 7.1f, 7.1f, 7.1f }, { 7.2f, 7.2f, 7.2f }, { 7.7f, 7.1f, 7.2f }, 7));
	triangles.push_back(Triangle::from3Points({ 8.1f, 8.1f, 8.1f }, { 8.2f, 8.2f, 8.2f }, { 8.8f, 8.1f, 8.2f }, 8));
	auto raw = buildBVH(std::move(triangles))->toRaw();

	Scene scene;
	scene.triangleData = *raw;

	std::stringstream ss;
	writeToStream(ss, scene);
	Scene result = readFromStream<Scene>(ss);

	REQUIRE(result.triangleData.triangles.size() == raw->triangles.size());
	for (unsigned i = 0; i < raw->triangles.size(); ++i)
	{
		REQUIRE(raw->triangles[0].toString() == result.triangleData.triangles[0].toString());
	}

	REQUIRE(result.triangleData.bvhNodes.size() == raw->bvhNodes.size());
	for (unsigned i = 0; i < raw->bvhNodes.size(); ++i)
	{
		REQUIRE(raw->bvhNodes[0].toString() == result.triangleData.bvhNodes[0].toString());
	}
}

TEST_CASE("Simple scenes are saved and loaded") {
	Scene scene;

	// Add lights
	scene.lights.push_back(PointLightSource{
		{ 1.5, 1.25, 1.125 },
		{ 0.5, 0.25, 0.125 }
	});
	scene.lights.push_back(PointLightSource{
		{ 4.5, 4.25, 4.125 },
		{ 2.5, 2.25, 2.125 }
	});

	// Add materials
	scene.materials.push_back(MaterialInfo{
		{
			{ 0.1f, 0.2f, 0.3f },
			{ 0.4f, 0.5f, 0.6f },
			{ 0.7f, 0.8f, 0.9f },
			1.0f,
			1.1f,
			false,
			{}
		},
		"blimey.texture"
	});

	// Add geometry
	std::vector<Triangle> triangles;
	triangles.push_back(Triangle::from3Points({ 0.1f, 0.1f, 0.1f }, { 0.2f, 0.2f, 0.2f }, { 0.0f, 0.1f, 0.2f }, 0));
	triangles.push_back(Triangle::from3Points({ 1.1f, 1.1f, 1.1f }, { 1.2f, 1.2f, 1.2f }, { 1.1f, 1.1f, 1.2f }, 1));
	triangles.push_back(Triangle::from3Points({ 2.1f, 2.1f, 2.1f }, { 2.2f, 2.2f, 2.2f }, { 2.2f, 2.1f, 2.2f }, 2));
	triangles.push_back(Triangle::from3Points({ 3.1f, 3.1f, 3.1f }, { 3.2f, 3.2f, 3.2f }, { 3.3f, 3.1f, 3.2f }, 3));
	triangles.push_back(Triangle::from3Points({ 4.1f, 4.1f, 4.1f }, { 4.2f, 4.2f, 4.2f }, { 4.4f, 4.1f, 4.2f }, 4));
	triangles.push_back(Triangle::from3Points({ 5.1f, 5.1f, 5.1f }, { 5.2f, 5.2f, 5.2f }, { 5.5f, 5.1f, 5.2f }, 5));
	triangles.push_back(Triangle::from3Points({ 6.1f, 6.1f, 6.1f }, { 6.2f, 6.2f, 6.2f }, { 6.6f, 6.1f, 6.2f }, 6));
	triangles.push_back(Triangle::from3Points({ 7.1f, 7.1f, 7.1f }, { 7.2f, 7.2f, 7.2f }, { 7.7f, 7.1f, 7.2f }, 7));
	triangles.push_back(Triangle::from3Points({ 8.1f, 8.1f, 8.1f }, { 8.2f, 8.2f, 8.2f }, { 8.8f, 8.1f, 8.2f }, 8));
	auto raw = buildBVH(std::move(triangles))->toRaw();
	scene.triangleData = *raw;

	// Save and load
	std::stringstream ss;
	writeToStream(ss, scene);
	Scene result = readFromStream<Scene>(ss);

	// Verify geometry
	REQUIRE(result.triangleData.triangles.size() == raw->triangles.size());
	for (unsigned i = 0; i < raw->triangles.size(); ++i)
	{
		REQUIRE(raw->triangles[0].toString() == result.triangleData.triangles[0].toString());
	}

	REQUIRE(result.triangleData.bvhNodes.size() == raw->bvhNodes.size());
	for (unsigned i = 0; i < raw->bvhNodes.size(); ++i)
	{
		REQUIRE(raw->bvhNodes[0].toString() == result.triangleData.bvhNodes[0].toString());
	}

	// Verify materials
	REQUIRE(result.materials.size() == 1);
	REQUIRE(result.materials[0].toString() == scene.materials[0].toString());

	// Verify lights
	REQUIRE(result.lights.size() == 2);

	REQUIRE(result.lights[0].position == make_float3(1.5, 1.25, 1.125));
	REQUIRE(result.lights[0].intensity == make_float3(0.5, 0.25, 0.125));

	REQUIRE(result.lights[1].position == make_float3(4.5, 4.25, 4.125));
	REQUIRE(result.lights[1].intensity == make_float3(2.5, 2.25, 2.125));
}
