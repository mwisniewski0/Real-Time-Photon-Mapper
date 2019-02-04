#ifndef CAMERA_H_INCLUDED__
#define CAMERA_H_INCLUDED__

#include <memory>
#include "GlHelp.h"
#include <glm/vec3.hpp>
#include <cuda_runtime_api.h>

struct CudaCamera
{
	float3 screenTopLeft;
	float3 screenTopRight;
	float3 screenBottomLeft;
	float3 screenBottomRight;
	float3 eyePos;
};

class Camera
{
	glm::vec3 position;
	glm::vec3 lookingAt;
	glm::vec3 upDir;
	glm::vec3 rightDir;

	float fovX;
	float fovY;

	glm::vec3 vectorToScreen() const;
	void refreshRightDir();

public:
	Camera(glm::vec3 position, glm::vec3 lookingAt, glm::vec3 upDir, float horizontalFov,
		float verticalFov);

	static std::unique_ptr<Camera> fromHorizontalFov(glm::vec3 position, glm::vec3 lookingAt,
		glm::vec3 upDir, float horizontalFov, float aspectRatio);

	CudaCamera getCudaInfo() const;
	glhelp::CameraInfo getGlslInfo();
	void moveForward(float delta);
	void moveBackward(float delta);
	void moveLeft(float delta);
	void moveRight(float delta);
	void turnLeft(float rad);
	void turnRight(float rad);
	void turnUp(float rad);
	void turnDown(float rad);
	void tiltLeft(float rad);
	void tiltRight(float rad);

	void absTurnRight(float rad);
	void absTurnUp(float rad);
};


#endif