#ifndef CAMERA_H_INCLUDED__
#define CAMERA_H_INCLUDED__

#include <memory>
#include "../common/cutil_math.h"
#include <cuda_runtime_api.h>

// Represents 
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
	float3 position;
	float3 lookingAt;
	float3 upDir;
	float3 rightDir;

	float fovX;
	float fovY;

	float3 vectorToScreen() const;
	void refreshRightDir();

public:
	Camera(float3 position, float3 lookingAt, float3 upDir, float horizontalFov,
		float verticalFov);

	static std::unique_ptr<Camera> fromHorizontalFov(float3 position, float3 lookingAt,
		float3 upDir, float horizontalFov, float aspectRatio);

	CudaCamera getCudaInfo() const;
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