#ifndef CAMERA_H_INCLUDED__
#define CAMERA_H_INCLUDED__

#include <memory>
#include "../common/cutil_math.h"
#include <cuda_runtime_api.h>

// Represents camera placement and direction on a GPU
struct CudaCamera
{
	float3 screenTopLeft;
	float3 screenTopRight;
	float3 screenBottomLeft;
	float3 screenBottomRight;
	float3 eyePos;
};

// Represents a 3D camera with its position and direction. Provides functions for moving the camera
// easily.
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
	// Creates a new camera. Position represents the point from which the cammera is observing the
	// scene. LookingAt specifies the point at which the camera is looking, whereas upDir
	// specifies a vector perpendicular to position->lookingAt which represents the upwards
	// direction. HorizontalFov and verticalFov represent fields of view in degrees. 
	Camera(float3 position, float3 lookingAt, float3 upDir, float horizontalFov,
		float verticalFov);

	// Same as regular contrstuctor but calculates vertical frame of view from aspect ratio.
	static std::unique_ptr<Camera> fromHorizontalFov(float3 position, float3 lookingAt,
		float3 upDir, float horizontalFov, float aspectRatio);

	// Turns a cpu camera into a gpu camera
	CudaCamera getCudaInfo() const;

	// Moves the camera forward by the given distance
	void moveForward(float delta);

	// Moves the camera backward by the given distance
	void moveBackward(float delta);

	// Moves the camera to the left by the given distance
	void moveLeft(float delta);

	// Moves the camera to the right by the given distance
	void moveRight(float delta);

	// Turns the camera left by the given angle in radians 
	void turnLeft(float rad);

	// Turns the camera right by the given angle in radians
	void turnRight(float rad);
	
	// Turns the camera up by the given angle in radians
	void turnUp(float rad);
	
	// Turns the camera down by the given angle in radians
	void turnDown(float rad);
	
	// Tilts the camera left by the given angle in radians
	void tiltLeft(float rad);

	// Tilts the camera right by the given angle in radians
	void tiltRight(float rad);

	// Used for video-game like controls. Turns the camera right by the given
	// angle, while assuming the axis of the turn to be the y axis
	void absTurnRight(float rad);

	// Used for video-game like controls. Turns the camera up by the given
	// angle, while assuming the axis of the turn not to have any tilt.
	void absTurnUp(float rad);
};


#endif