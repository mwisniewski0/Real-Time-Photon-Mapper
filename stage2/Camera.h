#ifndef CAMERA_H_INCLUDED__
#define CAMERA_H_INCLUDED__

#include <memory>
#include "GlHelp.h"
#include <glm/vec3.hpp>


class Camera
{
	glm::vec3 position;
	glm::vec3 lookingAt;
	glm::vec3 upDir;
	glm::vec3 rightDir;

	float fovX;
	float fovY;

	glm::vec3 vectorToScreen();
	void refreshRightDir();

public:
	Camera(glm::vec3 position, glm::vec3 lookingAt, glm::vec3 upDir, float horizontalFov,
		float verticalFov);

	static std::unique_ptr<Camera> fromHorizontalFov(glm::vec3 position, glm::vec3 lookingAt,
		glm::vec3 upDir, float horizontalFov, float aspectRatio);

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