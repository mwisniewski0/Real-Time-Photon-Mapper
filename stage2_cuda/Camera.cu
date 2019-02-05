#include "Camera.h"
#include "Transform3d.h"
#include <memory>

float3 Camera::vectorToScreen() const
{
	return this->lookingAt - this->position;
}

void Camera::refreshRightDir()
{
	this->rightDir = normalize(cross(this->vectorToScreen(), this->upDir));
}

Camera::Camera(float3 position, float3 lookingAt, float3 upDir, float horizontalFov, float verticalFov)
{
	this->position = position;
	this->lookingAt = lookingAt;
	this->fovX = horizontalFov;
	this->fovY = verticalFov;

	this->upDir = upDir;
	refreshRightDir();
}

std::unique_ptr<Camera> Camera::fromHorizontalFov(float3 position, float3 lookingAt,
	float3 upDir, float horizontalFov,
	float aspectRatio)
{
	float widthAtDistance1 = tan(horizontalFov / 2.0f) * 2.0f;
	float heightAtDistance1 = widthAtDistance1 / aspectRatio;
	float verticalFov = atan(heightAtDistance1 / 2.0f) * 2.0f;
	return std::make_unique<Camera>(position, lookingAt, upDir, horizontalFov, verticalFov);
}

float3 toFloat3(const float3& vec)
{
	float3 res;
	res.x = vec.x;
	res.y = vec.y;
	res.z = vec.z;
	return res;
}

CudaCamera Camera::getCudaInfo() const
{
	CudaCamera result;
	result.eyePos = toFloat3(position);

	float distanceToScreen = length(vectorToScreen());
	auto horizontalStretch = distanceToScreen * tanf(fovX / 2.f) * rightDir;
	auto verticalStretch = distanceToScreen * tanf(fovY / 2.f) * upDir;

	result.screenBottomLeft = toFloat3(lookingAt - horizontalStretch - verticalStretch);
	result.screenBottomRight = toFloat3(lookingAt + horizontalStretch - verticalStretch);
	result.screenTopLeft = toFloat3(lookingAt - horizontalStretch + verticalStretch);
	result.screenTopRight = toFloat3(lookingAt + horizontalStretch + verticalStretch);

	return result;
}

void Camera::moveForward(float delta)
{
	auto displacement = normalize(lookingAt - position) * delta;
	position += displacement;
	lookingAt += displacement;
}

void Camera::moveBackward(float delta)
{
	moveForward(-delta);
}

void Camera::moveLeft(float delta)
{
	moveRight(-delta);
}

void Camera::moveRight(float delta)
{
	auto displacement = this->rightDir * delta;
	position += displacement;
	lookingAt += displacement;
}

void Camera::turnLeft(float rad)
{
	turnRight(-rad);
}

void Camera::turnRight(float rad)
{
	lookingAt =
		Transform3D::rotateCCWAroundAxis(position, position + upDir, rad).transform(lookingAt);
	refreshRightDir();
}

void Camera::turnUp(float rad)
{
	lookingAt =
		Transform3D::rotateCCWAroundAxis(position, position + rightDir, rad).transform(lookingAt);
	upDir =
		Transform3D::rotateCCWAroundAxis(make_float3(0), rightDir, rad).transform(upDir);
	refreshRightDir();
}

void Camera::turnDown(float rad)
{
	turnUp(-rad);
}

void Camera::tiltLeft(float rad)
{
	tiltRight(-rad);
}

void Camera::tiltRight(float rad)
{
	upDir =
		Transform3D::rotateCCWAroundAxis(make_float3(0), vectorToScreen(), rad).transform(upDir);
	refreshRightDir();
}

void Camera::absTurnRight(float rad)
{
	auto transform = Transform3D::rotateCCWAroundAxis(position, position + make_float3(0, 1, 0), rad);
	auto temp = transform.transform(lookingAt + upDir);
	lookingAt = transform.transform(lookingAt);
	upDir = temp - lookingAt;
	refreshRightDir();
}

void Camera::absTurnUp(float rad)
{
	turnUp(rad);
}
