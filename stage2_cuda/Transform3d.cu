#include "Transform3d.h"
#include <exception>
#include <stdexcept>

float Transform3D::getAffineVectorCoord(const float3& vector, int index)
{
	switch (index)
	{
	case 0:
		return vector.x;
	case 1:
		return vector.y;
	case 2:
		return vector.z;
	case 3:
		return 1.0;
	}
	throw std::runtime_error("Index out of bounds");
}

float3 Transform3D::transform(const float3& toTransform)
{
	float affineCoords[4] = {0,0,0,0};
	for (int y = 0; y < 4; ++y)
	{
		for (int x = 0; x < 4; ++x)
		{
			affineCoords[y] += matrix[x][y] * getAffineVectorCoord(toTransform, x);
		}
	}
	return make_float3(
		affineCoords[0] / affineCoords[3],
		affineCoords[1] / affineCoords[3],
		affineCoords[2] / affineCoords[3]
	);
}

Transform3D Transform3D::rotateCCWAroundAxis(const float3& axisStart, const float3& axisEnd, float angle)
{
	float cosx = cos(angle);
	float sinx = sin(angle);

	// Unit vector from origin
	float3 a = normalize(axisEnd - axisStart);

	Transform3D result;
	result.matrix = {
		{
			{cosx + (1 - cosx) * a.x * a.x, (1 - cosx) * a.x * a.y - sinx * a.z, (1 - cosx) * a.x * a.z + sinx * a.y, 0},
			{(1 - cosx) * a.x * a.y + sinx * a.z, cosx + (1 - cosx) * a.y * a.y, (1 - cosx) * a.y * a.z - sinx * a.x, 0},
			{(1 - cosx) * a.x * a.z - sinx * a.y, (1 - cosx) * a.y * a.z + sinx * a.x, cosx + (1 - cosx) * a.z * a.z, 0},
			{0,0,0,1},
		}
	};

	// If the line is not going through the origin, we need to first translate to the origin, and then translate
	// back
	result.concatenate(getTranslateInstance(-axisStart));
	result.preConcatenate(getTranslateInstance(axisStart));
	return result;
}

void Transform3D::concatenate(const Transform3D& other)
{
	this->matrix = this->multiply(other).matrix;
}

void Transform3D::preConcatenate(const Transform3D& other)
{
	this->matrix = other.multiply(*this).matrix;
}

Transform3D Transform3D::multiply(const Transform3D& other) const
{
	Transform3D result;
	for (int x = 0; x < 4; x++)
	{
		for (int y = 0; y < 4; y++)
		{
			for (int i = 0; i < 4; ++i)
			{
				result.matrix[x][y] += this->matrix[i][y] * other.matrix[x][i];
			}
		}
	}
	return result;
}

Transform3D Transform3D::getTranslateInstance(const float3& translation)
{
	Transform3D result;
	result.matrix[0][0] = 1;
	result.matrix[3][0] = translation.x;

	result.matrix[1][1] = 1;
	result.matrix[3][1] = translation.y;

	result.matrix[2][2] = 1;
	result.matrix[3][2] = translation.z;

	result.matrix[3][3] = 1;
	return result;
}

Transform3D Transform3D::getIdentityMatrix()
{
	Transform3D result;
	result.matrix[0][0] = 1.0;
	result.matrix[1][1] = 1.0;
	result.matrix[2][2] = 1.0;
	result.matrix[3][3] = 1.0;
	return result;
}

Transform3D Transform3D::getScaleInstance(float scaleX, float scaleY, float scaleZ)
{
	Transform3D result;
	result.matrix[0][0] = scaleX;
	result.matrix[1][1] = scaleY;
	result.matrix[2][2] = scaleZ;
	result.matrix[3][3] = 1.0;
	return result;
}
