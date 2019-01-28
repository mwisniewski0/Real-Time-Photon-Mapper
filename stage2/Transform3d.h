#ifndef TRANSFORM_3D_INCLUDED__
#define TRANSFORM_3D_INCLUDED__

#include <glm/vec3.hpp>
#include <glm/glm.hpp>
#include <array>


/**
 * Transform3D is a very simple to use class for performing 3D transformations on points in 3-space.
 */
class Transform3D {
	std::array<std::array<float, 4>, 4> matrix;

	float getAffineVectorCoord(const glm::vec3& vector, int index);

	Transform3D()
	{
		matrix = { {{0,0,0,0}, {0,0,0,0}, {0,0,0,0}, {0,0,0,0}} };
	}


public:
	/**
	* Transforms the provided point using this matrix. Note that the passed point will not be modified.
	* @param toTransform The point to transform
	* @return The point after the transformation
	*/
	glm::vec3 transform(const glm::vec3& toTransform);

	/**
	* Builds a Transform3D that rotates around the specified axis counter clock-wise by the provided angle.
	* @param axisStart A point on the desired axis of rotation.
	* @param axisEnd A point on the desired axis of rotation that is different from axisStart
	* @param angle The angle of the rotation. The input will be treated as if it was in radians.
	* @return The rotation matrix
	*/
	static Transform3D rotateCCWAroundAxis(const glm::vec3& axisStart, const glm::vec3& axisEnd, float angle);

	/**
	* Multiplies this matrix by the provided matrix on the right. The result will be saved in this matrix.
	* @param other The matrix to multiply with
	*/
	void concatenate(const Transform3D& other);

	/**
	* Multiplies this matrix by the provided matrix on the left. The result will be saved in this matrix.
	* @param other The matrix to multiply with
	*/
	void preConcatenate(const Transform3D& other);

	/**
	* Multiplies this matrix by the provided matrix on the right. Neither this nor the provided matrix will be
	* modified
	* @param other The matrix to multiply with
	* @return The multiplied matrix
	*/
	Transform3D multiply(const Transform3D& other) const;

	static Transform3D getTranslateInstance(const glm::vec3& translation);

	/**
	* Creates a new identity matrix
	* @return The identity matrix
	*/
	static Transform3D getIdentityMatrix();

	/**
	* Creates a transform that scales points by the provided ratios
	* @param scaleX The scale factor in the x-direction
	* @param scaleY The scale factor in the y-direction
	* @param scaleZ The scale factor in the z-direction
	* @return A transform that scales points by the provided ratios
	*/
	static Transform3D getScaleInstance(float scaleX, float scaleY, float scaleZ);
};


#endif
