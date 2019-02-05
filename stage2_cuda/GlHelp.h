#ifndef GL_HELP_H_INCLUDED__
#define GL_HELP_H_INCLUDED__

#include <GL/glew.h>
#include <glm/vec3.hpp>
#include <string>

namespace glhelp {
	struct CameraInfo
	{
		glm::vec3 screenTopLeft;
		glm::vec3 screenTopRight;
		glm::vec3 screenBottomLeft;
		glm::vec3 screenBottomRight;
		glm::vec3 eyePos;

		void setInGlsl(GLuint programID,
			const char* topLeftUniformName = "cameraTopLeft",
			const char* topRightUniformName = "cameraTopRight",
			const char* bottomLeftUniformName = "cameraBottomLeft",
			const char* bottomRightUniformName = "cameraBottomRight",
			const char* eyePosUniformName = "cameraEyePos");
	};

	/**
	 * \brief createShaderProgram() creates a new shader program based on the provided source for
	 * the vertex and the fragment shader.
	 * \param vtxShader Source of the vertex shader
	 * \param fragShader Source of the fragment shader
	 * \return ID of the compiled GL program
	 */
	GLuint createShaderProgram(const std::string& vtxShader, const std::string& fragShader);

	void drawSquare(double x1, double y1, double sidelength);
}


#endif
