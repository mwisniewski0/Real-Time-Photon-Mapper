#ifndef GL_HELP_H_INCLUDED__
#define GL_HELP_H_INCLUDED__

#include <GL/glew.h>
#include <string>

namespace glhelp {
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
