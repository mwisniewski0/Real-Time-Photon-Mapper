#include "GlHelp.h"
#include <string>
#include <memory>

namespace glhelp {

	void CameraInfo::setInGlsl(GLuint programID, const char* topLeftUniformName,
		const char* topRightUniformName, const char* bottomLeftUniformName,
		const char* bottomRightUniformName, const char* eyePosUniformName)
	{
		glUniform3f(glGetUniformLocation(programID, topLeftUniformName),
			screenTopLeft.x, screenTopLeft.y, screenTopLeft.z);
		glUniform3f(glGetUniformLocation(programID, topRightUniformName),
			screenTopRight.x, screenTopRight.y, screenTopRight.z);
		glUniform3f(glGetUniformLocation(programID, bottomLeftUniformName),
			screenBottomLeft.x, screenBottomLeft.y, screenBottomLeft.z);
		glUniform3f(glGetUniformLocation(programID, bottomRightUniformName),
			screenBottomRight.x, screenBottomRight.y, screenBottomRight.z);
		glUniform3f(glGetUniformLocation(programID, eyePosUniformName),
			eyePos.x, eyePos.y, eyePos.z);
	}


	GLint compileShader(const std::string& shaderSource, GLuint shaderID)
	{
		GLint status = GLEW_OK;

		// Compile Vertex Shader
		char const* vtxSourceCString = shaderSource.c_str();
		glShaderSource(shaderID, 1, &vtxSourceCString, nullptr);
		glCompileShader(shaderID);

		// Check Vertex Shader
		glGetShaderiv(shaderID, GL_COMPILE_STATUS, &status);

		GLint infoLogLength;
		glGetShaderiv(shaderID, GL_INFO_LOG_LENGTH, &infoLogLength);
		if (infoLogLength > 0) {
			char* buffer = new char[infoLogLength + 1];
			glGetShaderInfoLog(shaderID, infoLogLength, nullptr, buffer);
			std::string errorString = buffer;
			delete[] buffer;

			throw std::runtime_error(errorString);
		}

		return status;
	}


	GLuint createShaderProgram(const std::string& vtxShader, const std::string& fragShader) {
		GLuint vtxShaderID = glCreateShader(GL_VERTEX_SHADER);
		GLuint fragShaderID = glCreateShader(GL_FRAGMENT_SHADER);
		GLuint programID = glCreateProgram();

		try {
			compileShader(vtxShader, vtxShaderID);
			compileShader(fragShader, fragShaderID);

			// Linking
			glAttachShader(programID, vtxShaderID);
			glAttachShader(programID, fragShaderID);
			glLinkProgram(programID);

			// Check the program
			GLint status = GL_FALSE;
			glGetProgramiv(programID, GL_LINK_STATUS, &status);

			GLint infoLogLength;
			glGetProgramiv(programID, GL_INFO_LOG_LENGTH, &infoLogLength);
			if (infoLogLength > 0) {
				char* buffer = new char[infoLogLength + 1];
				glGetProgramInfoLog(programID, infoLogLength, nullptr, buffer);
				std::string errorString = buffer;
				delete[] buffer;
			}
		}
		catch (...)
		{
			// Cleanup
			glDetachShader(programID, vtxShaderID);
			glDetachShader(programID, fragShaderID);

			glDeleteShader(vtxShaderID);
			glDeleteShader(fragShaderID);
			glDeleteProgram(programID);

			throw;
		}

		glDetachShader(programID, vtxShaderID);
		glDetachShader(programID, fragShaderID);

		glDeleteShader(vtxShaderID);
		glDeleteShader(fragShaderID);

		return programID;
	}

	void drawSquare(double x1, double y1, double sidelength)
	{
		double halfside = sidelength / 2;

		glColor3d(1.0, 0, 1.0);
		glBegin(GL_POLYGON);

		glVertex2d(x1 + halfside, y1 + halfside);
		glVertex2d(x1 + halfside, y1 - halfside);
		glVertex2d(x1 - halfside, y1 - halfside);
		glVertex2d(x1 - halfside, y1 + halfside);

		glEnd();
	}

}
