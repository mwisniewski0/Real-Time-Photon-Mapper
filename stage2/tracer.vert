#version 430 core
layout(location = 0) in vec3 vertexPosition;
out vec2 screenPos;

void main(){
  gl_Position.xyz = vertexPosition;
  gl_Position.w = 1.0;
  screenPos = vec2((vertexPosition.x + 1.0) / 2.0, (vertexPosition.y + 1.0) / 2.0);
}
