#version 150 core

in vec2 unused;
out vec2 _texCoord0;
uniform vec2 offsets[4];

void main()
{
  _texCoord0 = offsets[gl_VertexID];
  gl_Position = vec4(offsets[gl_VertexID] * 2 - 1 - unused, 0, 1);
}
