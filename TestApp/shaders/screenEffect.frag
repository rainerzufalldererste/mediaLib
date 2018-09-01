#version 150 core

out vec4 glFragColour;
in vec2 _texCoord0;

uniform sampler2D texture0;

void main()
{
    glFragColour = texture(texture0, _texCoord0).bgra;
}