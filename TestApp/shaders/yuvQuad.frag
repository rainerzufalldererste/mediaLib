#version 150 core

out vec4 outColour;
in vec2 _texCoord0;
uniform sampler2D textureY;
uniform sampler2D textureU;
uniform sampler2D textureV;
uniform vec2 offsets[4];

float cross(in vec2 a, in vec2 b) { return a.x*b.y - a.y*b.x; }

vec2 invBilinear(in vec2 p, in vec2 a, in vec2 b, in vec2 c, in vec2 d)
{
  vec2 e = b - a;
  vec2 f = d - a;
  vec2 g = a - b + c - d;
  vec2 h = p - a;

  float k2 = cross(g, f);
  float k1 = cross(e, f) + cross(h, g);
  float k0 = cross(h, e);

  float w = k1 * k1 - 4.0 * k0 * k2;

  if (w < 0.0)
    return vec2(-1.0);

  w = sqrt(w);

  float v1 = (-k1 - w) / (2.0 * k2);
  float u1 = (h.x - f.x * v1) / (e.x + g.x * v1);

  float v2 = (-k1 + w) / (2.0 * k2);
  float u2 = (h.x - f.x * v2) / (e.x + g.x * v2);

  float u = u1;
  float v = v1;

  if (v < 0.0 || v > 1.0 || u < 0.0 || u > 1.0)
  {
    u = u2;
    v = v2;
  }

  if (v < 0.0 || v > 1.0 || u < 0.0 || u > 1.0)
  {
    u = -1.0;
    v = -1.0;
  }

  return vec2(u, v);
}

void main()
{
  vec2 pos = invBilinear(_texCoord0, offsets[1], offsets[3], offsets[2], offsets[0]);

  float y = texture(textureY, pos).x;
  float u = texture(textureU, pos).x;
  float v = texture(textureV, pos).x;

  vec3 col = vec3(
    clamp(y + 1.370705 * (v - 0.5), 0, 1),
    clamp(y - 0.698001 * (v - 0.5) - 0.337633 * (u - 0.5), 0, 1),
    clamp(y + 1.732446 * (u - 0.5), 0, 1));

  outColour = vec4(col, 1);
}
