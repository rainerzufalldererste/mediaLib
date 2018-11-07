#include "mSpriteBatch.h"

mSBETextureCrop::mSBETextureCrop() : textureStartEndPoint(0.0f, 0.0f, 1.0f, 1.0f)
{ }

mSBETextureCrop::mSBETextureCrop(const mVec2f & startPoint, const mVec2f & endPoint) : textureStartEndPoint(startPoint.x, startPoint.y, endPoint.x, endPoint.y) { }

mSBETextureCrop::operator mVec4f() const
{
  return textureStartEndPoint;
}

mSBEColour::mSBEColour(const float_t r, const float_t g, const float_t b) : colour(r, g, b, 1)
{ }

mSBEColour::mSBEColour(const float_t r, const float_t g, const float_t b, const float_t a) : colour(r, g, b, a)
{ }

mSBEColour::mSBEColour(const mVec3f & colour) : colour(colour, 1.0f)
{ }

mSBEColour::mSBEColour(const mVec4f & colour) : colour(colour)
{ }

mSBEColour::operator mVec4f() const
{
  return colour;
}

mSBERotation::mSBERotation(const float_t rotation) : rotation(rotation)
{ }

mSBERotation::operator float_t() const
{
  return rotation;
}

mSBEMatrixTransform::mSBEMatrixTransform(const mMatrix & matrix) : matrix(matrix)
{ }

mSBEMatrixTransform::operator mMatrix() const
{
  return matrix;
}

mSBETextureFlip::mSBETextureFlip(const bool flipX, const bool flipY) : textureFlip(flipX ? 1.0f : 0.0f, flipY ? 1.0f : 0.0f)
{ }

mSBETextureFlip::operator mVec2f() const
{
  return textureFlip;
}
