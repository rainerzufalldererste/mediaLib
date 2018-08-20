// Copyright 2018 Christoph Stiller
// 
// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files(the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions :
// 
// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
// 
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "mSpriteBatch.h"

mSBETextureCrop::mSBETextureCrop(const mVec2f & startPoint, const mVec2f & endPoint) : textureStartEndPoint(startPoint.x, startPoint.y, endPoint.x, endPoint.y) { }

mSBETextureCrop::operator mVec4f() const
{
  return textureStartEndPoint;
}

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

mSBETextureFlip::mSBETextureFlip(const bool flipX, const bool flipY) : textureFlip(flipX ? -1.0f : 1.0f, flipY ? -1.0f : 1.0f)
{ }

mSBETextureFlip::operator mVec2f() const
{
  return textureFlip;
}
