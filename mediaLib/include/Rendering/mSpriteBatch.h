// Copyright 2018 Christoph Stiller
// 
// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files(the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions :
// 
// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
// 
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef mSpriteBatch_h__
#define mSpriteBatch_h__

#include "mRenderParams.h"
#include "mMesh.h"

enum mSpriteBatch_SpriteSortMode
{
  mSB_SSM_None,
  mSB_SSM_BackToFront,
  mSB_SSM_FrontToBack,
};

enum mSpriteBatch_AlphaMode
{
  mSB_AM_NoAlpha,
  mSB_AM_Additive,
  mSB_AM_AlphaBlend,
  mSB_AM_Premultiplied,
};

enum mSpriteBatch_TextureSampleMode
{
  mSB_TSM_NearestNeighbor,
  mSB_TSM_LinearFiltering,
};

template <typename ...Args>
struct mSpriteBatch_Internal_RenderObject
{
  mSpriteBatch_Internal_RenderObject() = default;

  mPtr<mTexture> texture;
  mVec3f position;
  mVec2f size;
  std::tuple<Args...> args;
};

struct mSpriteBatch_ShaderParams
{
  uint8_t textureCrop : 1;
  uint8_t textureCropIndex;
  uint8_t colour : 1;
  uint8_t colourIndex;
  uint8_t rotation : 1;
  uint8_t rotationIndex;
  uint8_t matrixTransform : 1;
  uint8_t matrixTransformIndex;
  uint8_t textureFlip : 1;
  uint8_t textureFlipIndex;
};

template <typename ...Args>
struct mSpriteBatch
{
  mSpriteBatch_SpriteSortMode spriteSortMode;
  mSpriteBatch_AlphaMode alphaMode;
  mSpriteBatch_TextureSampleMode textureSampleMode;
  bool isStarted;
  mPtr<mShader> shader;
  mPtr<mQueue<mSpriteBatch_Internal_RenderObject<Args...>>> enqueuedRenderObjects;
  mSpriteBatch_ShaderParams shaderParams;
#if defined (mRENDERER_OPENGL)
  GLuint vbo;
#endif
};

template <typename ...Args>
mFUNCTION(mSpriteBatch_Internal_RenderObject_Create, OUT mSpriteBatch_Internal_RenderObject<Args...> *pRenderObject, const mPtr<mTexture> &texture, const mVec2f position, const mVec2f size, const float_t depth, Args&& ...args);

template <typename ...Args>
mFUNCTION(mSpriteBatch_Internal_RenderObject_Destroy, IN_OUT mSpriteBatch_Internal_RenderObject<Args...> *pRenderObject);

template <typename ...Args>
mFUNCTION(mSpriteBatch_Internal_RenderObject_Render, mSpriteBatch_Internal_RenderObject<Args...> &renderObject, mPtr<mSpriteBatch<Args...>> &spriteBatch);

template <typename ...Args>
mFUNCTION(mSpriteBatch_Create, OUT mPtr<mSpriteBatch<Args...>> *pSpriteBatch, IN mAllocator *pAllocator);

template <typename ...Args>
mFUNCTION(mSpriteBatch_Destroy, IN_OUT mPtr<mSpriteBatch<Args...>> *pSpriteBatch);

template <typename ...Args>
mFUNCTION(mSpriteBatch_Begin, mPtr<mSpriteBatch<Args...>> &spriteBatch);

template <typename ...Args>
mFUNCTION(mSpriteBatch_Begin, mPtr<mSpriteBatch<Args...>> &spriteBatch, const mSpriteBatch_SpriteSortMode spriteSortMode, const mSpriteBatch_AlphaMode alphaMode, const mSpriteBatch_TextureSampleMode sampleMode = mSB_TSM_LinearFiltering);

template <typename ...Args>
mFUNCTION(mSpriteBatch_DrawWithDepth, mPtr<mSpriteBatch<Args...>> &spriteBatch, const mVec2f &position, const mPtr<mTexture> &texture, const float_t depth, Args&&... args);

template <typename ...Args>
mFUNCTION(mSpriteBatch_DrawWithDepth, mPtr<mSpriteBatch<Args...>> &spriteBatch, const mRectangle2D<float_t> &rect, const mPtr<mTexture> &texture, const float_t depth, Args&&... args);

template <typename ...Args>
mFUNCTION(mSpriteBatch_Draw, mPtr<mSpriteBatch<Args...>> &spriteBatch, const mVec2f &position, const mPtr<mTexture> &texture, Args&&... args);

template <typename ...Args>
mFUNCTION(mSpriteBatch_Draw, mPtr<mSpriteBatch<Args...>> &spriteBatch, const mRectangle2D<float_t> &rect, const mPtr<mTexture> &texture, Args&&... args);

template <typename ...Args>
mFUNCTION(mSpriteBatch_End, mPtr<mSpriteBatch<Args...>> &spriteBatch);

//////////////////////////////////////////////////////////////////////////

enum mSpriteBatchExtention_Type
{
  mSBE_T_TextureCrop,
  mSBE_T_Colour,
  mSBE_T_Rotation,
  mSBE_T_MatrixTransform,
  mSBE_T_TextureFlip,
};

struct mSpriteBatchExtention
{
  // Base Struct.
};

#define mSBETextureCrop_UniformName "mSBETextureCrop"

struct mSBETextureCrop : mSpriteBatchExtention
{
  mVec4f textureStartEndPoint;

  mSBETextureCrop() = default;
  explicit mSBETextureCrop(const mVec2f &startPoint, const mVec2f &endPoint);

  static const mSpriteBatchExtention_Type type = mSBE_T_TextureCrop;
  static const size_t parameterSize = sizeof(mVec4f);

  operator mVec4f() const;
};

#define mSBEColour_UniformName "mSBEColour"

struct mSBEColour : mSpriteBatchExtention
{
  mVec4f colour;

  mSBEColour() = default;
  explicit mSBEColour(const mVec3f &colour);
  explicit mSBEColour(const mVec4f &colour);

  static const mSpriteBatchExtention_Type type = mSBE_T_Colour;
  static const size_t parameterSize = sizeof(mVec4f);

  operator mVec4f() const;
};

#define mSBERotation_UniformName "mSBERotation"

struct mSBERotation : mSpriteBatchExtention
{
  float_t rotation;

  mSBERotation() = default;
  explicit mSBERotation(const float_t rotation);

  static const mSpriteBatchExtention_Type type = mSBE_T_Rotation;
  static const size_t parameterSize = sizeof(float_t);

  operator float_t() const;
};

#define mSBEMatrixTransform_UniformName "mSBEMatrixTransform"

struct mSBEMatrixTransform : mSpriteBatchExtention
{
  mMatrix matrix;

  mSBEMatrixTransform() = default;
  explicit mSBEMatrixTransform(const mMatrix &matrix);

  static const mSpriteBatchExtention_Type type = mSBE_T_MatrixTransform;
  static const size_t parameterSize = sizeof(mMatrix);

  operator mMatrix() const;
};

#define mSBETextureFlip_UniformName "mSBEMatrixTransform"

struct mSBETextureFlip : mSpriteBatchExtention
{
  mVec2f textureFlip;

  mSBETextureFlip() = default;
  explicit mSBETextureFlip(const bool flipX, const bool flipY);

  static const mSpriteBatchExtention_Type type = mSBE_T_TextureFlip;
  static const size_t parameterSize = sizeof(mVec2f);

  operator mVec2f() const;
};

#include "mSpriteBatch.inl"

#endif // mSpriteBatch_h__
