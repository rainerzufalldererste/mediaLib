// Copyright 2018 Christoph Stiller
// 
// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files(the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions :
// 
// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
// 
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef mShader_h__
#define mShader_h__

#include "mRenderParams.h"
#include "mTexture.h"

struct mShader
{
  bool initialized = false;

#if defined(mRENDERER_OPENGL)
  GLuint shaderProgram;
#endif
};

#if defined(mRENDERER_OPENGL)
#define mGLSL(src) "#version 150 core\n" #src
#endif

mFUNCTION(mShader_Create, OUT mShader *pShader, const std::string &vertexShader, const std::string &fragmentShader, IN OPTIONAL const char *fragDataLocation = nullptr);
mFUNCTION(mShader_CreateFromFile, OUT mShader *pShader, const std::wstring &filename);
mFUNCTION(mShader_CreateFromFile, OUT mShader *pShader, const std::wstring &vertexShaderPath, const std::wstring &fragmentShaderPath, IN OPTIONAL const char *fragDataLocation = nullptr);
mFUNCTION(mShader_Destroy, IN_OUT mShader *pShader);

mFUNCTION(mShader_SetTo, mPtr<mShader> &shader, const std::string &vertexShader, const std::string &fragmentShader, IN OPTIONAL const char *fragDataLocation = nullptr);
mFUNCTION(mShader_SetToFile, mPtr<mShader> &shader, const std::wstring &vertexShaderPath, const std::wstring &fragmentShaderPath, IN OPTIONAL const char *fragDataLocation = nullptr);
mFUNCTION(mShader_SetToFile, mPtr<mShader> &shader, const std::wstring &filename);

mFUNCTION(mShader_Bind, mShader &shader);

#if defined(mRENDERER_OPENGL)
typedef GLuint shaderAttributeIndex_t;
#else
typedef size_t shaderAttributeIndex_t;
#endif

mFUNCTION(mShader_SetUniformAtIndex, mShader &shader, const shaderAttributeIndex_t index, const float_t v);
mFUNCTION(mShader_SetUniformAtIndex, mShader &shader, const shaderAttributeIndex_t index, const mVec2f &v);
mFUNCTION(mShader_SetUniformAtIndex, mShader &shader, const shaderAttributeIndex_t index, const mVec3f &v);
mFUNCTION(mShader_SetUniformAtIndex, mShader &shader, const shaderAttributeIndex_t index, const mVec4f &v);
mFUNCTION(mShader_SetUniformAtIndex, mShader &shader, const shaderAttributeIndex_t index, const mVector &v);
mFUNCTION(mShader_SetUniformAtIndex, mShader &shader, const shaderAttributeIndex_t index, const mMatrix &v);
mFUNCTION(mShader_SetUniformAtIndex, mShader &shader, const shaderAttributeIndex_t index, mTexture &v);
mFUNCTION(mShader_SetUniformAtIndex, mShader &shader, const shaderAttributeIndex_t index, mPtr<mTexture> &v);

mFUNCTION(mShader_SetUniformAtIndex, mShader &shader, const shaderAttributeIndex_t index, const float_t *pV, const size_t count);
mFUNCTION(mShader_SetUniformAtIndex, mShader &shader, const shaderAttributeIndex_t index, const mVec2f *pV, const size_t count);
mFUNCTION(mShader_SetUniformAtIndex, mShader &shader, const shaderAttributeIndex_t index, const mVec3f *pV, const size_t count);
mFUNCTION(mShader_SetUniformAtIndex, mShader &shader, const shaderAttributeIndex_t index, const mVec4f *pV, const size_t count);
mFUNCTION(mShader_SetUniformAtIndex, mShader &shader, const shaderAttributeIndex_t index, const mVector *pV, const size_t count);
mFUNCTION(mShader_SetUniformAtIndex, mShader &shader, const shaderAttributeIndex_t index, const mTexture *pV, const size_t count);
mFUNCTION(mShader_SetUniformAtIndex, mShader &shader, const shaderAttributeIndex_t index, const mPtr<mTexture> *pV, const size_t count);

//////////////////////////////////////////////////////////////////////////

template <typename T>
mFUNCTION(mShader_SetUniform, mShader &shader, const char *uniformName, T v)
{
  mFUNCTION_SETUP();

  const shaderAttributeIndex_t uniformLocation =
#if defined (mRENDERER_OPENGL)
    glGetUniformLocation(shader.shaderProgram, uniformName);
#else
    0;
  mRETURN_RESULT(mR_NotImplemented);
#endif

  mERROR_CHECK(mShader_SetUniformAtIndex(shader, uniformLocation, v));

  mGL_DEBUG_ERROR_CHECK();

  mRETURN_SUCCESS();
}

template <typename T>
mFUNCTION(mShader_SetUniform, mPtr<mShader> &shader, const char *uniformName, T v)
{
  mFUNCTION_SETUP();

  const shaderAttributeIndex_t uniformLocation =
#if defined (mRENDERER_OPENGL)
    glGetUniformLocation(shader->shaderProgram, uniformName);
#else
    0;
  mRETURN_RESULT(mR_NotImplemented);
#endif

  mERROR_CHECK(mShader_SetUniformAtIndex(*shader.GetPointer(), uniformLocation, v));

  mGL_DEBUG_ERROR_CHECK();

  mRETURN_SUCCESS();
}
template <typename T>
mFUNCTION(mShader_SetUniform, mShader &shader, const std::string &uniformName, T v)
{
  mFUNCTION_SETUP();

  const shaderAttributeIndex_t uniformLocation =
#if defined (mRENDERER_OPENGL)
    glGetUniformLocation(shader.shaderProgram, uniformName.c_str());
#else
    0;
  mRETURN_RESULT(mR_NotImplemented);
#endif

  mERROR_CHECK(mShader_SetUniformAtIndex(shader, uniformLocation, v));

  mGL_DEBUG_ERROR_CHECK();

  mRETURN_SUCCESS();
}

template <typename T>
mFUNCTION(mShader_SetUniform, mPtr<mShader> &shader, const std::string &uniformName, T v)
{
  mFUNCTION_SETUP();

  const shaderAttributeIndex_t uniformLocation =
#if defined (mRENDERER_OPENGL)
    glGetUniformLocation(shader->shaderProgram, uniformName.c_str());
#else
    0;
  mRETURN_RESULT(mR_NotImplemented);
#endif

  mERROR_CHECK(mShader_SetUniformAtIndex(*shader.GetPointer(), uniformLocation, v));

  mGL_DEBUG_ERROR_CHECK();

  mRETURN_SUCCESS();
}

//////////////////////////////////////////////////////////////////////////

template <typename T, size_t TCount>
mFUNCTION(mShader_SetUniform, mShader &shader, const char *uniformName, T pV[TCount])
{
  mFUNCTION_SETUP();

  const shaderAttributeIndex_t uniformLocation =
#if defined (mRENDERER_OPENGL)
    glGetUniformLocation(shader.shaderProgram, uniformName);
#else
    0;
  mRETURN_RESULT(mR_NotImplemented);
#endif

  mERROR_CHECK(mShader_SetUniformAtIndex(shader, uniformLocation, pV, TCount));

  mGL_DEBUG_ERROR_CHECK();

  mRETURN_SUCCESS();
}

template <typename T, size_t TCount>
mFUNCTION(mShader_SetUniform, mPtr<mShader> &shader, const char *uniformName, T pV[TCount])
{
  mFUNCTION_SETUP();

  const shaderAttributeIndex_t uniformLocation =
#if defined (mRENDERER_OPENGL)
    glGetUniformLocation(shader->shaderProgram, uniformName);
#else
    0;
  mRETURN_RESULT(mR_NotImplemented);
#endif

  mERROR_CHECK(mShader_SetUniformAtIndex(*shader.GetPointer(), uniformLocation, pV, TCount));

  mGL_DEBUG_ERROR_CHECK();

  mRETURN_SUCCESS();
}
template <typename T, size_t TCount>
mFUNCTION(mShader_SetUniform, mShader &shader, const std::string &uniformName, T pV[TCount])
{
  mFUNCTION_SETUP();

  const shaderAttributeIndex_t uniformLocation =
#if defined (mRENDERER_OPENGL)
    glGetUniformLocation(shader.shaderProgram, uniformName.c_str());
#else
    0;
  mRETURN_RESULT(mR_NotImplemented);
#endif

  mERROR_CHECK(mShader_SetUniformAtIndex(shader, uniformLocation, pV, TCount));

  mGL_DEBUG_ERROR_CHECK();

  mRETURN_SUCCESS();
}

template <typename T, size_t TCount>
mFUNCTION(mShader_SetUniform, mPtr<mShader> &shader, const std::string &uniformName, T pV[TCount])
{
  mFUNCTION_SETUP();

  const shaderAttributeIndex_t uniformLocation =
#if defined (mRENDERER_OPENGL)
    glGetUniformLocation(shader->shaderProgram, uniformName.c_str());
#else
    0;
  mRETURN_RESULT(mR_NotImplemented);
#endif

  mERROR_CHECK(mShader_SetUniformAtIndex(*shader.GetPointer(), uniformLocation, pV, TCount));

  mGL_DEBUG_ERROR_CHECK();

  mRETURN_SUCCESS();
}

//////////////////////////////////////////////////////////////////////////

template <typename T>
mFUNCTION(mShader_SetUniform, mShader &shader, const char *uniformName, T *pV, const size_t count)
{
  mFUNCTION_SETUP();

  const shaderAttributeIndex_t uniformLocation =
#if defined (mRENDERER_OPENGL)
    glGetUniformLocation(shader.shaderProgram, uniformName);
#else
    0;
  mRETURN_RESULT(mR_NotImplemented);
#endif

  mERROR_CHECK(mShader_SetUniformAtIndex(shader, uniformLocation, pV, count));

  mGL_DEBUG_ERROR_CHECK();

  mRETURN_SUCCESS();
}

template <typename T>
mFUNCTION(mShader_SetUniform, mPtr<mShader> &shader, const char *uniformName, T *pV, const size_t count)
{
  mFUNCTION_SETUP();

  const shaderAttributeIndex_t uniformLocation =
#if defined (mRENDERER_OPENGL)
    glGetUniformLocation(shader->shaderProgram, uniformName);
#else
    0;
  mRETURN_RESULT(mR_NotImplemented);
#endif

  mERROR_CHECK(mShader_SetUniformAtIndex(*shader.GetPointer(), uniformLocation, pV, count));

  mGL_DEBUG_ERROR_CHECK();

  mRETURN_SUCCESS();
}
template <typename T>
mFUNCTION(mShader_SetUniform, mShader &shader, const std::string &uniformName, T *pV, const size_t count)
{
  mFUNCTION_SETUP();

  const shaderAttributeIndex_t uniformLocation =
#if defined (mRENDERER_OPENGL)
    glGetUniformLocation(shader.shaderProgram, uniformName.c_str());
#else
    0;
  mRETURN_RESULT(mR_NotImplemented);
#endif

  mERROR_CHECK(mShader_SetUniformAtIndex(shader, uniformLocation, pV, count));

  mGL_DEBUG_ERROR_CHECK();

  mRETURN_SUCCESS();
}

template <typename T>
mFUNCTION(mShader_SetUniform, mPtr<mShader> &shader, const std::string &uniformName, T *pV, const size_t count)
{
  mFUNCTION_SETUP();

  const shaderAttributeIndex_t uniformLocation =
#if defined (mRENDERER_OPENGL)
    glGetUniformLocation(shader->shaderProgram, uniformName.c_str());
#else
    0;
  mRETURN_RESULT(mR_NotImplemented);
#endif

  mERROR_CHECK(mShader_SetUniformAtIndex(*shader.GetPointer(), uniformLocation, pV, count));

  mGL_DEBUG_ERROR_CHECK();

  mRETURN_SUCCESS();
}

#endif // mShader_h__
