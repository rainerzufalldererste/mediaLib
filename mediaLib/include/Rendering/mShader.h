#ifndef mShader_h__
#define mShader_h__

#include "mRenderParams.h"
#include "mTexture.h"
#include "mFramebuffer.h"

struct mShader
{
  bool initialized = false;
  mString vertexShader;
  mString fragmentShader;

#if defined(mRENDERER_OPENGL)
  GLuint shaderProgram;
  bool loadedFromFile;
#endif
};

#if defined(mRENDERER_OPENGL)
#define mGLSL(src) "#version 150 core\n" #src
#endif

mFUNCTION(mShader_Allocate, OUT mPtr<mShader> *pShader, mAllocator *pAllocator);
mFUNCTION(mShader_Create, OUT mShader *pShader, const mString &vertexShader, const mString &fragmentShader, IN OPTIONAL const char *fragDataLocation = nullptr);
mFUNCTION(mShader_CreateFromFile, OUT mShader *pShader, const mString &filename);
mFUNCTION(mShader_CreateFromFile, OUT mShader *pShader, const mString &vertexShaderPath, const mString &fragmentShaderPath, IN OPTIONAL const char *fragDataLocation = nullptr);
mFUNCTION(mShader_Destroy, IN_OUT mShader *pShader);

mFUNCTION(mShader_SetTo, mPtr<mShader> &shader, const mString &vertexShader, const mString &fragmentShader, IN OPTIONAL const char *fragDataLocation = nullptr);
mFUNCTION(mShader_SetToFile, mPtr<mShader> &shader, const mString &vertexShaderPath, const mString &fragmentShaderPath, IN OPTIONAL const char *fragDataLocation = nullptr);
mFUNCTION(mShader_SetToFile, mPtr<mShader> &shader, const mString &filename);

mFUNCTION(mShader_ReloadFromFile, mPtr<mShader> &shader);

mFUNCTION(mShader_Bind, mShader &shader);

#if defined(mRENDERER_OPENGL)
typedef GLuint shaderAttributeIndex_t;
#else
typedef size_t shaderAttributeIndex_t;
#endif

mFUNCTION(mShader_SetUniformAtIndex, mShader &shader, const shaderAttributeIndex_t index, const int32_t v);
mFUNCTION(mShader_SetUniformAtIndex, mShader &shader, const shaderAttributeIndex_t index, const uint32_t v);
mFUNCTION(mShader_SetUniformAtIndex, mShader &shader, const shaderAttributeIndex_t index, const float_t v);
mFUNCTION(mShader_SetUniformAtIndex, mShader &shader, const shaderAttributeIndex_t index, const mVec2f &v);
mFUNCTION(mShader_SetUniformAtIndex, mShader &shader, const shaderAttributeIndex_t index, const mVec3f &v);
mFUNCTION(mShader_SetUniformAtIndex, mShader &shader, const shaderAttributeIndex_t index, const mVec4f &v);
mFUNCTION(mShader_SetUniformAtIndex, mShader &shader, const shaderAttributeIndex_t index, const mVector &v);
mFUNCTION(mShader_SetUniformAtIndex, mShader &shader, const shaderAttributeIndex_t index, const mMatrix &v);
mFUNCTION(mShader_SetUniformAtIndex, mShader &shader, const shaderAttributeIndex_t index, mTexture &v);
mFUNCTION(mShader_SetUniformAtIndex, mShader &shader, const shaderAttributeIndex_t index, mPtr<mTexture> &v);
mFUNCTION(mShader_SetUniformAtIndex, mShader &shader, const shaderAttributeIndex_t index, mTexture3D &v);
mFUNCTION(mShader_SetUniformAtIndex, mShader &shader, const shaderAttributeIndex_t index, mPtr<mTexture3D> &v);
mFUNCTION(mShader_SetUniformAtIndex, mShader &shader, const shaderAttributeIndex_t index, mPtr<mFramebuffer> &v);

mFUNCTION(mShader_SetUniformAtIndex, mShader &shader, const shaderAttributeIndex_t index, const int32_t *pV, const size_t count);
mFUNCTION(mShader_SetUniformAtIndex, mShader &shader, const shaderAttributeIndex_t index, const uint32_t *pV, const size_t count);
mFUNCTION(mShader_SetUniformAtIndex, mShader &shader, const shaderAttributeIndex_t index, const float_t *pV, const size_t count);
mFUNCTION(mShader_SetUniformAtIndex, mShader &shader, const shaderAttributeIndex_t index, const mVec2f *pV, const size_t count);
mFUNCTION(mShader_SetUniformAtIndex, mShader &shader, const shaderAttributeIndex_t index, const mVec3f *pV, const size_t count);
mFUNCTION(mShader_SetUniformAtIndex, mShader &shader, const shaderAttributeIndex_t index, const mVec4f *pV, const size_t count);
mFUNCTION(mShader_SetUniformAtIndex, mShader &shader, const shaderAttributeIndex_t index, const mVector *pV, const size_t count);
mFUNCTION(mShader_SetUniformAtIndex, mShader &shader, const shaderAttributeIndex_t index, const mTexture *pV, const size_t count);
mFUNCTION(mShader_SetUniformAtIndex, mShader &shader, const shaderAttributeIndex_t index, const mPtr<mTexture> *pV, const size_t count);
mFUNCTION(mShader_SetUniformAtIndex, mShader &shader, const shaderAttributeIndex_t index, const mTexture3D *pV, const size_t count);
mFUNCTION(mShader_SetUniformAtIndex, mShader &shader, const shaderAttributeIndex_t index, const mPtr<mTexture3D> *pV, const size_t count);
mFUNCTION(mShader_SetUniformAtIndex, mShader &shader, const shaderAttributeIndex_t index, const mPtr<mFramebuffer> *pV, const size_t count);

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
mFUNCTION(mShader_SetUniform, mShader &shader, const mString &uniformName, T v)
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
mFUNCTION(mShader_SetUniform, mPtr<mShader> &shader, const mString &uniformName, T v)
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
mFUNCTION(mShader_SetUniform, mShader &shader, const mString &uniformName, T pV[TCount])
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
mFUNCTION(mShader_SetUniform, mPtr<mShader> &shader, const mString &uniformName, T pV[TCount])
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
mFUNCTION(mShader_SetUniform, mShader &shader, const mString &uniformName, T *pV, const size_t count)
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
mFUNCTION(mShader_SetUniform, mPtr<mShader> &shader, const mString &uniformName, T *pV, const size_t count)
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
