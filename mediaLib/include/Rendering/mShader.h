#ifndef mShader_h__
#define mShader_h__

#include "mRenderParams.h"
#include "mTexture.h"
#include "mFramebuffer.h"

#ifdef GIT_BUILD // Define __M_FILE__
  #ifdef __M_FILE__
    #undef __M_FILE__
  #endif
  #define __M_FILE__ "m6zXzrHYpGnP1PcwHSP612cO34D06a/9f4WVFvg+f9ZFPndbd5bLL1LSRw5p+8w8UniYr3roqVRRTRso"
#endif

#if defined(mRENDERER_OPENGL)
typedef GLuint mShaderAttributeIndex_t;
#else
typedef size_t mShaderAttributeIndex_t;
#endif

struct mShader_NameReference
{
  char name[28];
  mShaderAttributeIndex_t index;
};

struct mShader
{
  bool initialized = false;

#if defined(mRENDERER_OPENGL)
  GLuint shaderProgram;
#endif

#ifndef GIT_BUILD
  bool loadedFromFile;
  mString vertexShaderText;
  mString fragmentShaderText;
  mString vertexShaderPath;
  mString fragmentShaderPath;
#endif

  mAllocator *pAllocator = nullptr;
  mShader_NameReference *pUniformReferences = nullptr;
  size_t uniformReferenceCount = 0;
  mShader_NameReference *pAttributeReferences = nullptr;
  size_t attributeReferenceCount = 0;
};

#if defined(mRENDERER_OPENGL)
#define mGLSL(...) "#version 150 core\n" #__VA_ARGS__
#endif

mFUNCTION(mShader_Allocate, OUT mPtr<mShader> *pShader, mAllocator *pAllocator);
mFUNCTION(mShader_Create, OUT mShader *pShader, const mString &vertexShader, const mString &fragmentShader, IN OPTIONAL const char *fragDataLocation = nullptr);
mFUNCTION(mShader_CreateFromFile, OUT mShader *pShader, const mString &filename);
mFUNCTION(mShader_CreateFromFile, OUT mShader *pShader, const mString &vertexShaderPath, const mString &fragmentShaderPath, IN OPTIONAL const char *fragDataLocation = nullptr);
mFUNCTION(mShader_Destroy, IN_OUT mShader *pShader);

mFUNCTION(mShader_SetTo, mPtr<mShader> &shader, const mString &vertexShader, const mString &fragmentShader, IN OPTIONAL const char *fragDataLocation = nullptr);
mFUNCTION(mShader_SetToFile, mPtr<mShader> &shader, const mString &vertexShaderPath, const mString &fragmentShaderPath, IN OPTIONAL const char *fragDataLocation = nullptr);
mFUNCTION(mShader_SetToFile, mPtr<mShader> &shader, const mString &filename);

#ifndef GIT_BUILD
mFUNCTION(mShader_ReloadFromFile, mPtr<mShader> &shader);
#endif

mFUNCTION(mShader_Bind, mShader &shader);
mFUNCTION(mShader_Bind, mPtr<mShader> &shader);
mFUNCTION(mShader_AfterForeign);

bool mShader_IsActive(const mShader &shader);

inline bool mShader_IsActive(const mPtr<mShader> &shader)
{
  if (shader == nullptr)
    return false;
  
  return mShader_IsActive(*shader);
}

mShaderAttributeIndex_t mShader_GetUniformIndex(mShader &shader, const char *uniformName);
inline mShaderAttributeIndex_t mShader_GetUniformIndex(mPtr<mShader> &shader, const char *uniformName) { return mShader_GetUniformIndex(*shader, uniformName); }
inline mShaderAttributeIndex_t mShader_GetUniformIndex(mShader &shader, const mString &uniformName) { return mShader_GetUniformIndex(shader, uniformName.c_str()); }
inline mShaderAttributeIndex_t mShader_GetUniformIndex(mPtr<mShader> &shader, const mString &uniformName) { return mShader_GetUniformIndex(*shader, uniformName.c_str()); }

mShaderAttributeIndex_t mShader_GetAttributeIndex(mShader &shader, const char *attributeName);
inline mShaderAttributeIndex_t mShader_GetAttributeIndex(mPtr<mShader> &shader, const char *attributeName) { return mShader_GetAttributeIndex(*shader, attributeName); }
inline mShaderAttributeIndex_t mShader_GetAttributeIndex(mShader &shader, const mString &attributeName) { return mShader_GetAttributeIndex(shader, attributeName.c_str()); }
inline mShaderAttributeIndex_t mShader_GetAttributeIndex(mPtr<mShader> &shader, const mString &attributeName) { return mShader_GetAttributeIndex(*shader, attributeName.c_str()); }

mFUNCTION(mShader_SetUniformAtIndex, mShader &shader, const mShaderAttributeIndex_t index, const int32_t v);
mFUNCTION(mShader_SetUniformAtIndex, mShader &shader, const mShaderAttributeIndex_t index, const uint32_t v);
mFUNCTION(mShader_SetUniformAtIndex, mShader &shader, const mShaderAttributeIndex_t index, const float_t v);
mFUNCTION(mShader_SetUniformAtIndex, mShader &shader, const mShaderAttributeIndex_t index, const mVec2f &v);
mFUNCTION(mShader_SetUniformAtIndex, mShader &shader, const mShaderAttributeIndex_t index, const mVec2i32 &v);
mFUNCTION(mShader_SetUniformAtIndex, mShader &shader, const mShaderAttributeIndex_t index, const mVec3f &v);
mFUNCTION(mShader_SetUniformAtIndex, mShader &shader, const mShaderAttributeIndex_t index, const mVec3i32 &v);
mFUNCTION(mShader_SetUniformAtIndex, mShader &shader, const mShaderAttributeIndex_t index, const mVec4f &v);
mFUNCTION(mShader_SetUniformAtIndex, mShader &shader, const mShaderAttributeIndex_t index, const mVec4i32 &v);
mFUNCTION(mShader_SetUniformAtIndex, mShader &shader, const mShaderAttributeIndex_t index, const mVector &v);
mFUNCTION(mShader_SetUniformAtIndex, mShader &shader, const mShaderAttributeIndex_t index, const mMatrix &v);
mFUNCTION(mShader_SetUniformAtIndex, mShader &shader, const mShaderAttributeIndex_t index, mTexture &v);
mFUNCTION(mShader_SetUniformAtIndex, mShader &shader, const mShaderAttributeIndex_t index, mPtr<mTexture> &v);
mFUNCTION(mShader_SetUniformAtIndex, mShader &shader, const mShaderAttributeIndex_t index, mTexture3D &v);
mFUNCTION(mShader_SetUniformAtIndex, mShader &shader, const mShaderAttributeIndex_t index, mPtr<mTexture3D> &v);
mFUNCTION(mShader_SetUniformAtIndex, mShader &shader, const mShaderAttributeIndex_t index, mPtr<mFramebuffer> &v);

mFUNCTION(mShader_SetUniformAtIndex, mShader &shader, const mShaderAttributeIndex_t index, const int32_t *pV, const size_t count);
mFUNCTION(mShader_SetUniformAtIndex, mShader &shader, const mShaderAttributeIndex_t index, const uint32_t *pV, const size_t count);
mFUNCTION(mShader_SetUniformAtIndex, mShader &shader, const mShaderAttributeIndex_t index, const float_t *pV, const size_t count);
mFUNCTION(mShader_SetUniformAtIndex, mShader &shader, const mShaderAttributeIndex_t index, const mVec2f *pV, const size_t count);
mFUNCTION(mShader_SetUniformAtIndex, mShader &shader, const mShaderAttributeIndex_t index, const mVec3f *pV, const size_t count);
mFUNCTION(mShader_SetUniformAtIndex, mShader &shader, const mShaderAttributeIndex_t index, const mVec4f *pV, const size_t count);
mFUNCTION(mShader_SetUniformAtIndex, mShader &shader, const mShaderAttributeIndex_t index, const mVector *pV, const size_t count);
mFUNCTION(mShader_SetUniformAtIndex, mShader &shader, const mShaderAttributeIndex_t index, const mTexture *pV, const size_t count);
mFUNCTION(mShader_SetUniformAtIndex, mShader &shader, const mShaderAttributeIndex_t index, const mPtr<mTexture> *pV, const size_t count);
mFUNCTION(mShader_SetUniformAtIndex, mShader &shader, const mShaderAttributeIndex_t index, const mTexture3D *pV, const size_t count);
mFUNCTION(mShader_SetUniformAtIndex, mShader &shader, const mShaderAttributeIndex_t index, const mPtr<mTexture3D> *pV, const size_t count);
mFUNCTION(mShader_SetUniformAtIndex, mShader &shader, const mShaderAttributeIndex_t index, const mPtr<mFramebuffer> *pV, const size_t count);

//////////////////////////////////////////////////////////////////////////

template <typename T>
mFUNCTION(mShader_SetUniform, mShader &shader, const char *uniformName, T v)
{
  mFUNCTION_SETUP();

  mERROR_CHECK(mShader_SetUniformAtIndex(shader, mShader_GetUniformIndex(shader, uniformName), v));

  mGL_DEBUG_ERROR_CHECK();

  mRETURN_SUCCESS();
}

template <typename T>
mFUNCTION(mShader_SetUniform, mPtr<mShader> &shader, const char *uniformName, T v)
{
  mFUNCTION_SETUP();

  mERROR_CHECK(mShader_SetUniformAtIndex(*shader.GetPointer(), mShader_GetUniformIndex(shader, uniformName), v));

  mGL_DEBUG_ERROR_CHECK();

  mRETURN_SUCCESS();
}
template <typename T>
mFUNCTION(mShader_SetUniform, mShader &shader, const mString &uniformName, T v)
{
  mFUNCTION_SETUP();

  mERROR_CHECK(mShader_SetUniformAtIndex(shader, mShader_GetUniformIndex(shader, uniformName), v));

  mGL_DEBUG_ERROR_CHECK();

  mRETURN_SUCCESS();
}

template <typename T>
mFUNCTION(mShader_SetUniform, mPtr<mShader> &shader, const mString &uniformName, T v)
{
  mFUNCTION_SETUP();

  mERROR_CHECK(mShader_SetUniformAtIndex(*shader.GetPointer(), mShader_GetUniformIndex(shader, uniformName), v));

  mGL_DEBUG_ERROR_CHECK();

  mRETURN_SUCCESS();
}

//////////////////////////////////////////////////////////////////////////

template <typename T, size_t TCount>
mFUNCTION(mShader_SetUniform, mShader &shader, const char *uniformName, T pV[TCount])
{
  mFUNCTION_SETUP();

  const mShaderAttributeIndex_t uniformLocation =
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

  const mShaderAttributeIndex_t uniformLocation =
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

  const mShaderAttributeIndex_t uniformLocation =
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

  const mShaderAttributeIndex_t uniformLocation =
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

  const mShaderAttributeIndex_t uniformLocation =
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

  const mShaderAttributeIndex_t uniformLocation =
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

  const mShaderAttributeIndex_t uniformLocation =
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

  const mShaderAttributeIndex_t uniformLocation =
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
