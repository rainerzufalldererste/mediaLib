#include "mShader.h"
#include "mFile.h"

#if defined (mRENDERER_OPENGL)
#ifdef GIT_BUILD // Define __M_FILE__
  #ifdef __M_FILE__
    #undef __M_FILE__
  #endif
  #define __M_FILE__ "CP/+iAj83Qa9F7fA/hEsrZHGN6ltysXmthtC0t9KsybDSrKZYCvAtcUdR2ubW1DXTgmibHGPVOGf9fg4"
#endif

GLuint mShader_CurrentlyBoundShader = (GLuint)-1;
#endif

mFUNCTION(mShader_Allocate, OUT mPtr<mShader> *pShader, mAllocator *pAllocator)
{
  mFUNCTION_SETUP();

  mERROR_IF(pShader == nullptr, mR_ArgumentNull);

  mERROR_CHECK(mSharedPointer_Allocate(pShader, pAllocator, (std::function<void(mShader *)>)[](mShader *pData) {mShader_Destroy(pData);}, 1));

  mRETURN_SUCCESS();
}

mFUNCTION(mShader_Create, OUT mShader *pShader, const mString &vertexShader, const mString &fragmentShader, IN OPTIONAL const char *fragDataLocation /* = nullptr */)
{
  mFUNCTION_SETUP();

  mERROR_IF(pShader == nullptr, mR_ArgumentNull);

  mERROR_CHECK(mZeroMemory(pShader));

#ifndef GIT_BUILD
  mERROR_CHECK(mString_Create(&pShader->fragmentShaderText, fragmentShader));
  mERROR_CHECK(mString_Create(&pShader->vertexShaderText, vertexShader));
#endif

#if defined(mRENDERER_OPENGL)

  char *vertexSource = nullptr;
  mDEFER(mAllocator_FreePtr(nullptr, &vertexSource));
  mERROR_CHECK(mAllocator_Allocate(nullptr, &vertexSource, vertexShader.bytes));

  size_t position = 0;

  const mchar_t carriageReturn = mToChar<1>("\r");

  for (auto &&_char : vertexShader)
  {
    if (_char.codePoint != carriageReturn)
    {
      char *s = (char *)_char.character;

      for (size_t i = 0; i < _char.characterSize; i++)
      {
        vertexSource[position] = *s;
        position++;
        s++;
      }
    }
  }

  vertexSource[position] = '\0';

  GLuint vertexShaderHandle = glCreateShader(GL_VERTEX_SHADER);
  mDEFER(glDeleteShader(vertexShaderHandle));
  glShaderSource(vertexShaderHandle, 1, &vertexSource, NULL);
  glCompileShader(vertexShaderHandle);

  GLint status;
  glGetShaderiv(vertexShaderHandle, GL_COMPILE_STATUS, &status);

  if (status != GL_TRUE)
  {
    mPRINT_ERROR("Error compiling vertex shader.\n");
#ifndef GIT_BUILD
    mPRINT_ERROR(vertexSource);
#endif
    mPRINT_ERROR("The following error occured:\n");
    char buffer[1024];
    glGetShaderInfoLog(vertexShaderHandle, sizeof(buffer), nullptr, buffer);
    mPRINT_ERROR(buffer);
    mRETURN_RESULT(mR_ResourceInvalid);
  }

  char *fragmentSource = nullptr;
  mDEFER(mAllocator_FreePtr(nullptr, &fragmentSource));
  mERROR_CHECK(mAllocator_Allocate(nullptr, &fragmentSource, fragmentShader.bytes));

  position = 0;

  for (auto &&_char : fragmentShader)
  {
    if (_char.codePoint != carriageReturn)
    {
      char *s = (char *)_char.character;

      for (size_t i = 0; i < _char.characterSize; i++)
      {
        fragmentSource[position] = *s;
        position++;
        s++;
      }
    }
  }

  fragmentSource[position] = '\0';

  GLuint fragmentShaderHandle = glCreateShader(GL_FRAGMENT_SHADER);
  mDEFER(glDeleteShader(fragmentShaderHandle));
  glShaderSource(fragmentShaderHandle, 1, &fragmentSource, NULL);
  glCompileShader(fragmentShaderHandle);

  status = GL_TRUE;
  glGetShaderiv(fragmentShaderHandle, GL_COMPILE_STATUS, &status);

  if (status != GL_TRUE)
  {
    mPRINT_ERROR("Error compiling fragment shader.\n");
#ifndef GIT_BUILD
    mPRINT_ERROR("%s", fragmentSource);
#endif
    mPRINT_ERROR("The following error occured:\n");
    char buffer[1024];
    glGetShaderInfoLog(fragmentShaderHandle, sizeof(buffer), nullptr, buffer);
    mPRINT_ERROR("%s", buffer);
    mRETURN_RESULT(mR_ResourceInvalid);
  }

  pShader->shaderProgram = glCreateProgram();
  glAttachShader(pShader->shaderProgram, vertexShaderHandle);
  glAttachShader(pShader->shaderProgram, fragmentShaderHandle);

  if (fragDataLocation != nullptr)
    glBindFragDataLocation(pShader->shaderProgram, 0, fragDataLocation);

  glLinkProgram(pShader->shaderProgram);

  glGetProgramiv(pShader->shaderProgram, GL_LINK_STATUS, &status);
  mERROR_IF(status != GL_TRUE, mR_ResourceInvalid);

  pShader->initialized = true;

  mGL_DEBUG_ERROR_CHECK();
#else
  mRETURN_RESULT(mR_NotImplemented);
#endif

  mRETURN_SUCCESS();
}

mFUNCTION(mShader_CreateFromFile, OUT mShader *pShader, const mString &filename)
{
  mFUNCTION_SETUP();

  mERROR_IF(pShader == nullptr, mR_ArgumentNull);

#if defined(mRENDERER_OPENGL)
  mERROR_CHECK(mShader_CreateFromFile(pShader, filename + ".vert", filename + ".frag", nullptr));
#else
  mRETURN_RESULT(mR_NotImplemented);
#endif

  mRETURN_SUCCESS();
}

mFUNCTION(mShader_CreateFromFile, OUT mShader *pShader, const mString &vertexShaderPath, const mString &fragmentShaderPath, IN OPTIONAL const char *fragDataLocation /* = nullptr */)
{
  mFUNCTION_SETUP();

  mERROR_IF(pShader == nullptr, mR_ArgumentNull);

#if defined(mRENDERER_OPENGL)
  mString vert;
  mString frag;

  mERROR_CHECK(mFile_ReadAllText(vertexShaderPath, nullptr, &vert));
  mERROR_CHECK(mFile_ReadAllText(fragmentShaderPath, nullptr, &frag));

  mERROR_CHECK(mShader_Create(pShader, vert, frag, fragDataLocation));

#ifndef GIT_BUILD
  mERROR_CHECK(mString_Create(&pShader->vertexShaderPath, vertexShaderPath));
  mERROR_CHECK(mString_Create(&pShader->fragmentShaderPath, fragmentShaderPath));
  pShader->loadedFromFile = true;
#endif

#else
  mRETURN_RESULT(mR_NotImplemented);
#endif

  mRETURN_SUCCESS();
}

mFUNCTION(mShader_Destroy, IN_OUT mShader *pShader)
{
  mFUNCTION_SETUP();

  mERROR_IF(pShader == nullptr, mR_ArgumentNull);

  mGL_DEBUG_ERROR_CHECK();

#if defined(mRENDERER_OPENGL)

  if (mShader_CurrentlyBoundShader == pShader->shaderProgram)
  {
    mShader_CurrentlyBoundShader = (GLuint)-1;
    mFAIL_DEBUG("The shader that's currently bound will be released. Please make sure you're not relying on this.");
  }

  if (pShader->initialized)
    glDeleteProgram(pShader->shaderProgram);

#else
  mRETURN_RESULT(mR_NotImplemented);
#endif

  pShader->initialized = false;

#ifndef GIT_BUILD
  pShader->loadedFromFile = false;
  mERROR_CHECK(mDestruct(&pShader->vertexShaderPath));
  mERROR_CHECK(mDestruct(&pShader->fragmentShaderPath));
  mERROR_CHECK(mDestruct(&pShader->fragmentShaderText));
  mERROR_CHECK(mDestruct(&pShader->vertexShaderText));
#endif

  mGL_DEBUG_ERROR_CHECK();

  mRETURN_SUCCESS();
}

mFUNCTION(mShader_SetTo, mPtr<mShader> &shader, const mString &vertexShader, const mString &fragmentShader, IN OPTIONAL const char *fragDataLocation)
{
  mFUNCTION_SETUP();

  mERROR_IF(shader == nullptr, mR_ArgumentNull);
  mERROR_IF(!shader->initialized, mR_NotInitialized);

#if defined(mRENDERER_OPENGL)

  GLuint subShaders[3];
  GLsizei subShaderCount;
  glGetAttachedShaders(shader->shaderProgram, 3, &subShaderCount, subShaders);

  for (size_t i = 0; i < subShaderCount; ++i)
  {
    glDeleteShader(subShaders[i]);
    glDetachShader(shader->shaderProgram, subShaders[i]);
  }

  char *vertexSource = nullptr;
  mDEFER(mAllocator_FreePtr(nullptr, &vertexSource));
  mERROR_CHECK(mAllocator_Allocate(nullptr, &vertexSource, vertexShader.bytes));

  size_t position = 0;

  const mchar_t carriageReturn = mToChar<1>("\r");

  for (auto &&_char : vertexShader)
  {
    if (_char.codePoint != carriageReturn)
    {
      char *s = (char *)_char.character;

      for (size_t i = 0; i < _char.characterSize; i++)
      {
        vertexSource[position] = *s;
        position++;
        s++;
      }
    }
  }

  vertexSource[position] = '\0';

  GLuint vertexShaderHandle = glCreateShader(GL_VERTEX_SHADER);
  mDEFER(glDeleteShader(vertexShaderHandle));
  glShaderSource(vertexShaderHandle, 1, &vertexSource, NULL);
  glCompileShader(vertexShaderHandle);

  GLint status;
  glGetShaderiv(vertexShaderHandle, GL_COMPILE_STATUS, &status);

  if (status != GL_TRUE)
  {
    mPRINT("Error compiling vertex shader:\n");
    mPRINT(vertexSource);
    mPRINT("\n\nThe following error occured:\n");
    char buffer[1024];
    glGetShaderInfoLog(vertexShaderHandle, sizeof(buffer), nullptr, buffer);
    mPRINT(buffer);
    mRETURN_RESULT(mR_ResourceInvalid);
  }

  char *fragmentSource = nullptr;
  mDEFER(mAllocator_FreePtr(nullptr, &fragmentSource));
  mERROR_CHECK(mAllocator_Allocate(nullptr, &fragmentSource, fragmentShader.bytes));

  position = 0;

  for (auto &&_char : fragmentShader)
  {
    if (_char.codePoint != carriageReturn)
    {
      char *s = (char *)_char.character;

      for (size_t i = 0; i < _char.characterSize; i++)
      {
        fragmentSource[position] = *s;
        position++;
        s++;
      }
    }
  }

  fragmentSource[position] = '\0';

  GLuint fragmentShaderHandle = glCreateShader(GL_FRAGMENT_SHADER);
  mDEFER(glDeleteShader(fragmentShaderHandle));
  glShaderSource(fragmentShaderHandle, 1, &fragmentSource, NULL);
  glCompileShader(fragmentShaderHandle);

  status = GL_TRUE;
  glGetShaderiv(fragmentShaderHandle, GL_COMPILE_STATUS, &status);

  if (status != GL_TRUE)
  {
    mPRINT("Error compiling fragment shader:\n");
    mPRINT(fragmentSource);
    mPRINT("\n\nThe following error occured:\n");
    char buffer[1024];
    glGetShaderInfoLog(fragmentShaderHandle, sizeof(buffer), nullptr, buffer);
    mPRINT(buffer);
    mRETURN_RESULT(mR_ResourceInvalid);
  }

  glUseProgram(shader->shaderProgram);

  glAttachShader(shader->shaderProgram, vertexShaderHandle);
  glAttachShader(shader->shaderProgram, fragmentShaderHandle);

  if (fragDataLocation != nullptr)
    glBindFragDataLocation(shader->shaderProgram, 0, fragDataLocation);

  glLinkProgram(shader->shaderProgram);

  glGetProgramiv(shader->shaderProgram, GL_LINK_STATUS, &status);
  mERROR_IF(status != GL_TRUE, mR_ResourceInvalid);

  mGL_DEBUG_ERROR_CHECK();
#else
  mRETURN_RESULT(mR_NotImplemented);
#endif

#ifndef GIT_BUILD
  mERROR_CHECK(mString_Create(&shader->fragmentShaderText, fragmentShader));
  mERROR_CHECK(mString_Create(&shader->vertexShaderText, vertexShader));
#endif

  mRETURN_SUCCESS();
}

mFUNCTION(mShader_SetToFile, mPtr<mShader> &shader, const mString &filename)
{
  mFUNCTION_SETUP();

  mERROR_CHECK(mShader_SetToFile(shader, filename + ".vert", filename + ".frag", nullptr));

  mRETURN_SUCCESS();
}

#ifndef GIT_BUILD
mFUNCTION(mShader_ReloadFromFile, mPtr<mShader> &shader)
{
  mFUNCTION_SETUP();

  mERROR_IF(shader == nullptr, mR_ArgumentNull);
  mERROR_IF(!shader->initialized, mR_NotInitialized);
  mERROR_IF(!shader->loadedFromFile, mR_ResourceIncompatible);

  mResult result = mShader_SetToFile(shader, shader->vertexShaderPath, shader->fragmentShaderPath);

  mERROR_IF(mFAILED(result) && result != mR_ResourceNotFound, result);

  mRETURN_SUCCESS();
}
#endif

mFUNCTION(mShader_SetToFile, mPtr<mShader> &shader, const mString &vertexShaderPath, const mString &fragmentShaderPath, IN OPTIONAL const char *fragDataLocation /* = nullptr */)
{
  mFUNCTION_SETUP();

  mERROR_IF(shader == nullptr, mR_ArgumentNull);

  mString vertexShader, fragmentShader;

  mERROR_CHECK(mFile_ReadAllText(vertexShaderPath, nullptr, &vertexShader));
  mERROR_CHECK(mFile_ReadAllText(fragmentShaderPath, nullptr, &fragmentShader));

  mERROR_CHECK(mShader_SetTo(shader, vertexShader, fragmentShader, fragDataLocation));

#ifndef GIT_BUILD
  mERROR_CHECK(mString_Create(&shader->vertexShaderPath, vertexShaderPath));
  mERROR_CHECK(mString_Create(&shader->fragmentShaderPath, fragmentShaderPath));
  shader->loadedFromFile = true;
#endif

  mRETURN_SUCCESS();
}

mFUNCTION(mShader_Bind, mShader &shader)
{
  mFUNCTION_SETUP();

#if defined(_DEBUG)
  mERROR_IF(!shader.initialized, mR_NotInitialized);
#endif

#if defined(mRENDERER_OPENGL)
  if (shader.shaderProgram == mShader_CurrentlyBoundShader)
  {
#ifdef _DEBUG
    GLint activeProgram = (GLint)-1;
    glGetIntegerv(GL_CURRENT_PROGRAM, &activeProgram);
    mASSERT_DEBUG((GLuint)activeProgram == mShader_CurrentlyBoundShader, "`mShader_CurrentlyBoundShader` is not up to date. Reset it after 3rdParty code by calling `mShader_AfterForeign()`");
#endif

    mRETURN_SUCCESS();
  }

  mShader_CurrentlyBoundShader = shader.shaderProgram;

  glUseProgram(shader.shaderProgram);

  mGL_DEBUG_ERROR_CHECK();
#else
  mRETURN_RESULT(mR_NotImplemented);
#endif

  mRETURN_SUCCESS();
}

mFUNCTION(mShader_Bind, mPtr<mShader> &shader)
{
  mFUNCTION_SETUP();

  mERROR_IF(shader == nullptr, mR_ArgumentNull);

  mRETURN_RESULT(mShader_Bind(*shader));
}

mFUNCTION(mShader_AfterForeign)
{
  mShader_CurrentlyBoundShader = (GLuint)-1;

  return mR_Success;
}

bool mShader_IsActive(const mShader &shader)
{
  if (!shader.initialized)
    return false;

#if defined (mRENDERER_OPENGL)
  return shader.shaderProgram == mShader_CurrentlyBoundShader;
#else
  mFAIL("Not Implemented!");
  return false;
#endif
}

//////////////////////////////////////////////////////////////////////////

mFUNCTION(mShader_SetUniformAtIndex, mShader &shader, const shaderAttributeIndex_t index, const int32_t v)
{
  mFUNCTION_SETUP();
  mERROR_CHECK(mShader_Bind(shader));

#if defined (mRENDERER_OPENGL)
  glUniform1i(index, v);
#else 
  mRETURN_RESULT(mR_NotImplemented);
#endif
  mRETURN_SUCCESS();
}

mFUNCTION(mShader_SetUniformAtIndex, mShader &shader, const shaderAttributeIndex_t index, const uint32_t v)
{
  mFUNCTION_SETUP();
  mERROR_CHECK(mShader_Bind(shader));

#if defined (mRENDERER_OPENGL)
  glUniform1ui(index, v);
#else 
  mRETURN_RESULT(mR_NotImplemented);
#endif
  mRETURN_SUCCESS();
}

mFUNCTION(mShader_SetUniformAtIndex, mShader &shader, const shaderAttributeIndex_t index, const float_t v)
{
  mFUNCTION_SETUP();
  mERROR_CHECK(mShader_Bind(shader));

#if defined (mRENDERER_OPENGL)
  glUniform1f(index, v);
#else 
  mRETURN_RESULT(mR_NotImplemented);
#endif
  mRETURN_SUCCESS();
}

mFUNCTION(mShader_SetUniformAtIndex, mShader &shader, const shaderAttributeIndex_t index, const mVec2f &v)
{
  mFUNCTION_SETUP();
  mERROR_CHECK(mShader_Bind(shader));

#if defined (mRENDERER_OPENGL)
  glUniform2f(index, v.x, v.y);
#else 
  mRETURN_RESULT(mR_NotImplemented);
#endif

  mRETURN_SUCCESS();
}

mFUNCTION(mShader_SetUniformAtIndex, mShader &shader, const shaderAttributeIndex_t index, const mVec3f &v)
{
  mFUNCTION_SETUP();
  mERROR_CHECK(mShader_Bind(shader));

#if defined (mRENDERER_OPENGL)
  glUniform3f(index, v.x, v.y, v.z);
#else 
  mRETURN_RESULT(mR_NotImplemented);
#endif

  mRETURN_SUCCESS();
}

mFUNCTION(mShader_SetUniformAtIndex, mShader &shader, const shaderAttributeIndex_t index, const mVec4f &v)
{
  mFUNCTION_SETUP();
  mERROR_CHECK(mShader_Bind(shader));

#if defined (mRENDERER_OPENGL)
  glUniform4f(index, v.x, v.y, v.z, v.w);
#else 
  mRETURN_RESULT(mR_NotImplemented);
#endif

  mRETURN_SUCCESS();
}

mFUNCTION(mShader_SetUniformAtIndex, mShader &shader, const shaderAttributeIndex_t index, const mVector &v)
{
  mFUNCTION_SETUP();
  mERROR_CHECK(mShader_Bind(shader));

#if defined (mRENDERER_OPENGL)
  glUniform4f(index, v.x, v.y, v.z, v.w);
#else 
  mRETURN_RESULT(mR_NotImplemented);
#endif

  mRETURN_SUCCESS();
}

mFUNCTION(mShader_SetUniformAtIndex, mShader &shader, const shaderAttributeIndex_t index, const mMatrix &v)
{
  mFUNCTION_SETUP();
  mERROR_CHECK(mShader_Bind(shader));

#if defined (mRENDERER_OPENGL)
  glUniformMatrix4fv(index, 1, false, &v._11);
#else 
  mRETURN_RESULT(mR_NotImplemented);
#endif

  mRETURN_SUCCESS();
}

mFUNCTION(mShader_SetUniformAtIndex, mShader &shader, const shaderAttributeIndex_t index, mTexture &v)
{
  mFUNCTION_SETUP();
  mERROR_CHECK(mShader_Bind(shader));

#if defined (mRENDERER_OPENGL)
  glUniform1i(index, (GLint)v.textureUnit);
#else 
  mRETURN_RESULT(mR_NotImplemented);
#endif

  mRETURN_SUCCESS();
}

mFUNCTION(mShader_SetUniformAtIndex, mShader &shader, const shaderAttributeIndex_t index, mPtr<mTexture> &v)
{
  mFUNCTION_SETUP();
  mERROR_CHECK(mShader_Bind(shader));

#if defined (mRENDERER_OPENGL)
  glUniform1i(index, (GLint)v->textureUnit);
#else 
  mRETURN_RESULT(mR_NotImplemented);
#endif

  mRETURN_SUCCESS();
}

mFUNCTION(mShader_SetUniformAtIndex, mShader &shader, const shaderAttributeIndex_t index, mTexture3D &v)
{
  mFUNCTION_SETUP();
  mERROR_CHECK(mShader_Bind(shader));

#if defined (mRENDERER_OPENGL)
  glUniform1i(index, (GLint)v.textureUnit);
#else 
  mRETURN_RESULT(mR_NotImplemented);
#endif

  mRETURN_SUCCESS();
}

mFUNCTION(mShader_SetUniformAtIndex, mShader &shader, const shaderAttributeIndex_t index, mPtr<mTexture3D> &v)
{
  mFUNCTION_SETUP();
  mERROR_CHECK(mShader_Bind(shader));

#if defined (mRENDERER_OPENGL)
  glUniform1i(index, (GLint)v->textureUnit);
#else 
  mRETURN_RESULT(mR_NotImplemented);
#endif

  mRETURN_SUCCESS();
}

mFUNCTION(mShader_SetUniformAtIndex, mShader &shader, const shaderAttributeIndex_t index, mPtr<mFramebuffer> &v)
{
  mFUNCTION_SETUP();
  mERROR_CHECK(mShader_Bind(shader));

#if defined (mRENDERER_OPENGL)
  glUniform1i(index, (GLint)v->textureUnit);
#else 
  mRETURN_RESULT(mR_NotImplemented);
#endif

  mRETURN_SUCCESS();
}

mFUNCTION(mShader_SetUniformAtIndex, mShader &shader, const shaderAttributeIndex_t index, const int32_t *pV, const size_t count)
{
  mFUNCTION_SETUP();
  mERROR_CHECK(mShader_Bind(shader));

#if defined (mRENDERER_OPENGL)
  glUniform1iv(index, (GLsizei)count, pV);
#else 
  mRETURN_RESULT(mR_NotImplemented);
#endif

  mRETURN_SUCCESS();
}

mFUNCTION(mShader_SetUniformAtIndex, mShader &shader, const shaderAttributeIndex_t index, const uint32_t *pV, const size_t count)
{
  mFUNCTION_SETUP();
  mERROR_CHECK(mShader_Bind(shader));

#if defined (mRENDERER_OPENGL)
  glUniform1uiv(index, (GLsizei)count, pV);
#else 
  mRETURN_RESULT(mR_NotImplemented);
#endif

  mRETURN_SUCCESS();
}

mFUNCTION(mShader_SetUniformAtIndex, mShader &shader, const shaderAttributeIndex_t index, const float_t *pV, const size_t count)
{
  mFUNCTION_SETUP();
  mERROR_CHECK(mShader_Bind(shader));

#if defined (mRENDERER_OPENGL)
  glUniform1fv(index, (GLsizei)count, pV);
#else 
  mRETURN_RESULT(mR_NotImplemented);
#endif

  mRETURN_SUCCESS();
}

mFUNCTION(mShader_SetUniformAtIndex, mShader &shader, const shaderAttributeIndex_t index, const mVec2f *pV, const size_t count)
{
  mFUNCTION_SETUP();
  mERROR_CHECK(mShader_Bind(shader));

#if defined (mRENDERER_OPENGL)
  glUniform2fv(index, (GLsizei)count, (float_t *)pV);
#else 
  mRETURN_RESULT(mR_NotImplemented);
#endif

  mRETURN_SUCCESS();
}

mFUNCTION(mShader_SetUniformAtIndex, mShader &shader, const shaderAttributeIndex_t index, const mVec3f *pV, const size_t count)
{
  mFUNCTION_SETUP();
  mERROR_CHECK(mShader_Bind(shader));

#if defined (mRENDERER_OPENGL)
  glUniform3fv(index, (GLsizei)count, (float_t *)pV);
#else 
  mRETURN_RESULT(mR_NotImplemented);
#endif

  mRETURN_SUCCESS();
}

mFUNCTION(mShader_SetUniformAtIndex, mShader &shader, const shaderAttributeIndex_t index, const mVec4f *pV, const size_t count)
{
  mFUNCTION_SETUP();
  mERROR_CHECK(mShader_Bind(shader));

#if defined (mRENDERER_OPENGL)
  glUniform4fv(index, (GLsizei)count, (float_t *)pV);
#else 
  mRETURN_RESULT(mR_NotImplemented);
#endif

  mRETURN_SUCCESS();
}

mFUNCTION(mShader_SetUniformAtIndex, mShader &shader, const shaderAttributeIndex_t index, const mVector *pV, const size_t count)
{
  mFUNCTION_SETUP();
  mERROR_CHECK(mShader_Bind(shader));

#if defined (mRENDERER_OPENGL)
  glUniform4fv(index, (GLsizei)count, (float_t *)pV);
#else 
  mRETURN_RESULT(mR_NotImplemented);
#endif

  mRETURN_SUCCESS();
}

mFUNCTION(mShader_SetUniformAtIndex, mShader &shader, const shaderAttributeIndex_t index, const mTexture *pV, const size_t count)
{
  mFUNCTION_SETUP();
  mERROR_CHECK(mShader_Bind(shader));

#if defined (mRENDERER_OPENGL)
  GLint *pValues = nullptr;
  mDEFER_CALL(&pValues, mFreePtrStack);
  mERROR_CHECK(mAllocStackZero(&pValues, count));

  for (size_t i = 0; i < count; ++i)
    pValues[i] = (GLint)pV[i].textureId;

  glUniform1iv(index, (GLsizei)count, pValues);
#else 
  mRETURN_RESULT(mR_NotImplemented);
#endif

  mRETURN_SUCCESS();
}

mFUNCTION(mShader_SetUniformAtIndex, mShader &shader, const shaderAttributeIndex_t index, const mPtr<mTexture> *pV, const size_t count)
{
  mFUNCTION_SETUP();
  mERROR_CHECK(mShader_Bind(shader));

#if defined (mRENDERER_OPENGL)
  GLint *pValues = nullptr;
  mDEFER_CALL(&pValues, mFreePtrStack);
  mERROR_CHECK(mAllocStackZero(&pValues, count));

  for (size_t i = 0; i < count; ++i)
    pValues[i] = (GLint)pV[i]->textureId;

  glUniform1iv(index, (GLsizei)count, pValues);
#else 
  mRETURN_RESULT(mR_NotImplemented);
#endif

  mRETURN_SUCCESS();
}

mFUNCTION(mShader_SetUniformAtIndex, mShader &shader, const shaderAttributeIndex_t index, const mTexture3D *pV, const size_t count)
{
  mFUNCTION_SETUP();
  mERROR_CHECK(mShader_Bind(shader));

#if defined (mRENDERER_OPENGL)
  GLint *pValues = nullptr;
  mDEFER_CALL(&pValues, mFreePtrStack);
  mERROR_CHECK(mAllocStackZero(&pValues, count));

  for (size_t i = 0; i < count; ++i)
    pValues[i] = (GLint)pV[i].textureId;

  glUniform1iv(index, (GLsizei)count, pValues);
#else 
  mRETURN_RESULT(mR_NotImplemented);
#endif

  mRETURN_SUCCESS();
}

mFUNCTION(mShader_SetUniformAtIndex, mShader &shader, const shaderAttributeIndex_t index, const mPtr<mTexture3D> *pV, const size_t count)
{
  mFUNCTION_SETUP();
  mERROR_CHECK(mShader_Bind(shader));

#if defined (mRENDERER_OPENGL)
  GLint *pValues = nullptr;
  mDEFER_CALL(&pValues, mFreePtrStack);
  mERROR_CHECK(mAllocStackZero(&pValues, count));

  for (size_t i = 0; i < count; ++i)
    pValues[i] = (GLint)pV[i]->textureId;

  glUniform1iv(index, (GLsizei)count, pValues);
#else 
  mRETURN_RESULT(mR_NotImplemented);
#endif

  mRETURN_SUCCESS();
}

mFUNCTION(mShader_SetUniformAtIndex, mShader &shader, const shaderAttributeIndex_t index, const mPtr<mFramebuffer> *pV, const size_t count)
{
  mFUNCTION_SETUP();
  mERROR_CHECK(mShader_Bind(shader));

#if defined (mRENDERER_OPENGL)
  GLint *pValues = nullptr;
  mDEFER_CALL(&pValues, mFreePtrStack);
  mERROR_CHECK(mAllocStackZero(&pValues, count));

  for (size_t i = 0; i < count; ++i)
    pValues[i] = (GLint)pV[i]->textureId;

  glUniform1iv(index, (GLsizei)count, pValues);
#else 
  mRETURN_RESULT(mR_NotImplemented);
#endif

  mRETURN_SUCCESS();
}
