#include "mScreenQuad.h"
#include "mFile.h"

#ifdef GIT_BUILD // Define __M_FILE__
  #ifdef __M_FILE__
    #undef __M_FILE__
  #endif
  #define __M_FILE__ "5d1lnZ43s32zNe4MCnudH9RQKWEIA4qdtHyil0o/LYyKKA0s6f7pRXWASgAmdq9IVNcsWmf8pmNHWIyU"
#endif

static mFUNCTION(mScreenQuad_Destroy_Internal, IN mScreenQuad *pScreenQuad);

//////////////////////////////////////////////////////////////////////////

mFUNCTION(mScreenQuad_Create, OUT mPtr<mScreenQuad> *pScreenQuad, IN mAllocator *pAllocator, const mString &fragmentShader, const size_t textureCount)
{
  mFUNCTION_SETUP();

  mERROR_CHECK(mSharedPointer_Allocate(pScreenQuad, pAllocator, (std::function<void(mScreenQuad *)>)[](mScreenQuad *pData) {mScreenQuad_Destroy_Internal(pData);}, 1));

#if defined(mRENDERER_OPENGL)
  mGL_ERROR_CHECK();

  char vertexShader[2048] = "#version 150 core\n\nin vec2 position0;";

  for (size_t i = 0; i < textureCount; ++i)
    mERROR_IF(0 > sprintf_s(vertexShader, "%s\nout vec2 _texCoord%" PRIu64 ";", vertexShader, i), mR_InternalError);

  mERROR_IF(0 > sprintf_s(vertexShader, "%s\n\nvoid main()\n{\n\tgl_Position = vec4(position0 * 2 - 1, 0, 1);\n\t", vertexShader), mR_InternalError);

  for (size_t i = 0; i < textureCount; ++i)
    mERROR_IF(0 > sprintf_s(vertexShader, "%s\n\t_texCoord%" PRIu64 " = position0;", vertexShader, i), mR_InternalError);

  mERROR_IF(0 > sprintf_s(vertexShader, "%s\n}\n", vertexShader), mR_InternalError);

  mERROR_CHECK(mSharedPointer_Allocate(&(*pScreenQuad)->shader, pAllocator, (std::function<void (mShader *)>)[](mShader *pShader) {mShader_Destroy(pShader);}, (size_t)1));

  mERROR_CHECK(mShader_Create((*pScreenQuad)->shader.GetPointer(), vertexShader, fragmentShader));

  mVec2f positions[4];

  for (size_t i = 0; i < 4; ++i)
    positions[i] = mVec2f((float_t)(i & 1), (float_t)((i & 2) >> 1));

  glGenBuffers(1, &(*pScreenQuad)->vbo);
  glBindBuffer(GL_ARRAY_BUFFER, (*pScreenQuad)->vbo);
  glBufferData(GL_ARRAY_BUFFER, sizeof(mVec2f) * 4, positions, GL_STATIC_DRAW);

  const GLint index = glGetAttribLocation((*pScreenQuad)->shader->shaderProgram, "position0");
  mERROR_IF(index < 0, mR_InternalError);

  glEnableVertexAttribArray(index);
  glVertexAttribPointer((GLuint)0, (GLint)2, GL_FLOAT, GL_FALSE, (GLsizei)sizeof(mVec2f), (const void *)0);

  mGL_ERROR_CHECK();
#else
  mRETURN_RESULT(mR_NotImplemented);
#endif

  mRETURN_SUCCESS();
}

mFUNCTION(mScreenQuad_Create, OUT mPtr<mScreenQuad> *pScreenQuad, IN mAllocator *pAllocator)
{
  mFUNCTION_SETUP();

  const mString defaultShader = "#version 150 core\n\nout vec4 outColour0;\n\nuniform sampler2D _texture0;\nin vec2 _texCoord0;\nvoid main()\n{\n\toutColour0 = texture(_texture0, _texCoord0);\n}\n";

  mERROR_CHECK(mScreenQuad_Create(pScreenQuad, pAllocator, defaultShader, 1));

  mRETURN_SUCCESS();
}

mFUNCTION(mScreenQuad_CreateForMultisampleTexture, OUT mPtr<mScreenQuad> *pScreenQuad, IN mAllocator *pAllocator)
{
  mFUNCTION_SETUP();

  const mString defaultShader = "#version 150 core\n\nout vec4 outColour0;\n\nuniform sampler2DMS _texture0;\nuniform int _texture0sampleCount;\nin vec2 _texCoord0;\n\nvoid main()\n{\n\toutColour0 = vec4(0);\n\tivec2 position = ivec2(_texCoord0 * textureSize(_texture0));\n\t\n\tfor (int i = 0; i < _texture0sampleCount; i++)\n\t\toutColour0 += texelFetch(_texture0, position, i);\n\t\n\toutColour0 /= float(_texture0sampleCount);\n}\n";

  mERROR_CHECK(mScreenQuad_Create(pScreenQuad, pAllocator, defaultShader, 1));

  mRETURN_SUCCESS();
}

mFUNCTION(mScreenQuad_CreateFrom, OUT mPtr<mScreenQuad> *pScreenQuad, IN mAllocator *pAllocator, const mString &fragmentShaderPath, const size_t textureCount /* = 1 */)
{
  mFUNCTION_SETUP();

  mString fragmentShader;
  mERROR_CHECK(mFile_ReadAllText(fragmentShaderPath, pAllocator, &fragmentShader));

  mERROR_CHECK(mScreenQuad_Create(pScreenQuad, pAllocator, fragmentShader, textureCount));

#ifndef GIT_BUILD
  (*pScreenQuad)->shader->fragmentShaderPath = fragmentShaderPath;
#endif

  mRETURN_SUCCESS();
}

mFUNCTION(mScreenQuad_Destroy, IN_OUT mPtr<mScreenQuad> *pScreenQuad)
{
  mFUNCTION_SETUP();

  mERROR_IF(pScreenQuad == nullptr, mR_ArgumentNull);

  mERROR_CHECK(mSharedPointer_Destroy(pScreenQuad));

  mRETURN_SUCCESS();
}

mFUNCTION(mScreenQuad_Render, mPtr<mScreenQuad> &screenQuad)
{
  mFUNCTION_SETUP();

  mERROR_CHECK(mShader_Bind(*screenQuad->shader));

#if defined(mRENDERER_OPENGL)
  glBindBuffer(GL_ARRAY_BUFFER, screenQuad->vbo);

  const GLuint index = glGetAttribLocation(screenQuad->shader->shaderProgram, "position0");
  glEnableVertexAttribArray(index);
  glVertexAttribPointer((GLuint)0, (GLint)2, GL_FLOAT, GL_FALSE, (GLsizei)sizeof(mVec2f), (const void *)0);

  glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
#else
  mRETURN_RESULT(mR_NotImplemented);
#endif

  mRETURN_SUCCESS();
}

//////////////////////////////////////////////////////////////////////////

static mFUNCTION(mScreenQuad_Destroy_Internal, IN mScreenQuad *pScreenQuad)
{
  mFUNCTION_SETUP();

  mERROR_IF(pScreenQuad == nullptr, mR_ArgumentNull);

  mERROR_CHECK(mSharedPointer_Destroy(&pScreenQuad->shader));

#if defined(mRENDERER_OPENGL)
  if (pScreenQuad->vbo)
    glDeleteBuffers(1, &pScreenQuad->vbo);
#endif

  mRETURN_SUCCESS();
}
