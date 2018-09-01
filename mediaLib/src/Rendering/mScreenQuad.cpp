// Copyright 2018 Christoph Stiller
// 
// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files(the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions :
// 
// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
// 
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "mScreenQuad.h"
#include "mFile.h"

mFUNCTION(mScreenQuad_Destroy_Internal, IN mScreenQuad *pScreenQuad);

//////////////////////////////////////////////////////////////////////////

mFUNCTION(mScreenQuad_Create, OUT mPtr<mScreenQuad> *pScreenQuad, IN mAllocator *pAllocator, const std::string &fragmentShader, const size_t textureCount)
{
  mFUNCTION_SETUP();

  mERROR_CHECK(mSharedPointer_Allocate(pScreenQuad, pAllocator, (std::function<void(mScreenQuad *)>)[](mScreenQuad *pData) {mScreenQuad_Destroy_Internal(pData);}, 1));

#if defined(mRENDERER_OPENGL)
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

  const GLuint index = glGetAttribLocation((*pScreenQuad)->shader->shaderProgram, "position0");
  glEnableVertexAttribArray(index);
  glVertexAttribPointer((GLuint)0, (GLint)2, GL_FLOAT, GL_FALSE, (GLsizei)sizeof(mVec2f), (const void *)0);

  mGL_ERROR_CHECK();
#else
  mRETURN_RESULT(mR_NotImplemented);
#endif

  mRETURN_SUCCESS();
}

mFUNCTION(mScreenQuad_CreateFrom, OUT mPtr<mScreenQuad> *pScreenQuad, IN mAllocator *pAllocator, const std::wstring &fragmentShaderPath, const size_t textureCount /* = 1 */)
{
  mFUNCTION_SETUP();

  std::string fragmentShader;
  mERROR_CHECK(mFile_ReadAllText(fragmentShaderPath, pAllocator, &fragmentShader));

  mERROR_CHECK(mScreenQuad_Create(pScreenQuad, pAllocator, fragmentShader, textureCount));

  (*pScreenQuad)->shader->fragmentShader = fragmentShaderPath;

  mRETURN_SUCCESS();
}

mFUNCTION(mScreenQuad_Destroy, IN_OUT mPtr<mScreenQuad> *pScreenQuad)
{
  mFUNCTION_SETUP();

  mERROR_IF(pScreenQuad == nullptr, mR_ArgumentNull);

  mERROR_CHECK(mSharedPointer_Destroy(pScreenQuad));

  mRETURN_SUCCESS();
}

mFUNCTION(mScreenQuad_Render, mPtr<mScreenQuad>& screenQuad)
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

mFUNCTION(mScreenQuad_Destroy_Internal, IN mScreenQuad *pScreenQuad)
{
  mFUNCTION_SETUP();

  mERROR_IF(pScreenQuad == nullptr, mR_ArgumentNull);

#if defined(mRENDERER_OPENGL)
  if (pScreenQuad->vbo)
    glDeleteBuffers(1, &pScreenQuad->vbo);
#endif

  mRETURN_SUCCESS();
}
