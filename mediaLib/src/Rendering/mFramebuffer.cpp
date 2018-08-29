// Copyright 2018 Christoph Stiller
// 
// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files(the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions :
// 
// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
// 
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "mFramebuffer.h"

mFUNCTION(mFramebuffer_Create_Internal, OUT mFramebuffer *pFramebuffer, const mVec2s &size);
mFUNCTION(mFramebuffer_Destroy_Internal, IN mFramebuffer *pFramebuffer);

//////////////////////////////////////////////////////////////////////////

mFUNCTION(mFramebuffer_Create, OUT mPtr<mFramebuffer>* pFramebuffer, IN mAllocator * pAllocator, const mVec2s & size)
{
  mFUNCTION_SETUP();

  mERROR_IF(pFramebuffer == nullptr, mR_ArgumentNull);

  mRETURN_SUCCESS();
}

mFUNCTION(mFramebuffer_Destroy, IN_OUT mPtr<mFramebuffer>* pFramebuffer)
{
  mFUNCTION_SETUP();

  mERROR_IF(pFramebuffer == nullptr, mR_ArgumentNull);

  mRETURN_SUCCESS();
}

mFUNCTION(mFramebuffer_Bind, mPtr<mFramebuffer>& framebuffer)
{
  mFUNCTION_SETUP();

  mERROR_IF(framebuffer == nullptr, mR_ArgumentNull);

  mRETURN_SUCCESS();
}

mFUNCTION(mFramebuffer_Unbind)
{
  mFUNCTION_SETUP();

  mRETURN_SUCCESS();
}

//////////////////////////////////////////////////////////////////////////

mFUNCTION(mFramebuffer_Create_Internal, mFramebuffer *pFramebuffer, const mVec2s &size)
{
  mFUNCTION_SETUP();

  mERROR_IF(pFramebuffer == nullptr, mR_ArgumentNull);
  mERROR_IF(size.x > UINT32_MAX || size.y > UINT32_MAX, mR_ArgumentOutOfBounds);

  glGenFramebuffers(1, &pFramebuffer->frameBufferHandle);
  glBindFramebuffer(GL_FRAMEBUFFER, pFramebuffer->frameBufferHandle);
  glGenTextures(1, &pFramebuffer->texColourBuffer);
  glBindTexture(GL_TEXTURE_2D, pFramebuffer->texColourBuffer);

  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, (GLsizei)size.x, (GLsizei)size.y, 0, GL_RGB, GL_UNSIGNED_BYTE, nullptr);

  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
  
  glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, pFramebuffer->texColourBuffer, 0);

  glGenRenderbuffers(1, &pFramebuffer->rboDepthStencil);
  glBindRenderbuffer(GL_RENDERBUFFER, pFramebuffer->rboDepthStencil);
  glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH24_STENCIL8, (GLsizei)size.x, (GLsizei)size.y);

  mGL_ERROR_CHECK();

  mRETURN_SUCCESS();
}

mFUNCTION(mFramebuffer_Destroy_Internal, IN mFramebuffer *pFramebuffer)
{
  mFUNCTION_SETUP();

  mERROR_IF(pFramebuffer == nullptr, mR_ArgumentNull);

#if defined(mRENDERER_OPENGL)
  mERROR_CHECK(mFramebuffer_Unbind());

  if (pFramebuffer->frameBufferHandle)
    glDeleteFramebuffers(1, &pFramebuffer->frameBufferHandle);

  pFramebuffer->frameBufferHandle = 0;

  if (pFramebuffer->texColourBuffer)
    glDeleteTextures(0, &pFramebuffer->texColourBuffer);

  pFramebuffer->texColourBuffer = 0;
#endif

  mRETURN_SUCCESS();
}
