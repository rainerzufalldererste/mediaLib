// Copyright 2018 Christoph Stiller
// 
// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files(the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions :
// 
// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
// 
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "mFramebuffer.h"

#if defined (mRENDERER_OPENGL)
GLuint mFrameBuffer_ActiveFrameBufferHandle = 0;
#endif

mFUNCTION(mFramebuffer_Create_Internal, OUT mFramebuffer *pFramebuffer, const mVec2s &size);
mFUNCTION(mFramebuffer_Destroy_Internal, IN mFramebuffer *pFramebuffer);

//////////////////////////////////////////////////////////////////////////

mFUNCTION(mFramebuffer_Create, OUT mPtr<mFramebuffer> *pFramebuffer, IN mAllocator *pAllocator, const mVec2s &size)
{
  mFUNCTION_SETUP();

  mERROR_IF(pFramebuffer == nullptr, mR_ArgumentNull);

  mERROR_CHECK(mSharedPointer_Allocate(pFramebuffer, pAllocator, (std::function<void (mFramebuffer *)>)[](mFramebuffer *pData) { mFramebuffer_Destroy_Internal(pData); }, 1));

  mERROR_CHECK(mFramebuffer_Create_Internal(pFramebuffer->GetPointer(), size));

  mRETURN_SUCCESS();
}

mFUNCTION(mFramebuffer_Destroy, IN_OUT mPtr<mFramebuffer> *pFramebuffer)
{
  mFUNCTION_SETUP();

  mERROR_IF(pFramebuffer == nullptr, mR_ArgumentNull);

  mERROR_CHECK(mSharedPointer_Destroy(pFramebuffer));

  mRETURN_SUCCESS();
}

mFUNCTION(mFramebuffer_Bind, mPtr<mFramebuffer> &framebuffer)
{
  mFUNCTION_SETUP();

  mERROR_IF(framebuffer == nullptr, mR_ArgumentNull);

#if defined(mRENDERER_OPENGL)
  glBindFramebuffer(GL_FRAMEBUFFER, framebuffer->frameBufferHandle);
  mERROR_CHECK(mRenderParams_SetCurrentRenderResolution(framebuffer->size));
  mFrameBuffer_ActiveFrameBufferHandle = framebuffer->frameBufferHandle;

  mGL_DEBUG_ERROR_CHECK();
#else
  mRETURN_RESULT(mR_NotImplemented);
#endif

  mRETURN_SUCCESS();
}

mFUNCTION(mFramebuffer_Unbind)
{
  mFUNCTION_SETUP();

#if defined(mRENDERER_OPENGL)
  glBindFramebuffer(GL_FRAMEBUFFER, 0);
  mFrameBuffer_ActiveFrameBufferHandle = 0;

  mGL_DEBUG_ERROR_CHECK();
#else
  mRETURN_RESULT(mR_NotImplemented);
#endif

  mRETURN_SUCCESS();
}

mFUNCTION(mFramebuffer_SetResolution, mPtr<mFramebuffer> &framebuffer, const mVec2s &size)
{
  mFUNCTION_SETUP();

  mERROR_IF(framebuffer == nullptr, mR_ArgumentNull);
  mERROR_IF(size.x > UINT32_MAX || size.y > UINT32_MAX, mR_ArgumentOutOfBounds);
  mERROR_IF(mFrameBuffer_ActiveFrameBufferHandle == framebuffer->frameBufferHandle, mR_ResourceStateInvalid);

  framebuffer->size = size;
  glBindTexture(GL_TEXTURE_2D, framebuffer->texColourBuffer);

  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, (GLsizei)size.x, (GLsizei)size.y, 0, GL_RGB, GL_UNSIGNED_BYTE, nullptr);

  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

  mRETURN_SUCCESS();
}

mFUNCTION(mFramebuffer_Download, mPtr<mFramebuffer> &framebuffer, OUT mPtr<mImageBuffer> *pImageBuffer, IN mAllocator *pAllocator)
{
  mFUNCTION_SETUP();

  mERROR_IF(framebuffer == nullptr, mR_ArgumentNull);

#if defined(mRENDERER_OPENGL)
  mERROR_IF(mFrameBuffer_ActiveFrameBufferHandle == framebuffer->frameBufferHandle, mR_ResourceStateInvalid);

  mERROR_CHECK(mImageBuffer_Create(pImageBuffer, pAllocator, framebuffer->size, mPF_R8G8B8A8));

  glBindTexture(GL_TEXTURE_2D, framebuffer->texColourBuffer);
  glGetTexImage(GL_TEXTURE_2D, 0, GL_RGBA, GL_UNSIGNED_BYTE, (*pImageBuffer)->pPixels);

  mERROR_CHECK(mImageBuffer_FlipY(*pImageBuffer));
#else
  mRETURN_RESULT(mR_NotImplemented);
#endif

  mRETURN_SUCCESS();
}

mFUNCTION(mTexture_Bind, mPtr<mFramebuffer> &framebuffer, const size_t textureUnit /* = 0 */)
{
  mFUNCTION_SETUP();

  mERROR_IF(framebuffer == nullptr, mR_ArgumentNull);

  framebuffer->textureUnit = textureUnit;

#if defined(mRENDERER_OPENGL)
  mERROR_IF(mFrameBuffer_ActiveFrameBufferHandle == framebuffer->frameBufferHandle, mR_ResourceStateInvalid);

  glActiveTexture(GL_TEXTURE0 + (GLuint)framebuffer->textureUnit);
  glBindTexture(GL_TEXTURE_2D, framebuffer->texColourBuffer);

  mGL_DEBUG_ERROR_CHECK();
#else
  mRETURN_RESULT(mR_NotImplemented);
#endif

  mRETURN_SUCCESS();
}

//////////////////////////////////////////////////////////////////////////

mFUNCTION(mFramebuffer_Create_Internal, mFramebuffer *pFramebuffer, const mVec2s &size)
{
  mFUNCTION_SETUP();

  mERROR_IF(pFramebuffer == nullptr, mR_ArgumentNull);
  mERROR_IF(size.x > UINT32_MAX || size.y > UINT32_MAX, mR_ArgumentOutOfBounds);

  pFramebuffer->size = size;
  pFramebuffer->textureUnit = (size_t)-1;

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

  glBindFramebuffer(GL_FRAMEBUFFER, 0);

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
