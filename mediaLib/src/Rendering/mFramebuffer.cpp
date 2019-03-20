#include "mFramebuffer.h"
#include "mScreenQuad.h"
#include "mHardwareWindow.h"

//#define DEBUG_FRAMBUFFER_CREATION

#if defined (mRENDERER_OPENGL)
GLuint mFrameBuffer_ActiveFrameBufferHandle = 0;
#endif

mPtr<mQueue<mPtr<mFramebuffer>>> mFramebuffer_Queue = nullptr;

mFUNCTION(mFramebuffer_Create_Internal, OUT mFramebuffer *pFramebuffer, const mVec2s &size, const mTexture2DParams &params, const size_t sampleCount);
mFUNCTION(mFramebuffer_Destroy_Internal, IN mFramebuffer *pFramebuffer);

//////////////////////////////////////////////////////////////////////////

mFUNCTION(mFramebuffer_Create, OUT mPtr<mFramebuffer> *pFramebuffer, IN mAllocator *pAllocator, const mVec2s &size, const mTexture2DParams &textureParams /* = mTexture2DParams() */, const size_t sampleCount /* = 0 */)
{
  mFUNCTION_SETUP();

  mERROR_IF(pFramebuffer == nullptr, mR_ArgumentNull);

  if (mFramebuffer_Queue == nullptr)
    mERROR_CHECK(mQueue_Create(&mFramebuffer_Queue, pAllocator));

  mERROR_CHECK(mSharedPointer_Allocate(pFramebuffer, pAllocator, (std::function<void (mFramebuffer *)>)[](mFramebuffer *pData) { mFramebuffer_Destroy_Internal(pData); }, 1));

  mERROR_CHECK(mFramebuffer_Create_Internal(pFramebuffer->GetPointer(), size, textureParams, sampleCount));

  mRETURN_SUCCESS();
}

mFUNCTION(mFramebuffer_Destroy, IN_OUT mPtr<mFramebuffer> *pFramebuffer)
{
  mFUNCTION_SETUP();

  mERROR_IF(pFramebuffer == nullptr, mR_ArgumentNull);
  
  mGL_DEBUG_ERROR_CHECK();

  mERROR_CHECK(mSharedPointer_Destroy(pFramebuffer));

  mGL_DEBUG_ERROR_CHECK();

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

  if (mFramebuffer_Queue != nullptr)
    mERROR_CHECK(mQueue_Clear(mFramebuffer_Queue));

#if defined(mRENDERER_OPENGL)
  mGL_DEBUG_ERROR_CHECK();
  glBindFramebuffer(GL_FRAMEBUFFER, 0);
  mFrameBuffer_ActiveFrameBufferHandle = 0;
  mGL_DEBUG_ERROR_CHECK();

#else
  mRETURN_RESULT(mR_NotImplemented);
#endif

  mRETURN_SUCCESS();
}

mFUNCTION(mFramebuffer_Push, mPtr<mFramebuffer> &framebuffer)
{
  mFUNCTION_SETUP();

  mERROR_IF(framebuffer == nullptr, mR_ArgumentNull);

  mERROR_CHECK(mFramebuffer_Bind(framebuffer));

  mERROR_CHECK(mQueue_PushBack(mFramebuffer_Queue, framebuffer));

  mRETURN_SUCCESS();
}

mFUNCTION(mFramebuffer_Pop)
{
  mFUNCTION_SETUP();

  mERROR_IF(mFramebuffer_Queue == nullptr, mR_Success);

  size_t count = 0;
  mERROR_CHECK(mQueue_GetCount(mFramebuffer_Queue, &count));

  if (count <= 1)
  {
    if (count == 1)
      mERROR_CHECK(mQueue_Clear(mFramebuffer_Queue));

    mERROR_CHECK(mFramebuffer_Unbind());
  }
  else
  {
    // Remove last framebuffer.
    {
      mPtr<mFramebuffer> framebuffer;
      mDEFER_CALL(&framebuffer, mFramebuffer_Destroy);
      mERROR_CHECK(mQueue_PopBack(mFramebuffer_Queue, &framebuffer));
    }

    // bind the previous framebuffer.
    {
      mPtr<mFramebuffer> framebuffer;
      mDEFER_CALL(&framebuffer, mFramebuffer_Destroy);
      mERROR_CHECK(mQueue_PeekBack(mFramebuffer_Queue, &framebuffer));
      mERROR_CHECK(mFramebuffer_Bind(framebuffer));
    }
  }

  mRETURN_SUCCESS();
}

mFUNCTION(mFramebuffer_SetResolution, mPtr<mFramebuffer> &framebuffer, const mVec2s &size)
{
  mFUNCTION_SETUP();

  mERROR_IF(framebuffer == nullptr, mR_ArgumentNull);
  mERROR_IF(size.x > UINT32_MAX || size.y > UINT32_MAX, mR_ArgumentOutOfBounds);
  mERROR_IF(mFrameBuffer_ActiveFrameBufferHandle == framebuffer->frameBufferHandle, mR_ResourceStateInvalid);
  mERROR_IF(framebuffer->size == size, mR_Success);

  framebuffer->allocatedSize = framebuffer->size = size;

  mGL_DEBUG_ERROR_CHECK();

  glBindTexture((framebuffer->sampleCount > 0) ? GL_TEXTURE_2D_MULTISAMPLE : GL_TEXTURE_2D, framebuffer->texColourBuffer);

  if (framebuffer->sampleCount > 0)
    glTexImage2DMultisample(GL_TEXTURE_2D_MULTISAMPLE, (GLsizei)framebuffer->sampleCount, GL_RGBA, (GLsizei)size.x, (GLsizei)size.y, false);
  else
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, (GLsizei)size.x, (GLsizei)size.y, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);

  mERROR_CHECK(mTexture2DParams_ApplyToBoundTexture(framebuffer->textureParams, framebuffer->sampleCount > 0));

  mGL_DEBUG_ERROR_CHECK();

  mRETURN_SUCCESS();
}

mFUNCTION(mFramebuffer_Download, mPtr<mFramebuffer> &framebuffer, OUT mPtr<mImageBuffer> *pImageBuffer, IN mAllocator *pAllocator, const mPixelFormat pixelFormat /* = mPF_R8G8B8A8 */)
{
  mFUNCTION_SETUP();

  mERROR_IF(framebuffer == nullptr || pImageBuffer == nullptr, mR_ArgumentNull);

#if defined(mRENDERER_OPENGL)

  GLenum glPixelFormat;

  switch (pixelFormat)
  {
  case mPF_R8G8B8A8:
    glPixelFormat = GL_RGBA;
    break;

  case mPF_R8G8B8:
    glPixelFormat = GL_RGB;
    break;

  case mPF_B8G8R8A8:
    glPixelFormat = GL_BGRA;
    break;

  case mPF_B8G8R8:
    glPixelFormat = GL_BGR;
    break;

  default:
    mRETURN_RESULT(mR_NotSupported);
    break;
  }

  if (framebuffer->sampleCount > 0)
  {
    mPtr<mFramebuffer> tempFramebuffer;
    mDEFER_CALL(&tempFramebuffer, mFramebuffer_Destroy);
    mERROR_CHECK(mFramebuffer_Create(&tempFramebuffer, &mDefaultTempAllocator, framebuffer->size, mTexture2DParams(), 0));

    mERROR_CHECK(mFramebuffer_BlitToFramebuffer(tempFramebuffer, framebuffer));
    mERROR_CHECK(mFramebuffer_Download(tempFramebuffer, pImageBuffer, pAllocator, pixelFormat));
  }
  else
  {
    mERROR_IF(mFrameBuffer_ActiveFrameBufferHandle == framebuffer->frameBufferHandle, mR_ResourceStateInvalid);

    mERROR_CHECK(mImageBuffer_Create(pImageBuffer, pAllocator, framebuffer->size, pixelFormat));

    glBindTexture(GL_TEXTURE_2D, framebuffer->texColourBuffer);
    glGetTexImage(GL_TEXTURE_2D, 0, glPixelFormat, GL_UNSIGNED_BYTE, (*pImageBuffer)->pPixels);

    mGL_DEBUG_ERROR_CHECK();
  }

  mERROR_CHECK(mImageBuffer_FlipY(*pImageBuffer));
#else
  mRETURN_RESULT(mR_NotImplemented);
#endif

  mRETURN_SUCCESS();
}

mFUNCTION(mFramebuffer_BlitToScreen, mPtr<mHardwareWindow> &window, mPtr<mFramebuffer> &framebuffer)
{
  mFUNCTION_SETUP();

  mERROR_IF(window == nullptr || framebuffer == nullptr, mR_ArgumentNull);

  mERROR_CHECK(mHardwareWindow_SetAsActiveRenderTarget(window));
  mGL_DEBUG_ERROR_CHECK();

  glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0);
  mGL_DEBUG_ERROR_CHECK();
  glBindFramebuffer(GL_READ_FRAMEBUFFER, framebuffer->frameBufferHandle);
  mGL_DEBUG_ERROR_CHECK();

  glDrawBuffer(GL_BACK);
  mGL_DEBUG_ERROR_CHECK();

  glBlitFramebuffer(0, 0, (GLsizei)framebuffer->size.x, (GLsizei)framebuffer->size.y, 0, 0, (GLsizei)mRenderParams_CurrentRenderResolution.x, (GLsizei)mRenderParams_CurrentRenderResolution.y, GL_COLOR_BUFFER_BIT, GL_NEAREST);

  mGL_DEBUG_ERROR_CHECK();

  mERROR_CHECK(mHardwareWindow_SetAsActiveRenderTarget(window));

  mRETURN_SUCCESS();
}

mFUNCTION(mFramebuffer_BlitToFramebuffer, mPtr<mFramebuffer> &target, mPtr<mFramebuffer> &source)
{
  mFUNCTION_SETUP();

  mERROR_IF(target == nullptr || source == nullptr, mR_ArgumentNull);

  if (source->sampleCount == target->sampleCount)
  {
    mERROR_CHECK(mFramebuffer_Push(target));
    mDEFER(mFramebuffer_Pop());
    mGL_DEBUG_ERROR_CHECK();

    glBindFramebuffer(GL_DRAW_FRAMEBUFFER, target->frameBufferHandle);
    glBindFramebuffer(GL_READ_FRAMEBUFFER, source->frameBufferHandle);

    glDrawBuffer(GL_BACK);

    glBlitFramebuffer(0, 0, (GLsizei)source->size.x, (GLsizei)source->size.y, 0, 0, (GLsizei)target->size.x, (GLsizei)target->size.y, GL_COLOR_BUFFER_BIT, GL_LINEAR);

    mGL_DEBUG_ERROR_CHECK();
  }
  else if (source->sampleCount != 0)
  {
    mERROR_CHECK(mFramebuffer_Push(target));
    mDEFER(mFramebuffer_Pop());
    mGL_DEBUG_ERROR_CHECK();

    mPtr<mScreenQuad> screenQuad;
    mDEFER_CALL(&screenQuad, mScreenQuad_Destroy);
    mERROR_CHECK(mScreenQuad_CreateForMultisampleTexture(&screenQuad, &mDefaultTempAllocator));

    mERROR_CHECK(mTexture_Bind(source, 0));
    mERROR_CHECK(mShader_SetUniform(screenQuad->shader, "_texture0", source));
    mERROR_CHECK(mShader_SetUniform(screenQuad->shader, "_texture0sampleCount", (int32_t)source->sampleCount));
    mERROR_CHECK(mScreenQuad_Render(screenQuad));
  }
  else
  {
    mERROR_CHECK(mFramebuffer_Push(target));
    mDEFER(mFramebuffer_Pop());
    mGL_DEBUG_ERROR_CHECK();

    mPtr<mScreenQuad> screenQuad;
    mDEFER_CALL(&screenQuad, mScreenQuad_Destroy);
    mERROR_CHECK(mScreenQuad_Create(&screenQuad, &mDefaultTempAllocator));

    mERROR_CHECK(mTexture_Bind(source, 0));
    mERROR_CHECK(mShader_SetUniform(screenQuad->shader, "_texture0", source));
    mERROR_CHECK(mScreenQuad_Render(screenQuad));
  }

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
  glBindTexture((framebuffer->sampleCount > 0) ? GL_TEXTURE_2D_MULTISAMPLE : GL_TEXTURE_2D, framebuffer->texColourBuffer);

  mGL_DEBUG_ERROR_CHECK();
#else
  mRETURN_RESULT(mR_NotImplemented);
#endif

  mRETURN_SUCCESS();
}

mFUNCTION(mTexture_Copy, mTexture &destination, mPtr<mFramebuffer> &source)
{
  mFUNCTION_SETUP();

  mERROR_IF(destination.uploadState != mRP_US_Ready, mR_ResourceStateInvalid);

#if defined(mRENDERER_OPENGL)
  {
    GLuint framebufferHandle;
    glGenFramebuffers(1, &framebufferHandle);
    mDEFER(glDeleteFramebuffers(1, &framebufferHandle));

    glBindFramebuffer(GL_FRAMEBUFFER, framebufferHandle);
    mDEFER(glBindFramebuffer(GL_FRAMEBUFFER, 0));

    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, (source->sampleCount > 0) ? GL_TEXTURE_2D_MULTISAMPLE : GL_TEXTURE_2D, destination.textureId, 0);

    mPtr<mScreenQuad> screenQuadRenderer;
    mDEFER_CALL(&screenQuadRenderer, mScreenQuad_Destroy);
    mERROR_CHECK(mScreenQuad_Create(&screenQuadRenderer, &mDefaultTempAllocator));

    mERROR_CHECK(mTexture_Bind(source));
    mERROR_CHECK(mShader_SetUniform(screenQuadRenderer->shader, "_texture0", source));
    mERROR_CHECK(mScreenQuad_Render(screenQuadRenderer));
    
    // TODO: Get a solution to work using the this: (Just untested. This should technically work - just remove the screenQuadRenderer.)
    //glBindFramebuffer(GL_DRAW_FRAMEBUFFER, framebufferHandle);
    //glBindFramebuffer(GL_READ_FRAMEBUFFER, source->frameBufferHandle);
    //glDrawBuffer(GL_BACK);
    //glBlitFramebuffer(0, 0, (GLsizei)source->size.x, (GLsizei)source->size.y, 0, 0, (GLsizei)destination.resolution.x, (GLsizei)destination.resolution.y, GL_COLOR_BUFFER_BIT, GL_NEAREST);
  }
#else
  mRETURN_RESULT(mR_NotImplemented);
#endif

  mGL_DEBUG_ERROR_CHECK();

  size_t queueCount = 0;
  mERROR_CHECK(mQueue_GetCount(mFramebuffer_Queue, &queueCount));

  // bind previous framebuffer (if any)
  if (queueCount > 0)
  {
    mPtr<mFramebuffer> framebuffer;
    mDEFER_CALL(&framebuffer, mFramebuffer_Destroy);
    mERROR_CHECK(mQueue_PeekBack(mFramebuffer_Queue, &framebuffer));
    mERROR_CHECK(mFramebuffer_Bind(framebuffer));
  }

  mRETURN_SUCCESS();
}

//////////////////////////////////////////////////////////////////////////

mFUNCTION(mFramebuffer_Create_Internal, mFramebuffer *pFramebuffer, const mVec2s &size, const mTexture2DParams &params, const size_t sampleCount)
{
  mFUNCTION_SETUP();

  mERROR_IF(pFramebuffer == nullptr, mR_ArgumentNull);
  mERROR_IF(size.x > UINT32_MAX || size.y > UINT32_MAX, mR_ArgumentOutOfBounds);

  pFramebuffer->allocatedSize = pFramebuffer->size = size;
  pFramebuffer->textureUnit = (size_t)-1;
  pFramebuffer->sampleCount = sampleCount;
  pFramebuffer->textureParams = params;

#ifdef mRENDERER_OPENGL
  glGenFramebuffers(1, &pFramebuffer->frameBufferHandle);
  glBindFramebuffer(GL_FRAMEBUFFER, pFramebuffer->frameBufferHandle);
  glGenTextures(1, &pFramebuffer->texColourBuffer);
  glBindTexture((pFramebuffer->sampleCount > 0) ? GL_TEXTURE_2D_MULTISAMPLE : GL_TEXTURE_2D, pFramebuffer->texColourBuffer);

  if (pFramebuffer->sampleCount > 0)
    glTexImage2DMultisample(GL_TEXTURE_2D_MULTISAMPLE, (GLsizei)sampleCount, GL_RGBA, (GLsizei)size.x, (GLsizei)size.y, false);
  else
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, (GLsizei)size.x, (GLsizei)size.y, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);

  mERROR_CHECK(mTexture2DParams_ApplyToBoundTexture(params, pFramebuffer->sampleCount > 0));
  
  glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, (pFramebuffer->sampleCount > 0) ? GL_TEXTURE_2D_MULTISAMPLE : GL_TEXTURE_2D, pFramebuffer->texColourBuffer, 0);

  glGenRenderbuffers(1, &pFramebuffer->rboDepthStencil);
  glBindRenderbuffer(GL_RENDERBUFFER, pFramebuffer->rboDepthStencil);

  if (sampleCount > 0)
    glRenderbufferStorageMultisample(GL_RENDERBUFFER, (GLint)sampleCount, GL_DEPTH24_STENCIL8, (GLsizei)size.x, (GLsizei)size.y);
  else
    glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH24_STENCIL8, (GLsizei)size.x, (GLsizei)size.y);
  
  const GLenum result = glCheckFramebufferStatus(GL_FRAMEBUFFER);
  mERROR_IF(result != GL_FRAMEBUFFER_COMPLETE, mR_RenderingError);

  glBindFramebuffer(GL_FRAMEBUFFER, 0);

  mGL_DEBUG_ERROR_CHECK();
#endif

#ifdef DEBUG_FRAMBUFFER_CREATION
  mPRINT_DEBUG("Created Framebuffer %" PRIu64 " with texture %" PRIu64 ".\n", (uint64_t)pFramebuffer->frameBufferHandle, (uint64_t)pFramebuffer->texColourBuffer);
#endif

  mGL_ERROR_CHECK();

  size_t queueCount = 0;
  mERROR_CHECK(mQueue_GetCount(mFramebuffer_Queue, &queueCount));

  // bind previous framebuffer (if any)
  if (queueCount > 0)
  {
    mPtr<mFramebuffer> framebuffer;
    mDEFER_CALL(&framebuffer, mFramebuffer_Destroy);
    mERROR_CHECK(mQueue_PeekBack(mFramebuffer_Queue, &framebuffer));
    mERROR_CHECK(mFramebuffer_Bind(framebuffer));
  }

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
