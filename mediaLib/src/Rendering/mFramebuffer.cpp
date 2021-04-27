#include "mFramebuffer.h"
#include "mScreenQuad.h"
#include "mHardwareWindow.h"

//#define DEBUG_FRAMBUFFER_CREATION

#if defined (mRENDERER_OPENGL)
#ifdef GIT_BUILD // Define __M_FILE__
  #ifdef __M_FILE__
    #undef __M_FILE__
  #endif
  #define __M_FILE__ "1fNBDeKfYQkAJQSQhx6M314RdQis15VJ++OE3e/ZzoZ1izXkI8uLSr1ef2wIzw9teQ3VDdElhcuqYIOr"
#endif

GLuint mFrameBuffer_ActiveFrameBufferHandle = 0;
#endif

mPtr<mQueue<mPtr<mFramebuffer>>> mFramebuffer_Queue = nullptr;

static mFUNCTION(mFramebuffer_Create_Internal, OUT mFramebuffer *pFramebuffer, const mVec2s &size, const mTexture2DParams &params, const size_t sampleCount, const mPixelFormat pixelFormat);
static mFUNCTION(mFramebuffer_Destroy_Internal, IN mFramebuffer *pFramebuffer);

#ifdef mRENDERER_OPENGL
static mFUNCTION(mFramebuffer_PixelFormatToGLenumChannels_Internal, const mPixelFormat pixelFormat, OUT GLenum *pPixelFormat);
static mFUNCTION(mFramebuffer_PixelFormatToGLenumDataType_Internal, const mPixelFormat pixelFormat, OUT GLenum *pPixelFormat);
#endif

//////////////////////////////////////////////////////////////////////////

mFUNCTION(mFramebuffer_Create, OUT mPtr<mFramebuffer> *pFramebuffer, IN mAllocator *pAllocator, const mVec2s &size, const mTexture2DParams &textureParams /* = mTexture2DParams() */, const size_t sampleCount /* = 0 */)
{
  mFUNCTION_SETUP();

  mERROR_IF(pFramebuffer == nullptr, mR_ArgumentNull);

  if (mFramebuffer_Queue == nullptr)
    mERROR_CHECK(mQueue_Create(&mFramebuffer_Queue, &mDefaultAllocator));

  mERROR_CHECK(mSharedPointer_Allocate(pFramebuffer, pAllocator, (std::function<void (mFramebuffer *)>)[](mFramebuffer *pData) { mFramebuffer_Destroy_Internal(pData); }, 1));

  mERROR_CHECK(mFramebuffer_Create_Internal(pFramebuffer->GetPointer(), size, textureParams, sampleCount, mPF_R8G8B8A8));

  mRETURN_SUCCESS();
}

mFUNCTION(mFramebuffer_Create, OUT mPtr<mFramebuffer> *pFramebuffer, IN mAllocator *pAllocator, const mVec2s &size, const mPixelFormat pixelFormat, const mTexture2DParams &textureParams /* = mTexture2DParams() */, const size_t sampleCount /* = 0 */)
{
  mFUNCTION_SETUP();

  mERROR_IF(pFramebuffer == nullptr, mR_ArgumentNull);

  if (mFramebuffer_Queue == nullptr)
    mERROR_CHECK(mQueue_Create(&mFramebuffer_Queue, &mDefaultAllocator));

  mERROR_CHECK(mSharedPointer_Allocate(pFramebuffer, pAllocator, (std::function<void (mFramebuffer *)>)[](mFramebuffer *pData) { mFramebuffer_Destroy_Internal(pData); }, 1));

  mERROR_CHECK(mFramebuffer_Create_Internal(pFramebuffer->GetPointer(), size, textureParams, sampleCount, pixelFormat));

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
  mGL_DEBUG_ERROR_CHECK();

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
  mERROR_CHECK(mRenderParams_SetCurrentRenderResolution(mRenderParams_BackBufferResolution));
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

  if (mFramebuffer_Queue == nullptr)
    mERROR_CHECK(mQueue_Create(&mFramebuffer_Queue, &mDefaultAllocator));

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

  GLenum glPixelFormat = GL_RGBA;
  mERROR_CHECK(mFramebuffer_PixelFormatToGLenumChannels_Internal(framebuffer->pixelFormat, &glPixelFormat));

  framebuffer->allocatedSize = framebuffer->size = size;

  mGL_DEBUG_ERROR_CHECK();

  glBindTexture((framebuffer->sampleCount > 0) ? GL_TEXTURE_2D_MULTISAMPLE : GL_TEXTURE_2D, framebuffer->textureId);

  if (framebuffer->sampleCount > 0)
    glTexImage2DMultisample(GL_TEXTURE_2D_MULTISAMPLE, (GLsizei)framebuffer->sampleCount, glPixelFormat, (GLsizei)size.x, (GLsizei)size.y, false);
  else
    glTexImage2D(GL_TEXTURE_2D, 0, glPixelFormat, (GLsizei)size.x, (GLsizei)size.y, 0, glPixelFormat, GL_UNSIGNED_BYTE, nullptr);

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
  GLenum type = GL_UNSIGNED_BYTE;

  switch (pixelFormat)
  {
  case mPF_R8G8B8A8:
  case mPF_Rf16Gf16Bf16Af16:
  case mPF_Rf32Gf32Bf32Af32:
    glPixelFormat = GL_RGBA;
    break;

  case mPF_R8G8B8:
  case mPF_Rf16Gf16Bf16:
  case mPF_Rf32Gf32Bf32:
    glPixelFormat = GL_RGB;
    break;

  case mPF_B8G8R8A8:
    glPixelFormat = GL_BGRA;
    break;

  case mPF_B8G8R8:
    glPixelFormat = GL_BGR;
    break;

  case mPF_Monochrome8:
    glPixelFormat = GL_RED;
    break;

  default:
    mRETURN_RESULT(mR_NotSupported);
    break;
  }

  switch (pixelFormat)
  {
  case mPF_Rf16Gf16Bf16:
  case mPF_Rf16Gf16Bf16Af16:
    type = GL_HALF_FLOAT;
    break;

  case mPF_Rf32Gf32Bf32:
  case mPF_Rf32Gf32Bf32Af32:
    type = GL_FLOAT;
    break;

  default:
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

    glBindTexture(GL_TEXTURE_2D, framebuffer->textureId);

    mGL_ERROR_CHECK();

    glGetTexImage(GL_TEXTURE_2D, 0, glPixelFormat, type, (*pImageBuffer)->pPixels);

    mGL_ERROR_CHECK();
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

#if defined(mRENDERER_OPENGL)
  {
    mGL_DEBUG_ERROR_CHECK();

    mERROR_CHECK(mFramebuffer_Push(target));
    mDEFER_CALL_0(mFramebuffer_Pop);
    mGL_DEBUG_ERROR_CHECK();

    glBindFramebuffer(GL_DRAW_FRAMEBUFFER, target->frameBufferHandle);
    glBindFramebuffer(GL_READ_FRAMEBUFFER, source->frameBufferHandle);

    glBlitFramebuffer(0, 0, (GLsizei)source->size.x, (GLsizei)source->size.y, 0, 0, (GLsizei)target->size.x, (GLsizei)target->size.y, GL_COLOR_BUFFER_BIT, GL_LINEAR);

    mGL_DEBUG_ERROR_CHECK();
  }
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
  glBindTexture((framebuffer->sampleCount > 0) ? GL_TEXTURE_2D_MULTISAMPLE : GL_TEXTURE_2D, framebuffer->textureId);

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
    glBindFramebuffer(GL_DRAW_FRAMEBUFFER, framebufferHandle);
    glBindFramebuffer(GL_READ_FRAMEBUFFER, source->frameBufferHandle);
    glBlitFramebuffer(0, 0, (GLsizei)source->size.x, (GLsizei)source->size.y, 0, 0, (GLsizei)destination.resolution.x, (GLsizei)destination.resolution.y, GL_COLOR_BUFFER_BIT, GL_NEAREST);
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

static mFUNCTION(mFramebuffer_Create_Internal, mFramebuffer *pFramebuffer, const mVec2s &size, const mTexture2DParams &params, const size_t sampleCount, const mPixelFormat pixelFormat)
{
  mFUNCTION_SETUP();

  mERROR_IF(pFramebuffer == nullptr, mR_ArgumentNull);
  mERROR_IF(size.x > UINT32_MAX || size.y > UINT32_MAX, mR_ArgumentOutOfBounds);

  pFramebuffer->allocatedSize = pFramebuffer->size = size;
  pFramebuffer->textureUnit = (size_t)-1;
  pFramebuffer->sampleCount = sampleCount;
  pFramebuffer->textureParams = params;
  pFramebuffer->pixelFormat = pixelFormat;

#ifdef mRENDERER_OPENGL
  mGL_ERROR_CHECK();

  GLenum glPixelFormat = GL_RGBA;
  mERROR_CHECK(mFramebuffer_PixelFormatToGLenumChannels_Internal(pixelFormat, &glPixelFormat));

  GLenum glType = GL_UNSIGNED_BYTE;
  mERROR_CHECK(mFramebuffer_PixelFormatToGLenumDataType_Internal(pixelFormat, &glType));

  glGenFramebuffers(1, &pFramebuffer->frameBufferHandle);
  glBindFramebuffer(GL_FRAMEBUFFER, pFramebuffer->frameBufferHandle);
  glGenTextures(1, &pFramebuffer->textureId);
  glBindTexture((pFramebuffer->sampleCount > 0) ? GL_TEXTURE_2D_MULTISAMPLE : GL_TEXTURE_2D, pFramebuffer->textureId);

  if (pFramebuffer->sampleCount > 0)
    glTexImage2DMultisample(GL_TEXTURE_2D_MULTISAMPLE, (GLsizei)sampleCount, glPixelFormat, (GLsizei)size.x, (GLsizei)size.y, false);
  else
    glTexImage2D(GL_TEXTURE_2D, 0, glPixelFormat, (GLsizei)size.x, (GLsizei)size.y, 0, glPixelFormat, glType, nullptr);

  mERROR_CHECK(mTexture2DParams_ApplyToBoundTexture(params, pFramebuffer->sampleCount > 0));
  
  glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, (pFramebuffer->sampleCount > 0) ? GL_TEXTURE_2D_MULTISAMPLE : GL_TEXTURE_2D, pFramebuffer->textureId, 0);

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
  mPRINT_DEBUG("Created Framebuffer %" PRIu64 " with texture %" PRIu64 ".\n", (uint64_t)pFramebuffer->frameBufferHandle, (uint64_t)pFramebuffer->textureId);
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

static mFUNCTION(mFramebuffer_Destroy_Internal, IN mFramebuffer *pFramebuffer)
{
  mFUNCTION_SETUP();

  mERROR_IF(pFramebuffer == nullptr, mR_ArgumentNull);

#if defined(mRENDERER_OPENGL)
  mERROR_CHECK(mFramebuffer_Unbind());

  if (pFramebuffer->frameBufferHandle)
    glDeleteFramebuffers(1, &pFramebuffer->frameBufferHandle);

  pFramebuffer->frameBufferHandle = 0;

  if (pFramebuffer->textureId)
    glDeleteTextures(0, &pFramebuffer->textureId);

  pFramebuffer->textureId = 0;
#endif

  mRETURN_SUCCESS();
}

#ifdef mRENDERER_OPENGL
static mFUNCTION(mFramebuffer_PixelFormatToGLenumChannels_Internal, const mPixelFormat pixelFormat, OUT GLenum *pPixelFormat)
{
  mFUNCTION_SETUP();

  switch (pixelFormat)
  {
  case mPF_R8G8B8:
  case mPF_Rf16Gf16Bf16:
  case mPF_Rf32Gf32Bf32:
    *pPixelFormat = GL_RGB;
    break;

  case mPF_R8G8B8A8:
  case mPF_Rf16Gf16Bf16Af16:
  case mPF_Rf32Gf32Bf32Af32:
    *pPixelFormat = GL_RGBA;
    break;

  case mPF_Monochrome8:
    *pPixelFormat = GL_RED;
    break;

  default:
    mRETURN_RESULT(mR_InvalidParameter);
  }

  mRETURN_SUCCESS();
}

static mFUNCTION(mFramebuffer_PixelFormatToGLenumDataType_Internal, const mPixelFormat pixelFormat, OUT GLenum *pPixelFormat)
{
  mFUNCTION_SETUP();

  switch (pixelFormat)
  {
  case mPF_Rf16Gf16Bf16:
  case mPF_Rf16Gf16Bf16Af16:
    *pPixelFormat = GL_HALF_FLOAT;
    break;

  case mPF_Rf32Gf32Bf32:
  case mPF_Rf32Gf32Bf32Af32:
    *pPixelFormat = GL_FLOAT;
    break;

  default:
    *pPixelFormat = GL_UNSIGNED_BYTE;
  }

  mRETURN_SUCCESS();
}
#endif
