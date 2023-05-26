#include "mFramebuffer.h"

#include "mScreenQuad.h"
#include "mHardwareWindow.h"
#include "mProfiler.h"

//#define DEBUG_FRAMBUFFER_CREATION

#if defined (mRENDERER_OPENGL)
#ifdef GIT_BUILD // Define __M_FILE__
  #ifdef __M_FILE__
    #undef __M_FILE__
  #endif
  #define __M_FILE__ "1fNBDeKfYQkAJQSQhx6M314RdQis15VJ++OE3e/ZzoZ1izXkI8uLSr1ef2wIzw9teQ3VDdElhcuqYIOr"
#endif

thread_local GLuint mFrameBuffer_ActiveFrameBufferHandle = 0;
#endif

static thread_local mPtr<mQueue<mPtr<mFramebuffer>>> mFramebuffer_Queue = nullptr;

static mFUNCTION(mFramebuffer_Create_Internal, OUT mFramebuffer *pFramebuffer, const mVec2s &size, const mTexture2DParams &params, const size_t sampleCount, const mPixelFormat pixelFormat, const mFramebufferFlags_ flags);
static mFUNCTION(mFramebuffer_CreateFromForeign_Internal, OUT mFramebuffer *pFramebuffer, const GLuint textureId, const size_t sampleCount, const mFramebufferFlags_ flags);
static mFUNCTION(mFramebuffer_Destroy_Internal, IN mFramebuffer *pFramebuffer);

//////////////////////////////////////////////////////////////////////////

mFUNCTION(mFramebuffer_Create, OUT mPtr<mFramebuffer> *pFramebuffer, IN mAllocator *pAllocator, const mVec2s &size, const mTexture2DParams &textureParams /* = mTexture2DParams() */, const size_t sampleCount /* = 0 */, const mFramebufferFlags_ flags /* = mFF_HasDepthStencil */)
{
  mFUNCTION_SETUP();

  mERROR_IF(pFramebuffer == nullptr, mR_ArgumentNull);

  if (mFramebuffer_Queue == nullptr)
    mERROR_CHECK(mQueue_Create(&mFramebuffer_Queue, &mDefaultAllocator));

  mDEFER_CALL_ON_ERROR(pFramebuffer, mSharedPointer_Destroy);
  mERROR_CHECK(mSharedPointer_Allocate(pFramebuffer, pAllocator, (std::function<void (mFramebuffer *)>)[](mFramebuffer *pData) { mFramebuffer_Destroy_Internal(pData); }, 1));

  mERROR_CHECK(mFramebuffer_Create_Internal(pFramebuffer->GetPointer(), size, textureParams, sampleCount, mPF_R8G8B8A8, flags));

  mRETURN_SUCCESS();
}

mFUNCTION(mFramebuffer_Create, OUT mPtr<mFramebuffer> *pFramebuffer, IN mAllocator *pAllocator, const mVec2s &size, const mPixelFormat pixelFormat, const mTexture2DParams &textureParams /* = mTexture2DParams() */, const size_t sampleCount /* = 0 */, const mFramebufferFlags_ flags /* = mFF_HasDepthStencil */)
{
  mFUNCTION_SETUP();

  mERROR_IF(pFramebuffer == nullptr, mR_ArgumentNull);

  if (mFramebuffer_Queue == nullptr)
    mERROR_CHECK(mQueue_Create(&mFramebuffer_Queue, &mDefaultAllocator));

  mDEFER_CALL_ON_ERROR(pFramebuffer, mSharedPointer_Destroy);
  mERROR_CHECK(mSharedPointer_Allocate(pFramebuffer, pAllocator, (std::function<void (mFramebuffer *)>)[](mFramebuffer *pData) { mFramebuffer_Destroy_Internal(pData); }, 1));

  mERROR_CHECK(mFramebuffer_Create_Internal(pFramebuffer->GetPointer(), size, textureParams, sampleCount, pixelFormat, flags));

  mRETURN_SUCCESS();
}

mFUNCTION(mFramebuffer_CreateFromForeignTextureId, OUT mPtr<mFramebuffer> *pFramebuffer, IN mAllocator *pAllocator, const GLuint textureId, const size_t sampleCount /* = 0 */, const mFramebufferFlags_ flags /* = mFF_HasDepthStencil */)
{
  mFUNCTION_SETUP();

  mERROR_IF(pFramebuffer == nullptr, mR_ArgumentNull);

  if (mFramebuffer_Queue == nullptr)
    mERROR_CHECK(mQueue_Create(&mFramebuffer_Queue, &mDefaultAllocator));

  mDEFER_CALL_ON_ERROR(pFramebuffer, mSharedPointer_Destroy);
  mERROR_CHECK(mSharedPointer_Allocate(pFramebuffer, pAllocator, (std::function<void(mFramebuffer *)>)[](mFramebuffer *pData) { mFramebuffer_Destroy_Internal(pData); }, 1));

  mERROR_CHECK(mFramebuffer_CreateFromForeign_Internal(pFramebuffer->GetPointer(), textureId, sampleCount, flags));

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

  mPROFILE_SCOPED("mFramebuffer_Bind");

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

  mPROFILE_SCOPED("mFramebuffer_Unbind");

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

  mPROFILE_SCOPED("mFramebuffer_SetResolution");

  GLenum glPixelFormat = GL_RGBA;
  mERROR_CHECK(mRenderParams_PixelFormatToGLenumChannels(framebuffer->pixelFormat, &glPixelFormat));

  GLenum glType = GL_UNSIGNED_BYTE;
  mERROR_CHECK(mRenderParams_PixelFormatToGLenumDataType(framebuffer->pixelFormat, &glType));

  framebuffer->allocatedSize = framebuffer->size = size;

  mGL_DEBUG_ERROR_CHECK();

  glBindTexture((framebuffer->sampleCount > 0) ? GL_TEXTURE_2D_MULTISAMPLE : GL_TEXTURE_2D, framebuffer->textureId);

  if (framebuffer->sampleCount > 0)
    glTexImage2DMultisample(GL_TEXTURE_2D_MULTISAMPLE, (GLsizei)framebuffer->sampleCount, glPixelFormat, (GLsizei)size.x, (GLsizei)size.y, true);
  else
    glTexImage2D(GL_TEXTURE_2D, 0, glPixelFormat, (GLsizei)size.x, (GLsizei)size.y, 0, glPixelFormat, glType, nullptr);

  mERROR_CHECK(mTexture2DParams_ApplyToBoundTexture(framebuffer->textureParams, framebuffer->sampleCount > 0));

  if (!!(framebuffer->flags & mFF_HasDepthStencil))
  {
    if (!(framebuffer->flags & mFF_DepthIsTexture))
    {
      glBindRenderbuffer(GL_RENDERBUFFER, framebuffer->depthStencilRBO);

      if (framebuffer->sampleCount > 0)
        glRenderbufferStorageMultisample(GL_RENDERBUFFER, (GLint)framebuffer->sampleCount, GL_DEPTH24_STENCIL8, (GLsizei)framebuffer->allocatedSize.x, (GLsizei)framebuffer->allocatedSize.y);
      else
        glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH24_STENCIL8, (GLsizei)framebuffer->allocatedSize.x, (GLsizei)framebuffer->allocatedSize.y);
    }
    else
    {
      glBindTexture((framebuffer->sampleCount > 0) ? GL_TEXTURE_2D_MULTISAMPLE : GL_TEXTURE_2D, framebuffer->depthStencilTextureId);

      if (framebuffer->sampleCount > 0)
        glTexImage2DMultisample(GL_TEXTURE_2D_MULTISAMPLE, (GLsizei)framebuffer->sampleCount, GL_DEPTH24_STENCIL8, (GLsizei)size.x, (GLsizei)size.y, true);
      else
        glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH24_STENCIL8, (GLsizei)size.x, (GLsizei)size.y, 0, GL_DEPTH_STENCIL, GL_UNSIGNED_BYTE, nullptr);

      if (framebuffer->sampleCount == 0)
      {
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
      }
    }
  }

  mGL_ERROR_CHECK();

  mRETURN_SUCCESS();
}

mFUNCTION(mFramebuffer_Download, mPtr<mFramebuffer> &framebuffer, OUT mPtr<mImageBuffer> *pImageBuffer, IN mAllocator *pAllocator, const mPixelFormat pixelFormat /* = mPF_R8G8B8A8 */)
{
  mFUNCTION_SETUP();

  mERROR_IF(framebuffer == nullptr || pImageBuffer == nullptr, mR_ArgumentNull);

  mPROFILE_SCOPED("mFramebuffer_Download");

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

  mPROFILE_SCOPED("mFramebuffer_BlitToScreen");

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

  mPROFILE_SCOPED("mFramebuffer_BlitToFramebuffer");

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

  mPROFILE_SCOPED("mTexture_Bind (mFramebuffer)");

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

  mPROFILE_SCOPED("mTexture_Copy (mFramebuffer)");

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

static mFUNCTION(mFramebuffer_CreateDepthStencil, mFramebuffer *pFramebuffer)
{
  mFUNCTION_SETUP();

  mERROR_IF(!(pFramebuffer->flags & mFF_HasDepthStencil), mR_Success);

  if (!(pFramebuffer->flags & mFF_DepthIsTexture))
  {
    glGenRenderbuffers(1, &pFramebuffer->depthStencilRBO);
    glBindRenderbuffer(GL_RENDERBUFFER, pFramebuffer->depthStencilRBO);

    if (pFramebuffer->sampleCount > 0)
      glRenderbufferStorageMultisample(GL_RENDERBUFFER, (GLsizei)pFramebuffer->sampleCount, GL_DEPTH24_STENCIL8, (GLsizei)pFramebuffer->size.x, (GLsizei)pFramebuffer->size.y);
    else
      glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH24_STENCIL8, (GLsizei)pFramebuffer->size.x, (GLsizei)pFramebuffer->size.y);

    glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_STENCIL_ATTACHMENT, GL_RENDERBUFFER, pFramebuffer->depthStencilRBO);
  }
  else
  {
    glGenTextures(1, &pFramebuffer->depthStencilTextureId);
    glBindTexture((pFramebuffer->sampleCount > 0) ? GL_TEXTURE_2D_MULTISAMPLE : GL_TEXTURE_2D, pFramebuffer->depthStencilTextureId);

    mERROR_CHECK(mTexture2DParams_ApplyToBoundTexture(pFramebuffer->textureParams, pFramebuffer->sampleCount > 0));

    if (pFramebuffer->sampleCount > 0)
      glTexImage2DMultisample(GL_TEXTURE_2D_MULTISAMPLE, (GLsizei)pFramebuffer->sampleCount, GL_DEPTH24_STENCIL8, (GLsizei)pFramebuffer->size.x, (GLsizei)pFramebuffer->size.y, true);
    else
      glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH24_STENCIL8, (GLsizei)pFramebuffer->size.x, (GLsizei)pFramebuffer->size.y, 0, GL_DEPTH_STENCIL, GL_UNSIGNED_BYTE, nullptr);

    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_STENCIL_ATTACHMENT, (pFramebuffer->sampleCount > 0) ? GL_TEXTURE_2D_MULTISAMPLE : GL_TEXTURE_2D, pFramebuffer->depthStencilTextureId, 0);
  }

  mRETURN_SUCCESS();
}

static mFUNCTION(mFramebuffer_Create_Internal, mFramebuffer *pFramebuffer, const mVec2s &size, const mTexture2DParams &params, const size_t sampleCount, const mPixelFormat pixelFormat, const mFramebufferFlags_ flags)
{
  mFUNCTION_SETUP();

  mERROR_IF(pFramebuffer == nullptr, mR_ArgumentNull);
  mERROR_IF(size.x > UINT32_MAX || size.y > UINT32_MAX, mR_ArgumentOutOfBounds);
  mERROR_IF(!!(flags & mFF_DepthIsTexture) && !(flags & mFF_HasDepthStencil), mR_InvalidParameter);

  mPROFILE_SCOPED("mFramebuffer_Create_Internal");

  pFramebuffer->flags = (mFramebufferFlags)flags;
  pFramebuffer->allocatedSize = pFramebuffer->size = size;
  pFramebuffer->textureUnit = (size_t)-1;
  pFramebuffer->sampleCount = sampleCount;
  pFramebuffer->textureParams = params;
  pFramebuffer->pixelFormat = pixelFormat;

#ifdef mRENDERER_OPENGL
  mGL_ERROR_CHECK();

  GLenum glPixelFormat = GL_RGBA;
  mERROR_CHECK(mRenderParams_PixelFormatToGLenumChannels(pixelFormat, &glPixelFormat));

  GLenum glType = GL_UNSIGNED_BYTE;
  mERROR_CHECK(mRenderParams_PixelFormatToGLenumDataType(pixelFormat, &glType));

  glGenFramebuffers(1, &pFramebuffer->frameBufferHandle);
  glBindFramebuffer(GL_FRAMEBUFFER, pFramebuffer->frameBufferHandle);

  glGenTextures(1, &pFramebuffer->textureId);
  glBindTexture((pFramebuffer->sampleCount > 0) ? GL_TEXTURE_2D_MULTISAMPLE : GL_TEXTURE_2D, pFramebuffer->textureId);

  if (pFramebuffer->sampleCount > 0)
    glTexImage2DMultisample(GL_TEXTURE_2D_MULTISAMPLE, (GLsizei)pFramebuffer->sampleCount, glPixelFormat, (GLsizei)size.x, (GLsizei)size.y, true);
  else
    glTexImage2D(GL_TEXTURE_2D, 0, glPixelFormat, (GLsizei)size.x, (GLsizei)size.y, 0, glPixelFormat, glType, nullptr);

  mERROR_CHECK(mTexture2DParams_ApplyToBoundTexture(params, pFramebuffer->sampleCount > 0));

  glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, (pFramebuffer->sampleCount > 0) ? GL_TEXTURE_2D_MULTISAMPLE : GL_TEXTURE_2D, pFramebuffer->textureId, 0);

  mERROR_CHECK(mFramebuffer_CreateDepthStencil(pFramebuffer));
  
  const GLenum result = glCheckFramebufferStatus(GL_FRAMEBUFFER);
  mERROR_IF(result != GL_FRAMEBUFFER_COMPLETE, mR_RenderingError);

  glBindFramebuffer(GL_FRAMEBUFFER, 0);

  mGL_DEBUG_ERROR_CHECK();
#endif

#ifdef DEBUG_FRAMBUFFER_CREATION
  mPRINT_DEBUG("Created Framebuffer ", (uint64_t)pFramebuffer->frameBufferHandle, " with texture ", (uint64_t)pFramebuffer->textureId, ".");
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

static mFUNCTION(mFramebuffer_CreateFromForeign_Internal, OUT mFramebuffer *pFramebuffer, const GLuint textureId, const size_t sampleCount, const mFramebufferFlags_ flags)
{
  mFUNCTION_SETUP();

  mERROR_IF(pFramebuffer == nullptr, mR_ArgumentNull);
  mERROR_IF(!!(flags & mFF_DepthIsTexture) && !(flags & mFF_HasDepthStencil), mR_InvalidParameter);

  mPROFILE_SCOPED("mFramebuffer_CreateFromForeign_Internal");

  pFramebuffer->flags = (mFramebufferFlags)(flags | mFF_IsForeignTexture);
  pFramebuffer->allocatedSize = pFramebuffer->size = mVec2s(0);
  pFramebuffer->textureUnit = (size_t)-1;
  pFramebuffer->sampleCount = sampleCount;
  pFramebuffer->pixelFormat = mPixelFormat_Count;
  pFramebuffer->textureParams = mTexture2DParams();

#ifdef mRENDERER_OPENGL
  mGL_ERROR_CHECK();

  glGenFramebuffers(1, &pFramebuffer->frameBufferHandle);
  glBindFramebuffer(GL_FRAMEBUFFER, pFramebuffer->frameBufferHandle);
  
  pFramebuffer->textureId = textureId;
  
  glBindTexture((pFramebuffer->sampleCount > 0) ? GL_TEXTURE_2D_MULTISAMPLE : GL_TEXTURE_2D, pFramebuffer->textureId);

  mVec2t<GLint> size;
  glGetTexLevelParameteriv(pFramebuffer->sampleCount > 0 ? GL_TEXTURE_2D_MULTISAMPLE : GL_TEXTURE_2D, 0, GL_TEXTURE_WIDTH, &size.x);
  glGetTexLevelParameteriv(pFramebuffer->sampleCount > 0 ? GL_TEXTURE_2D_MULTISAMPLE : GL_TEXTURE_2D, 0, GL_TEXTURE_HEIGHT, &size.y);

  pFramebuffer->allocatedSize = pFramebuffer->size = mVec2s(size);

  glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, (pFramebuffer->sampleCount > 0) ? GL_TEXTURE_2D_MULTISAMPLE : GL_TEXTURE_2D, pFramebuffer->textureId, 0);

  mERROR_CHECK(mFramebuffer_CreateDepthStencil(pFramebuffer));

  const GLenum result = glCheckFramebufferStatus(GL_FRAMEBUFFER);
  mERROR_IF(result != GL_FRAMEBUFFER_COMPLETE, mR_RenderingError);

  glBindFramebuffer(GL_FRAMEBUFFER, 0);

  mGL_DEBUG_ERROR_CHECK();
#endif

#ifdef DEBUG_FRAMBUFFER_CREATION
  mPRINT_DEBUG("Created Framebuffer ", (uint64_t)pFramebuffer->frameBufferHandle, " with foreign texture ", (uint64_t)pFramebuffer->textureId, ".");
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

  if (pFramebuffer->textureId && !(pFramebuffer->flags & mFF_IsForeignTexture))
    glDeleteTextures(1, &pFramebuffer->textureId);

  if (!!(pFramebuffer->flags & mFF_HasDepthStencil))
  {
    if (!!(pFramebuffer->flags & mFF_DepthIsTexture))
    {
      if (&pFramebuffer->depthStencilTextureId)
        glDeleteTextures(1, &pFramebuffer->depthStencilTextureId);

      pFramebuffer->depthStencilTextureId = 0;
    }
    else
    {
      if (pFramebuffer->depthStencilRBO)
        glDeleteRenderbuffers(1, &pFramebuffer->depthStencilRBO);

      pFramebuffer->depthStencilRBO = 0;
    }
  }

  pFramebuffer->textureId = 0;
#endif

  mRETURN_SUCCESS();
}
