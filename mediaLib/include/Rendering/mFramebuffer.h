#ifndef mFramebuffer_h__
#define mFramebuffer_h__

#include "mRenderParams.h"
#include "mImageBuffer.h"
#include "mTexture.h"

#if defined (mRENDERER_OPENGL)
#ifdef GIT_BUILD // Define __M_FILE__
  #ifdef __M_FILE__
    #undef __M_FILE__
  #endif
  #define __M_FILE__ "59KY3lHCw9BWZQbqjoOuIQPyT5r5Q/6NZ5R5q8rzNPRYrJeHN9UgEBfosKbnLgkWUK/cTaMSAQH+KFtU"
#endif

extern thread_local GLuint mFrameBuffer_ActiveFrameBufferHandle;
#endif

enum mFramebufferFlags : uint32_t
{
  mFF_None = 0,
  mFF_HasDepthStencil = 1 << 0,
  mFF_DepthIsTexture = 1 << 1,
  mFF_IsForeignTexture = 1 << 2,
};

typedef uint32_t mFramebufferFlags_;

struct mFramebuffer
{
  mVec2s size;
  mVec2s allocatedSize;
  size_t textureUnit, depthTextureUnit;
  mFramebufferFlags flags;
#if defined(mRENDERER_OPENGL)
  GLuint frameBufferHandle;
  GLuint textureId;

#pragma warning(push)
#pragma warning(disable: 4201)
  union
  {
    GLuint depthStencilRBO; // if depth/stencil is a renderbuffer.
    GLuint depthStencilTextureId; // if depth/stencil is a texture.
  };
#pragma warning(pop)
#endif

  size_t sampleCount;
  mTexture2DParams textureParams;
  mPixelFormat pixelFormat;
};

mFUNCTION(mFramebuffer_Create, OUT mPtr<mFramebuffer> *pFramebuffer, IN mAllocator *pAllocator, const mVec2s &size, const mTexture2DParams &textureParams = mTexture2DParams(), const size_t sampleCount = 0, const mFramebufferFlags_ flags = mFF_HasDepthStencil);
mFUNCTION(mFramebuffer_Create, OUT mPtr<mFramebuffer> *pFramebuffer, IN mAllocator *pAllocator, const mVec2s &size, const mPixelFormat pixelFormat, const mTexture2DParams &textureParams = mTexture2DParams(), const size_t sampleCount = 0, const mFramebufferFlags_ flags = mFF_HasDepthStencil);

#if defined(mRENDERER_OPENGL)
mFUNCTION(mFramebuffer_CreateFromForeignTextureId, OUT mPtr<mFramebuffer> *pFramebuffer, IN mAllocator *pAllocator, const GLuint textureId, const size_t sampleCount = 0, const mFramebufferFlags_ flags = mFF_HasDepthStencil);
#endif

mFUNCTION(mFramebuffer_Destroy, IN_OUT mPtr<mFramebuffer> *pFramebuffer);
mFUNCTION(mFramebuffer_Bind, mPtr<mFramebuffer> &framebuffer);
mFUNCTION(mFramebuffer_Unbind);

mFUNCTION(mFramebuffer_Push, mPtr<mFramebuffer> &framebuffer);
mFUNCTION(mFramebuffer_Pop);

mFUNCTION(mFramebuffer_SetResolution, mPtr<mFramebuffer> &framebuffer, const mVec2s &size);
mFUNCTION(mFramebuffer_Download, mPtr<mFramebuffer> &framebuffer, OUT mPtr<mImageBuffer> *pImageBuffer, IN mAllocator *pAllocator, const mPixelFormat pixelFormat = mPF_R8G8B8A8);
mFUNCTION(mFramebuffer_BlitToScreen, mPtr<struct mHardwareWindow> &window, mPtr<mFramebuffer> &framebuffer);
mFUNCTION(mFramebuffer_BlitToFramebuffer, mPtr<mFramebuffer> &target, mPtr<mFramebuffer> &source);

mFUNCTION(mTexture_Bind, mPtr<mFramebuffer> &framebuffer, const size_t textureUnit = 0);
mFUNCTION(mTexture_Copy, mTexture &destination, mPtr<mFramebuffer> &source);

#endif // mFramebuffer_h__
