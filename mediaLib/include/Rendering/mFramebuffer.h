#ifndef mFramebuffer_h__
#define mFramebuffer_h__

#include "mRenderParams.h"
#include "mImageBuffer.h"
#include "mTexture.h"

#if defined (mRENDERER_OPENGL)
extern GLuint mFrameBuffer_ActiveFrameBufferHandle;
#endif

struct mFramebuffer
{
  mVec2s size;
  mVec2s allocatedSize;
  size_t textureUnit;
#if defined(mRENDERER_OPENGL)
  GLuint frameBufferHandle;
  GLuint texColourBuffer;
  GLuint rboDepthStencil;
#endif

  size_t sampleCount;
  mTexture2DParams textureParams;
};

mFUNCTION(mFramebuffer_Create, OUT mPtr<mFramebuffer> *pFramebuffer, IN mAllocator *pAllocator, const mVec2s &size, const mTexture2DParams &textureParams = mTexture2DParams(), const size_t sampleCount = 0);
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
