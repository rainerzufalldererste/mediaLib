#ifndef mTexture_h__
#define mTexture_h__

#include "mRenderParams.h"
#include "mImageBuffer.h"
#include "mResourceManager.h"

struct mTexture
{
  mVec2s resolution;
  mVec2f resolutionF;
  mRenderParams_UploadState uploadState = mRenderParams_UploadState::mRP_US_NotInitialized;
  mPtr<mImageBuffer> imageBuffer;

#if defined(mRENDERER_OPENGL)
  GLuint textureId;
  GLuint textureUnit;
#endif

  size_t sampleCount;
  bool foreignTexture;
};

mFUNCTION(mTexture_Create, OUT mTexture *pTexture, mPtr<mImageBuffer> &imageBuffer, const bool upload = true, const size_t textureUnit = 0, const mTexture2DParams &textureParams = mTexture2DParams());
mFUNCTION(mTexture_Create, OUT mTexture *pTexture, const mString &filename, const bool upload = true, const size_t textureUnit = 0, const mTexture2DParams &textureParams = mTexture2DParams());
mFUNCTION(mTexture_Create, OUT mTexture *pTexture, const uint8_t *pData, const mVec2s &size, const mPixelFormat pixelFormat = mPF_B8G8R8A8, const bool upload = true, const size_t textureUnit = 0, const mTexture2DParams &textureParams = mTexture2DParams());
mFUNCTION(mTexture_CreateFromUnownedIndex, OUT mTexture *pTexture, int textureIndex, const size_t textureUnit = 0);
mFUNCTION(mTexture_Allocate, OUT mTexture *pTexture, const mVec2s size, const mPixelFormat pixelFormat = mPF_R8G8B8A8, const size_t textureUnit = 0, const mTexture2DParams &textureParams = mTexture2DParams());
mFUNCTION(mTexture_Destroy, IN_OUT mTexture *pTexture);

mFUNCTION(mTexture_GetUploadState, mTexture &texture, OUT mRenderParams_UploadState *pUploadState);
mFUNCTION(mTexture_Upload, mTexture &texture);
mFUNCTION(mTexture_Bind, mTexture &texture, const size_t textureUnit = 0);

mFUNCTION(mTexture_SetTo, mTexture &texture, mPtr<mImageBuffer> &imageBuffer, const bool upload = true);
mFUNCTION(mTexture_SetTo, mTexture &texture, const uint8_t *pData, const mVec2s &size, const mPixelFormat pixelFormat = mPF_B8G8R8A8, const bool upload = true);

mFUNCTION(mTexture_Download, mTexture &texture, OUT mPtr<mImageBuffer> *pImageBuffer, IN mAllocator *pAllocator, const mPixelFormat pixelFormat = mPF_R8G8B8A8);

// For mResourceManager:

mFUNCTION(mCreateResource, OUT mTexture *pTexture, const mString &filename);
mFUNCTION(mDestroyResource, IN_OUT mTexture *pTexture);

mFUNCTION(mDestruct, IN_OUT mTexture *pTexture);

#endif // mTexture_h__

