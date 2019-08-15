#ifndef mTexture_h__
#define mTexture_h__

#include "mRenderParams.h"
#include "mImageBuffer.h"
#include "mResourceManager.h"

struct mTexture
{
  mVec2s resolution;
  mVec2f resolutionF;
  mRenderParams_UploadState uploadState = mRP_US_NotInitialized;
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
mFUNCTION(mTexture_CreateFromUnownedIndex, OUT mTexture *pTexture, int textureIndex, const size_t textureUnit = 0, const size_t sampleCount = 0);
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

//////////////////////////////////////////////////////////////////////////

struct mTexture3D
{
  mVec3s resolution;
  mVec3f resolutionF;
  mRenderParams_UploadState uploadState = mRP_US_NotInitialized;

#if defined(mRENDERER_OPENGL)
  GLuint textureId;
  GLuint textureUnit;
#endif
};

mFUNCTION(mTexture3D_Create, OUT mTexture3D *pTexture, const uint8_t *pData, const mVec3s &size, const mPixelFormat pixelFormat = mPF_B8G8R8A8, const size_t textureUnit = 0, const mTexture3DParams &textureParams = mTexture3DParams());
mFUNCTION(mTexture3D_Allocate, OUT mTexture3D *pTexture, const mVec3s &size, const mPixelFormat pixelFormat = mPF_B8G8R8A8, const size_t textureUnit = 0, const mTexture3DParams &textureParams = mTexture3DParams());
mFUNCTION(mTexture3D_Destroy, IN_OUT mTexture3D *pTexture);

mFUNCTION(mTexture3D_SetTo, mTexture3D &texture, const uint8_t *pData, const mVec3s &size, const mPixelFormat pixelFormat = mPF_B8G8R8A8);
mFUNCTION(mTexture3D_Bind, mTexture3D &texture, const size_t textureUnit = 0);

mFUNCTION(mDestruct, IN_OUT mTexture3D *pTexture);

#endif // mTexture_h__

