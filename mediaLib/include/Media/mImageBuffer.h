#ifndef mImageBuffer_h__
#define mImageBuffer_h__

#include "mediaLib.h"
#include "mPixelFormat.h"

#ifdef GIT_BUILD // Define __M_FILE__
  #ifdef __M_FILE__
    #undef __M_FILE__
  #endif
  #define __M_FILE__ "tnnAGkeojpHNbyvX55Ucm+fYEnrzRXB1sgAsL1LIT0RGQoepVzttNxDJN982s3WZ0HzEeft8YW6aH0XV"
#endif

enum mImageBuffer_CopyFlags
{
  mIB_CF_None = 0,
  mIB_CF_ResizeAllowed = 1 << 0,
  mIB_CF_PixelFormatChangeAllowed = 1 << 1,
};

struct mImageBuffer
{
  uint8_t *pPixels;
  size_t allocatedSize;
  bool ownedResource;
  mPixelFormat pixelFormat;
  mVec2s currentSize;
  size_t lineStride;
  mAllocator *pAllocator;
};

mFUNCTION(mImageBuffer_Create, OUT mPtr<mImageBuffer> *pImageBuffer, IN OPTIONAL mAllocator *pAllocator);
mFUNCTION(mImageBuffer_CreateFromFile, OUT mPtr<mImageBuffer> *pImageBuffer, IN OPTIONAL mAllocator *pAllocator, const mString &filename, const mPixelFormat pixelFormat = mPF_R8G8B8A8);
mFUNCTION(mImageBuffer_CreateFromData, OUT mPtr<mImageBuffer> *pImageBuffer, IN OPTIONAL mAllocator *pAllocator, IN const uint8_t *pData, const size_t size, const mPixelFormat pixelFormat = mPF_R8G8B8A8);
mFUNCTION(mImageBuffer_Create, OUT mPtr<mImageBuffer> *pImageBuffer, IN OPTIONAL mAllocator *pAllocator, const mVec2s &size, const mPixelFormat pixelFormat = mPF_B8G8R8A8);
mFUNCTION(mImageBuffer_Create, OUT mPtr<mImageBuffer> *pImageBuffer, IN OPTIONAL mAllocator *pAllocator, IN const void *pData, const mVec2s &size, const mPixelFormat pixelFormat = mPF_B8G8R8A8);
mFUNCTION(mImageBuffer_Create, OUT mPtr<mImageBuffer> *pImageBuffer, IN OPTIONAL mAllocator *pAllocator, IN const void *pData, const mVec2s &size, const size_t stride, const mPixelFormat pixelFormat = mPF_B8G8R8A8);
mFUNCTION(mImageBuffer_Create, OUT mPtr<mImageBuffer> *pImageBuffer, IN OPTIONAL mAllocator *pAllocator, IN const void *pData, const mVec2s &size, const mRectangle2D<size_t> &rect, const mPixelFormat pixelFormat = mPF_B8G8R8A8);
mFUNCTION(mImageBuffer_Destroy, OUT mPtr<mImageBuffer> *pImageBuffer);

mFUNCTION(mImageBuffer_AllocateBuffer, mPtr<mImageBuffer> &imageBuffer, const mVec2s &size, const mPixelFormat pixelFormat = mPF_B8G8R8A8);
mFUNCTION(mImageBuffer_SetBuffer, mPtr<mImageBuffer> &imageBuffer, IN const void *pData, const mVec2s &size, const mPixelFormat pixelFormat = mPF_B8G8R8A8);
mFUNCTION(mImageBuffer_SetBuffer, mPtr<mImageBuffer> &imageBuffer, IN const void *pData, const mVec2s &size, const size_t stride, const mPixelFormat pixelFormat = mPF_B8G8R8A8);
mFUNCTION(mImageBuffer_SetBuffer, mPtr<mImageBuffer> &imageBuffer, IN const void *pData, const mVec2s &size, const mRectangle2D<size_t> &rect, const mPixelFormat pixelFormat = mPF_B8G8R8A8);
mFUNCTION(mImageBuffer_CopyTo, mPtr<mImageBuffer> &source, mPtr<mImageBuffer> &target, const mImageBuffer_CopyFlags copyFlags);

mFUNCTION(mImageBuffer_SaveAsPng, mPtr<mImageBuffer> &imageBuffer, const mString &filename);
mFUNCTION(mImageBuffer_SaveAsJpeg, mPtr<mImageBuffer> &imageBuffer, const mString &filename);
mFUNCTION(mImageBuffer_SaveAsBmp, mPtr<mImageBuffer> &imageBuffer, const mString &filename);
mFUNCTION(mImageBuffer_SaveAsTga, mPtr<mImageBuffer> &imageBuffer, const mString &filename);
mFUNCTION(mImageBuffer_SaveAsRaw, mPtr<mImageBuffer> &imageBuffer, const mString &filename);

mFUNCTION(mImageBuffer_SetToFile, mPtr<mImageBuffer> &imageBuffer, const mString &filename, const mPixelFormat pixelFormat = mPF_B8G8R8A8);
mFUNCTION(mImageBuffer_SetToData, mPtr<mImageBuffer> &imageBuffer, IN const uint8_t *pData, const size_t size, const mPixelFormat pixelFormat = mPF_B8G8R8A8);

mFUNCTION(mImageBuffer_FlipY, mPtr<mImageBuffer> &imageBuffer);

#endif // mImageBuffer_h__
