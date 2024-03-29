#ifndef mPixelFormat_h__
#define mPixelFormat_h__

#include "mediaLib.h"
#include "mThreadPool.h"

#ifdef GIT_BUILD // Define __M_FILE__
  #ifdef __M_FILE__
    #undef __M_FILE__
  #endif
  #define __M_FILE__ "qN3buM09rfR4aXIp0+OMli5NtPtokbxdwmnOsji1Y2v9fYqBgleaaXc2D2tIx5Ycmg9HuyIEvurE9VK+"
#endif

struct mImageBuffer;

enum mPixelFormat
{
  mPF_Monochrome8,
  mPF_Monochrome16,
  mPF_Monochromef16,
  mPF_Monochromef32,

  mPF_R8G8,
  mPF_R16G16,
  mPF_Rf16Gf16,
  mPF_Rf32Gf32,

  mPF_R8G8B8,
  mPF_R16G16B16,
  mPF_Rf16Gf16Bf16,
  mPF_Rf32Gf32Bf32,

  mPF_R8G8B8A8,
  mPF_R16G16B16A16,
  mPF_Rf16Gf16Bf16Af16,
  mPF_Rf32Gf32Bf32Af32,

  mPF_B8G8R8,
  mPF_B8G8R8A8,

  mPF_YUV444,
  mPF_YUV422,
  mPF_YUV440,
  mPF_YUV420,
  mPF_YUV411,

  mPixelFormat_Count,
};

mFUNCTION(mPixelFormat_HasSubBuffers, const mPixelFormat pixelFormat, OUT bool *pValue);
mFUNCTION(mPixelFormat_IsChromaSubsampled, const mPixelFormat pixelFormat, OUT bool *pValue);
mFUNCTION(mPixelFormat_GetSize, const mPixelFormat pixelFormat, const mVec2s &size, OUT size_t *pBytes);
mFUNCTION(mPixelFormat_GetSize, const mPixelFormat pixelFormat, const mVec3s &size, OUT size_t *pBytes);
mFUNCTION(mPixelFormat_GetUnitSize, const mPixelFormat pixelFormat, OUT size_t *pBytes);
mFUNCTION(mPixelFormat_GetComponentCount, const mPixelFormat pixelFormat, OUT size_t *pCount);
mFUNCTION(mPixelFormat_GetComponentSize, const mPixelFormat pixelFormat, OUT size_t *pBytes);
mFUNCTION(mPixelFormat_IsComponentIntegral, const mPixelFormat pixelFormat, OUT bool *pIsIntegral);
mFUNCTION(mPixelFormat_GetSubBufferCount, const mPixelFormat pixelFormat, OUT size_t *pBufferCount);
mFUNCTION(mPixelFormat_GetSubBufferOffset, const mPixelFormat pixelFormat, const size_t bufferIndex, const mVec2s &size, OUT size_t *pOffset);
mFUNCTION(mPixelFormat_GetSubBufferSize, const mPixelFormat pixelFormat, const size_t bufferIndex, const mVec2s &size, OUT mVec2s *pSubBufferSize);
mFUNCTION(mPixelFormat_GetSubBufferPixelFormat, const mPixelFormat pixelFormat, const size_t bufferIndex, OUT mPixelFormat *pSubBufferPixelFormat);
mFUNCTION(mPixelFormat_GetSubBufferStride, const mPixelFormat pixelFormat, const size_t bufferIndex, const size_t originalLineStride, OUT size_t *pSubBufferLineStride);

mFUNCTION(mPixelFormat_TransformBuffer, mPtr<mImageBuffer> &source, mPtr<mImageBuffer> &target);
mFUNCTION(mPixelFormat_TransformBuffer, mPtr<mImageBuffer> &source, mPtr<mImageBuffer> &target, mPtr<mThreadPool> &asyncTaskHandler);

mFUNCTION(mPixelFormat_CanInplaceTransform, const mPixelFormat source, const mPixelFormat target, OUT bool *pCanInplaceTransform);

mFUNCTION(mPixelFormat_InplaceTransformBuffer, mPtr<mImageBuffer> &source, const mPixelFormat target);
mFUNCTION(mPixelFormat_InplaceTransformBuffer, mPtr<mImageBuffer> &source, const mPixelFormat target, mPtr<mThreadPool> &asyncTaskHandler);

struct mPixelFormatMapping
{
  mPixelFormat basePixelFormat;
  bool isSigned; // only relevant if the base pixel format is based on integers.
  bool isNormalized; // only relevant if the base pixel format is based on integers.

  mPixelFormatMapping() = default;
  inline mPixelFormatMapping(const mPixelFormat pixelFormat) : basePixelFormat(pixelFormat), isSigned(false), isNormalized(true) {}
  inline mPixelFormatMapping(const mPixelFormat pixelFormat, const bool _signed, const bool normalized) : basePixelFormat(pixelFormat), isSigned(_signed), isNormalized(normalized) {}
};

#endif // mPixelFormat_h__
