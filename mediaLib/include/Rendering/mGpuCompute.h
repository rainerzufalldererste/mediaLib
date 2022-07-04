#ifndef mGpuCompute_h__
#define mGpuCompute_h__

#include "mediaLib.h"
#include "mPixelFormat.h"

struct mGpuComputeContext;

mFUNCTION(mGpuComputeContext_Create, OUT mPtr<mGpuComputeContext> *pGpuComputeContext, IN mAllocator *pAllocator, const bool enableRendererSharing);

mFUNCTION(mGpuComputeContext_CompleteQueue, OUT mPtr<mGpuComputeContext> &gpuComputeContext);
mFUNCTION(mGpuComputeContext_FinalizeRenderer, OUT mPtr<mGpuComputeContext> &gpuComputeContext);

//////////////////////////////////////////////////////////////////////////

struct mGpuComputeEvent;

mFUNCTION(mGpuComputeEvent_Await, mPtr<mGpuComputeEvent> &_event);

//////////////////////////////////////////////////////////////////////////

enum mGpuComputeBuffer_ReadWriteConfiguration
{
  mGCB_RWC_Read = 1 << 0,
  mGCB_RWC_Write = 1 << 1,
  mGCB_RWC_ReadWrite = mGCB_RWC_Read | mGCB_RWC_Write,
};

struct mGpuComputeBuffer;

mFUNCTION(mGpuComputeBuffer_CreateDataBuffer, OUT mPtr<mGpuComputeBuffer> *pGpuBuffer, IN mAllocator *pAllocator, mPtr<mGpuComputeContext> &context, const mGpuComputeBuffer_ReadWriteConfiguration rwconfig, const size_t bytes, OPTIONAL IN const void *pDataToCopy = nullptr);
mFUNCTION(mGpuComputeBuffer_CreateTexture2D, OUT mPtr<mGpuComputeBuffer> *pGpuBuffer, IN mAllocator *pAllocator, mPtr<mGpuComputeContext> &context, const mGpuComputeBuffer_ReadWriteConfiguration rwconfig, const mVec2s resolution, const mPixelFormat pixelFormat, OPTIONAL const void *pDataToCopy = nullptr);
mFUNCTION(mGpuComputeBuffer_CreateTexture3D, OUT mPtr<mGpuComputeBuffer> *pGpuBuffer, IN mAllocator *pAllocator, mPtr<mGpuComputeContext> &context, const mGpuComputeBuffer_ReadWriteConfiguration rwconfig, const mVec3s resolution, const mPixelFormat pixelFormat, OPTIONAL const void *pDataToCopy = nullptr);
mFUNCTION(mGpuComputeBuffer_CreateTexture2DFromRendererTexture, OUT mPtr<mGpuComputeBuffer> *pGpuBuffer, IN mAllocator *pAllocator, mPtr<mGpuComputeContext> &context, const mGpuComputeBuffer_ReadWriteConfiguration rwconfig, mPtr<struct mTexture> &rendererTexture, const mPixelFormat pixelFormat);
mFUNCTION(mGpuComputeBuffer_CreateTexture3DFromRendererTexture, OUT mPtr<mGpuComputeBuffer> *pGpuBuffer, IN mAllocator *pAllocator, mPtr<mGpuComputeContext> &context, const mGpuComputeBuffer_ReadWriteConfiguration rwconfig, mPtr<struct mTexture3D> &rendererTexture, const mPixelFormat pixelFormat);

mFUNCTION(mGpuComputeBuffer_EnqueueAcquire, mPtr<mGpuComputeBuffer> &buffer, OUT OPTIONAL mPtr<mGpuComputeEvent> *pEvent = nullptr, IN OPTIONAL mAllocator *pAllocator = nullptr);
mFUNCTION(mGpuComputeBuffer_EnqueueRelease, mPtr<mGpuComputeBuffer> &buffer, OUT OPTIONAL mPtr<mGpuComputeEvent> *pEvent = nullptr, IN OPTIONAL mAllocator *pAllocator = nullptr);

mFUNCTION(mGpuComputeBuffer_EnqueueWriteToBuffer, mPtr<mGpuComputeBuffer> &buffer, const void *pData, const size_t size, const size_t writeOffset = 0, const bool blocking = false, OUT OPTIONAL mPtr<mGpuComputeEvent> *pEvent = nullptr, IN OPTIONAL mAllocator *pAllocator = nullptr);
mFUNCTION(mGpuComputeBuffer_EnqueueWriteToTexture2D, mPtr<mGpuComputeBuffer> &buffer, mPtr<mImageBuffer> &imageBuffer, const mVec2s writeOffset = mVec2s(0), const bool blocking = false, OUT OPTIONAL mPtr<mGpuComputeEvent> *pEvent = nullptr, IN OPTIONAL mAllocator *pAllocator = nullptr);
mFUNCTION(mGpuComputeBuffer_EnqueueWriteToTexture2D, mPtr<mGpuComputeBuffer> &buffer, const void *pData, const mPixelFormat pixelFormat, const mVec2s writeSize, const mVec2s writeOffset = mVec2s(0), const size_t lineStrideBytes = 0, const bool blocking = false, OUT OPTIONAL mPtr<mGpuComputeEvent> *pEvent = nullptr, IN OPTIONAL mAllocator *pAllocator = nullptr);
mFUNCTION(mGpuComputeBuffer_EnqueueWriteToTexture3D, mPtr<mGpuComputeBuffer> &buffer, const void *pData, const mPixelFormat pixelFormat, const mVec3s writeSize, const mVec3s writeOffset = mVec3s(0), const size_t lineStrideBytes = 0, const size_t rowStrideLines = 0, const bool blocking = false, OUT OPTIONAL mPtr<mGpuComputeEvent> *pEvent = nullptr, IN OPTIONAL mAllocator *pAllocator = nullptr);

mFUNCTION(mGpuComputeBuffer_EnqueueReadFromBuffer, mPtr<mGpuComputeBuffer> &buffer, OUT void *pData, const size_t size, const size_t readOffset = 0, const bool blocking = false, OUT OPTIONAL mPtr<mGpuComputeEvent> *pEvent = nullptr, IN OPTIONAL mAllocator *pAllocator = nullptr);
mFUNCTION(mGpuComputeBuffer_EnqueueReadFromTexture2D, mPtr<mGpuComputeBuffer> &buffer, OUT void *pData, const mPixelFormat pixelFormat, const mVec2s readSize, const mVec2s readOffset = mVec2s(0), const size_t lineStrideBytes = 0, const bool blocking = false, OUT OPTIONAL mPtr<mGpuComputeEvent> *pEvent = nullptr, IN OPTIONAL mAllocator *pAllocator = nullptr);
mFUNCTION(mGpuComputeBuffer_EnqueueReadFromTexture3D, mPtr<mGpuComputeBuffer> &buffer, OUT void *pData, const mPixelFormat pixelFormat, const mVec3s readSize, const mVec3s readOffset = mVec3s(0), const size_t lineStrideBytes = 0, const size_t rowStrideLines = 0, const bool blocking = false, OUT OPTIONAL mPtr<mGpuComputeEvent> *pEvent = nullptr, IN OPTIONAL mAllocator *pAllocator = nullptr);

mFUNCTION(mGpuComputeBuffer_EnqueueReadFromBuffer, mPtr<mGpuComputeBuffer> &buffer, OUT void **ppData, IN mAllocator *pDataAllocator, OPTIONAL OUT size_t *pBytes = nullptr, const size_t readOffset = 0, const bool blocking = false, OUT OPTIONAL mPtr<mGpuComputeEvent> *pEvent = nullptr, IN OPTIONAL mAllocator *pAllocator = nullptr);
mFUNCTION(mGpuComputeBuffer_EnqueueReadFromTexture2D, mPtr<mGpuComputeBuffer> &buffer, OUT void **ppData, IN mAllocator *pDataAllocator, OPTIONAL OUT size_t *pBytes = nullptr, const mVec2s readOffset = mVec2s(0), const bool blocking = false, OUT OPTIONAL mPtr<mGpuComputeEvent> *pEvent = nullptr, IN OPTIONAL mAllocator *pAllocator = nullptr);
mFUNCTION(mGpuComputeBuffer_EnqueueReadFromTexture3D, mPtr<mGpuComputeBuffer> &buffer, OUT void **ppData, IN mAllocator *pDataAllocator, OPTIONAL OUT size_t *pBytes = nullptr, const mVec3s readOffset = mVec3s(0), const bool blocking = false, OUT OPTIONAL mPtr<mGpuComputeEvent> *pEvent = nullptr, IN OPTIONAL mAllocator *pAllocator = nullptr);

mFUNCTION(mGpuComputeBuffer_GetPixelFormat, mPtr<mGpuComputeBuffer> &buffer, OUT mPixelFormat *pPixelFormat);

mFUNCTION(mGpuComputeBuffer_GetSize, mPtr<mGpuComputeBuffer> &buffer, OUT size_t *pSize);
mFUNCTION(mGpuComputeBuffer_GetSize, mPtr<mGpuComputeBuffer> &buffer, OUT mVec2s *pSize);
mFUNCTION(mGpuComputeBuffer_GetSize, mPtr<mGpuComputeBuffer> &buffer, OUT mVec3s *pSize);

//////////////////////////////////////////////////////////////////////////

struct mGpuComputeSampler;

mFUNCTION(mGpuComputeSampler_Create, OUT mPtr<mGpuComputeSampler> *pTextureSampler, IN mAllocator *pAllocator, mPtr<mGpuComputeContext> &context, const bool normalizedCoordinates, const enum mRenderParams_TextureWrapMode wrapMode, const enum mRenderParams_TextureMagnificationFilteringMode filterMode);

//////////////////////////////////////////////////////////////////////////

struct mGpuComputeKernel;

mFUNCTION(mGpuComputeKernel_Create, OUT mPtr<mGpuComputeKernel> *pComputeKernel, IN mAllocator *pAllocator, mPtr<mGpuComputeContext> &context, const char *kernelName, const char *source, const size_t sourceLength);

mFUNCTION(mGpuComputeKernel_SetArgAtIndex, mPtr<mGpuComputeKernel> &kernel, const uint32_t index, const uint32_t val);
mFUNCTION(mGpuComputeKernel_SetArgAtIndex, mPtr<mGpuComputeKernel> &kernel, const uint32_t index, const int32_t val);
mFUNCTION(mGpuComputeKernel_SetArgAtIndex, mPtr<mGpuComputeKernel> &kernel, const uint32_t index, const float_t val);
mFUNCTION(mGpuComputeKernel_SetArgAtIndex, mPtr<mGpuComputeKernel> &kernel, const uint32_t index, const mPtr<mGpuComputeBuffer> &buffer);
mFUNCTION(mGpuComputeKernel_SetArgAtIndex, mPtr<mGpuComputeKernel> &kernel, const uint32_t index, const mPtr<mGpuComputeSampler> &sampler);

template <uint32_t index, typename T>
inline mFUNCTION(_mGpuComputeKernel_CallSetArg, mPtr<mGpuComputeKernel> &kernel, T arg)
{
  return mGpuComputeKernel_SetArgAtIndex(kernel, index, arg);
}

template <uint32_t index, typename T, typename ... Args>
inline mFUNCTION(_mGpuComputeKernel_CallSetArg, mPtr<mGpuComputeKernel> &kernel, T arg, Args&& ...args)
{
  mFUNCTION_SETUP();

  mERROR_CHECK(mGpuComputeKernel_SetArgAtIndex(kernel, index, arg));

  return _mGpuComputeKernel_CallSetArg<index + 1>(kernel, args...);
}

template <typename ... Args>
inline mFUNCTION(mGpuComputeKernel_SetArgs, mPtr<mGpuComputeKernel> &kernel, Args&& ...args)
{
  mFUNCTION_SETUP();

  mERROR_IF(kernel == nullptr, mR_ArgumentNull);

  return _mGpuComputeKernel_CallSetArg<0>(kernel, args...);
}

inline mFUNCTION(mGpuComputeKernel_SetArgs, mPtr<mGpuComputeKernel> &kernel)
{
  mFUNCTION_SETUP();

  mERROR_IF(kernel == nullptr, mR_ArgumentNull);

  mRETURN_SUCCESS();
}

mFUNCTION(mGpuComputeKernel_EnqueueExecution, mPtr<mGpuComputeKernel> &kernel, const size_t globalWorkSize, const size_t localWorkSize, const size_t globalWorkOffset = 0, OUT OPTIONAL mPtr<mGpuComputeEvent> *pEvent = nullptr, IN OPTIONAL mAllocator *pAllocator = nullptr);
mFUNCTION(mGpuComputeKernel_EnqueueExecution, mPtr<mGpuComputeKernel> &kernel, const mVec2s globalWorkSize, const mVec2s localWorkSize, const mVec2s globalWorkOffset = mVec2s(0), OUT OPTIONAL mPtr<mGpuComputeEvent> *pEvent = nullptr, IN OPTIONAL mAllocator *pAllocator = nullptr);
mFUNCTION(mGpuComputeKernel_EnqueueExecution, mPtr<mGpuComputeKernel> &kernel, const mVec3s globalWorkSize, const mVec3s localWorkSize, const mVec3s globalWorkOffset = mVec3s(0), OUT OPTIONAL mPtr<mGpuComputeEvent> *pEvent = nullptr, IN OPTIONAL mAllocator *pAllocator = nullptr);

#endif // mGpuCompute_h__
