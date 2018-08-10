// Copyright 2018 Christoph Stiller
// 
// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files(the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions :
// 
// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
// 
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "mImageBuffer.h"

mFUNCTION(mImageBuffer_Create_Iternal, mPtr<mImageBuffer> *pImageBuffer, IN OPTIONAL mAllocator *pAllocator);
mFUNCTION(mImageBuffer_Destroy_Iternal, mImageBuffer *pImageBuffer);

mFUNCTION(mImageBuffer_Create, OUT mPtr<mImageBuffer> *pImageBuffer, IN OPTIONAL mAllocator *pAllocator)
{
  mFUNCTION_SETUP();

  mERROR_IF(pImageBuffer == nullptr, mR_ArgumentNull);
  mERROR_CHECK(mImageBuffer_Create_Iternal(pImageBuffer, pAllocator));

  mRETURN_SUCCESS();
}

mFUNCTION(mImageBuffer_Create, OUT mPtr<mImageBuffer> *pImageBuffer, IN OPTIONAL mAllocator *pAllocator, const mVec2s &size, const mPixelFormat pixelFormat /* = mPF_B8G8R8A8 */)
{
  mFUNCTION_SETUP();

  mERROR_IF(pImageBuffer == nullptr, mR_ArgumentNull);
  mERROR_CHECK(mImageBuffer_Create_Iternal(pImageBuffer, pAllocator));
  mERROR_CHECK(mImageBuffer_AllocateBuffer(*pImageBuffer, size, pixelFormat));

  mRETURN_SUCCESS();
}

mFUNCTION(mImageBuffer_Create, OUT mPtr<mImageBuffer> *pImageBuffer, IN OPTIONAL mAllocator *pAllocator, IN void *pData, const mVec2s &size, const mPixelFormat pixelFormat /* = mPF_B8G8R8A8 */)
{
  mFUNCTION_SETUP();

  mERROR_IF(pImageBuffer == nullptr, mR_ArgumentNull);
  mERROR_CHECK(mImageBuffer_Create_Iternal(pImageBuffer, pAllocator));
  mERROR_CHECK(mImageBuffer_SetBuffer(*pImageBuffer, pData, size, pixelFormat));

  mRETURN_SUCCESS();
}

mFUNCTION(mImageBuffer_Create, OUT mPtr<mImageBuffer> *pImageBuffer, IN OPTIONAL mAllocator *pAllocator, IN void *pData, const mVec2s &size, const size_t stride, const mPixelFormat pixelFormat /* = mPF_B8G8R8A8 */)
{
  mFUNCTION_SETUP();

  mERROR_IF(pImageBuffer == nullptr, mR_ArgumentNull);
  mERROR_CHECK(mImageBuffer_Create_Iternal(pImageBuffer, pAllocator));
  mERROR_CHECK(mImageBuffer_SetBuffer(*pImageBuffer, pData, size, stride, pixelFormat));

  mRETURN_SUCCESS();
}

mFUNCTION(mImageBuffer_Create, OUT mPtr<mImageBuffer> *pImageBuffer, IN OPTIONAL mAllocator *pAllocator, IN void *pData, const mVec2s &size, const mRectangle2D<size_t> &rect, const mPixelFormat pixelFormat /* = mPF_B8G8R8A8 */)
{
  mFUNCTION_SETUP();

  mERROR_IF(pImageBuffer == nullptr, mR_ArgumentNull);
  mERROR_CHECK(mImageBuffer_Create_Iternal(pImageBuffer, pAllocator));
  mERROR_CHECK(mImageBuffer_SetBuffer(*pImageBuffer, pData, size, rect, pixelFormat));

  mRETURN_SUCCESS();
}

mFUNCTION(mImageBuffer_Destroy, OUT mPtr<mImageBuffer> *pImageBuffer)
{
  mFUNCTION_SETUP();

  mERROR_IF(pImageBuffer == nullptr, mR_ArgumentNull);
  mERROR_CHECK(mSharedPointer_Destroy(pImageBuffer));

  mRETURN_SUCCESS();
}

mFUNCTION(mImageBuffer_AllocateBuffer, mPtr<mImageBuffer> &imageBuffer, const mVec2s &size, const mPixelFormat pixelFormat /* = mPF_B8G8R8A8 */)
{
  mFUNCTION_SETUP();

  if (!imageBuffer->ownedResource && imageBuffer->pPixels != nullptr)
    mERROR_CHECK(mImageBuffer_Destroy_Iternal(imageBuffer.GetPointer()));

  size_t allocatedSize;
  mERROR_CHECK(mPixelFormat_GetSize(pixelFormat, size, &allocatedSize));

  if (imageBuffer->pPixels == nullptr)
  {
    uint8_t *pPixels = nullptr;
    mERROR_CHECK(mAllocator_Allocate(imageBuffer->pAllocator, &pPixels, allocatedSize));

    imageBuffer->ownedResource = true;
    imageBuffer->pPixels = pPixels;
    imageBuffer->allocatedSize = allocatedSize;
    imageBuffer->currentSize = size;
    imageBuffer->lineStride = size.x;
    imageBuffer->pixelFormat = pixelFormat;
  }
  else
  {
    if (imageBuffer->allocatedSize < allocatedSize)
    {
      uint8_t *pPixels = nullptr;
      mERROR_CHECK(mAllocator_Reallocate(imageBuffer->pAllocator, &pPixels, allocatedSize));

      imageBuffer->allocatedSize = allocatedSize;
      imageBuffer->ownedResource = true;
      imageBuffer->pPixels = pPixels;
    }

    imageBuffer->currentSize = size;
    imageBuffer->pixelFormat = pixelFormat;
    imageBuffer->lineStride = size.x;
  }

  mRETURN_SUCCESS();
}

mFUNCTION(mImageBuffer_SetBuffer, mPtr<mImageBuffer> &imageBuffer, IN void *pData, const mVec2s &size, const mPixelFormat pixelFormat /* = mPF_B8G8R8A8 */)
{
  mFUNCTION_SETUP();

  mERROR_CHECK(mImageBuffer_SetBuffer(imageBuffer, pData, size, size.x, pixelFormat));

  mRETURN_SUCCESS();
}

mFUNCTION(mImageBuffer_SetBuffer, mPtr<mImageBuffer> &imageBuffer, IN void *pData, const mVec2s &size, const size_t stride, const mPixelFormat pixelFormat /* = mPF_B8G8R8A8 */)
{
  mFUNCTION_SETUP();

  if (imageBuffer->pPixels != nullptr)
    mERROR_CHECK(mImageBuffer_Destroy_Iternal(imageBuffer.GetPointer()));

  mERROR_CHECK(mPixelFormat_GetSize(pixelFormat, size, &imageBuffer->allocatedSize));

  imageBuffer->ownedResource = false;
  imageBuffer->pPixels = (uint8_t *)pData;
  imageBuffer->pixelFormat = pixelFormat;
  imageBuffer->currentSize = size;
  imageBuffer->lineStride = stride;

  mRETURN_SUCCESS();
}

mFUNCTION(mImageBuffer_SetBuffer, mPtr<mImageBuffer> &imageBuffer, IN void *pData, const mVec2s &size, const mRectangle2D<size_t> &rect, const mPixelFormat pixelFormat)
{
  mFUNCTION_SETUP();

  if (imageBuffer->pPixels != nullptr)
    mERROR_CHECK(mImageBuffer_Destroy_Iternal(imageBuffer.GetPointer()));

  mERROR_IF(rect.x + rect.w >= size.x, mR_InvalidParameter);
  mERROR_IF(rect.y + rect.h >= size.y, mR_InvalidParameter);

  bool hasSubBuffers;
  mERROR_CHECK(mPixelFormat_HasSubBuffers(pixelFormat, &hasSubBuffers));
  mERROR_IF(hasSubBuffers, mR_InvalidParameter);

  size_t pixelFormatUnitSize;
  mERROR_CHECK(mPixelFormat_GetUnitSize(pixelFormat, &pixelFormatUnitSize));

  imageBuffer->ownedResource = false;
  imageBuffer->pPixels = ((uint8_t *)pData) + (size.x * rect.y + rect.x) * pixelFormatUnitSize;
  imageBuffer->lineStride = size.x;
  imageBuffer->currentSize = mVec2s(rect.w, rect.h);
  imageBuffer->allocatedSize = (size.x * size.y - rect.x - rect.y * size.x) * pixelFormatUnitSize;
  imageBuffer->pixelFormat = pixelFormat;

  mRETURN_SUCCESS();
}

mFUNCTION(mImageBuffer_CopyTo, mPtr<mImageBuffer> &source, mPtr<mImageBuffer> &target, const mImageBuffer_CopyFlags copyFlags)
{
  mFUNCTION_SETUP();

  mERROR_IF(source->pPixels == nullptr, mR_NotInitialized);
  
  if (target->pPixels == nullptr)
    mERROR_CHECK(mImageBuffer_AllocateBuffer(target, source->currentSize, source->pixelFormat));

  if ((copyFlags & mImageBuffer_CopyFlags::mIB_CF_ResizeAllowed) && target->ownedResource)
  {
    if (source->currentSize != target->currentSize)
    {
      mERROR_IF(source->pixelFormat != target->pixelFormat && (copyFlags & mImageBuffer_CopyFlags::mIB_CF_PixelFormatChangeAllowed) == 0, mR_InvalidParameter);
      
      mERROR_CHECK(mImageBuffer_AllocateBuffer(target, source->currentSize, source->pixelFormat));
    }
  }
  else
  {
    mERROR_IF(source->currentSize != target->currentSize, mR_InvalidParameter);
  }

  if ((copyFlags & mImageBuffer_CopyFlags::mIB_CF_PixelFormatChangeAllowed) && target->ownedResource)
  {
    mERROR_IF(source->currentSize != target->currentSize && (copyFlags & mImageBuffer_CopyFlags::mIB_CF_ResizeAllowed) == 0, mR_InvalidParameter);

    mERROR_CHECK(mImageBuffer_AllocateBuffer(target, source->currentSize, source->pixelFormat));
  }
  else
  {
    mERROR_IF(source->pixelFormat != target->pixelFormat, mR_InvalidParameter);
  }

  if (source->lineStride == target->lineStride)
  {
    size_t copiedSize;
    mERROR_CHECK(mPixelFormat_GetSize(source->pixelFormat, mVec2s(source->lineStride, source->currentSize.y), &copiedSize));
    mERROR_CHECK(mAllocator_Copy(source->pAllocator, target->pPixels, source->pPixels, copiedSize));
  }
  else
  {
    size_t subBufferCount;
    mERROR_CHECK(mPixelFormat_GetSubBufferCount(source->pixelFormat, &subBufferCount));

    for (size_t i = 0; i < subBufferCount; i++)
    {
      size_t sourceSubBufferOffset;
      mERROR_CHECK(mPixelFormat_GetSubBufferOffset(source->pixelFormat, i, mVec2s(source->lineStride, source->currentSize.y), &sourceSubBufferOffset));

      size_t targetSubBufferOffset;
      mERROR_CHECK(mPixelFormat_GetSubBufferOffset(target->pixelFormat, i, mVec2s(target->lineStride, target->currentSize.y), &targetSubBufferOffset));

      mVec2s subBufferSize;
      mERROR_CHECK(mPixelFormat_GetSubBufferSize(source->pixelFormat, i, source->currentSize, &subBufferSize));

      size_t sourceSubBufferStride;
      mERROR_CHECK(mPixelFormat_GetSubBufferStride(source->pixelFormat, i, source->lineStride, &sourceSubBufferStride));

      size_t targetSubBufferStride;
      mERROR_CHECK(mPixelFormat_GetSubBufferStride(target->pixelFormat, i, target->lineStride, &targetSubBufferStride));

      mPixelFormat subBufferPixelFormat;
      mERROR_CHECK(mPixelFormat_GetSubBufferPixelFormat(source->pixelFormat, i, &subBufferPixelFormat));

      size_t subBufferUnitSize;
      mERROR_CHECK(mPixelFormat_GetUnitSize(subBufferPixelFormat, &subBufferUnitSize));

      uint8_t *pSourceSubBufferPixels = source->pPixels + sourceSubBufferOffset;
      uint8_t *pTargetSubBufferPixels = target->pPixels + targetSubBufferOffset;

      size_t targetLineStride = 0;
      size_t sourceLineStride = 0;

      for (size_t y = 0; y < source->currentSize.y; y++)
      {
        mERROR_CHECK(mAllocator_Copy(source->pAllocator, pTargetSubBufferPixels + targetLineStride, pSourceSubBufferPixels + sourceLineStride, subBufferSize.x));

        targetLineStride += sourceSubBufferStride * subBufferUnitSize;
        sourceLineStride += targetSubBufferStride * subBufferUnitSize;
      }
    }
  }

  mRETURN_SUCCESS();
}

//////////////////////////////////////////////////////////////////////////

mFUNCTION(mImageBuffer_Destroy_Iternal, mImageBuffer *pImageBuffer)
{
  mFUNCTION_SETUP();

  mERROR_IF(pImageBuffer == nullptr, mR_ArgumentNull);

  if (pImageBuffer->ownedResource && pImageBuffer->pPixels != nullptr)
    mSTDRESULT = mAllocator_FreePtr(pImageBuffer->pAllocator, &pImageBuffer->pPixels);

  pImageBuffer->ownedResource = false;
  pImageBuffer->pPixels = nullptr;
  pImageBuffer->allocatedSize = 0;
  pImageBuffer->currentSize = 0;
  pImageBuffer->lineStride = 0;

  mERROR_IF(mFAILED(mSTDRESULT), mSTDRESULT);

  mRETURN_SUCCESS();
}

mFUNCTION(mImageBuffer_Create_Iternal, mPtr<mImageBuffer> *pImageBuffer, IN OPTIONAL mAllocator *pAllocator)
{
  mFUNCTION_SETUP();

  mERROR_IF(pImageBuffer == nullptr, mR_ArgumentNull);

  if (pImageBuffer != nullptr)
  {
    mERROR_CHECK(mSharedPointer_Destroy(pImageBuffer));
    *pImageBuffer = nullptr;
  }

  mImageBuffer *pImageBufferRaw = nullptr;
  mDEFER_ON_ERROR(mAllocator_FreePtr(pAllocator, &pImageBufferRaw));
  mERROR_CHECK(mAllocator_AllocateZero(pAllocator, &pImageBufferRaw, 1));

  mDEFER_DESTRUCTION_ON_ERROR(pImageBuffer, mSharedPointer_Destroy);
  mERROR_CHECK(mSharedPointer_Create<mImageBuffer>(pImageBuffer, pImageBufferRaw, [](mImageBuffer *pData) { mImageBuffer_Destroy_Iternal(pData); }, pAllocator));
  pImageBufferRaw = nullptr; // to not be destroyed on error.

  (*pImageBuffer)->pAllocator = pAllocator;

  mRETURN_SUCCESS();
}
