// Copyright 2018 Christoph Stiller
// 
// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files(the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions :
// 
// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
// 
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "mImageBuffer.h"

mFUNCTION(mImageBuffer_Destroy_Iternal, mImageBuffer *pImageBuffer);
mFUNCTION(mImageBuffer_Create_Iternal, mPtr<mImageBuffer> *pImageBuffer);

mFUNCTION(mImageBuffer_Create, OUT mPtr<mImageBuffer> *pImageBuffer)
{
  mFUNCTION_SETUP();

  mERROR_IF(pImageBuffer == nullptr, mR_ArgumentNull);
  mERROR_CHECK(mImageBuffer_Create_Iternal(pImageBuffer));

  mRETURN_SUCCESS();
}

mFUNCTION(mImageBuffer_Create, OUT mPtr<mImageBuffer> *pImageBuffer, const mVec2s &size, const mPixelFormat pixelFormat /* = mPF_B8G8R8A8 */)
{
  mFUNCTION_SETUP();

  mERROR_IF(pImageBuffer == nullptr, mR_ArgumentNull);
  mERROR_CHECK(mImageBuffer_Create_Iternal(pImageBuffer));
  mERROR_CHECK(mImageBuffer_AllocateBuffer(*pImageBuffer, size, pixelFormat));

  mRETURN_SUCCESS();
}

mFUNCTION(mImageBuffer_Create, OUT mPtr<mImageBuffer> *pImageBuffer, IN void *pData, const mVec2s &size, const mPixelFormat pixelFormat /* = mPF_B8G8R8A8 */)
{
  mFUNCTION_SETUP();

  mERROR_IF(pImageBuffer == nullptr, mR_ArgumentNull);
  mERROR_CHECK(mImageBuffer_Create_Iternal(pImageBuffer));
  mERROR_CHECK(mImageBuffer_SetBuffer(*pImageBuffer, pData, size, pixelFormat));

  mRETURN_SUCCESS();
}

mFUNCTION(mImageBuffer_Create, OUT mPtr<mImageBuffer> *pImageBuffer, IN void *pData, const mVec2s &size, const size_t stride, const mPixelFormat pixelFormat /* = mPF_B8G8R8A8 */)
{
  mFUNCTION_SETUP();

  mERROR_IF(pImageBuffer == nullptr, mR_ArgumentNull);
  mERROR_CHECK(mImageBuffer_Create_Iternal(pImageBuffer));
  mERROR_CHECK(mImageBuffer_SetBuffer(*pImageBuffer, pData, size, stride, pixelFormat));

  mRETURN_SUCCESS();
}

mFUNCTION(mImageBuffer_Create, OUT mPtr<mImageBuffer> *pImageBuffer, IN void *pData, const mVec2s &size, const mRectangle2D<size_t> &rect, const mPixelFormat pixelFormat /* = mPF_B8G8R8A8 */)
{
  mFUNCTION_SETUP();

  mERROR_IF(pImageBuffer == nullptr, mR_ArgumentNull);
  mERROR_CHECK(mImageBuffer_Create_Iternal(pImageBuffer));
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

  if (imageBuffer->pPixels != nullptr)
    mERROR_CHECK(mImageBuffer_Destroy_Iternal(imageBuffer.GetPointer()));

  size_t allocatedSize;
  mERROR_CHECK(mPixelFormat_GetSize(pixelFormat, size, &allocatedSize));

  uint8_t *pPixels = nullptr;
  mERROR_CHECK(mAlloc(&pPixels, allocatedSize));

  imageBuffer->ownedResource = true;
  imageBuffer->pPixels = pPixels;
  imageBuffer->allocatedSize = allocatedSize;
  imageBuffer->currentSize = size;
  imageBuffer->lineStride = size.x;

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

//////////////////////////////////////////////////////////////////////////

mFUNCTION(mImageBuffer_Destroy_Iternal, mImageBuffer *pImageBuffer)
{
  mFUNCTION_SETUP();

  mERROR_IF(pImageBuffer == nullptr, mR_ArgumentNull);

  if (pImageBuffer->ownedResource && pImageBuffer->pPixels != nullptr)
    mFreePtr(&pImageBuffer->pPixels);

  pImageBuffer->ownedResource = false;
  pImageBuffer->pPixels = nullptr;
  pImageBuffer->allocatedSize = 0;
  pImageBuffer->currentSize = 0;
  pImageBuffer->lineStride = 0;

  mRETURN_SUCCESS();
}

mFUNCTION(mImageBuffer_Create_Iternal, mPtr<mImageBuffer> *pImageBuffer)
{
  mFUNCTION_SETUP();

  mERROR_IF(pImageBuffer == nullptr, mR_ArgumentNull);

  if (pImageBuffer != nullptr)
  {
    mERROR_CHECK(mSharedPointer_Destroy(pImageBuffer));
    *pImageBuffer = nullptr;
  }

  mImageBuffer *pImageBufferRaw = nullptr;
  mDEFER_DESTRUCTION_ON_ERROR(&pImageBufferRaw, mFreePtr);
  mERROR_CHECK(mAllocZero(&pImageBufferRaw, 1));

  mDEFER_DESTRUCTION_ON_ERROR(pImageBuffer, mSharedPointer_Destroy);
  mERROR_CHECK(mSharedPointer_Create<mImageBuffer>(pImageBuffer, pImageBufferRaw, [](mImageBuffer *pData) { mImageBuffer_Destroy_Iternal(pData); }, mAT_mAlloc));
  pImageBufferRaw = nullptr; // to not be destroyed on error.

  mRETURN_SUCCESS();
}
