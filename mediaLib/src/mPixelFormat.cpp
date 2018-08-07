// Copyright 2018 Christoph Stiller
// 
// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files(the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions :
// 
// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
// 
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "mPixelFormat.h"
#include "mImageBuffer.h"

mFUNCTION(mPixelFormat_HasSubBuffers, const mPixelFormat pixelFormat, OUT bool *pValue)
{
  mFUNCTION_SETUP();

  mERROR_IF(pValue == nullptr, mR_ArgumentNull);

  switch (pixelFormat)
  {
  case mPF_B8G8R8:
  case mPF_B8G8R8A8:
  case mPF_Monochrome8:
  case mPF_Monochrome16:
    *pValue = false;
    break;

  case mPF_YUV422:
  case mPF_YUV411:
    *pValue = true;
    break;

  default:
    mRETURN_RESULT(mR_InvalidParameter);
  }

  mRETURN_SUCCESS();
}

mFUNCTION(mPixelFormat_IsChromaSubsampled, const mPixelFormat pixelFormat, OUT bool *pValue)
{
  mFUNCTION_SETUP();

  mERROR_IF(pValue == nullptr, mR_ArgumentNull);

  switch (pixelFormat)
  {
  case mPF_B8G8R8:
  case mPF_B8G8R8A8:
  case mPF_Monochrome8:
  case mPF_Monochrome16:
    *pValue = false;
    break;

  case mPF_YUV422:
  case mPF_YUV411:
    *pValue = true;
    break;

  default:
    mRETURN_RESULT(mR_InvalidParameter);
  }

  mRETURN_SUCCESS();
}

mFUNCTION(mPixelFormat_GetSize, const mPixelFormat pixelFormat, const mVec2s &size, OUT size_t *pBytes)
{
  mFUNCTION_SETUP();

  mERROR_IF(pBytes == nullptr, mR_ArgumentNull);

  switch (pixelFormat)
  {
  case mPF_B8G8R8:
    *pBytes = sizeof(uint8_t) * 3 * size.x * size.y;
    break;

  case mPF_B8G8R8A8:
    *pBytes = sizeof(uint32_t) * size.x * size.y;
    break;

  case mPF_YUV422:
    *pBytes = sizeof(uint8_t) * size.x * size.y * 2;
    break;

  case mPF_YUV411:
    *pBytes = sizeof(uint8_t) * size.x * size.y * 3 / 2;
    break;

  case mPF_Monochrome8:
    *pBytes = sizeof(uint8_t) * size.x * size.y;
    break;

  case mPF_Monochrome16:
    *pBytes = sizeof(uint16_t) * size.x * size.y;
    break;

  default:
    mRETURN_RESULT(mR_InvalidParameter);
  }

  mRETURN_SUCCESS();
}

mFUNCTION(mPixelFormat_GetUnitSize, const mPixelFormat pixelFormat, OUT size_t *pBytes)
{
  mFUNCTION_SETUP();

  mERROR_IF(pBytes == nullptr, mR_ArgumentNull);

  switch (pixelFormat)
  {
  case mPF_B8G8R8:
    *pBytes = sizeof(uint8_t) * 3;
    break;

  case mPF_B8G8R8A8:
    *pBytes = sizeof(uint32_t);
    break;

  case mPF_Monochrome8:
    *pBytes = sizeof(uint8_t);
    break;

  case mPF_Monochrome16:
    *pBytes = sizeof(uint16_t);
    break;

  default:
    mRETURN_RESULT(mR_InvalidParameter);
  }

  mRETURN_SUCCESS();
}

mFUNCTION(mPixelFormat_GetSubBufferCount, const mPixelFormat pixelFormat, OUT size_t *pBufferCount)
{
  mFUNCTION_SETUP();

  mERROR_IF(pBufferCount == nullptr, mR_ArgumentNull);

  switch (pixelFormat)
  {
  case mPF_B8G8R8:
  case mPF_B8G8R8A8:
  case mPF_Monochrome8:
  case mPF_Monochrome16:
    *pBufferCount = 1;
    break;

  case mPF_YUV422:
  case mPF_YUV411:
    *pBufferCount = 3;
    break;

  default:
    mRETURN_RESULT(mR_InvalidParameter);
  }

  mRETURN_SUCCESS();
}

mFUNCTION(mPixelFormat_GetSubBufferOffset, const mPixelFormat pixelFormat, const size_t bufferIndex, const mVec2s &size, OUT size_t *pOffset)
{
  mFUNCTION_SETUP();

  mERROR_IF(pOffset == nullptr, mR_ArgumentNull);

  size_t subBufferCount = 0;
  mERROR_CHECK(mPixelFormat_GetSubBufferCount(pixelFormat, &subBufferCount));
  mERROR_IF(subBufferCount <= bufferIndex, mR_IndexOutOfBounds);

  *pOffset = 0;

  if (bufferIndex > 0)
  {
    for (size_t i = 0; i <= bufferIndex; i++)
    {
      mPixelFormat subBufferPixelFormat;
      mERROR_CHECK(mPixelFormat_GetSubBufferPixelFormat(pixelFormat, bufferIndex, &subBufferPixelFormat));

      size_t subBufferUnitSize;
      mERROR_CHECK(mPixelFormat_GetUnitSize(subBufferPixelFormat, &subBufferUnitSize));

      mVec2s subBufferSize;
      mERROR_CHECK(mPixelFormat_GetSubBufferSize(pixelFormat, bufferIndex, size, &subBufferSize));

      *pOffset += subBufferSize.x * subBufferSize.y * subBufferUnitSize;
    }
  }

  mRETURN_SUCCESS();
}

mFUNCTION(mPixelFormat_GetSubBufferSize, const mPixelFormat pixelFormat, const size_t bufferIndex, const mVec2s &size, OUT mVec2s *pSubBufferSize)
{
  mFUNCTION_SETUP();

  mERROR_IF(pSubBufferSize == nullptr, mR_ArgumentNull);

  size_t subBufferCount = 0;
  mERROR_CHECK(mPixelFormat_GetSubBufferCount(pixelFormat, &subBufferCount));
  mERROR_IF(subBufferCount <= bufferIndex, mR_IndexOutOfBounds);

  switch (pixelFormat)
  {
  case mPF_YUV422:
    switch (bufferIndex)
    {
    case 0:
      *pSubBufferSize = size;
      break;

    case 1:
    case 2:
      *pSubBufferSize = mVec2s(size.x, size.y / 2);
      break;

    default:
      mRETURN_RESULT(mR_IndexOutOfBounds);
    }
    break;

  case mPF_YUV411:
    switch (bufferIndex)
    {
    case 0:
      *pSubBufferSize = size;
      break;

    case 1:
    case 2:
      *pSubBufferSize = mVec2s(size.x / 2, size.y / 2);
      break;

    default:
      mRETURN_RESULT(mR_IndexOutOfBounds);
    }
    break;

  case mPF_B8G8R8:
  case mPF_B8G8R8A8:
  case mPF_Monochrome8:
  case mPF_Monochrome16:
    *pSubBufferSize = size;
    break;

  default:
    mRETURN_RESULT(mR_InvalidParameter);
    break;
  }

  mRETURN_SUCCESS();
}

mFUNCTION(mPixelFormat_GetSubBufferPixelFormat, const mPixelFormat pixelFormat, const size_t bufferIndex, OUT mPixelFormat *pSubBufferPixelFormat)
{
  mFUNCTION_SETUP();

  mERROR_IF(pSubBufferPixelFormat == nullptr, mR_ArgumentNull);

  size_t subBufferCount = 0;
  mERROR_CHECK(mPixelFormat_GetSubBufferCount(pixelFormat, &subBufferCount));
  mERROR_IF(subBufferCount <= bufferIndex, mR_IndexOutOfBounds);

  switch (pixelFormat)
  {
  case mPF_YUV422:
  case mPF_YUV411:
    *pSubBufferPixelFormat = mPixelFormat::mPF_Monochrome8;
    break;

  case mPF_B8G8R8:
  case mPF_B8G8R8A8:
  case mPF_Monochrome8:
  case mPF_Monochrome16:
    *pSubBufferPixelFormat = pixelFormat;
    break;

  default:
    mRETURN_RESULT(mR_InvalidParameter);
    break;
  }

  mRETURN_SUCCESS();
}

mFUNCTION(mPixelFormat_GetSubBufferStride, const mPixelFormat pixelFormat, const size_t bufferIndex, const size_t originalLineStride, OUT size_t *pSubBufferLineStride)
{
  mFUNCTION_SETUP();

  mERROR_IF(pSubBufferLineStride == nullptr, mR_ArgumentNull);

  size_t subBufferCount = 0;
  mERROR_CHECK(mPixelFormat_GetSubBufferCount(pixelFormat, &subBufferCount));
  mERROR_IF(subBufferCount <= bufferIndex, mR_IndexOutOfBounds);

  switch (pixelFormat)
  {
  case mPF_YUV422:
  case mPF_YUV411:
    if (bufferIndex == 0)
      *pSubBufferLineStride = originalLineStride;
    else
      *pSubBufferLineStride = originalLineStride / 2;
    break;

  case mPF_B8G8R8:
  case mPF_B8G8R8A8:
  case mPF_Monochrome8:
  case mPF_Monochrome16:
    *pSubBufferLineStride = originalLineStride;
    break;

  default:
    mRETURN_RESULT(mR_InvalidParameter);
    break;
  }

  mRETURN_SUCCESS();
}

//////////////////////////////////////////////////////////////////////////



static inline void YUV444_to_RGB(const uint8_t y, const uint8_t u, const uint8_t v, OUT uint8_t *pR, OUT uint8_t *pG, OUT uint8_t *pB)
{
  const int32_t c = y - 16;
  const int32_t d = u - 128;
  const int32_t e = v - 128;

  const int32_t c_ = 298 * c;

  *pR = (uint8_t)mClamp((c_ + 409 * e + 128) >> 8, 0, 255);
  *pG = (uint8_t)mClamp((c_ - 100 * d - 208 * e + 128) >> 8, 0, 255);
  *pB = (uint8_t)mClamp((c_ + 516 * d + 128) >> 8, 0, 255);
}

static inline void RGB_to_YUV444(const uint8_t r, const uint8_t g, const uint8_t b, OUT uint8_t *pY, OUT uint8_t *pU, OUT uint8_t *pV)
{
  const int32_t b_ = b + 128;
  const int32_t y = ((66 * r + 129 * g + 25 * b_) >> 8) + 16;
  const int32_t u = ((-38 * r + 74 * g + 112 * b_) >> 8) + 128;
  const int32_t v = ((112 * r + 94 * g + 18 * b_) >> 8) + 128;

  *pY = (uint8_t)y;
  *pU = (uint8_t)u;
  *pV = (uint8_t)v;
}

static inline void YUV422_to_RGB(const uint8_t y0, const uint8_t y1, const uint8_t u, const uint8_t v, OUT uint8_t *pR0, OUT uint8_t *pG0, OUT uint8_t *pB0, OUT uint8_t *pR1, OUT uint8_t *pG1, OUT uint8_t *pB1)
{
  const int32_t c0 = y0 - 16;
  const int32_t c1 = y1 - 16;
  const int32_t d = u - 128;
  const int32_t e = v - 128;

  const int32_t c0_ = 298 * c0;
  const int32_t c1_ = 298 * c1;

  const int32_t v0 = 409 * e + 128;
  const int32_t v1 = -100 * d - 208 * e + 128;
  const int32_t v2 = 516 * d + 128;

  *pR0 = (uint8_t)mClamp((c0_ + v0) >> 8, 0, 255);
  *pR1 = (uint8_t)mClamp((c1_ + v0) >> 8, 0, 255);

  *pG0 = (uint8_t)mClamp((c0_ + v1) >> 8, 0, 255);
  *pG1 = (uint8_t)mClamp((c1_ + v1) >> 8, 0, 255);

  *pB0 = (uint8_t)mClamp((c0_ + v2) >> 8, 0, 255);
  *pB1 = (uint8_t)mClamp((c1_ + v2) >> 8, 0, 255);
}

static inline void YUV411_to_RGB(const uint8_t y0, const uint8_t y1, const uint8_t y2, const uint8_t y3, const uint8_t u, const uint8_t v, OUT uint8_t *pR0, OUT uint8_t *pG0, OUT uint8_t *pB0, OUT uint8_t *pR1, OUT uint8_t *pG1, OUT uint8_t *pB1, OUT uint8_t *pR2, OUT uint8_t *pG2, OUT uint8_t *pB2, OUT uint8_t *pR3, OUT uint8_t *pG3, OUT uint8_t *pB3)
{
  const int32_t c0 = y0 - 16;
  const int32_t c1 = y1 - 16;
  const int32_t c2 = y2 - 16;
  const int32_t c3 = y3 - 16;
  const int32_t d = u - 128;
  const int32_t e = v - 128;

  const int32_t c0_ = 298 * c0;
  const int32_t c1_ = 298 * c1;
  const int32_t c2_ = 298 * c2;
  const int32_t c3_ = 298 * c3;

  const int32_t v0 = 409 * e + 128;
  const int32_t v1 = -100 * d - 208 * e + 128;
  const int32_t v2 = 516 * d + 128;

  *pR0 = (uint8_t)mClamp((c0_ + v0) >> 8, 0, 255);
  *pR1 = (uint8_t)mClamp((c1_ + v0) >> 8, 0, 255);
  *pR2 = (uint8_t)mClamp((c2_ + v0) >> 8, 0, 255);
  *pR3 = (uint8_t)mClamp((c3_ + v0) >> 8, 0, 255);

  *pG0 = (uint8_t)mClamp((c0_ + v1) >> 8, 0, 255);
  *pG1 = (uint8_t)mClamp((c1_ + v1) >> 8, 0, 255);
  *pG2 = (uint8_t)mClamp((c2_ + v1) >> 8, 0, 255);
  *pG3 = (uint8_t)mClamp((c3_ + v1) >> 8, 0, 255);

  *pB0 = (uint8_t)mClamp((c0_ + v2) >> 8, 0, 255);
  *pB1 = (uint8_t)mClamp((c1_ + v2) >> 8, 0, 255);
  *pB2 = (uint8_t)mClamp((c2_ + v2) >> 8, 0, 255);
  *pB3 = (uint8_t)mClamp((c3_ + v2) >> 8, 0, 255);
}


static inline void YUV411_to_RGB(const uint8_t y0, const uint8_t y1, const uint8_t y2, const uint8_t y3, const uint8_t u, const uint8_t v, OUT uint32_t *pColor0, OUT uint32_t *pColor1, OUT uint32_t *pColor2, OUT uint32_t *pColor3)
{
  const int32_t c0 = y0 - 16;
  const int32_t c1 = y1 - 16;
  const int32_t c2 = y2 - 16;
  const int32_t c3 = y3 - 16;
  const int32_t d = u - 128;
  const int32_t e = v - 128;

  const int32_t c0_ = 298 * c0;
  const int32_t c1_ = 298 * c1;
  const int32_t c2_ = 298 * c2;
  const int32_t c3_ = 298 * c3;

  const int32_t v0 = 409 * e + 128;
  const int32_t v1 = -100 * d - 208 * e + 128;
  const int32_t v2 = 516 * d + 128;

  *pColor0 = ((uint32_t)mClamp((c0_ + v2) << 8, 0, 0xFF0000) & 0xFF0000) | ((uint32_t)mClamp((c0_ + v1), 0, 0xFF00) & 0xFF00) | (uint32_t)mClamp((c0_ + v0) >> 8, 0, 0xFF);
  *pColor1 = ((uint32_t)mClamp((c1_ + v2) << 8, 0, 0xFF0000) & 0xFF0000) | ((uint32_t)mClamp((c1_ + v1), 0, 0xFF00) & 0xFF00) | (uint32_t)mClamp((c1_ + v0) >> 8, 0, 0xFF);
  *pColor2 = ((uint32_t)mClamp((c2_ + v2) << 8, 0, 0xFF0000) & 0xFF0000) | ((uint32_t)mClamp((c2_ + v1), 0, 0xFF00) & 0xFF00) | (uint32_t)mClamp((c2_ + v0) >> 8, 0, 0xFF);
  *pColor3 = ((uint32_t)mClamp((c3_ + v2) << 8, 0, 0xFF0000) & 0xFF0000) | ((uint32_t)mClamp((c3_ + v1), 0, 0xFF00) & 0xFF00) | (uint32_t)mClamp((c3_ + v0) >> 8, 0, 0xFF);
}

static inline void YUV420_to_RGB(const uint8_t y0, const uint8_t y1, const uint8_t y2, const uint8_t y3, const uint8_t u, const uint8_t v, OUT uint8_t *pR0, OUT uint8_t *pG0, OUT uint8_t *pB0, OUT uint8_t *pR1, OUT uint8_t *pG1, OUT uint8_t *pB1, OUT uint8_t *pR2, OUT uint8_t *pG2, OUT uint8_t *pB2, OUT uint8_t *pR3, OUT uint8_t *pG3, OUT uint8_t *pB3)
{
  YUV411_to_RGB(y0, y1, y2, y3, u, v, pR0, pG0, pB0, pR1, pG1, pB1, pR2, pG2, pB2, pR3, pG3, pB3);
}

//////////////////////////////////////////////////////////////////////////

mFUNCTION(mPixelFormat_TransformBuffer, mPtr<mImageBuffer> &source, mPtr<mImageBuffer> &target)
{
  mFUNCTION_SETUP();

  mERROR_IF(source->pPixels == nullptr || target->pPixels == nullptr, mR_NotInitialized);

  if (source->pixelFormat == target->pixelFormat)
    mERROR_CHECK(mImageBuffer_CopyTo(source, target, mImageBuffer_CopyFlags::mIB_CF_None));

  mERROR_IF(source->currentSize != target->currentSize, mR_InvalidParameter);

  switch (target->pixelFormat)
  {
  case mPF_B8G8R8A8:
  {
    uint32_t *pOutPixels = (uint32_t *)target->pPixels;

    switch (source->pixelFormat)
    {
    case mPF_YUV411:
    {
      uint8_t *pBuffer0 = source->pPixels;
      uint8_t *pBuffer1 = source->pPixels;
      uint8_t *pBuffer2 = source->pPixels;

      size_t offset;
      mERROR_CHECK(mPixelFormat_GetSubBufferOffset(source->pixelFormat, 0, mVec2s(source->lineStride, source->currentSize.y), &offset));
      pBuffer0 += offset;

      mERROR_CHECK(mPixelFormat_GetSubBufferOffset(source->pixelFormat, 1, mVec2s(source->lineStride, source->currentSize.y), &offset));
      pBuffer1 += offset;

      mERROR_CHECK(mPixelFormat_GetSubBufferOffset(source->pixelFormat, 2, mVec2s(source->lineStride, source->currentSize.y), &offset));
      pBuffer2 += offset;

      size_t sourceLineStride0, sourceLineStride1, sourceLineStride2;
      mERROR_CHECK(mPixelFormat_GetSubBufferStride(source->pixelFormat, 0, source->lineStride, &sourceLineStride0));
      mERROR_CHECK(mPixelFormat_GetSubBufferStride(source->pixelFormat, 1, source->lineStride, &sourceLineStride1));
      mERROR_CHECK(mPixelFormat_GetSubBufferStride(source->pixelFormat, 2, source->lineStride, &sourceLineStride2));

      for (size_t y = 0; y < target->currentSize.y - 1; y += 2)
      {
        const size_t yline = y * target->lineStride;
        const size_t yhalf = y >> 1;
        const size_t ySourceLine0 = y * sourceLineStride0;
        const size_t ySourceLine1 = y * sourceLineStride1;

        for (size_t x = 0; x < target->currentSize.x - 1; x += 2)
        {
          size_t xhalf = x >> 1;
          size_t xySourceLine0 = x + ySourceLine0;
          size_t xyTargetLine0 = x + yline;
          size_t sourcePosSubRes = xhalf + ySourceLine1;

          YUV411_to_RGB(
            pBuffer0[xySourceLine0], pBuffer0[xySourceLine0 + 1], 
            pBuffer0[xySourceLine0 + sourceLineStride0], pBuffer0[xySourceLine0 + sourceLineStride0 + 1],

            pBuffer1[sourcePosSubRes],
            
            pBuffer2[sourcePosSubRes],
            
            &pOutPixels[xyTargetLine0], &pOutPixels[xyTargetLine0 + 1], 
            &pOutPixels[xyTargetLine0 + target->lineStride], &pOutPixels[xyTargetLine0 + target->lineStride + 1]);
        }
      }

      break;
    }

    default:
    {
      mRETURN_RESULT(mR_InvalidParameter);
    }
    }

    break;
  }

  case mPF_B8G8R8:
  case mPF_Monochrome8:
  case mPF_Monochrome16:
  case mPF_YUV422:
  case mPF_YUV411:
  {
    mRETURN_RESULT(mR_NotImplemented);
  }

  default:
  {
    mRETURN_RESULT(mR_InvalidParameter);
  }
  }

  mRETURN_SUCCESS();
}
