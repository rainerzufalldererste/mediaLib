#include "mPixelFormat.h"
#include "mImageBuffer.h"

#ifdef GIT_BUILD // Define __M_FILE__
  #ifdef __M_FILE__
    #undef __M_FILE__
  #endif
  #define __M_FILE__ "WxZHdolxaOsVMVztMcLWewBh6hSpEpuquqmpB8m6HcA9zdae0FuZIHcxgKjRdXKBj5HQQfLkazzf6tq9"
#endif

mFUNCTION(mPixelFormat_HasSubBuffers, const mPixelFormat pixelFormat, OUT bool *pValue)
{
  mFUNCTION_SETUP();

  mERROR_IF(pValue == nullptr, mR_ArgumentNull);

  switch (pixelFormat)
  {
  case mPF_Monochrome8:
  case mPF_Monochrome16:
  case mPF_Monochromef16:
  case mPF_Monochromef32:
  case mPF_R8G8:
  case mPF_R16G16:
  case mPF_Rf16Gf16:
  case mPF_Rf32Gf32:
  case mPF_R8G8B8:
  case mPF_R16G16B16:
  case mPF_Rf16Gf16Bf16:
  case mPF_Rf32Gf32Bf32:
  case mPF_R8G8B8A8:
  case mPF_R16G16B16A16:
  case mPF_Rf16Gf16Bf16Af16:
  case mPF_Rf32Gf32Bf32Af32:
  case mPF_B8G8R8:
  case mPF_B8G8R8A8:
    *pValue = false;
    break;

  case mPF_YUV444:
  case mPF_YUV422:
  case mPF_YUV440:
  case mPF_YUV420:
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
  case mPF_Monochrome8:
  case mPF_Monochrome16:
  case mPF_Monochromef16:
  case mPF_Monochromef32:
  case mPF_R8G8:
  case mPF_R16G16:
  case mPF_Rf16Gf16:
  case mPF_Rf32Gf32:
  case mPF_R8G8B8:
  case mPF_R16G16B16:
  case mPF_Rf16Gf16Bf16:
  case mPF_Rf32Gf32Bf32:
  case mPF_R8G8B8A8:
  case mPF_R16G16B16A16:
  case mPF_Rf16Gf16Bf16Af16:
  case mPF_Rf32Gf32Bf32Af32:
  case mPF_B8G8R8:
  case mPF_B8G8R8A8:
    *pValue = false;
    break;

  case mPF_YUV444:
  case mPF_YUV422:
  case mPF_YUV440:
  case mPF_YUV420:
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
  case mPF_Monochrome8:
    *pBytes = sizeof(uint8_t) * size.x * size.y;
    break;

  case mPF_Monochrome16:
  case mPF_Monochromef16:
    *pBytes = sizeof(uint16_t) * size.x * size.y;
    break;

  case mPF_Monochromef32:
    *pBytes = sizeof(float_t) * size.x * size.y;
    break;

  case mPF_R8G8:
    *pBytes = sizeof(uint8_t) * 2 * size.x * size.y;
    break;

  case mPF_R16G16:
  case mPF_Rf16Gf16:
    *pBytes = sizeof(uint16_t) * 2 * size.x * size.y;
    break;

  case mPF_Rf32Gf32:
    *pBytes = sizeof(float_t) * 2 * size.x * size.y;
    break;

  case mPF_R8G8B8:
  case mPF_B8G8R8:
    *pBytes = sizeof(uint8_t) * 3 * size.x * size.y;
    break;

  case mPF_R16G16B16:
  case mPF_Rf16Gf16Bf16:
    *pBytes = sizeof(uint16_t) * 3 * size.x * size.y;
    break;

  case mPF_Rf32Gf32Bf32:
    *pBytes = sizeof(float_t) * 3 * size.x * size.y;
    break;

  case mPF_R8G8B8A8:
  case mPF_B8G8R8A8:
    *pBytes = sizeof(uint8_t) * 4 * size.x * size.y;
    break;

  case mPF_R16G16B16A16:
  case mPF_Rf16Gf16Bf16Af16:
    *pBytes = sizeof(uint16_t) * 4 * size.x * size.y;
    break;

  case mPF_Rf32Gf32Bf32Af32:
    *pBytes = sizeof(float_t) * 4 * size.x * size.y;
    break;

  case mPF_YUV444:
    *pBytes = sizeof(uint8_t) * size.x * size.y * 3;
    break;

  case mPF_YUV422:
    *pBytes = sizeof(uint8_t) * size.x * size.y * 2;
    break;

  case mPF_YUV440:
    *pBytes = sizeof(uint8_t) * size.x * size.y * 2;
    break;

  case mPF_YUV420:
    *pBytes = sizeof(uint8_t) * size.x * size.y * 3 / 2;
    break;

  case mPF_YUV411:
    *pBytes = sizeof(uint8_t) * size.x * size.y * 3 / 2;
    break;

  default:
    mRETURN_RESULT(mR_InvalidParameter);
  }

  mRETURN_SUCCESS();
}

mFUNCTION(mPixelFormat_GetComponentCount, const mPixelFormat pixelFormat, OUT size_t *pCount)
{
  mFUNCTION_SETUP();

  mERROR_IF(pCount == nullptr, mR_ArgumentNull);

  switch (pixelFormat)
  {
  case mPF_Monochrome8:
  case mPF_Monochrome16:
  case mPF_Monochromef16:
  case mPF_Monochromef32:
    *pCount = 1;
    break;

  case mPF_R8G8:
  case mPF_R16G16:
  case mPF_Rf16Gf16:
  case mPF_Rf32Gf32:
    *pCount = 2;
    break;

  case mPF_R8G8B8:
  case mPF_R16G16B16:
  case mPF_Rf16Gf16Bf16:
  case mPF_Rf32Gf32Bf32:
  case mPF_B8G8R8:
  case mPF_B8G8R8A8:
    *pCount = 3;
    break;

  case mPF_R8G8B8A8:
  case mPF_R16G16B16A16:
  case mPF_Rf16Gf16Bf16Af16:
  case mPF_Rf32Gf32Bf32Af32:
    *pCount = 4;
    break;

  default:
    mRETURN_RESULT(mR_InvalidParameter);
  }

  mRETURN_SUCCESS();
}

mFUNCTION(mPixelFormat_GetComponentSize, const mPixelFormat pixelFormat, OUT size_t *pBytes)
{
  mFUNCTION_SETUP();

  mERROR_IF(pBytes == nullptr, mR_ArgumentNull);

  switch (pixelFormat)
  {
  case mPF_Monochrome8:
  case mPF_R8G8:
  case mPF_R8G8B8:
  case mPF_R8G8B8A8:
  case mPF_B8G8R8:
  case mPF_B8G8R8A8:
    *pBytes = sizeof(uint8_t);
    break;

  case mPF_Monochrome16:
  case mPF_Monochromef16:
  case mPF_R16G16:
  case mPF_Rf16Gf16:
  case mPF_R16G16B16:
  case mPF_Rf16Gf16Bf16:
  case mPF_R16G16B16A16:
  case mPF_Rf16Gf16Bf16Af16:
    *pBytes = sizeof(uint16_t);
    break;

  case mPF_Monochromef32:
  case mPF_Rf32Gf32:
  case mPF_Rf32Gf32Bf32:
  case mPF_Rf32Gf32Bf32Af32:
    *pBytes = sizeof(float_t);
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

  size_t componentCount = 0;
  mERROR_CHECK(mPixelFormat_GetComponentCount(pixelFormat, &componentCount));

  size_t componentSize = 0;
  mERROR_CHECK(mPixelFormat_GetComponentSize(pixelFormat, &componentSize));

  *pBytes = componentCount * componentSize;

  mRETURN_SUCCESS();
}

mFUNCTION(mPixelFormat_IsComponentIntegral, const mPixelFormat pixelFormat, OUT bool *pIsIntegral)
{
  mFUNCTION_SETUP();

  mERROR_IF(pIsIntegral == nullptr, mR_ArgumentNull);

  switch (pixelFormat)
  {
  case mPF_Monochrome8:
  case mPF_Monochrome16:
  case mPF_R8G8:
  case mPF_R8G8B8:
  case mPF_R8G8B8A8:
  case mPF_B8G8R8:
  case mPF_B8G8R8A8:
  case mPF_R16G16:
  case mPF_R16G16B16:
  case mPF_R16G16B16A16:
    *pIsIntegral = true;
    break;

  case mPF_Monochromef16:
  case mPF_Monochromef32:
  case mPF_Rf16Gf16:
  case mPF_Rf32Gf32:
  case mPF_Rf16Gf16Bf16:
  case mPF_Rf32Gf32Bf32:
  case mPF_Rf16Gf16Bf16Af16:
  case mPF_Rf32Gf32Bf32Af32:
    *pIsIntegral = false;
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
  case mPF_Monochrome8:
  case mPF_Monochrome16:
  case mPF_Monochromef16:
  case mPF_Monochromef32:
  case mPF_R8G8:
  case mPF_R16G16:
  case mPF_Rf16Gf16:
  case mPF_Rf32Gf32:
  case mPF_R8G8B8:
  case mPF_R16G16B16:
  case mPF_Rf16Gf16Bf16:
  case mPF_Rf32Gf32Bf32:
  case mPF_R8G8B8A8:
  case mPF_R16G16B16A16:
  case mPF_Rf16Gf16Bf16Af16:
  case mPF_Rf32Gf32Bf32Af32:
  case mPF_B8G8R8:
  case mPF_B8G8R8A8:
    *pBufferCount = 1;
    break;

  case mPF_YUV444:
  case mPF_YUV422:
  case mPF_YUV440:
  case mPF_YUV420:
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
    for (size_t i = 0; i < bufferIndex; i++)
    {
      mPixelFormat subBufferPixelFormat;
      mERROR_CHECK(mPixelFormat_GetSubBufferPixelFormat(pixelFormat, i, &subBufferPixelFormat));

      size_t subBufferUnitSize;
      mERROR_CHECK(mPixelFormat_GetUnitSize(subBufferPixelFormat, &subBufferUnitSize));

      mVec2s subBufferSize;
      mERROR_CHECK(mPixelFormat_GetSubBufferSize(pixelFormat, i, size, &subBufferSize));

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
  case mPF_YUV444:
    switch (bufferIndex)
    {
    case 0:
    case 1:
    case 2:
      *pSubBufferSize = size;
      break;

    default:
      mRETURN_RESULT(mR_IndexOutOfBounds);
    }
    break;

  case mPF_YUV422:
    switch (bufferIndex)
    {
    case 0:
      *pSubBufferSize = size;
      break;

    case 1:
    case 2:
      *pSubBufferSize = mVec2s(size.x / 2, size.y);
      break;

    default:
      mRETURN_RESULT(mR_IndexOutOfBounds);
    }
    break;

  case mPF_YUV440:
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

  case mPF_YUV420:
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

  case mPF_YUV411:
    switch (bufferIndex)
    {
    case 0:
      *pSubBufferSize = size;
      break;

    case 1:
    case 2:
      *pSubBufferSize = mVec2s(size.x / 4, size.y);
      break;

    default:
      mRETURN_RESULT(mR_IndexOutOfBounds);
    }
    break;

  case mPF_B8G8R8:
  case mPF_R8G8B8:
  case mPF_B8G8R8A8:
  case mPF_R8G8B8A8:
  case mPF_Monochrome8:
  case mPF_Monochrome16:
  case mPF_Rf16Gf16Bf16Af16:
  case mPF_Rf16Gf16Bf16:
  case mPF_Rf32Gf32Bf32Af32:
  case mPF_Rf32Gf32Bf32:
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
  case mPF_YUV444:
  case mPF_YUV422:
  case mPF_YUV440:
  case mPF_YUV420:
  case mPF_YUV411:
    *pSubBufferPixelFormat = mPixelFormat::mPF_Monochrome8;
    break;

  case mPF_Monochrome8:
  case mPF_Monochrome16:
  case mPF_Monochromef16:
  case mPF_Monochromef32:
  case mPF_R8G8:
  case mPF_R16G16:
  case mPF_Rf16Gf16:
  case mPF_Rf32Gf32:
  case mPF_R8G8B8:
  case mPF_R16G16B16:
  case mPF_Rf16Gf16Bf16:
  case mPF_Rf32Gf32Bf32:
  case mPF_R8G8B8A8:
  case mPF_R16G16B16A16:
  case mPF_Rf16Gf16Bf16Af16:
  case mPF_Rf32Gf32Bf32Af32:
  case mPF_B8G8R8:
  case mPF_B8G8R8A8:
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
  case mPF_YUV444:
  case mPF_YUV440:
    *pSubBufferLineStride = originalLineStride;
    break;

  case mPF_YUV422:
  case mPF_YUV420:
    if (bufferIndex == 0)
      *pSubBufferLineStride = originalLineStride;
    else
      *pSubBufferLineStride = originalLineStride / 2;
    break;

  case mPF_YUV411:
    if (bufferIndex == 0)
      *pSubBufferLineStride = originalLineStride;
    else
      *pSubBufferLineStride = originalLineStride / 4;
    break;

  case mPF_Monochrome8:
  case mPF_Monochrome16:
  case mPF_Monochromef16:
  case mPF_Monochromef32:
  case mPF_R8G8:
  case mPF_R16G16:
  case mPF_Rf16Gf16:
  case mPF_Rf32Gf32:
  case mPF_R8G8B8:
  case mPF_R16G16B16:
  case mPF_Rf16Gf16Bf16:
  case mPF_Rf32Gf32Bf32:
  case mPF_R8G8B8A8:
  case mPF_R16G16B16A16:
  case mPF_Rf16Gf16Bf16Af16:
  case mPF_Rf32Gf32Bf32Af32:
  case mPF_B8G8R8:
  case mPF_B8G8R8A8:
    *pSubBufferLineStride = originalLineStride;
    break;

  default:
    mRETURN_RESULT(mR_InvalidParameter);
    break;
  }

  mRETURN_SUCCESS();
}

//////////////////////////////////////////////////////////////////////////

namespace mPixelFormat_Transform
{
#ifdef SSE2

  /*
  * Simd Library (http://ermig1979.github.io/Simd).
  *
  * Copyright (c) 2011-2017 Yermalayeu Ihar.
  *
  * Permission is hereby granted, free of charge, to any person obtaining a copy
  * of this software and associated documentation files (the "Software"), to deal
  * in the Software without restriction, including without limitation the rights
  * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
  * copies of the Software, and to permit persons to whom the Software is
  * furnished to do so, subject to the following conditions:
  *
  * The above copyright notice and this permission notice shall be included in
  * all copies or substantial portions of the Software.
  *
  * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
  * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
  * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
  * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
  * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
  * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
  * SOFTWARE.
  */

#include "mSimd.h"

#ifdef GIT_BUILD // Define __M_FILE__
  #ifdef __M_FILE__
    #undef __M_FILE__
  #endif
  #define __M_FILE__ "WxZHdolxaOsVMVztMcLWewBh6hSpEpuquqmpB8m6HcA9zdae0FuZIHcxgKjRdXKBj5HQQfLkazzf6tq9"
#endif

  const int32_t mPixelFormatTransform_YUV_SIMD_Y_ADJUST = 16;
  const int32_t mPixelFormatTransform_YUV_SIMD_UV_ADJUST = 128;
  const int32_t mPixelFormatTransform_YUV_SIMD_YUV_TO_BGR_AVERAGING_SHIFT = 13;
  const int32_t mPixelFormatTransform_YUV_SIMD_YUV_TO_BGR_ROUND_TERM = 1 << (mPixelFormatTransform_YUV_SIMD_YUV_TO_BGR_AVERAGING_SHIFT - 1);
  const int32_t mPixelFormatTransform_YUV_SIMD_Y_TO_RGB_WEIGHT = int32_t(1.164 * (1 << mPixelFormatTransform_YUV_SIMD_YUV_TO_BGR_AVERAGING_SHIFT) + 0.5);
  const int32_t mPixelFormatTransform_YUV_SIMD_U_TO_BLUE_WEIGHT = int32_t(2.018 * (1 << mPixelFormatTransform_YUV_SIMD_YUV_TO_BGR_AVERAGING_SHIFT) + 0.5);
  const int32_t mPixelFormatTransform_YUV_SIMD_U_TO_GREEN_WEIGHT = -int32_t(0.391 * (1 << mPixelFormatTransform_YUV_SIMD_YUV_TO_BGR_AVERAGING_SHIFT) + 0.5);
  const int32_t mPixelFormatTransform_YUV_SIMD_V_TO_GREEN_WEIGHT = -int32_t(0.813 * (1 << mPixelFormatTransform_YUV_SIMD_YUV_TO_BGR_AVERAGING_SHIFT) + 0.5);
  const int32_t mPixelFormatTransform_YUV_SIMD_V_TO_RED_WEIGHT = int32_t(1.596 * (1 << mPixelFormatTransform_YUV_SIMD_YUV_TO_BGR_AVERAGING_SHIFT) + 0.5);

  const __m128i mPixelFormatTransform_YUV_SIMD_K16_Y_ADJUST = mSIMD_MM_SET1_EPI16(mPixelFormatTransform_YUV_SIMD_Y_ADJUST);
  const __m128i mPixelFormatTransform_YUV_SIMD_K16_UV_ADJUST = mSIMD_MM_SET1_EPI16(mPixelFormatTransform_YUV_SIMD_UV_ADJUST);

  const __m128i mPixelFormatTransform_YUV_SIMD_K16_YRGB_RT = mSIMD_MM_SET2_EPI16(mPixelFormatTransform_YUV_SIMD_Y_TO_RGB_WEIGHT, mPixelFormatTransform_YUV_SIMD_YUV_TO_BGR_ROUND_TERM);
  const __m128i mPixelFormatTransform_YUV_SIMD_K16_VR_0 = mSIMD_MM_SET2_EPI16(mPixelFormatTransform_YUV_SIMD_V_TO_RED_WEIGHT, 0);
  const __m128i mPixelFormatTransform_YUV_SIMD_K16_UG_VG = mSIMD_MM_SET2_EPI16(mPixelFormatTransform_YUV_SIMD_U_TO_GREEN_WEIGHT, mPixelFormatTransform_YUV_SIMD_V_TO_GREEN_WEIGHT);
  const __m128i mPixelFormatTransform_YUV_SIMD_K16_UB_0 = mSIMD_MM_SET2_EPI16(mPixelFormatTransform_YUV_SIMD_U_TO_BLUE_WEIGHT, 0);

  mINLINE __m128i AdjustY16(__m128i y16)
  {
    return _mm_subs_epi16(y16, mPixelFormatTransform_YUV_SIMD_K16_Y_ADJUST);
  }

  mINLINE __m128i AdjustUV16(__m128i uv16)
  {
    return _mm_subs_epi16(uv16, mPixelFormatTransform_YUV_SIMD_K16_UV_ADJUST);
  }

  mINLINE __m128i AdjustedYuvToRed32(__m128i y16_1, __m128i v16_0)
  {
    return _mm_srai_epi32(_mm_add_epi32(_mm_madd_epi16(y16_1, mPixelFormatTransform_YUV_SIMD_K16_YRGB_RT), _mm_madd_epi16(v16_0, mPixelFormatTransform_YUV_SIMD_K16_VR_0)), mPixelFormatTransform_YUV_SIMD_YUV_TO_BGR_AVERAGING_SHIFT);
  }

  mINLINE __m128i AdjustedYuvToRed16(__m128i y16, __m128i v16)
  {
    return mSimd_SaturateI16ToU8(_mm_packs_epi32(AdjustedYuvToRed32(_mm_unpacklo_epi16(y16, mSimd_K16_0001), _mm_unpacklo_epi16(v16, mSimdZero)), AdjustedYuvToRed32(_mm_unpackhi_epi16(y16, mSimd_K16_0001), _mm_unpackhi_epi16(v16, mSimdZero))));
  }

  mINLINE __m128i AdjustedYuvToGreen32(__m128i y16_1, __m128i u16_v16)
  {
    return _mm_srai_epi32(_mm_add_epi32(_mm_madd_epi16(y16_1, mPixelFormatTransform_YUV_SIMD_K16_YRGB_RT), _mm_madd_epi16(u16_v16, mPixelFormatTransform_YUV_SIMD_K16_UG_VG)), mPixelFormatTransform_YUV_SIMD_YUV_TO_BGR_AVERAGING_SHIFT);
  }

  mINLINE __m128i AdjustedYuvToGreen16(__m128i y16, __m128i u16, __m128i v16)
  {
    return mSimd_SaturateI16ToU8(_mm_packs_epi32(AdjustedYuvToGreen32(_mm_unpacklo_epi16(y16, mSimd_K16_0001), _mm_unpacklo_epi16(u16, v16)), AdjustedYuvToGreen32(_mm_unpackhi_epi16(y16, mSimd_K16_0001), _mm_unpackhi_epi16(u16, v16))));
  }

  mINLINE __m128i AdjustedYuvToBlue32(__m128i y16_1, __m128i u16_0)
  {
    return _mm_srai_epi32(_mm_add_epi32(_mm_madd_epi16(y16_1, mPixelFormatTransform_YUV_SIMD_K16_YRGB_RT), _mm_madd_epi16(u16_0, mPixelFormatTransform_YUV_SIMD_K16_UB_0)), mPixelFormatTransform_YUV_SIMD_YUV_TO_BGR_AVERAGING_SHIFT);
  }

  mINLINE __m128i AdjustedYuvToBlue16(__m128i y16, __m128i u16)
  {
    return mSimd_SaturateI16ToU8(_mm_packs_epi32(AdjustedYuvToBlue32(_mm_unpacklo_epi16(y16, mSimd_K16_0001), _mm_unpacklo_epi16(u16, mSimdZero)), AdjustedYuvToBlue32(_mm_unpackhi_epi16(y16, mSimd_K16_0001), _mm_unpackhi_epi16(u16, mSimdZero))));
  }

  mINLINE __m128i YuvToRed(__m128i y, __m128i v)
  {
    __m128i lo = AdjustedYuvToRed16(AdjustY16(_mm_unpacklo_epi8(y, mSimdZero)), AdjustUV16(_mm_unpacklo_epi8(v, mSimdZero)));
    __m128i hi = AdjustedYuvToRed16(AdjustY16(_mm_unpackhi_epi8(y, mSimdZero)), AdjustUV16(_mm_unpackhi_epi8(v, mSimdZero)));
    return _mm_packus_epi16(lo, hi);
  }

  mINLINE __m128i YuvToGreen(__m128i y, __m128i u, __m128i v)
  {
    __m128i lo = AdjustedYuvToGreen16(AdjustY16(_mm_unpacklo_epi8(y, mSimdZero)), AdjustUV16(_mm_unpacklo_epi8(u, mSimdZero)), AdjustUV16(_mm_unpacklo_epi8(v, mSimdZero)));
    __m128i hi = AdjustedYuvToGreen16(AdjustY16(_mm_unpackhi_epi8(y, mSimdZero)), AdjustUV16(_mm_unpackhi_epi8(u, mSimdZero)), AdjustUV16(_mm_unpackhi_epi8(v, mSimdZero)));
    return _mm_packus_epi16(lo, hi);
  }

  mINLINE __m128i YuvToBlue(__m128i y, __m128i u)
  {
    __m128i lo = AdjustedYuvToBlue16(AdjustY16(_mm_unpacklo_epi8(y, mSimdZero)), AdjustUV16(_mm_unpacklo_epi8(u, mSimdZero)));
    __m128i hi = AdjustedYuvToBlue16(AdjustY16(_mm_unpackhi_epi8(y, mSimdZero)), AdjustUV16(_mm_unpackhi_epi8(u, mSimdZero)));

    return _mm_packus_epi16(lo, hi);
  }

  template <bool align> mINLINE void AdjustedYuv16ToBgra(__m128i y16, __m128i u16, __m128i v16, const __m128i &a_0, __m128i *pBgra)
  {
    const __m128i b16 = AdjustedYuvToBlue16(y16, u16);
    const __m128i g16 = AdjustedYuvToGreen16(y16, u16, v16);
    const __m128i r16 = AdjustedYuvToRed16(y16, v16);
    const __m128i bg8 = _mm_or_si128(b16, _mm_slli_si128(g16, 1));
    const __m128i ra8 = _mm_or_si128(r16, a_0);

    mSimd_Store<align>(pBgra + 0, _mm_unpacklo_epi16(bg8, ra8));
    mSimd_Store<align>(pBgra + 1, _mm_unpackhi_epi16(bg8, ra8));
  }

  template <bool align> mINLINE void Yuv16ToBgra(__m128i y16, __m128i u16, __m128i v16, const __m128i &a_0, __m128i *pBgra)
  {
    AdjustedYuv16ToBgra<align>(AdjustY16(y16), AdjustUV16(u16), AdjustUV16(v16), a_0, pBgra);
  }

  template <bool align> mINLINE void Yuv8ToBgra(__m128i y8, __m128i u8, __m128i v8, const __m128i &a_0, __m128i *pBgra)
  {
    Yuv16ToBgra<align>(_mm_unpacklo_epi8(y8, mSimdZero), _mm_unpacklo_epi8(u8, mSimdZero), _mm_unpacklo_epi8(v8, mSimdZero), a_0, pBgra + 0);
    Yuv16ToBgra<align>(_mm_unpackhi_epi8(y8, mSimdZero), _mm_unpackhi_epi8(u8, mSimdZero), _mm_unpackhi_epi8(v8, mSimdZero), a_0, pBgra + 2);
  }

  template <bool align> mINLINE void Yuv444pToBgra(const uint8_t *pY, const uint8_t *pU, const uint8_t *pV, const __m128i &a_0, uint8_t *pBgra)
  {
    Yuv8ToBgra<align>(mSimd_Load<align>((__m128i*)pY), mSimd_Load<align>((__m128i*)pU), mSimd_Load<align>((__m128i*)pV), a_0, (__m128i*)pBgra);
  }

  template <bool align> mFUNCTION(Yuv444pToBgra, const uint8_t *pY, size_t yStride, const uint8_t *pU, size_t uStride, const uint8_t *pV, size_t vStride,
    size_t width, size_t height, uint8_t *bgra, size_t bgraStride, uint8_t alpha)
  {
    mFUNCTION_SETUP();

    mERROR_IF(!(width >= mSimd128bit), mR_InvalidParameter);

#ifdef _DEBUG
    if (align)
    {
      bool yIsAligned, yStrideIsAligned, uIsAligned, uStrideIsAligned, vIsAligned, vStrideIsAligned, bgraIsAligned, bgraStrideIsAligned;

      mERROR_CHECK(mMemoryIsAligned(pY, sizeof(__m128), &yIsAligned));
      mERROR_CHECK(mMemoryIsAligned(yStride, sizeof(__m128), &yStrideIsAligned));
      mERROR_CHECK(mMemoryIsAligned(pU, sizeof(__m128), &uIsAligned));
      mERROR_CHECK(mMemoryIsAligned(uStride, sizeof(__m128), &uStrideIsAligned));
      mERROR_CHECK(mMemoryIsAligned(pV, sizeof(__m128), &vIsAligned));
      mERROR_CHECK(mMemoryIsAligned(vStride, sizeof(__m128), &vStrideIsAligned));
      mERROR_CHECK(mMemoryIsAligned(bgra, sizeof(__m128), &bgraIsAligned));
      mERROR_CHECK(mMemoryIsAligned(bgraStride, sizeof(__m128), &bgraStrideIsAligned));

      mASSERT(yIsAligned && yStrideIsAligned && uIsAligned && uStrideIsAligned && vIsAligned && vStrideIsAligned && bgraIsAligned && bgraStrideIsAligned, "Unaligned memory passed to aligned function.");
    }
#endif // _DEBUG

    __m128i a_0 = _mm_slli_si128(_mm_set1_epi16(alpha), 1);
    size_t bodyWidth;
    mERROR_CHECK(mMemoryAlignLo(width, mSimd128bit, &bodyWidth));
    size_t tail = width - bodyWidth;

    for (size_t row = 0; row < height; ++row)
    {
      for (size_t colYuv = 0, colBgra = 0; colYuv < bodyWidth; colYuv += mSimd128bit, colBgra += mSimd512bit)
      {
        Yuv444pToBgra<align>(pY + colYuv, pU + colYuv, pV + colYuv, a_0, bgra + colBgra);
      }

      if (tail)
      {
        size_t col = width - mSimd128bit;
        Yuv444pToBgra<false>(pY + col, pU + col, pV + col, a_0, bgra + 4 * col);
      }

      pY += yStride;
      pU += uStride;
      pV += vStride;
      bgra += bgraStride;
    }

    mRETURN_SUCCESS();
  }

  mFUNCTION(mPixelFormat_Yuv444pToBgra, const uint8_t *pY, size_t yStride, const uint8_t *pU, size_t uStride, const uint8_t *pV, size_t vStride, size_t width, size_t height, uint8_t *pBgra, size_t bgraStride, uint8_t alpha)
  {
    mFUNCTION_SETUP();

    bool yIsAligned, yStrideIsAligned, uIsAligned, uStrideIsAligned, vIsAligned, vStrideIsAligned, bgraIsAligned, bgraStrideIsAligned;

    mERROR_CHECK(mMemoryIsAligned(pY, sizeof(__m128), &yIsAligned));
    mERROR_CHECK(mMemoryIsAligned(yStride, sizeof(__m128), &yStrideIsAligned));
    mERROR_CHECK(mMemoryIsAligned(pU, sizeof(__m128), &uIsAligned));
    mERROR_CHECK(mMemoryIsAligned(uStride, sizeof(__m128), &uStrideIsAligned));
    mERROR_CHECK(mMemoryIsAligned(pV, sizeof(__m128), &vIsAligned));
    mERROR_CHECK(mMemoryIsAligned(vStride, sizeof(__m128), &vStrideIsAligned));
    mERROR_CHECK(mMemoryIsAligned(pBgra, sizeof(__m128), &bgraIsAligned));
    mERROR_CHECK(mMemoryIsAligned(bgraStride, sizeof(__m128), &bgraStrideIsAligned));

    if (yIsAligned && yStrideIsAligned && uIsAligned && uStrideIsAligned && vIsAligned && vStrideIsAligned && bgraIsAligned && bgraStrideIsAligned)
      mERROR_CHECK(Yuv444pToBgra<true>(pY, yStride, pU, uStride, pV, vStride, width, height, pBgra, bgraStride, alpha));
    else
      mERROR_CHECK(Yuv444pToBgra<false>(pY, yStride, pU, uStride, pV, vStride, width, height, pBgra, bgraStride, alpha));

    mRETURN_SUCCESS();
  }

  template <bool align> mINLINE void Yuv422pToBgra(const uint8_t *pY, const __m128i &u, const __m128i &v, const __m128i &a_0, uint8_t *pBgra)
  {
    Yuv8ToBgra<align>(mSimd_Load<align>((__m128i*)pY + 0), _mm_unpacklo_epi8(u, u), _mm_unpacklo_epi8(v, v), a_0, (__m128i*)pBgra + 0);
    Yuv8ToBgra<align>(mSimd_Load<align>((__m128i*)pY + 1), _mm_unpackhi_epi8(u, u), _mm_unpackhi_epi8(v, v), a_0, (__m128i*)pBgra + 4);
  }

  template <bool align> mFUNCTION(mPixelFormat_Transform_Yuv420pToBgra_SSE2, const uint8_t *pY, size_t yStride, const uint8_t *pU, size_t uStride, const uint8_t *pV, size_t vStride, size_t width, size_t height, uint8_t *pBgra, size_t bgraStride, uint8_t alpha)
  {
    mFUNCTION_SETUP();

    mERROR_IF(!((width % 2 == 0) && (height % 2 == 0) && (width >= mSimd256bit) && (height >= 2)), mR_InvalidParameter);

#ifdef _DEBUG
    if (align)
    {
      bool yIsAligned, yStrideIsAligned, uIsAligned, uStrideIsAligned, vIsAligned, vStrideIsAligned, bgraIsAligned, bgraStrideIsAligned;

      mERROR_CHECK(mMemoryIsAligned(pY, sizeof(__m128), &yIsAligned));
      mERROR_CHECK(mMemoryIsAligned(yStride, sizeof(__m128), &yStrideIsAligned));
      mERROR_CHECK(mMemoryIsAligned(pU, sizeof(__m128), &uIsAligned));
      mERROR_CHECK(mMemoryIsAligned(uStride, sizeof(__m128), &uStrideIsAligned));
      mERROR_CHECK(mMemoryIsAligned(pV, sizeof(__m128), &vIsAligned));
      mERROR_CHECK(mMemoryIsAligned(vStride, sizeof(__m128), &vStrideIsAligned));
      mERROR_CHECK(mMemoryIsAligned(pBgra, sizeof(__m128), &bgraIsAligned));
      mERROR_CHECK(mMemoryIsAligned(bgraStride, sizeof(__m128), &bgraStrideIsAligned));

      mASSERT(yIsAligned && yStrideIsAligned && uIsAligned && uStrideIsAligned && vIsAligned && vStrideIsAligned && bgraIsAligned && bgraStrideIsAligned, "Unaligned memory passed to aligned function.");
    }
#endif // _DEBUG

    __m128i a_0 = _mm_slli_si128(_mm_set1_epi16(alpha), 1);
    size_t bodyWidth;
    mERROR_CHECK(mMemoryAlignLo(width, mSimd256bit, &bodyWidth));
    const size_t tail = width - bodyWidth;

    for (size_t row = 0; row < height; row += 2)
    {
      for (size_t colUV = 0, colY = 0, colBgra = 0; colY < bodyWidth; colY += mSimd256bit, colUV += mSimd128bit, colBgra += mSimd1024bit)
      {
        __m128i u_ = mSimd_Load<align>((__m128i*)(pU + colUV));
        __m128i v_ = mSimd_Load<align>((__m128i*)(pV + colUV));
        Yuv422pToBgra<align>(pY + colY, u_, v_, a_0, pBgra + colBgra);
        Yuv422pToBgra<align>(pY + colY + yStride, u_, v_, a_0, pBgra + colBgra + bgraStride);
      }

      if (tail)
      {
        size_t offset = width - mSimd256bit;
        __m128i u_ = mSimd_Load<false>((__m128i*)(pU + offset / 2));
        __m128i v_ = mSimd_Load<false>((__m128i*)(pV + offset / 2));
        Yuv422pToBgra<false>(pY + offset, u_, v_, a_0, pBgra + 4 * offset);
        Yuv422pToBgra<false>(pY + offset + yStride, u_, v_, a_0, pBgra + 4 * offset + bgraStride);
      }

      pY += 2 * yStride;
      pU += uStride;
      pV += vStride;
      pBgra += 2 * bgraStride;
    }
    mRETURN_SUCCESS();
  }

  mFUNCTION(mPixelFormat_Transform_Yuv420pToBgra_SSE2, const uint8_t *pY, const size_t yStride, const uint8_t *pU, const size_t uStride, const uint8_t *pV, const size_t vStride, const size_t width, const size_t height, uint8_t *pBgra, const size_t bgraStride, const uint8_t alpha)
  {
    mFUNCTION_SETUP();

    bool yIsAligned, yStrideIsAligned, uIsAligned, uStrideIsAligned, vIsAligned, vStrideIsAligned, bgraIsAligned, bgraStrideIsAligned;

    mERROR_CHECK(mMemoryIsAligned(pY, sizeof(__m128), &yIsAligned));
    mERROR_CHECK(mMemoryIsAligned(yStride, sizeof(__m128), &yStrideIsAligned));
    mERROR_CHECK(mMemoryIsAligned(pU, sizeof(__m128), &uIsAligned));
    mERROR_CHECK(mMemoryIsAligned(uStride, sizeof(__m128), &uStrideIsAligned));
    mERROR_CHECK(mMemoryIsAligned(pV, sizeof(__m128), &vIsAligned));
    mERROR_CHECK(mMemoryIsAligned(vStride, sizeof(__m128), &vStrideIsAligned));
    mERROR_CHECK(mMemoryIsAligned(pBgra, sizeof(__m128), &bgraIsAligned));
    mERROR_CHECK(mMemoryIsAligned(bgraStride, sizeof(__m128), &bgraStrideIsAligned));

    if (yIsAligned && yStrideIsAligned && uIsAligned && uStrideIsAligned && vIsAligned && vStrideIsAligned && bgraIsAligned && bgraStrideIsAligned)
      mERROR_CHECK(mPixelFormat_Transform_Yuv420pToBgra_SSE2<true>(pY, yStride, pU, uStride, pV, vStride, width, height, pBgra, bgraStride, alpha));
    else
      mERROR_CHECK(mPixelFormat_Transform_Yuv420pToBgra_SSE2<false>(pY, yStride, pU, uStride, pV, vStride, width, height, pBgra, bgraStride, alpha));

    mRETURN_SUCCESS();
  }

  template <bool align> mINLINE void Yuv422pToBgra(const uint8_t *y, const uint8_t *u, const uint8_t *v, const __m128i & a_0, uint8_t *bgra)
  {
    Yuv422pToBgra<align>(y, mSimd_Load<align>((__m128i*)u), mSimd_Load<align>((__m128i*)v), a_0, bgra);
  }

  template <bool align> mFUNCTION(Yuv422pToBgra, const uint8_t *y, size_t yStride, const uint8_t *u, size_t uStride, const uint8_t *v, size_t vStride, size_t width, size_t height, uint8_t *bgra, size_t bgraStride, uint8_t alpha)
  {
    mFUNCTION_SETUP();

    mERROR_IF(!((width % 2 == 0) && (width >= mSimd256bit)), mR_InvalidParameter);

#ifdef _DEBUG
    if (align)
    {
      bool yIsAligned, yStrideIsAligned, uIsAligned, uStrideIsAligned, vIsAligned, vStrideIsAligned, bgraIsAligned, bgraStrideIsAligned;

      mERROR_CHECK(mMemoryIsAligned(y, sizeof(__m128), &yIsAligned));
      mERROR_CHECK(mMemoryIsAligned(yStride, sizeof(__m128), &yStrideIsAligned));
      mERROR_CHECK(mMemoryIsAligned(u, sizeof(__m128), &uIsAligned));
      mERROR_CHECK(mMemoryIsAligned(uStride, sizeof(__m128), &uStrideIsAligned));
      mERROR_CHECK(mMemoryIsAligned(v, sizeof(__m128), &vIsAligned));
      mERROR_CHECK(mMemoryIsAligned(vStride, sizeof(__m128), &vStrideIsAligned));
      mERROR_CHECK(mMemoryIsAligned(bgra, sizeof(__m128), &bgraIsAligned));
      mERROR_CHECK(mMemoryIsAligned(bgraStride, sizeof(__m128), &bgraStrideIsAligned));

      mASSERT(yIsAligned && yStrideIsAligned && uIsAligned && uStrideIsAligned && vIsAligned && vStrideIsAligned && bgraIsAligned && bgraStrideIsAligned, "Unaligned memory passed to aligned function.");
    }
#endif // _DEBUG

    __m128i a_0 = _mm_slli_si128(_mm_set1_epi16(alpha), 1);
    size_t bodyWidth;
    mERROR_CHECK(mMemoryAlignLo(width, mSimd256bit, &bodyWidth));
    size_t tail = width - bodyWidth;
    for (size_t row = 0; row < height; ++row)
    {
      for (size_t colUV = 0, colY = 0, colBgra = 0; colY < bodyWidth; colY += mSimd256bit, colUV += mSimd128bit, colBgra += mSimd1024bit)
        Yuv422pToBgra<align>(y + colY, u + colUV, v + colUV, a_0, bgra + colBgra);

      if (tail)
      {
        size_t offset = width - mSimd256bit;
        Yuv422pToBgra<false>(y + offset, u + offset / 2, v + offset / 2, a_0, bgra + 4 * offset);
      }

      y += yStride;
      u += uStride;
      v += vStride;
      bgra += bgraStride;
    }

    mRETURN_SUCCESS();
  }

  mFUNCTION(mPixelFormat_Yuv422pToBgra, const uint8_t *y, size_t yStride, const uint8_t *u, size_t uStride, const uint8_t *v, size_t vStride, size_t width, size_t height, uint8_t *bgra, size_t bgraStride, uint8_t alpha)
  {
    mFUNCTION_SETUP();

    bool yIsAligned, yStrideIsAligned, uIsAligned, uStrideIsAligned, vIsAligned, vStrideIsAligned, bgraIsAligned, bgraStrideIsAligned;

    mERROR_CHECK(mMemoryIsAligned(y, sizeof(__m128), &yIsAligned));
    mERROR_CHECK(mMemoryIsAligned(yStride, sizeof(__m128), &yStrideIsAligned));
    mERROR_CHECK(mMemoryIsAligned(u, sizeof(__m128), &uIsAligned));
    mERROR_CHECK(mMemoryIsAligned(uStride, sizeof(__m128), &uStrideIsAligned));
    mERROR_CHECK(mMemoryIsAligned(v, sizeof(__m128), &vIsAligned));
    mERROR_CHECK(mMemoryIsAligned(vStride, sizeof(__m128), &vStrideIsAligned));
    mERROR_CHECK(mMemoryIsAligned(bgra, sizeof(__m128), &bgraIsAligned));
    mERROR_CHECK(mMemoryIsAligned(bgraStride, sizeof(__m128), &bgraStrideIsAligned));

    if (yIsAligned && yStrideIsAligned && uIsAligned && uStrideIsAligned && vIsAligned && vStrideIsAligned && bgraIsAligned && bgraStrideIsAligned)
      Yuv422pToBgra<true>(y, yStride, u, uStride, v, vStride, width, height, bgra, bgraStride, alpha);
    else
      Yuv422pToBgra<false>(y, yStride, u, uStride, v, vStride, width, height, bgra, bgraStride, alpha);

    mRETURN_SUCCESS();
  }

  mFUNCTION(mPixelFormat_Transform_YUV422ToBgra, mPtr<mImageBuffer> &source, mPtr<mImageBuffer> &target, mPtr<mThreadPool> & /* asyncTaskHandler */)
  {
    mFUNCTION_SETUP();

    uint8_t *pOutPixels = target->pPixels;
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

    size_t sourceLineStride0, sourceLineStride1;
    mERROR_CHECK(mPixelFormat_GetSubBufferStride(source->pixelFormat, 0, source->lineStride, &sourceLineStride0));
    mERROR_CHECK(mPixelFormat_GetSubBufferStride(source->pixelFormat, 1, source->lineStride, &sourceLineStride1));

    size_t targetUnitSize;
    mERROR_CHECK(mPixelFormat_GetUnitSize(target->pixelFormat, &targetUnitSize));

    mERROR_CHECK(mPixelFormat_Yuv422pToBgra(pBuffer0, sourceLineStride0, pBuffer1, sourceLineStride1, pBuffer2, sourceLineStride1, target->currentSize.x, target->currentSize.y, pOutPixels, target->lineStride, 0xFF));

    mRETURN_SUCCESS();
  }

  mFUNCTION(mPixelFormat_Transform_YUV444ToBgra, mPtr<mImageBuffer> &source, mPtr<mImageBuffer> &target, mPtr<mThreadPool> & /* asyncTaskHandler */)
  {
    mFUNCTION_SETUP();

    uint8_t *pOutPixels = target->pPixels;
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

    size_t sourceLineStride0, sourceLineStride1;
    mERROR_CHECK(mPixelFormat_GetSubBufferStride(source->pixelFormat, 0, source->lineStride, &sourceLineStride0));
    mERROR_CHECK(mPixelFormat_GetSubBufferStride(source->pixelFormat, 1, source->lineStride, &sourceLineStride1));

    size_t targetUnitSize;
    mERROR_CHECK(mPixelFormat_GetUnitSize(target->pixelFormat, &targetUnitSize));

    mERROR_CHECK(mPixelFormat_Yuv444pToBgra(pBuffer0, sourceLineStride0, pBuffer1, sourceLineStride1, pBuffer2, sourceLineStride1, target->currentSize.x, target->currentSize.y, pOutPixels, target->lineStride, 0xFF));

    mRETURN_SUCCESS();
  }

#endif // SSE2

  static mINLINE void YUV444_to_RGB(const uint8_t y, const uint8_t u, const uint8_t v, OUT uint8_t *pR, OUT uint8_t *pG, OUT uint8_t *pB)
  {
    const int32_t c = y - 16;
    const int32_t d = u - 128;
    const int32_t e = v - 128;

    const int32_t c_ = 298 * c;

    *pR = (uint8_t)mClamp((c_ + 409 * e + 128) >> 8, 0, 255);
    *pG = (uint8_t)mClamp((c_ - 100 * d - 208 * e + 128) >> 8, 0, 255);
    *pB = (uint8_t)mClamp((c_ + 516 * d + 128) >> 8, 0, 255);
  }

  static mINLINE void RGB_to_YUV444(const uint8_t r, const uint8_t g, const uint8_t b, OUT uint8_t *pY, OUT uint8_t *pU, OUT uint8_t *pV)
  {
    const int32_t b_ = b + 128;
    const int32_t y = ((66 * r + 129 * g + 25 * b_) >> 8) + 16;
    const int32_t u = ((-38 * r + 74 * g + 112 * b_) >> 8) + 128;
    const int32_t v = ((112 * r + 94 * g + 18 * b_) >> 8) + 128;

    *pY = (uint8_t)y;
    *pU = (uint8_t)u;
    *pV = (uint8_t)v;
  }

  static mINLINE void YUV422_to_RGB(const uint8_t y0, const uint8_t y1, const uint8_t u, const uint8_t v, OUT uint8_t *pR0, OUT uint8_t *pG0, OUT uint8_t *pB0, OUT uint8_t *pR1, OUT uint8_t *pG1, OUT uint8_t *pB1)
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

  static mINLINE void YUV420_to_RGB(const uint8_t y0, const uint8_t y1, const uint8_t y2, const uint8_t y3, const uint8_t u, const uint8_t v, OUT uint8_t *pR0, OUT uint8_t *pG0, OUT uint8_t *pB0, OUT uint8_t *pR1, OUT uint8_t *pG1, OUT uint8_t *pB1, OUT uint8_t *pR2, OUT uint8_t *pG2, OUT uint8_t *pB2, OUT uint8_t *pR3, OUT uint8_t *pG3, OUT uint8_t *pB3)
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

  static mINLINE void YUV420_to_BGRA(const uint8_t y0, const uint8_t y1, const uint8_t y2, const uint8_t y3, const uint8_t u, const uint8_t v, OUT uint32_t *pColor0, OUT uint32_t *pColor1, OUT uint32_t *pColor2, OUT uint32_t *pColor3)
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

    *pColor0 = ((uint32_t)mClamp((c0_ + v0) << 8, 0, 0xFF0000) & 0xFF0000) | ((uint32_t)mClamp((c0_ + v1), 0, 0xFF00) & 0xFF00) | (uint32_t)mClamp((c0_ + v2) >> 8, 0, 0xFF);
    *pColor1 = ((uint32_t)mClamp((c1_ + v0) << 8, 0, 0xFF0000) & 0xFF0000) | ((uint32_t)mClamp((c1_ + v1), 0, 0xFF00) & 0xFF00) | (uint32_t)mClamp((c1_ + v2) >> 8, 0, 0xFF);
    *pColor2 = ((uint32_t)mClamp((c2_ + v0) << 8, 0, 0xFF0000) & 0xFF0000) | ((uint32_t)mClamp((c2_ + v1), 0, 0xFF00) & 0xFF00) | (uint32_t)mClamp((c2_ + v2) >> 8, 0, 0xFF);
    *pColor3 = ((uint32_t)mClamp((c3_ + v0) << 8, 0, 0xFF0000) & 0xFF0000) | ((uint32_t)mClamp((c3_ + v1), 0, 0xFF00) & 0xFF00) | (uint32_t)mClamp((c3_ + v2) >> 8, 0, 0xFF);
  }

  mFUNCTION(mPixelFormat_Transform_YUV420ToBgra_Base, uint8_t *pY, uint8_t *pU, uint8_t *pV, uint32_t *pBgra, const size_t width, const size_t height, const size_t yStride, const size_t uvStride, const size_t bgraStride)
  {
    mFUNCTION_SETUP();

    for (size_t y = 0; y < height - 1; y += 2)
    {
      const size_t yline = y * bgraStride;
      const size_t yhalf = y >> 1;
      const size_t ySourceLine0 = y * yStride;
      const size_t ySourceLine1 = (y >> 1) * uvStride;

      for (size_t x = 0; x < width - 1; x += 2)
      {
        const size_t xhalf = x >> 1;
        const size_t xySourceLine0 = x + ySourceLine0;
        const size_t xyTargetLine0 = x + yline;
        const size_t sourcePosSubRes = xhalf + ySourceLine1;

        YUV420_to_BGRA(
          pY[xySourceLine0], pY[xySourceLine0 + 1],
          pY[xySourceLine0 + yStride], pY[xySourceLine0 + yStride + 1],

          pU[sourcePosSubRes],

          pV[sourcePosSubRes],

          &pBgra[xyTargetLine0], &pBgra[xyTargetLine0 + 1],
          &pBgra[xyTargetLine0 + bgraStride], &pBgra[xyTargetLine0 + bgraStride + 1]);
      }
    }

    mRETURN_SUCCESS();
  }

  mFUNCTION(mPixelFormat_Transform_YUV420ToBgra_Wrapper, uint8_t *pY, uint8_t *pU, uint8_t *pV, uint32_t *pBgra, const size_t width, const size_t height, const size_t bgraStride, const size_t yStride, size_t uvStride, size_t targetUnitSize)
  {
    mFUNCTION_SETUP();

    mUnused(targetUnitSize);

#ifndef SSE2
    mERROR_CHECK(mPixelFormat_Transform_YUV420ToBgra_Base(pY, pU, pV, pBgra, width, height, yStride, uvStride, bgraStride));
#else
    mERROR_CHECK(mPixelFormat_Transform_Yuv420pToBgra_SSE2(pY, yStride, pU, uvStride, pV, uvStride, width, height, (uint8_t *)pBgra, bgraStride * targetUnitSize, 0xFF));
#endif

    mRETURN_SUCCESS();
  }

  mFUNCTION(mPixelFormat_Transform_YUV420ToBgra, mPtr<mImageBuffer> &source, mPtr<mImageBuffer> &target, mPtr<mThreadPool> &asyncTaskHandler)
  {
    mFUNCTION_SETUP();

    uint32_t *pOutPixels = reinterpret_cast<uint32_t *>(target->pPixels);
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

    size_t sourceLineStride0, sourceLineStride1;
    mERROR_CHECK(mPixelFormat_GetSubBufferStride(source->pixelFormat, 0, source->lineStride, &sourceLineStride0));
    mERROR_CHECK(mPixelFormat_GetSubBufferStride(source->pixelFormat, 1, source->lineStride, &sourceLineStride1));

    size_t targetUnitSize;
    mERROR_CHECK(mPixelFormat_GetUnitSize(target->pixelFormat, &targetUnitSize));

    if (asyncTaskHandler == nullptr)
    {
      mERROR_CHECK(mPixelFormat_Transform_YUV420ToBgra_Wrapper(pBuffer0, pBuffer1, pBuffer2, pOutPixels, target->currentSize.x, target->currentSize.y, target->lineStride, sourceLineStride0, sourceLineStride1, targetUnitSize));
    }
    else
    {
      mTask **ppTasks = nullptr;
      size_t threadCount;
      mERROR_CHECK(mThreadPool_GetThreadCount(asyncTaskHandler, &threadCount));

      mAllocator *pAllocator = &mDefaultAllocator;
      mDEFER_CALL_2(mAllocator_FreePtr, pAllocator, &ppTasks);
      mERROR_CHECK(mAllocator_AllocateZero(pAllocator, &ppTasks, threadCount));

      const size_t subHeight = target->currentSize.y / threadCount;
      const size_t yOffset = sourceLineStride0 * subHeight;
      const size_t uvOffset = sourceLineStride1 * (subHeight / 2);
      const size_t bgraOffset = target->lineStride *subHeight;

      mResult result = mR_Success;

      for (size_t i = 0; i < threadCount; ++i)
        mERROR_CHECK_GOTO(mTask_CreateWithLambda(&ppTasks[i], pAllocator, [=]() { return mPixelFormat_Transform_YUV420ToBgra_Wrapper(pBuffer0 + i * yOffset, pBuffer1 + i * uvOffset, pBuffer2 + i * uvOffset, pOutPixels + i * bgraOffset, target->currentSize.x, subHeight, target->lineStride, sourceLineStride0, sourceLineStride1, targetUnitSize); }), result, epilogue);

      for (size_t i = 0; i < threadCount; ++i)
        mERROR_CHECK_GOTO(mThreadPool_EnqueueTask(asyncTaskHandler, ppTasks[i]), result, epilogue);

      for (size_t i = 0; i < threadCount; ++i)
        mERROR_CHECK_GOTO(mTask_Join(ppTasks[i]), result, epilogue);

    epilogue:
      for (size_t i = 0; i < threadCount; ++i)
        if (ppTasks[i] != nullptr)
          mERROR_CHECK(mTask_Destroy(&ppTasks[i]));

      mRETURN_RESULT(result);
    }

    mRETURN_SUCCESS();
  }

  mFUNCTION(mPixelFormat_Transform_RgbaToBgra_BgraToRgba, const uint32_t *pSource, const size_t sourceStride, uint32_t *pTarget, const size_t targetStride, const mVec2s &size)
  {
    mFUNCTION_SETUP();

#ifndef SSE2
    const size_t maxX = size.x >= 1 ? size.x - 1 : 0;
#else
    const size_t maxX = size.x >= 3 ? size.x - 3 : 0;
#endif

    for (size_t y = 0; y < size.y; ++y)
    {
      const size_t ySourceOffset = y * sourceStride;
      const size_t yTargetOffset = y * targetStride;

      size_t x = 0;
#ifndef SSE2
      for (; x < maxX; x += 2)
      {
        uint64_t color = *(uint64_t *)&pSource[x + ySourceOffset];
        uint64_t swap0 = __ll_lshift(color, 0x10);
        uint64_t swap1 = __ull_rshift(color, 0x10);
        
        *(uint64_t *)&pTarget[x + yTargetOffset] = ((color & 0xFF00FF00FF00FF00) | (swap0 & 0x00FF000000FF0000) | (swap1 & 0x000000FF000000FF));
      }
#else
      const __m128i _G_A_Pattern = mSIMD_MM_SET1_EPI32(0xFF00FF00);
      const __m128i X____Pattern = mSIMD_MM_SET1_EPI32(0x00FF0000);
      const __m128i __X__Pattern = mSIMD_MM_SET1_EPI32(0x000000FF);

      for (; x < maxX; x += 4)
      {
        __m128i color = _mm_loadu_si128(reinterpret_cast<const __m128i *>(&pSource[x + ySourceOffset]));
        __m128i swap0 = _mm_slli_si128(color, 2);
        __m128i swap1 = _mm_srli_si128(color, 2);

        color = _mm_and_si128(color, _G_A_Pattern);
        swap0 = _mm_and_si128(swap0, X____Pattern);
        swap1 = _mm_and_si128(swap1, __X__Pattern);

        color = _mm_or_si128(color, _mm_or_si128(swap0, swap1));

        _mm_storeu_si128(reinterpret_cast<__m128i *>(&pTarget[x + yTargetOffset]), color);
      }
#endif

      for (; x < size.x; x++)
      {
        const uint32_t color = pSource[x + ySourceOffset];
        pTarget[x + yTargetOffset] = (color & 0xFF00FF00) | ((color & 0x00FF0000) >> 0x10) | ((color & 0x000000FF) << 0x10);
      }
    }

    mRETURN_SUCCESS();
  }

  mFUNCTION(mPixelFormat_Transform_RgbaToBgra, mPtr<mImageBuffer> &source, mPtr<mImageBuffer> &target, mPtr<mThreadPool> &asyncTaskHandler)
  {
    mFUNCTION_SETUP();

    uint32_t *pTarget = (uint32_t *)target->pPixels;
    uint32_t *pSource = (uint32_t *)source->pPixels;
    const mVec2s size = source->currentSize;
    const size_t targetStride = target->lineStride;
    const size_t sourceStride = source->lineStride;

    if (asyncTaskHandler == nullptr)
    {
      mERROR_CHECK(mPixelFormat_Transform_RgbaToBgra_BgraToRgba(pSource, sourceStride, pTarget, targetStride, size));
    }
    else
    {
      mTask **ppTasks = nullptr;
      size_t threadCount;
      mERROR_CHECK(mThreadPool_GetThreadCount(asyncTaskHandler, &threadCount));

      mAllocator *pAllocator = &mDefaultAllocator;
      mDEFER_CALL_2(mAllocator_FreePtr, pAllocator, &ppTasks);
      mERROR_CHECK(mAllocator_AllocateZero(pAllocator, &ppTasks, threadCount));

      const size_t subHeight = size.y / threadCount;
      const size_t targetOffset = target->lineStride * subHeight;
      const size_t sourceOffset = source->lineStride * subHeight;
      const mVec2s subSize = mVec2s(size.x, subHeight);

      mResult result = mR_Success;

      for (size_t i = 0; i < threadCount; ++i)
        mERROR_CHECK_GOTO(mTask_CreateWithLambda(&ppTasks[i], pAllocator, [=]() {return mPixelFormat_Transform_RgbaToBgra_BgraToRgba(pSource + sourceOffset, sourceStride, pTarget + targetOffset, targetStride, subSize);}), result, epilogue);

      for (size_t i = 0; i < threadCount; ++i)
        mERROR_CHECK_GOTO(mThreadPool_EnqueueTask(asyncTaskHandler, ppTasks[i]), result, epilogue);

      for (size_t i = 0; i < threadCount; ++i)
        mERROR_CHECK_GOTO(mTask_Join(ppTasks[i]), result, epilogue);

    epilogue:
      for (size_t i = 0; i < threadCount; ++i)
        if (ppTasks[i] != nullptr)
          mERROR_CHECK(mTask_Destroy(&ppTasks[i]));

      mRETURN_RESULT(result);
    }

    mRETURN_SUCCESS();
  }

  mFUNCTION(mPixelFormat_Transform_BgraToRgba, mPtr<mImageBuffer> &source, mPtr<mImageBuffer> &target, mPtr<mThreadPool> &asyncTaskHandler)
  {
    mFUNCTION_SETUP();

    mERROR_CHECK(mPixelFormat_Transform_RgbaToBgra(source, target, asyncTaskHandler));

    mRETURN_SUCCESS();
  }

  void mPixelFormat_Transform_RgbToBgrSameStrideSSSE3(const uint8_t *pSource, uint8_t *pTarget, const size_t index, OUT size_t *pIndex, const size_t size)
  {
    const __m128i shuffle = _mm_set_epi8(15, 12, 13, 14, 9, 10, 11, 6, 7, 8, 3, 4, 5, 0, 1, 2);
    size_t i = index;

    for (; i < size - (sizeof(__m128i) - 1); i += (sizeof(__m128i) - 1))
    {
      const __m128i src = _mm_loadu_si128(reinterpret_cast<const __m128i *>(pSource + i));
      _mm_storeu_si128(reinterpret_cast<__m128i *>(pTarget + i), _mm_shuffle_epi8(src, shuffle));
    }

    *pIndex = i;
  }

  mFUNCTION(mPixelFormat_Transform_RgbToBgr, mPtr<mImageBuffer> &source, mPtr<mImageBuffer> &target, mPtr<mThreadPool> & /*asyncTaskHandler */)
  {
    mFUNCTION_SETUP();

    if (source->lineStride == target->lineStride)
    {
      const size_t size = source->currentSize.y * source->lineStride * 3;

      size_t i = 0;
      const uint8_t *pSource = source->pPixels;
      uint8_t *pTarget = target->pPixels;
      
      if (size > sizeof(__m128i))
      {
        mCpuExtensions::Detect();

        if (mCpuExtensions::ssse3Supported)
        {
          mPixelFormat_Transform_RgbToBgrSameStrideSSSE3(pSource, pTarget, i, &i, size);
        }
        else
        {
          const __m128i maskG = _mm_set_epi8(-1, 0, -1, 0, 0, -1, 0, 0, -1, 0, 0, -1, 0, 0, -1, 1);
          const __m128i maskRB = _mm_set_epi8(0, -1, 0, 0, -1, 0, 0, -1, 0, 0, -1, 0, 0, -1, 0, 0);
          const __m128i maskBR = _mm_set_epi8(0, 0, 0, -1, 0, 0, -1, 0, 0, -1, 0, 0, -1, 0, 0, -1);

          for (; i < size - (sizeof(__m128i) - 1); i += (sizeof(__m128i) - 1))
          {
            const __m128i src = _mm_loadu_si128(reinterpret_cast<const __m128i *>(pSource + i));
            const __m128i g = _mm_and_si128(maskG, src);
            const __m128i rb = _mm_and_si128(maskRB, _mm_slli_si128(src, 2));
            const __m128i br = _mm_and_si128(maskBR, _mm_srli_si128(src, 2));

            _mm_storeu_si128(reinterpret_cast<__m128i *>(pTarget + i), _mm_or_si128(g, _mm_or_si128(rb, br)));
          }
        }
      }

      for (; i < size; i += 3)
      {
        pTarget[i + 0] = pSource[i + 2];
        pTarget[i + 1] = pSource[i + 1];
        pTarget[i + 2] = pSource[i + 0];
      }
    }
    else
    {
      mRETURN_RESULT(mR_NotImplemented);
    }

    mRETURN_SUCCESS();
  }

  mFUNCTION(mPixelFormat_Transform_BgrToRgb, mPtr<mImageBuffer> &source, mPtr<mImageBuffer> &target, mPtr<mThreadPool> &asyncTaskHandler)
  {
    mFUNCTION_SETUP();

    mERROR_CHECK(mPixelFormat_Transform_RgbToBgr(source, target, asyncTaskHandler));

    mRETURN_SUCCESS();
  }

  mFUNCTION(mPixelFormat_Transform_Yuv444ToYuv420, mPtr<mImageBuffer> &source, mPtr<mImageBuffer> &target, mPtr<mThreadPool> & /*asyncTaskHandler */)
  {
    mFUNCTION_SETUP();

    const uint8_t *pSource = source->pPixels;
    uint8_t *pTarget = target->pPixels;

    const mVec2s size = source->currentSize;

    if (pSource != pTarget)
      mMemcpy(pTarget, pSource, size.x * size.y);

    pSource += size.x * size.y;
    pTarget += size.x * size.y;

    const mVec2s targetSubBufferSize = size / 2;

    const __m128i mask00FF = _mm_set1_epi16(0x00FF);
    const __m128i two_epi16 = _mm_set1_epi16(2);

    for (size_t subBuffer = 0; subBuffer < 2; subBuffer++)
    {
      for (size_t y = 0; y < targetSubBufferSize.y; y++)
      {
        const uint8_t *pSourceLine2 = pSource + size.x;

        size_t x = 0;

        if (targetSubBufferSize.x >= sizeof(__m128i))
        {
          for (; x < targetSubBufferSize.x - (sizeof(__m128i) - 1); x += sizeof(__m128i))
          {
            // Presume pSource/pSourceLine2 are
            // A B C D E F G H A B C D E F G H I J K L M N O P I J K L M N O P
            // a b c d e f g h a b c d e f g h i j k l m n o p i j k l m n o p

            // A B C D E F G H A B C D E F G H
            const __m128i line00 = _mm_loadu_si128(reinterpret_cast<const __m128i *>(pSource));
            // a b c d e f g h a b c d e f g h
            const __m128i line10 = _mm_loadu_si128(reinterpret_cast<const __m128i *>(pSourceLine2));

            // A   C   E   G   A   C   E   G
            const __m128i line00_16a = _mm_and_si128(line00, mask00FF);
            // B   D   F   H   B   D   F   H
            const __m128i line00_16b = _mm_srli_epi16(line00, 8);
            // a   c   e   g   a   c   e   g
            const __m128i line01_16a = _mm_and_si128(line10, mask00FF);
            // b   d   f   h   b   d   f   h
            const __m128i line01_16b = _mm_srli_epi16(line10, 8);

            // Calculate Averages & Round to the next integer.
            // (A + B + a + b + 2) / 4, 0, (C + D + c + d + 2) / 4, 0, (E + F + e + f + 2) / 4, 0, (G + H + g + h + 2) / 4, 0, ...
            // => avg0, 0, avg1, 0, avg2, 0, avg3, ...
            const __m128i add0 = _mm_srli_epi16(_mm_add_epi16(two_epi16, _mm_add_epi16(_mm_add_epi16(line00_16a, line00_16b), _mm_add_epi16(line01_16a, line01_16b))), 2);

            // I J K L M N O P I J K L M N O P
            const __m128i line01 = _mm_loadu_si128(reinterpret_cast<const __m128i *>(pSource) + 1);
            // i j k l m n o p i j k l m n o p
            const __m128i line11 = _mm_loadu_si128(reinterpret_cast<const __m128i *>(pSourceLine2) + 1);

            // I   K   M   O   I   K   M   O
            const __m128i line10_16a = _mm_and_si128(line01, mask00FF);
            // J   L   N   P   J   L   N   P
            const __m128i line10_16b = _mm_srli_epi16(line01, 8);
            // i   k   m   o   i   k   m   o
            const __m128i line11_16a = _mm_and_si128(line11, mask00FF);
            // j   l   n   p   j   l   n   p
            const __m128i line11_16b = _mm_srli_epi16(line11, 8);

            // Calculate Averages & Round to the next integer.
            // (I + J + i + j + 2) / 4, 0, (K + L + k + l + 2) / 4, 0, (M + N + m + n + 2) / 4, 0, (O + P + o + p + 2) / 4, 0, ...
            // => avg8, 0, avg9, 0, avg10, 0, avg11, ...
            const __m128i add1 = _mm_srli_epi16(_mm_add_epi16(two_epi16, _mm_add_epi16(_mm_add_epi16(line10_16a, line10_16b), _mm_add_epi16(line11_16a, line11_16b))), 2);

            // avg0, avg1, avg2, avg3, avg4, avg5, avg6, avg7, avg8, avg9, avg10, avg11, avg12, avg13, avg14, avg15
            const __m128i added = _mm_packus_epi16(add0, add1);

            _mm_storeu_si128(reinterpret_cast<__m128i *>(pTarget), added);

            pTarget += sizeof(__m128i);
            pSource += sizeof(__m128i) * 2;
            pSourceLine2 += sizeof(__m128i) * 2;
          }
        }

        for (; x < targetSubBufferSize.x; x++)
        {
          *pTarget = (uint8_t)(((uint16_t)pSource[0] + (uint16_t)pSource[1] + (uint16_t)pSourceLine2[0] + (uint16_t)pSourceLine2[1] + 2) >> 2);

          pTarget++;
          pSource += 2;
          pSourceLine2 += 2;
        }

        pSource = pSourceLine2;
      }
    }

    mRETURN_SUCCESS();
  }

  mFUNCTION(mPixelFormat_Transform_Yuv422ToYuv420, mPtr<mImageBuffer> &source, mPtr<mImageBuffer> &target, mPtr<mThreadPool> & /*asyncTaskHandler */)
  {
    mFUNCTION_SETUP();

    const uint8_t *pSource = source->pPixels;
    uint8_t *pTarget = target->pPixels;

    const mVec2s size = source->currentSize;

    if (pSource != pTarget)
      mMemcpy(pTarget, pSource, size.x * size.y);

    pSource += size.x * size.y;
    pTarget += size.x * size.y;

    const mVec2s targetSubBufferSize = size / 2;

    const __m128i mask00FF = _mm_set1_epi16(0x00FF);
    const __m128i one_epi16 = _mm_set1_epi16(1);

    for (size_t subBuffer = 0; subBuffer < 2; subBuffer++)
    {
      for (size_t y = 0; y < targetSubBufferSize.y; y++)
      {
        const uint8_t *pSourceLine2 = pSource + size.x / 2;

        size_t x = 0;

        if (targetSubBufferSize.x >= sizeof(__m128i))
        {
          for (; x < targetSubBufferSize.x - (sizeof(__m128i) - 1); x += sizeof(__m128i))
          {
            // Presume pSource/pSourceLine2 are
            // A B C D E F G H ...
            // a b c d e f g h ...

            // A B C D E F G H ...
            const __m128i line0 = _mm_loadu_si128(reinterpret_cast<const __m128i *>(pSource));
            // a b c d e f g h ...
            const __m128i line1 = _mm_loadu_si128(reinterpret_cast<const __m128i *>(pSourceLine2));

            // A   C   E   G   ...
            const __m128i line0_16a = _mm_and_si128(line0, mask00FF);
            // B   D   F   H   ...
            const __m128i line0_16b = _mm_srli_epi16(line0, 8);
            // a   c   e   g   ...
            const __m128i line1_16a = _mm_and_si128(line1, mask00FF);
            // b   d   f   h   ...
            const __m128i line1_16b = _mm_srli_epi16(line1, 8);

            // Calculate Averages & Round to the next integer.

            // (A + a + 1) / 2, 0, (C + c + 1) / 2, 0, (E + e + 1) / 2, 0, (G + g + 1) / 2, 0, ...
            // avg0, 0, avg2, 0, avg4, 0, avg6, 0, ...
            const __m128i add0 = _mm_srli_epi16(_mm_add_epi16(one_epi16, _mm_add_epi16(line0_16a, line1_16a)), 1);

            // (B + b + 1) / 2, 0, (D + d + 1) / 2, 0, (F + f + 1) / 2, 0, (H + h + 1) / 2, 0, ...
            // avg1, 0, avg3, 0, avg5, 0, avg7, 0, ...
            const __m128i add1 = _mm_srli_epi16(_mm_add_epi16(one_epi16, _mm_add_epi16(line0_16b, line1_16b)), 1);

            // avg0, avg1, avg2, avg3, avg4, avg5, avg6, avg7, ...
            const __m128i packed = _mm_packus_epi16(_mm_unpacklo_epi16(add0, add1), _mm_unpackhi_epi16(add0, add1));

            _mm_storeu_si128(reinterpret_cast<__m128i *>(pTarget), packed);

            pTarget += sizeof(__m128i);
            pSource += sizeof(__m128i);
            pSourceLine2 += sizeof(__m128i);
          }
        }

        for (; x < targetSubBufferSize.x; x++)
        {
          *pTarget = (uint8_t)(((uint16_t)*pSource + (uint16_t)*pSourceLine2 + 1) >> 1);

          pTarget++;
          pSource++;
          pSourceLine2++;
        }

        pSource = pSourceLine2;
      }
    }

    mRETURN_SUCCESS();
  }

  mFUNCTION(mPixelFormat_Transform_Yuv440ToYuv420, mPtr<mImageBuffer> &source, mPtr<mImageBuffer> &target, mPtr<mThreadPool> & /*asyncTaskHandler */)
  {
    mFUNCTION_SETUP();

    const uint8_t *pSource = source->pPixels;
    uint8_t *pTarget = target->pPixels;

    const mVec2s size = source->currentSize;

    if (pSource != pTarget)
      mMemcpy(pTarget, pSource, size.x * size.y);

    pSource += size.x * size.y;
    pTarget += size.x * size.y;

    const mVec2s targetSubBufferSize = size / 2;

    const __m128i mask00FF = _mm_set1_epi16(0x00FF);
    const __m128i one_epi16 = _mm_set1_epi16(1);

    for (size_t subBuffer = 0; subBuffer < 2; subBuffer++)
    {
      for (size_t y = 0; y < targetSubBufferSize.y; y++)
      {
        size_t x = 0;

        if (targetSubBufferSize.x >= sizeof(__m128i))
        {
          for (; x < targetSubBufferSize.x - (sizeof(__m128i) - 1); x += sizeof(__m128i))
          {
            // Presume pSource is
            // A B C D E F G H A B C D E F G H I J K L M N O P I J K L M N O P

            // A B C D E F G H ...
            const __m128i values0 = _mm_loadu_si128(reinterpret_cast<const __m128i *>(pSource));

            // I J K L M N O P ...
            const __m128i values1 = _mm_loadu_si128(reinterpret_cast<const __m128i *>(pSource) + 1);

            // A   C   E   G   ...
            const __m128i values0_16a = _mm_and_si128(values0, mask00FF);
            // B   D   F   H   ...
            const __m128i values0_16b = _mm_srli_epi16(values0, 8);
            // I   K   M   O   ...
            const __m128i values1_16a = _mm_and_si128(values1, mask00FF);
            // J   L   N   P   ...
            const __m128i values1_16b = _mm_srli_epi16(values1, 8);

            // Calculate Averages & Round to the next integer.

            // (A + B + 1) / 2, 0, (C + D + 1) / 2, 0, (E + F + 1) / 2, 0, (G + H + 1) / 2, 0
            // avg0, 0, avg1, 0, avg2, 0, avg3, 0, ...
            const __m128i add0 = _mm_srli_epi16(_mm_add_epi16(one_epi16, _mm_add_epi16(values0_16a, values0_16b)), 1);

            // (I + J + 1) / 2, 0, (K + L + 1) / 2, 0, (M + N + 1) / 2, 0, (O + P + 1) / 2, 0
            // avg8, 0, avg9, 0, avg10, 0, avg11, 0, ...
            const __m128i add1 = _mm_srli_epi16(_mm_add_epi16(one_epi16, _mm_add_epi16(values1_16a, values1_16b)), 1);

            // avg0, avg1, avg2, avg3, avg4, avg5, avg6, avg7, avg8, avg9, avg10, avg11, avg12, avg13, avg14, avg15
            const __m128i added = _mm_packus_epi16(add0, add1);

            _mm_storeu_si128(reinterpret_cast<__m128i *>(pTarget), added);

            pTarget += sizeof(__m128i);
            pSource += sizeof(__m128i) * 2;
          }
        }

        for (; x < targetSubBufferSize.x; x++)
        {
          *pTarget = (uint8_t)(((uint16_t)pSource[0] + (uint16_t)pSource[1] + 1) >> 1);

          pTarget++;
          pSource += 2;
        }
      }
    }

    mRETURN_SUCCESS();
  }

  mFUNCTION(mPixelFormat_Transform_Yuv411ToYuv420, mPtr<mImageBuffer> &source, mPtr<mImageBuffer> &target, mPtr<mThreadPool> & /*asyncTaskHandler */)
  {
    mFUNCTION_SETUP();

    const uint8_t *pSource = source->pPixels;
    uint8_t *pTarget = target->pPixels;

    const mVec2s size = source->currentSize;

    if (pSource != pTarget)
      mMemcpy(pTarget, pSource, size.x * size.y);

    pSource += size.x * size.y;
    pTarget += size.x * size.y;

    const mVec2s targetSubBufferSize = size / 2;

    const __m128i mask00FF = _mm_set1_epi16(0x00FF);
    const __m128i one_epi16 = _mm_set1_epi16(1);

    for (size_t subBuffer = 0; subBuffer < 2; subBuffer++)
    {
      for (size_t y = 0; y < targetSubBufferSize.y; y++)
      {
        const uint8_t *pSourceLine2 = pSource + size.x / 4;

        size_t x = 0;

        if (targetSubBufferSize.x >= sizeof(__m128i) * 2)
        {
          for (; x < targetSubBufferSize.x - (sizeof(__m128i) * 2 - 1); x += sizeof(__m128i) * 2)
          {
            // Presume pSource/pSourceLine2 are
            // A B C D E F G H ...
            // a b c d e f g h ...

            // A B C D E F G H ...
            const __m128i line0 = _mm_loadu_si128(reinterpret_cast<const __m128i *>(pSource));
            // a b c d e f g h ...
            const __m128i line1 = _mm_loadu_si128(reinterpret_cast<const __m128i *>(pSourceLine2));

            // A   C   E   G   ...
            const __m128i line0_16a = _mm_and_si128(line0, mask00FF);
            // B   D   F   H   ...
            const __m128i line0_16b = _mm_srli_epi16(line0, 8);
            // a   c   e   g   ...
            const __m128i line1_16a = _mm_and_si128(line1, mask00FF);
            // b   d   f   h   ...
            const __m128i line1_16b = _mm_srli_epi16(line1, 8);

            // Calculate Averages & Round to the next integer.

            // (A + a + 1) / 2, 0, (C + c + 1) / 2, 0, (E + e + 1) / 2, 0, (G + g + 1) / 2, 0, ...
            // avg0, 0, avg2, 0, avg4, 0, avg6, 0, ...
            const __m128i add0 = _mm_srli_epi16(_mm_add_epi16(one_epi16, _mm_add_epi16(line0_16a, line1_16a)), 1);

            // (B + b + 1) / 2, 0, (D + d + 1) / 2, 0, (F + f + 1) / 2, 0, (H + h + 1) / 2, 0, ...
            // avg1, 0, avg3, 0, avg5, 0, avg7, 0, ...
            const __m128i add1 = _mm_srli_epi16(_mm_add_epi16(one_epi16, _mm_add_epi16(line0_16b, line1_16b)), 1);

            // avg0, avg1, avg2, avg3, avg4, avg5, avg6, avg7, avg8, avg9, avg10, avg11, avg12, avg13, avg14, avg15
            const __m128i packed = _mm_packus_epi16(_mm_unpacklo_epi16(add0, add1), _mm_unpackhi_epi16(add0, add1));

            // avg0, avg0, avg1, avg1, avg2, avg2, avg3, avg3, avg4, avg4, avg5, avg5, avg6, avg6, avg7, avg7
            const __m128i packed0 = _mm_unpacklo_epi8(packed, packed);

            // avg8, avg8, avg9, avg9, avg10, avg10, avg11, avg11, avg12, avg12, avg13, avg13, avg14, avg14, avg15, avg15
            const __m128i packed1 = _mm_unpackhi_epi8(packed, packed);

            _mm_storeu_si128(reinterpret_cast<__m128i *>(pTarget), packed0);
            _mm_storeu_si128(reinterpret_cast<__m128i *>(pTarget) + 1, packed1);

            pTarget += sizeof(__m128i) * 2;
            pSource += sizeof(__m128i);
            pSourceLine2 += sizeof(__m128i);
          }
        }

        for (; x < targetSubBufferSize.x - 1; x += 2)
        {
          pTarget[0] = pTarget[1] = (uint8_t)(((uint16_t)*pSource + (uint16_t)*pSourceLine2 + 1) >> 1);

          pTarget += 2;
          pSource++;
          pSourceLine2++;
        }

        if (x < targetSubBufferSize.x)
        {
          if (x > 0)
            *pTarget = *(pTarget - 1);
          else
            *pTarget = 0x7F;

          pTarget++;
        }

        pSource = pSourceLine2;
      }
    }

    mRETURN_SUCCESS();
  }

  mFUNCTION(mPixelFormat_Transform_YuvXXXToMonochrome8, mPtr<mImageBuffer> &source, mPtr<mImageBuffer> &target, mPtr<mThreadPool> & /*asyncTaskHandler */)
  {
    mFUNCTION_SETUP();

    const uint8_t *pSource = source->pPixels;
    uint8_t *pTarget = target->pPixels;

    const mVec2s size = source->currentSize;

    if (pSource != pTarget)
      mMemcpy(pTarget, pSource, size.x * size.y);

    mRETURN_SUCCESS();
  }

  mFUNCTION(mPixelFormat_Transform_Monochrome8ToYuvXXX, mPtr<mImageBuffer> &source, mPtr<mImageBuffer> &target, const mPixelFormat targetPixelFormat, mPtr<mThreadPool> & /*asyncTaskHandler */)
  {
    mFUNCTION_SETUP();

    const uint8_t *pSource = source->pPixels;
    uint8_t *pTarget = target->pPixels;

    const mVec2s size = source->currentSize;

    if (pSource != pTarget)
      mMemcpy(pTarget, pSource, size.x * size.y);

    pTarget += size.x * size.y;

    mVec2s subBuffer1Size, subBuffer2Size;
    mERROR_CHECK(mPixelFormat_GetSubBufferSize(targetPixelFormat, 1, size, &subBuffer1Size));
    mERROR_CHECK(mPixelFormat_GetSubBufferSize(targetPixelFormat, 2, size, &subBuffer2Size));

    mMemset(pTarget, subBuffer1Size.x * subBuffer1Size.y + subBuffer2Size.x * subBuffer2Size.y, 0x7F);

    mRETURN_SUCCESS();
  }
} // end namespace mPixelFormat_Transform

//////////////////////////////////////////////////////////////////////////

mFUNCTION(mPixelFormat_TransformBuffer, mPtr<mImageBuffer> &source, mPtr<mImageBuffer> &target, mPtr<mThreadPool> &asyncTaskHandler)
{
  using namespace mPixelFormat_Transform;

  mFUNCTION_SETUP();

  mERROR_IF(source->pPixels == nullptr || target->pPixels == nullptr, mR_NotInitialized);

  if (source->pixelFormat == target->pixelFormat)
  {
    mERROR_CHECK(mImageBuffer_CopyTo(source, target, mImageBuffer_CopyFlags::mIB_CF_None));

    mRETURN_SUCCESS();
  }

  mERROR_IF(source->currentSize != target->currentSize, mR_InvalidParameter);
  mERROR_IF(source->allocatedSize == 0, mR_InvalidParameter);

  switch (target->pixelFormat)
  {
  case mPF_B8G8R8A8:
  {
    switch (source->pixelFormat)
    {
    case mPF_YUV420:
    {
      mERROR_CHECK(mPixelFormat_Transform_YUV420ToBgra(source, target, asyncTaskHandler));
      break;
    }

    case mPF_YUV422:
    {
      mERROR_CHECK(mPixelFormat_Transform_YUV422ToBgra(source, target, asyncTaskHandler));
      break;
    }

    case mPF_YUV444:
    {
      mERROR_CHECK(mPixelFormat_Transform_YUV444ToBgra(source, target, asyncTaskHandler));
      break;
    }

    case mPF_R8G8B8A8:
    {
      mERROR_CHECK(mPixelFormat_Transform_RgbaToBgra(source, target, asyncTaskHandler));
      break;
    }

    default:
    {
      mRETURN_RESULT(mR_NotImplemented);
    }
    }

    break;
  }

  //////////////////////////////////////////////////////////////////////////

  case mPF_R8G8B8A8:
  {
    switch (source->pixelFormat)
    {
    case mPF_B8G8R8A8:
    {
      mERROR_CHECK(mPixelFormat_Transform_BgraToRgba(source, target, asyncTaskHandler));
      break;
    }

    default:
    {
      mRETURN_RESULT(mR_NotImplemented);
    }
    }

    break;
  }

  //////////////////////////////////////////////////////////////////////////

  case mPF_B8G8R8:
  {
    switch (source->pixelFormat)
    {
    case mPF_R8G8B8:
      mERROR_CHECK(mPixelFormat_Transform_BgrToRgb(source, target, asyncTaskHandler));
      break;

    default:
      mRETURN_RESULT(mR_NotImplemented);
    }

    break;
  }
  
  //////////////////////////////////////////////////////////////////////////
  
  case mPF_R8G8B8:
  {
    switch (source->pixelFormat)
    {
    case mPF_B8G8R8:
      mERROR_CHECK(mPixelFormat_Transform_RgbToBgr(source, target, asyncTaskHandler));
      break;

    default:
      mRETURN_RESULT(mR_NotImplemented);
    }

    break;
  }

  //////////////////////////////////////////////////////////////////////////

  case mPF_YUV420:
  {
    switch (source->pixelFormat)
    {
    case mPF_YUV444:
      mERROR_CHECK(mPixelFormat_Transform_Yuv444ToYuv420(source, target, asyncTaskHandler));
      break;

    case mPF_YUV422:
      mERROR_CHECK(mPixelFormat_Transform_Yuv422ToYuv420(source, target, asyncTaskHandler));
      break;
    
    case mPF_YUV440:
      mERROR_CHECK(mPixelFormat_Transform_Yuv440ToYuv420(source, target, asyncTaskHandler));
      break;
    
    case mPF_YUV411:
      mERROR_CHECK(mPixelFormat_Transform_Yuv411ToYuv420(source, target, asyncTaskHandler));
      break;

    case mPF_Monochrome8:
      mERROR_CHECK(mPixelFormat_Transform_Monochrome8ToYuvXXX(source, target, target->pixelFormat, asyncTaskHandler));
      break;

    default:
      mRETURN_RESULT(mR_NotImplemented);
    }

    break;
  }

  //////////////////////////////////////////////////////////////////////////

  case mPF_YUV444:
  {
    switch (source->pixelFormat)
    {
    case mPF_Monochrome8:
      mERROR_CHECK(mPixelFormat_Transform_Monochrome8ToYuvXXX(source, target, target->pixelFormat, asyncTaskHandler));
      break;

    default:
      mRETURN_RESULT(mR_NotImplemented);
    }

    break;
  }

  //////////////////////////////////////////////////////////////////////////

  case mPF_YUV422:
  {
    switch (source->pixelFormat)
    {
    case mPF_Monochrome8:
      mERROR_CHECK(mPixelFormat_Transform_Monochrome8ToYuvXXX(source, target, target->pixelFormat, asyncTaskHandler));
      break;

    default:
      mRETURN_RESULT(mR_NotImplemented);
    }

    break;
  }

  //////////////////////////////////////////////////////////////////////////

  case mPF_YUV440:
  {
    switch (source->pixelFormat)
    {
    case mPF_Monochrome8:
      mERROR_CHECK(mPixelFormat_Transform_Monochrome8ToYuvXXX(source, target, target->pixelFormat, asyncTaskHandler));
      break;

    default:
      mRETURN_RESULT(mR_NotImplemented);
    }

    break;
  }

  //////////////////////////////////////////////////////////////////////////

  case mPF_YUV411:
  {
    switch (source->pixelFormat)
    {
    case mPF_Monochrome8:
      mERROR_CHECK(mPixelFormat_Transform_Monochrome8ToYuvXXX(source, target, target->pixelFormat, asyncTaskHandler));
      break;

    default:
      mRETURN_RESULT(mR_NotImplemented);
    }

    break;
  }

  //////////////////////////////////////////////////////////////////////////

  case mPF_Monochrome8:
  {
    switch (source->pixelFormat)
    {
    case mPF_YUV444:
    case mPF_YUV422:
    case mPF_YUV440:
    case mPF_YUV420:
    case mPF_YUV411:
      mERROR_CHECK(mPixelFormat_Transform_YuvXXXToMonochrome8(source, target, asyncTaskHandler));
      break;

    default:
      mRETURN_RESULT(mR_NotImplemented);
    }

    break;
  }

  //////////////////////////////////////////////////////////////////////////

  case mPF_Monochrome16:
  case mPF_Monochromef16:
  case mPF_Monochromef32:
  case mPF_R8G8:
  case mPF_R16G16:
  case mPF_Rf16Gf16:
  case mPF_Rf32Gf32:
  case mPF_R16G16B16:
  case mPF_Rf16Gf16Bf16:
  case mPF_Rf32Gf32Bf32:
  case mPF_R16G16B16A16:
  case mPF_Rf16Gf16Bf16Af16:
  case mPF_Rf32Gf32Bf32Af32:
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

mFUNCTION(mPixelFormat_TransformBuffer, mPtr<mImageBuffer> &source, mPtr<mImageBuffer> &target)
{
  mFUNCTION_SETUP();

  mPtr<mThreadPool> nullThreadPool = nullptr;
  mERROR_CHECK(mPixelFormat_TransformBuffer(source, target, nullThreadPool));

  mRETURN_SUCCESS();
}

mFUNCTION(mPixelFormat_CanInplaceTransform, const mPixelFormat source, const mPixelFormat target, OUT bool *pCanInplaceTransform)
{
  mFUNCTION_SETUP();

  mERROR_IF(pCanInplaceTransform == nullptr, mR_ArgumentNull);

  if (source == target)
  {
    *pCanInplaceTransform = true;
  }
  else
  {
    bool subSamplingSource = false;
    mERROR_CHECK(mPixelFormat_IsChromaSubsampled(source, &subSamplingSource));

    bool subSamplingTarget = false;
    mERROR_CHECK(mPixelFormat_IsChromaSubsampled(target, &subSamplingTarget));

    if (subSamplingSource != subSamplingTarget)
    {
      *pCanInplaceTransform = (subSamplingSource && target == mPF_Monochrome8) || (subSamplingTarget && source == mPF_Monochrome8);
    }
    else if (subSamplingSource) // both are chroma subsampled.
    {
      mVec2s sourceChromaSubBufferSize, targetChromaSubBufferSize;
      mERROR_CHECK(mPixelFormat_GetSubBufferSize(source, 1, mVec2s(16, 16), &sourceChromaSubBufferSize));
      mERROR_CHECK(mPixelFormat_GetSubBufferSize(target, 1, mVec2s(16, 16), &targetChromaSubBufferSize));

      *pCanInplaceTransform = (targetChromaSubBufferSize.x <= sourceChromaSubBufferSize.x && targetChromaSubBufferSize.y <= sourceChromaSubBufferSize.y);
    }
    else // none are chroma subsampled.
    {
      size_t sourceUnitSize, targetUnitSize;
      mERROR_CHECK(mPixelFormat_GetUnitSize(source, &sourceUnitSize));
      mERROR_CHECK(mPixelFormat_GetUnitSize(target, &targetUnitSize));

      *pCanInplaceTransform = (sourceUnitSize >= targetUnitSize);
    }
  }

  mRETURN_SUCCESS();
}

mFUNCTION(mPixelFormat_InplaceTransformBuffer, mPtr<mImageBuffer> &source, const mPixelFormat target)
{
  mFUNCTION_SETUP();

  mPtr<mThreadPool> nullThreadPool = nullptr;
  mERROR_CHECK(mPixelFormat_InplaceTransformBuffer(source, target, nullThreadPool));

  mRETURN_SUCCESS();
}

mFUNCTION(mPixelFormat_InplaceTransformBuffer, mPtr<mImageBuffer> &source, const mPixelFormat target, mPtr<mThreadPool> &asyncTaskHandler)
{
  mFUNCTION_SETUP();

  mERROR_IF(source == nullptr, mR_ArgumentNull);
  mERROR_IF(source->allocatedSize == 0, mR_InvalidParameter);
  mERROR_IF(!source->ownedResource, mR_ResourceIncompatible);

  bool canTechnicallyInplaceTransform = false;
  mERROR_CHECK(mPixelFormat_CanInplaceTransform(source->pixelFormat, target, &canTechnicallyInplaceTransform));
  mERROR_IF(!canTechnicallyInplaceTransform, mR_ResourceIncompatible);

  size_t requiredSize = 0;
  mERROR_CHECK(mPixelFormat_GetSize(target, source->currentSize, &requiredSize));

  if (requiredSize > source->allocatedSize)
  {
    const mResult result = mAllocator_Reallocate(source->pAllocator, &source->pPixels, requiredSize);
    
    if (mFAILED(result))
    {
      source->allocatedSize = 0;
      source->currentSize = mVec2s(0);

      mRETURN_RESULT(result);
    }

    source->allocatedSize = requiredSize;
  }

  using namespace mPixelFormat_Transform;

  switch (target)
  {
  case mPF_B8G8R8:
  {
    switch (source->pixelFormat)
    {
    case mPF_R8G8B8:
      mERROR_CHECK(mPixelFormat_Transform_RgbToBgr(source, source, asyncTaskHandler));
      break;

    default:
      mRETURN_RESULT(mR_NotImplemented);
    }

    break;
  }

  //////////////////////////////////////////////////////////////////////////

  case mPF_R8G8B8:
  {
    switch (source->pixelFormat)
    {
    case mPF_B8G8R8:
      mERROR_CHECK(mPixelFormat_Transform_BgrToRgb(source, source, asyncTaskHandler));
      break;

    default:
      mRETURN_RESULT(mR_NotImplemented);
    }

    break;
  }

  //////////////////////////////////////////////////////////////////////////

  case mPF_R8G8B8A8:
  {
    switch (source->pixelFormat)
    {
    case mPF_B8G8R8A8:
      mERROR_CHECK(mPixelFormat_Transform_BgraToRgba(source, source, asyncTaskHandler));
      break;

    default:
      mRETURN_RESULT(mR_NotImplemented);
    }

    break;
  }

  //////////////////////////////////////////////////////////////////////////

  case mPF_B8G8R8A8:
  {
    switch (source->pixelFormat)
    {
    case mPF_B8G8R8A8:
      mERROR_CHECK(mPixelFormat_Transform_RgbaToBgra(source, source, asyncTaskHandler));
      break;

    default:
      mRETURN_RESULT(mR_NotImplemented);
    }

    break;
  }

  //////////////////////////////////////////////////////////////////////////

  case mPF_YUV420:
  {
    switch (source->pixelFormat)
    {
    case mPF_YUV444:
      mERROR_CHECK(mPixelFormat_Transform_Yuv444ToYuv420(source, source, asyncTaskHandler));
      break;

    case mPF_YUV422:
      mERROR_CHECK(mPixelFormat_Transform_Yuv422ToYuv420(source, source, asyncTaskHandler));
      break;

    case mPF_YUV440:
      mERROR_CHECK(mPixelFormat_Transform_Yuv440ToYuv420(source, source, asyncTaskHandler));
      break;

    case mPF_Monochrome8:
      mERROR_CHECK(mPixelFormat_Transform_Monochrome8ToYuvXXX(source, source, target, asyncTaskHandler));
      break;

    default:
      mRETURN_RESULT(mR_NotImplemented);
    }

    break;
  }

  //////////////////////////////////////////////////////////////////////////

  case mPF_YUV444:
  {
    switch (source->pixelFormat)
    {
    case mPF_Monochrome8:
      mERROR_CHECK(mPixelFormat_Transform_Monochrome8ToYuvXXX(source, source, target, asyncTaskHandler));
      break;

    default:
      mRETURN_RESULT(mR_NotImplemented);
    }

    break;
  }

  //////////////////////////////////////////////////////////////////////////

  case mPF_YUV422:
  {
    switch (source->pixelFormat)
    {
    case mPF_Monochrome8:
      mERROR_CHECK(mPixelFormat_Transform_Monochrome8ToYuvXXX(source, source, target, asyncTaskHandler));
      break;

    default:
      mRETURN_RESULT(mR_NotImplemented);
    }

    break;
  }

  //////////////////////////////////////////////////////////////////////////

  case mPF_YUV440:
  {
    switch (source->pixelFormat)
    {
    case mPF_Monochrome8:
      mERROR_CHECK(mPixelFormat_Transform_Monochrome8ToYuvXXX(source, source, target, asyncTaskHandler));
      break;

    default:
      mRETURN_RESULT(mR_NotImplemented);
    }

    break;
  }

  //////////////////////////////////////////////////////////////////////////

  case mPF_YUV411:
  {
    switch (source->pixelFormat)
    {
    case mPF_Monochrome8:
      mERROR_CHECK(mPixelFormat_Transform_Monochrome8ToYuvXXX(source, source, target, asyncTaskHandler));
      break;

    default:
      mRETURN_RESULT(mR_NotImplemented);
    }

    break;
  }

  //////////////////////////////////////////////////////////////////////////

  case mPF_Monochrome8:
  {
    switch (source->pixelFormat)
    {
    case mPF_YUV444:
    case mPF_YUV422:
    case mPF_YUV440:
    case mPF_YUV420:
    case mPF_YUV411:
      mERROR_CHECK(mPixelFormat_Transform_YuvXXXToMonochrome8(source, source, asyncTaskHandler));
      break;

    default:
      mRETURN_RESULT(mR_NotImplemented);
    }

    break;
  }

  //////////////////////////////////////////////////////////////////////////

  default:
    mRETURN_RESULT(mR_NotImplemented);
  }

  source->pixelFormat = target;

  mRETURN_SUCCESS();
}
