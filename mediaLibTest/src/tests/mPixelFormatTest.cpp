#include "mTestLib.h"
#include "mImageBuffer.h"

mTEST(mPixelFormat, TestConvertYuv444ToYuv420)
{
  mTEST_ALLOCATOR_SETUP();

  const mVec2s size = mVec2s(40, 6);
  const mVec2s halfSize = size / 2;

  mPtr<mImageBuffer> source;
  mTEST_ASSERT_SUCCESS(mImageBuffer_Create(&source, pAllocator, size, mPF_YUV444));
  mTEST_ASSERT_EQUAL(mPF_YUV444, source->pixelFormat);
  mTEST_ASSERT_EQUAL(size.x * size.y * 3, source->allocatedSize);

  mPtr<mImageBuffer> target;
  mTEST_ASSERT_SUCCESS(mImageBuffer_Create(&target, pAllocator, size, mPF_YUV420));
  mTEST_ASSERT_EQUAL(mPF_YUV420, target->pixelFormat);
  mTEST_ASSERT_EQUAL(size.x * size.y * 3 / 2, target->allocatedSize);

  // Write Test Data.
  {
    uint8_t *pSource = source->pPixels;

    for (size_t channel = 0; channel < 3; channel++)
    {
      for (size_t i = 0; i < size.x * size.y; i++)
        pSource[i] = (uint8_t)i;

      pSource += size.x * size.y;
    }
  }

  mTEST_ASSERT_SUCCESS(mPixelFormat_TransformBuffer(source, target));

  // Check Transformed Data.
  {
    size_t subBufferCount;
    mTEST_ASSERT_SUCCESS(mPixelFormat_GetSubBufferCount(source->pixelFormat, &subBufferCount));
    mTEST_ASSERT_EQUAL(3, subBufferCount);

    mTEST_ASSERT_SUCCESS(mPixelFormat_GetSubBufferCount(target->pixelFormat, &subBufferCount));
    mTEST_ASSERT_EQUAL(3, subBufferCount);

    mVec2s subBufferSize;

    for (size_t channel = 0; channel < 3; channel++)
    {
      mTEST_ASSERT_SUCCESS(mPixelFormat_GetSubBufferSize(source->pixelFormat, channel, size, &subBufferSize));
      mTEST_ASSERT_EQUAL(size, subBufferSize);
    }

    mTEST_ASSERT_SUCCESS(mPixelFormat_GetSubBufferSize(target->pixelFormat, 0, size, &subBufferSize));
    mTEST_ASSERT_EQUAL(size, subBufferSize);

    for (size_t channel = 1; channel < 3; channel++)
    {
      mTEST_ASSERT_SUCCESS(mPixelFormat_GetSubBufferSize(target->pixelFormat, channel, size, &subBufferSize));
      mTEST_ASSERT_EQUAL(halfSize, subBufferSize);
    }

    uint8_t *pTarget = target->pPixels;

    for (size_t i = 0; i < size.x * size.y; i++)
      mTEST_ASSERT_EQUAL(pTarget[i], (uint8_t)i);

    pTarget += size.x * size.y;

    for (size_t channel = 1; channel < 3; channel++)
    {
      for (size_t y = 0; y < halfSize.y; y++)
        for (size_t x = 0; x < halfSize.x; x++)
          mTEST_ASSERT_EQUAL(pTarget[y * subBufferSize.x + x], (uint8_t)mRound(((uint16_t)(y * 2 * size.x + x * 2) + (uint16_t)(y * 2 * size.x + x * 2 + 2) + (uint16_t)((y * 2 + 1) * size.x + x * 2) + (uint16_t)((y * 2 + 1) * size.x + x * 2 + 1)) / 4.0f));

      pTarget += halfSize.x * halfSize.y;
    }
  }

  bool canInplaceTransform = false;
  mTEST_ASSERT_SUCCESS(mPixelFormat_CanInplaceTransform(source->pixelFormat, target->pixelFormat, &canInplaceTransform));
  mTEST_ASSERT_TRUE(canInplaceTransform);

  mTEST_ASSERT_SUCCESS(mPixelFormat_InplaceTransformBuffer(source, target->pixelFormat));
  mTEST_ASSERT_EQUAL(source->pixelFormat, target->pixelFormat);

  {
    uint8_t *pSource = source->pPixels;

    for (size_t i = 0; i < size.x * size.y; i++)
      mTEST_ASSERT_EQUAL(pSource[i], (uint8_t)i);

    pSource += size.x * size.y;

    for (size_t channel = 1; channel < 3; channel++)
    {
      for (size_t y = 0; y < halfSize.y; y++)
        for (size_t x = 0; x < halfSize.x; x++)
          mTEST_ASSERT_EQUAL(pSource[y * halfSize.x + x], (uint8_t)mRound(((uint16_t)(y * 2 * size.x + x * 2) + (uint16_t)(y * 2 * size.x + x * 2 + 2) + (uint16_t)((y * 2 + 1) * size.x + x * 2) + (uint16_t)((y * 2 + 1) * size.x + x * 2 + 1)) / 4.0f));

      pSource += halfSize.x * halfSize.y;
    }
  }

  mTEST_ALLOCATOR_ZERO_CHECK();
}

mTEST(mPixelFormat, TestConvertYuv422ToYuv420)
{
  mTEST_ALLOCATOR_SETUP();

  const mVec2s size = mVec2s(40, 6);
  const mVec2s halfXSize = mVec2s(size.x / 2, size.y);
  const mVec2s halfSize = size / 2;

  mPtr<mImageBuffer> source;
  mTEST_ASSERT_SUCCESS(mImageBuffer_Create(&source, pAllocator, size, mPF_YUV422));
  mTEST_ASSERT_EQUAL(mPF_YUV422, source->pixelFormat);
  mTEST_ASSERT_EQUAL(size.x * size.y * 2, source->allocatedSize);

  mPtr<mImageBuffer> target;
  mTEST_ASSERT_SUCCESS(mImageBuffer_Create(&target, pAllocator, size, mPF_YUV420));
  mTEST_ASSERT_EQUAL(mPF_YUV420, target->pixelFormat);
  mTEST_ASSERT_EQUAL(size.x * size.y * 3 / 2, target->allocatedSize);

  // Write Test Data.
  {
    uint8_t *pSource = source->pPixels;

    for (size_t i = 0; i < size.x * size.y; i++)
      pSource[i] = (uint8_t)i;

    pSource += size.x * size.y;

    for (size_t channel = 1; channel < 3; channel++)
    {
      for (size_t i = 0; i < halfXSize.x * halfXSize.y; i++)
        pSource[i] = (uint8_t)i;

      pSource += halfXSize.x * halfXSize.y;
    }
  }

  mTEST_ASSERT_SUCCESS(mPixelFormat_TransformBuffer(source, target));

  // Check Transformed Data.
  {
    size_t subBufferCount;
    mTEST_ASSERT_SUCCESS(mPixelFormat_GetSubBufferCount(source->pixelFormat, &subBufferCount));
    mTEST_ASSERT_EQUAL(3, subBufferCount);

    mTEST_ASSERT_SUCCESS(mPixelFormat_GetSubBufferCount(target->pixelFormat, &subBufferCount));
    mTEST_ASSERT_EQUAL(3, subBufferCount);

    mVec2s subBufferSize;
    mTEST_ASSERT_SUCCESS(mPixelFormat_GetSubBufferSize(source->pixelFormat, 0, size, &subBufferSize));
    mTEST_ASSERT_EQUAL(size, subBufferSize);

    for (size_t channel = 1; channel < 3; channel++)
    {
      mTEST_ASSERT_SUCCESS(mPixelFormat_GetSubBufferSize(source->pixelFormat, channel, size, &subBufferSize));
      mTEST_ASSERT_EQUAL(halfXSize, subBufferSize);
    }

    mTEST_ASSERT_SUCCESS(mPixelFormat_GetSubBufferSize(target->pixelFormat, 0, size, &subBufferSize));
    mTEST_ASSERT_EQUAL(size, subBufferSize);

    for (size_t channel = 1; channel < 3; channel++)
    {
      mTEST_ASSERT_SUCCESS(mPixelFormat_GetSubBufferSize(target->pixelFormat, channel, size, &subBufferSize));
      mTEST_ASSERT_EQUAL(halfSize, subBufferSize);
    }

    uint8_t *pTarget = target->pPixels;

    for (size_t i = 0; i < size.x * size.y; i++)
      mTEST_ASSERT_EQUAL(pTarget[i], (uint8_t)i);

    pTarget += size.x * size.y;

    for (size_t channel = 1; channel < 3; channel++)
    {
      for (size_t y = 0; y < halfSize.y; y++)
        for (size_t x = 0; x < halfSize.x; x++)
          mTEST_ASSERT_EQUAL(pTarget[y * subBufferSize.x + x], (uint8_t)mRound(((uint16_t)(y * 2 * halfXSize.x + x) + (uint16_t)((y * 2 + 1) * halfXSize.x + x)) / 2.0f));

      pTarget += halfSize.x * halfSize.y;
    }
  }

  bool canInplaceTransform = false;
  mTEST_ASSERT_SUCCESS(mPixelFormat_CanInplaceTransform(source->pixelFormat, target->pixelFormat, &canInplaceTransform));
  mTEST_ASSERT_TRUE(canInplaceTransform);

  mTEST_ASSERT_SUCCESS(mPixelFormat_InplaceTransformBuffer(source, target->pixelFormat));
  mTEST_ASSERT_EQUAL(source->pixelFormat, target->pixelFormat);

  {
    uint8_t *pSource = source->pPixels;

    for (size_t i = 0; i < size.x * size.y; i++)
      mTEST_ASSERT_EQUAL(pSource[i], (uint8_t)i);

    pSource += size.x * size.y;

    for (size_t channel = 1; channel < 3; channel++)
    {
      for (size_t y = 0; y < halfSize.y; y++)
        for (size_t x = 0; x < halfSize.x; x++)
          mTEST_ASSERT_EQUAL(pSource[y * halfSize.x + x], (uint8_t)mRound(((uint16_t)(y * 2 * halfXSize.x + x) + (uint16_t)((y * 2 + 1) * halfXSize.x + x)) / 2.0f));

      pSource += halfSize.x * halfSize.y;
    }
  }

  mTEST_ALLOCATOR_ZERO_CHECK();
}

mTEST(mPixelFormat, TestConvertYuv440ToYuv420)
{
  mTEST_ALLOCATOR_SETUP();

  const mVec2s size = mVec2s(40, 6);
  const mVec2s halfYSize = mVec2s(size.x, size.y / 2);
  const mVec2s halfSize = size / 2;

  mPtr<mImageBuffer> source;
  mTEST_ASSERT_SUCCESS(mImageBuffer_Create(&source, pAllocator, size, mPF_YUV440));
  mTEST_ASSERT_EQUAL(mPF_YUV440, source->pixelFormat);
  mTEST_ASSERT_EQUAL(size.x * size.y * 2, source->allocatedSize);

  mPtr<mImageBuffer> target;
  mTEST_ASSERT_SUCCESS(mImageBuffer_Create(&target, pAllocator, size, mPF_YUV420));
  mTEST_ASSERT_EQUAL(mPF_YUV420, target->pixelFormat);
  mTEST_ASSERT_EQUAL(size.x * size.y * 3 / 2, target->allocatedSize);

  // Write Test Data.
  {
    uint8_t *pSource = source->pPixels;

    for (size_t i = 0; i < size.x * size.y; i++)
      pSource[i] = (uint8_t)i;

    pSource += size.x * size.y;

    for (size_t channel = 1; channel < 3; channel++)
    {
      for (size_t i = 0; i < halfYSize.x * halfYSize.y; i++)
        pSource[i] = (uint8_t)i;

      pSource += halfYSize.x * halfYSize.y;
    }
  }

  mTEST_ASSERT_SUCCESS(mPixelFormat_TransformBuffer(source, target));

  // Check Transformed Data.
  {
    size_t subBufferCount;
    mTEST_ASSERT_SUCCESS(mPixelFormat_GetSubBufferCount(source->pixelFormat, &subBufferCount));
    mTEST_ASSERT_EQUAL(3, subBufferCount);

    mTEST_ASSERT_SUCCESS(mPixelFormat_GetSubBufferCount(target->pixelFormat, &subBufferCount));
    mTEST_ASSERT_EQUAL(3, subBufferCount);

    mVec2s subBufferSize;
    mTEST_ASSERT_SUCCESS(mPixelFormat_GetSubBufferSize(source->pixelFormat, 0, size, &subBufferSize));
    mTEST_ASSERT_EQUAL(size, subBufferSize);

    for (size_t channel = 1; channel < 3; channel++)
    {
      mTEST_ASSERT_SUCCESS(mPixelFormat_GetSubBufferSize(source->pixelFormat, channel, size, &subBufferSize));
      mTEST_ASSERT_EQUAL(halfYSize, subBufferSize);
    }

    mTEST_ASSERT_SUCCESS(mPixelFormat_GetSubBufferSize(target->pixelFormat, 0, size, &subBufferSize));
    mTEST_ASSERT_EQUAL(size, subBufferSize);

    for (size_t channel = 1; channel < 3; channel++)
    {
      mTEST_ASSERT_SUCCESS(mPixelFormat_GetSubBufferSize(target->pixelFormat, channel, size, &subBufferSize));
      mTEST_ASSERT_EQUAL(halfSize, subBufferSize);
    }

    uint8_t *pTarget = target->pPixels;

    for (size_t i = 0; i < size.x * size.y; i++)
      mTEST_ASSERT_EQUAL(pTarget[i], (uint8_t)i);

    pTarget += size.x * size.y;

    for (size_t channel = 1; channel < 3; channel++)
    {
      for (size_t y = 0; y < halfSize.y; y++)
        for (size_t x = 0; x < halfSize.x; x++)
          mTEST_ASSERT_EQUAL(pTarget[y * subBufferSize.x + x], (uint8_t)mRound(((uint16_t)(y * halfYSize.x + x * 2) + (uint16_t)(y * halfYSize.x + x * 2 + 1)) / 2.0f));

      pTarget += halfSize.x * halfSize.y;
    }
  }

  bool canInplaceTransform = false;
  mTEST_ASSERT_SUCCESS(mPixelFormat_CanInplaceTransform(source->pixelFormat, target->pixelFormat, &canInplaceTransform));
  mTEST_ASSERT_TRUE(canInplaceTransform);

  mTEST_ASSERT_SUCCESS(mPixelFormat_InplaceTransformBuffer(source, target->pixelFormat));
  mTEST_ASSERT_EQUAL(source->pixelFormat, target->pixelFormat);

  {
    uint8_t *pSource = source->pPixels;

    for (size_t i = 0; i < size.x * size.y; i++)
      mTEST_ASSERT_EQUAL(pSource[i], (uint8_t)i);

    pSource += size.x * size.y;

    for (size_t channel = 1; channel < 3; channel++)
    {
      for (size_t y = 0; y < halfSize.y; y++)
        for (size_t x = 0; x < halfSize.x; x++)
          mTEST_ASSERT_EQUAL(pSource[y * halfSize.x + x], (uint8_t)mRound(((uint16_t)(y * halfYSize.x + x * 2) + (uint16_t)(y * halfYSize.x + x * 2 + 1)) / 2.0f));

      pSource += halfSize.x * halfSize.y;
    }
  }

  mTEST_ALLOCATOR_ZERO_CHECK();
}

mTEST(mPixelFormat, TestConvertYuv411ToYuv420)
{
  mTEST_ALLOCATOR_SETUP();

  const mVec2s size = mVec2s(40, 6);
  const mVec2s quarterXSize = mVec2s(size.x / 4, size.y);
  const mVec2s halfSize = size / 2;

  mPtr<mImageBuffer> source;
  mTEST_ASSERT_SUCCESS(mImageBuffer_Create(&source, pAllocator, size, mPF_YUV411));
  mTEST_ASSERT_EQUAL(mPF_YUV411, source->pixelFormat);
  mTEST_ASSERT_EQUAL(size.x * size.y * 3 / 2, source->allocatedSize);

  mPtr<mImageBuffer> target;
  mTEST_ASSERT_SUCCESS(mImageBuffer_Create(&target, pAllocator, size, mPF_YUV420));
  mTEST_ASSERT_EQUAL(mPF_YUV420, target->pixelFormat);
  mTEST_ASSERT_EQUAL(size.x * size.y * 3 / 2, target->allocatedSize);

  // Write Test Data.
  {
    uint8_t *pSource = source->pPixels;

    for (size_t i = 0; i < size.x * size.y; i++)
      pSource[i] = (uint8_t)i;

    pSource += size.x * size.y;

    for (size_t channel = 1; channel < 3; channel++)
    {
      for (size_t i = 0; i < quarterXSize.x * quarterXSize.y; i++)
        pSource[i] = (uint8_t)i;

      pSource += quarterXSize.x * quarterXSize.y;
    }
  }

  mTEST_ASSERT_SUCCESS(mPixelFormat_TransformBuffer(source, target));

  // Check Transformed Data.
  {
    size_t subBufferCount;
    mTEST_ASSERT_SUCCESS(mPixelFormat_GetSubBufferCount(source->pixelFormat, &subBufferCount));
    mTEST_ASSERT_EQUAL(3, subBufferCount);

    mTEST_ASSERT_SUCCESS(mPixelFormat_GetSubBufferCount(target->pixelFormat, &subBufferCount));
    mTEST_ASSERT_EQUAL(3, subBufferCount);

    mVec2s subBufferSize;
    mTEST_ASSERT_SUCCESS(mPixelFormat_GetSubBufferSize(source->pixelFormat, 0, size, &subBufferSize));
    mTEST_ASSERT_EQUAL(size, subBufferSize);

    for (size_t channel = 1; channel < 3; channel++)
    {
      mTEST_ASSERT_SUCCESS(mPixelFormat_GetSubBufferSize(source->pixelFormat, channel, size, &subBufferSize));
      mTEST_ASSERT_EQUAL(quarterXSize, subBufferSize);
    }

    mTEST_ASSERT_SUCCESS(mPixelFormat_GetSubBufferSize(target->pixelFormat, 0, size, &subBufferSize));
    mTEST_ASSERT_EQUAL(size, subBufferSize);

    for (size_t channel = 1; channel < 3; channel++)
    {
      mTEST_ASSERT_SUCCESS(mPixelFormat_GetSubBufferSize(target->pixelFormat, channel, size, &subBufferSize));
      mTEST_ASSERT_EQUAL(halfSize, subBufferSize);
    }

    uint8_t *pTarget = target->pPixels;

    for (size_t i = 0; i < size.x * size.y; i++)
      mTEST_ASSERT_EQUAL(pTarget[i], (uint8_t)i);

    pTarget += size.x * size.y;

    for (size_t channel = 1; channel < 3; channel++)
    {
      for (size_t y = 0; y < halfSize.y; y++)
      {
        for (size_t x = 0; x < halfSize.x; x += 2)
        {
          const uint8_t expectedValue = (uint8_t)mRound(((uint16_t)(y * 2 * quarterXSize.x + x / 2) + (uint16_t)((y * 2 + 1) * quarterXSize.x + x / 2)) / 2.0f);

          mTEST_ASSERT_EQUAL(pTarget[y * subBufferSize.x + x], expectedValue);
          mTEST_ASSERT_EQUAL(pTarget[y * subBufferSize.x + x + 1], expectedValue);
        }
      }

      pTarget += halfSize.x * halfSize.y;
    }
  }

  bool canInplaceTransform = true;
  mTEST_ASSERT_SUCCESS(mPixelFormat_CanInplaceTransform(source->pixelFormat, target->pixelFormat, &canInplaceTransform));
  mTEST_ASSERT_FALSE(canInplaceTransform);

  mTEST_ALLOCATOR_ZERO_CHECK();
}
