#include "mTestLib.h"
#include "mAudio.h"

template <size_t size, size_t offset>
mFUNCTION(TestApplyVolume)
{
  mFUNCTION_SETUP();

  constexpr float_t volume = 0.24f;
  
  mALIGN(32) uint8_t f[sizeof(float_t *) * size + offset];
  float_t *pData = reinterpret_cast<float_t *>(f + offset);

  for (size_t i = 0; i < size; i++)
    pData[i] = (float_t)i;

  mERROR_CHECK(mAudio_ApplyVolumeFloat(pData, volume, size));

  for (size_t i = 0; i < size; i++)
    mERROR_IF(i * volume != pData[i], mR_Failure);

  mRETURN_SUCCESS();
}

mTEST(mAudio, ApplyVolumeSmall)
{
  mTEST_ASSERT_SUCCESS((TestApplyVolume<2, 1>()));

  if (mCpuExtensions::avxSupported)
  {
    mResult result;

    {
      mDEFER(mCpuExtensions::avxSupported = true);
      mCpuExtensions::avxSupported = false;

      result = TestApplyVolume<2, 1>();
    }

    mTEST_ASSERT_SUCCESS(result);
  }

  mTEST_RETURN_SUCCESS();
}
mTEST(mAudio, ApplyVolumeAligned)
{
  mTEST_ASSERT_SUCCESS((TestApplyVolume<1024, 0>()));

  if (mCpuExtensions::avxSupported)
  {
    mResult result;

    {
      mDEFER(mCpuExtensions::avxSupported = true);
      mCpuExtensions::avxSupported = false;

      result = TestApplyVolume<1024, 0>();
    }

    mTEST_ASSERT_SUCCESS(result);
  }

  mTEST_RETURN_SUCCESS();
}

mTEST(mAudio, ApplyVolumeFloatAligned)
{
  mTEST_ASSERT_SUCCESS((TestApplyVolume<1021, sizeof(float_t)>()));

  if (mCpuExtensions::avxSupported)
  {
    mResult result;

    {
      mDEFER(mCpuExtensions::avxSupported = true);
      mCpuExtensions::avxSupported = false;

      result = TestApplyVolume<1021, sizeof(float_t)>();
    }

    mTEST_ASSERT_SUCCESS(result);
  }

  mTEST_RETURN_SUCCESS();
}

mTEST(mAudio, ApplyVolumeFloatUnaligned)
{
  mTEST_ASSERT_SUCCESS((TestApplyVolume<1021, 1>()));

  if (mCpuExtensions::avxSupported)
  {
    mResult result;

    {
      mDEFER(mCpuExtensions::avxSupported = true);
      mCpuExtensions::avxSupported = false;

      result = TestApplyVolume<1021, 1>();
    }

    mTEST_ASSERT_SUCCESS(result);
  }

  mTEST_RETURN_SUCCESS();
}

template <size_t channelCount, size_t sampleCount>
mFUNCTION(TestSetInterleavedChannelFloat)
{
  mFUNCTION_SETUP();

  float_t interleaved[sampleCount * channelCount];
  mERROR_CHECK(mZeroMemory(interleaved, mARRAYSIZE(interleaved)));

  float_t channelFactor[channelCount];

  for (size_t channelIndex = 0; channelIndex < channelCount; channelIndex++)
    channelFactor[channelIndex] = mPow(-1.f, channelIndex) * (1 + channelIndex / 2);

  for (size_t channelIndex = 0; channelIndex < channelCount; channelIndex++)
  {
    float_t channel[sampleCount];

    for (size_t sampleIndex = 0; sampleIndex < sampleCount; sampleIndex++)
      channel[sampleIndex] = (float_t)sampleIndex * channelFactor[channelIndex];

    mERROR_CHECK(mAudio_SetInterleavedChannelFloat(interleaved, channel, channelIndex, channelCount, sampleCount));
  }

  for (size_t sampleIndex = 0; sampleIndex < sampleCount; sampleIndex++)
  {
    float_t *pSample = &interleaved[sampleIndex * channelCount];

    for (size_t channelIndex = 0; channelIndex < channelCount; channelIndex++)
      mERROR_IF(pSample[channelIndex] != sampleIndex * channelFactor[channelIndex], mR_Failure);
  }

  mRETURN_SUCCESS();
}

mTEST(mAudio, SetInterleavedChannelFloatMono)
{
  mTEST_ASSERT_SUCCESS((TestSetInterleavedChannelFloat<1, 1023>()));
  mTEST_RETURN_SUCCESS();
}

mTEST(mAudio, SetInterleavedChannelFloatStereo)
{
  mTEST_ASSERT_SUCCESS((TestSetInterleavedChannelFloat<2, 1023>()));
  mTEST_RETURN_SUCCESS();
}

mTEST(mAudio, SetInterleavedChannelFloatQuadro)
{
  mTEST_ASSERT_SUCCESS((TestSetInterleavedChannelFloat<4, 1023>()));
  mTEST_RETURN_SUCCESS();
}

template <size_t channelCount, size_t sampleCount>
mFUNCTION(TestAddToInterleavedFromChannelFloat)
{
  mFUNCTION_SETUP();

  constexpr float_t volume = 0.64f;
  constexpr float_t initialValueVolume = 0.13f;
  
  float_t channelFactor[channelCount];
  float_t interleaved[sampleCount * channelCount];

  for (size_t channelIndex = 0; channelIndex < channelCount; channelIndex++)
    channelFactor[channelIndex] = mPow(-1.f, channelIndex) * (1 + channelIndex / 2);

  for (size_t sampleIndex = 0; sampleIndex < sampleCount; sampleIndex++)
  {
    float_t *pSample = &interleaved[sampleIndex * channelCount];

    for (size_t channelIndex = 0; channelIndex < channelCount; channelIndex++)
      pSample[channelIndex] = (float_t)(sampleCount - sampleIndex) * channelFactor[channelIndex] * initialValueVolume;
  }

  for (size_t channelIndex = 0; channelIndex < channelCount; channelIndex++)
  {
    float_t channel[sampleCount];

    for (size_t sampleIndex = 0; sampleIndex < sampleCount; sampleIndex++)
      channel[sampleIndex] = (float_t)sampleIndex * channelFactor[channelIndex];

    mERROR_CHECK(mAudio_AddToInterleavedFromChannelWithVolumeFloat(interleaved, channel, channelIndex, channelCount, sampleCount, volume));
  }

  for (size_t sampleIndex = 0; sampleIndex < sampleCount; sampleIndex++)
  {
    float_t *pSample = &interleaved[sampleIndex * channelCount];

    for (size_t channelIndex = 0; channelIndex < channelCount; channelIndex++)
      mERROR_IF(pSample[channelIndex] != (float_t)(sampleCount - sampleIndex) * channelFactor[channelIndex] * initialValueVolume + volume * (sampleIndex * channelFactor[channelIndex]), mR_Failure);
  }

  mRETURN_SUCCESS();
}

mTEST(mAudio, TestAddToInterleavedFromChannelFloatMono)
{
  mTEST_ASSERT_SUCCESS((TestAddToInterleavedFromChannelFloat<1, 1023>()));
  mTEST_RETURN_SUCCESS();
}

mTEST(mAudio, TestAddToInterleavedFromChannelFloatStereo)
{
  mTEST_ASSERT_SUCCESS((TestAddToInterleavedFromChannelFloat<2, 1023>()));
  mTEST_RETURN_SUCCESS();
}

mTEST(mAudio, TestAddToInterleavedFromChannelFloatQuadro)
{
  mTEST_ASSERT_SUCCESS((TestAddToInterleavedFromChannelFloat<4, 1023>()));
  mTEST_RETURN_SUCCESS();
}

template <size_t channelCount, size_t sampleCount>
mFUNCTION(TestAddToInterleavedBufferFromMonoWithVolumeFloat)
{
  mFUNCTION_SETUP();

  constexpr float_t volume = 0.64f;
  constexpr float_t initialValueVolume = 0.13f;

  float_t channelFactor[channelCount];
  float_t interleaved[sampleCount * channelCount];

  for (size_t channelIndex = 0; channelIndex < channelCount; channelIndex++)
    channelFactor[channelIndex] = mPow(-1.f, channelIndex) * (1 + channelIndex / 2);

  for (size_t sampleIndex = 0; sampleIndex < sampleCount; sampleIndex++)
  {
    float_t *pSample = &interleaved[sampleIndex * channelCount];

    for (size_t channelIndex = 0; channelIndex < channelCount; channelIndex++)
      pSample[channelIndex] = (float_t)(sampleCount - sampleIndex) * channelFactor[channelIndex] * initialValueVolume;
  }

  {
    float_t channel[sampleCount];

    for (size_t sampleIndex = 0; sampleIndex < sampleCount; sampleIndex++)
      channel[sampleIndex] = (float_t)sampleIndex;

    mERROR_CHECK(mAudio_AddToInterleavedBufferFromMonoWithVolumeFloat(interleaved, channel, channelCount, sampleCount, volume));
  }

  for (size_t sampleIndex = 0; sampleIndex < sampleCount; sampleIndex++)
  {
    float_t *pSample = &interleaved[sampleIndex * channelCount];

    for (size_t channelIndex = 0; channelIndex < channelCount; channelIndex++)
      mERROR_IF(pSample[channelIndex] != (float_t)(sampleCount - sampleIndex) * channelFactor[channelIndex] * initialValueVolume + volume * sampleIndex, mR_Failure);
  }

  mRETURN_SUCCESS();
}

mTEST(mAudio, TestAddToInterleavedBufferFromMonoWithVolumeFloatMono)
{
  mTEST_ASSERT_SUCCESS((TestAddToInterleavedBufferFromMonoWithVolumeFloat<1, 1023>()));
  mTEST_RETURN_SUCCESS();
}

mTEST(mAudio, TestAddToInterleavedBufferFromMonoWithVolumeFloatStereo)
{
  mTEST_ASSERT_SUCCESS((TestAddToInterleavedBufferFromMonoWithVolumeFloat<2, 1023>()));
  mTEST_RETURN_SUCCESS();
}

mTEST(mAudio, TestAddToInterleavedBufferFromMonoWithVolumeFloatQuadro)
{
  mTEST_ASSERT_SUCCESS((TestAddToInterleavedBufferFromMonoWithVolumeFloat<4, 1023>()));
  mTEST_RETURN_SUCCESS();
}

template <size_t channelCount, size_t sampleCount>
mFUNCTION(TestExtractFloatChannelFromInterleavedInt16)
{
  mFUNCTION_SETUP();

  int16_t interleaved[sampleCount * channelCount];
  mERROR_CHECK(mZeroMemory(interleaved, mARRAYSIZE(interleaved)));

  float_t channelFactor[channelCount];

  for (size_t channelIndex = 0; channelIndex < channelCount; channelIndex++)
    channelFactor[channelIndex] = mPow(-1.f, channelIndex) * (1 + channelIndex / 2);

  for (size_t sampleIndex = 0; sampleIndex < sampleCount; sampleIndex++)
  {
    int16_t *pSample = &interleaved[sampleIndex * channelCount];

    for (size_t channelIndex = 0; channelIndex < channelCount; channelIndex++)
      pSample[channelIndex] = static_cast<int16_t>(roundf(mSin(sampleIndex * channelFactor[channelIndex]) * mMaxValue<int16_t>()));
  }

  const float_t range = 1.f / (float_t)mMaxValue<int16_t>();

  for (size_t channelIndex = 0; channelIndex < channelCount; channelIndex++)
  {
    float_t channel[sampleCount];

    mERROR_CHECK(mAudio_ExtractFloatChannelFromInterleavedInt16(channel, channelIndex, interleaved, channelCount, sampleCount));

    for (size_t sampleIndex = 0; sampleIndex < sampleCount; sampleIndex++)
      mERROR_IF(!mTEST_FLOAT_IN_RANGE(mSin(sampleIndex * channelFactor[channelIndex]), (float_t)channel[sampleIndex], range), mR_Failure);
  }

  mRETURN_SUCCESS();
}

mTEST(mAudio, TestExtractFloatChannelFromInterleavedInt16Mono)
{
  mTEST_ASSERT_SUCCESS((TestExtractFloatChannelFromInterleavedInt16<1, 1023>()));

  if (mCpuExtensions::sse41Supported)
  {
    mResult result;

    {
      mDEFER(mCpuExtensions::sse41Supported = true);
      mCpuExtensions::sse41Supported = false;

      result = TestExtractFloatChannelFromInterleavedInt16<1, 1023>();
    }

    mTEST_ASSERT_SUCCESS(result);
  }

  mTEST_RETURN_SUCCESS();
}

mTEST(mAudio, TestExtractFloatChannelFromInterleavedInt16Stereo)
{
  mTEST_ASSERT_SUCCESS((TestExtractFloatChannelFromInterleavedInt16<2, 1023>()));

  if (mCpuExtensions::sse41Supported)
  {
    mResult result;

    {
      mDEFER(mCpuExtensions::sse41Supported = true);
      mCpuExtensions::sse41Supported = false;

      result = TestExtractFloatChannelFromInterleavedInt16<2, 1023>();
    }

    mTEST_ASSERT_SUCCESS(result);
  }

  mTEST_RETURN_SUCCESS();
}

mTEST(mAudio, TestExtractFloatChannelFromInterleavedInt16Quadro)
{
  mTEST_ASSERT_SUCCESS((TestExtractFloatChannelFromInterleavedInt16<4, 1023>()));

  if (mCpuExtensions::sse41Supported)
  {
    mResult result;

    {
      mDEFER(mCpuExtensions::sse41Supported = true);
      mCpuExtensions::sse41Supported = false;

      result = TestExtractFloatChannelFromInterleavedInt16<4, 1023>();
    }

    mTEST_ASSERT_SUCCESS(result);
  }

  mTEST_RETURN_SUCCESS();
}

template <size_t channelCount, size_t sampleCount>
mFUNCTION(TestExtractFloatChannelFromInterleavedFloat)
{
  mFUNCTION_SETUP();

  float_t interleaved[sampleCount * channelCount];
  mERROR_CHECK(mZeroMemory(interleaved, mARRAYSIZE(interleaved)));

  float_t channelFactor[channelCount];

  for (size_t channelIndex = 0; channelIndex < channelCount; channelIndex++)
    channelFactor[channelIndex] = mPow(-1.f, channelIndex) * (1 + channelIndex / 2);

  for (size_t sampleIndex = 0; sampleIndex < sampleCount; sampleIndex++)
  {
    float_t *pSample = &interleaved[sampleIndex * channelCount];

    for (size_t channelIndex = 0; channelIndex < channelCount; channelIndex++)
      pSample[channelIndex] = mSin(sampleIndex * channelFactor[channelIndex]);
  }

  for (size_t channelIndex = 0; channelIndex < channelCount; channelIndex++)
  {
    float_t channel[sampleCount];

    mERROR_CHECK(mAudio_ExtractFloatChannelFromInterleavedFloat(channel, channelIndex, interleaved, channelCount, sampleCount));

    for (size_t sampleIndex = 0; sampleIndex < sampleCount; sampleIndex++)
      mERROR_IF(mSin(sampleIndex * channelFactor[channelIndex]) != (float_t)channel[sampleIndex], mR_Failure);
  }

  mRETURN_SUCCESS();
}

mTEST(mAudio, TestExtractFloatChannelFromInterleavedFloatMono)
{
  mTEST_ASSERT_SUCCESS((TestExtractFloatChannelFromInterleavedFloat<1, 1023>()));

  if (mCpuExtensions::avx2Supported)
  {
    mResult result;

    {
      mDEFER(mCpuExtensions::avx2Supported = true);
      mCpuExtensions::avx2Supported = false;

      result = TestExtractFloatChannelFromInterleavedFloat<1, 1023>();
    }

    mTEST_ASSERT_SUCCESS(result);
  }

  mTEST_RETURN_SUCCESS();
}

mTEST(mAudio, TestExtractFloatChannelFromInterleavedFloatStereo)
{
  mTEST_ASSERT_SUCCESS((TestExtractFloatChannelFromInterleavedFloat<2, 1023>()));

  if (mCpuExtensions::avx2Supported)
  {
    mResult result;

    {
      mDEFER(mCpuExtensions::avx2Supported = true);
      mCpuExtensions::avx2Supported = false;

      result = TestExtractFloatChannelFromInterleavedFloat<2, 1023>();
    }

    mTEST_ASSERT_SUCCESS(result);
  }

  mTEST_RETURN_SUCCESS();
}

mTEST(mAudio, TestExtractFloatChannelFromInterleavedFloatQuadro)
{
  mTEST_ASSERT_SUCCESS((TestExtractFloatChannelFromInterleavedFloat<4, 1023>()));

  if (mCpuExtensions::avx2Supported)
  {
    mResult result;

    {
      mDEFER(mCpuExtensions::avx2Supported = true);
      mCpuExtensions::avx2Supported = false;

      result = TestExtractFloatChannelFromInterleavedFloat<4, 1023>();
    }

    mTEST_ASSERT_SUCCESS(result);
  }

  mTEST_RETURN_SUCCESS();
}

mTEST(mAudio, ConvertInt16ToFloat)
{
  int16_t ibuffer[1023];
  float_t fbuffer[mARRAYSIZE(ibuffer)];

  for (size_t i = 0; i < mARRAYSIZE(ibuffer); i++)
    ibuffer[i] = (int16_t)i;

  mTEST_ASSERT_SUCCESS(mAudio_ConvertInt16ToFloat(fbuffer, ibuffer, mARRAYSIZE(ibuffer)));

  const float_t div = 1.f / (float_t)mMaxValue<int16_t>();

  for (size_t i = 0; i < mARRAYSIZE(ibuffer); i++)
    mTEST_ASSERT_EQUAL(fbuffer[i], (float_t)i * div);

  if (mCpuExtensions::sse41Supported)
  {
    mTEST_ASSERT_SUCCESS(mZeroMemory(fbuffer, mARRAYSIZE(fbuffer)));

    mResult result;

    {
      mDEFER(mCpuExtensions::sse41Supported = true);
      mCpuExtensions::sse41Supported = false;

      result = mAudio_ConvertInt16ToFloat(fbuffer, ibuffer, mARRAYSIZE(ibuffer));
    }

    mTEST_ASSERT_SUCCESS(result);

    for (size_t i = 0; i < mARRAYSIZE(ibuffer); i++)
      mTEST_ASSERT_EQUAL(fbuffer[i], (float_t)i * div);
  }

  if (mCpuExtensions::avx2Supported)
  {
    mTEST_ASSERT_SUCCESS(mZeroMemory(fbuffer, mARRAYSIZE(fbuffer)));

    mResult result;

    {
      mDEFER(mCpuExtensions::sse41Supported = true);
      mDEFER(mCpuExtensions::avx2Supported = true);
      mCpuExtensions::sse41Supported = false;
      mCpuExtensions::avx2Supported = false;

      result = mAudio_ConvertInt16ToFloat(fbuffer, ibuffer, mARRAYSIZE(ibuffer));
    }

    mTEST_ASSERT_SUCCESS(result);

    for (size_t i = 0; i < mARRAYSIZE(ibuffer); i++)
      mTEST_ASSERT_EQUAL(fbuffer[i], (float_t)i * div);
  }

  mTEST_RETURN_SUCCESS();
}

mTEST(mAudio, InplaceDecodeMidSideToStereo)
{
  constexpr size_t sampleCount = 1023;

  float_t left[sampleCount];
  float_t right[sampleCount];

  for (size_t i = 0; i < sampleCount; i++)
  {
    left[i] = (float_t)i;
    right[i] = mSin(-(float_t)i * .25f);
  }

  mTEST_ASSERT_SUCCESS(mAudio_InplaceMidSideToStereo(left, right, sampleCount));

  for (size_t i = 0; i < sampleCount; i++)
  {
    const float_t mid = (float_t)i;
    const float_t side = mSin(-(float_t)i * .25f) * .5f;

    mTEST_ASSERT_EQUAL(mid + side, left[i]);
    mTEST_ASSERT_EQUAL(mid - side, right[i]);
  }

  if (mCpuExtensions::avxSupported)
  {
    for (size_t i = 0; i < sampleCount; i++)
    {
      left[i] = (float_t)i;
      right[i] = mSin(-(float_t)i * .25f);
    }

    mResult result;

    {
      mDEFER(mCpuExtensions::avxSupported = true);
      mCpuExtensions::avxSupported = false;

      result = mAudio_InplaceMidSideToStereo(left, right, sampleCount);
    }

    mTEST_ASSERT_SUCCESS(result);

    for (size_t i = 0; i < sampleCount; i++)
    {
      const float_t mid = (float_t)i;
      const float_t side = mSin(-(float_t)i * .25f) * .5f;

      mTEST_ASSERT_EQUAL(mid + side, left[i]);
      mTEST_ASSERT_EQUAL(mid - side, right[i]);
    }
  }

  mTEST_RETURN_SUCCESS();
}

mTEST(mAudio, InplaceMidLateralLongitudinalToQuadro)
{
  constexpr size_t sampleCount = 1023;

  float_t frontLeft[sampleCount];
  float_t frontRight[sampleCount];
  float_t backLeft[sampleCount];
  float_t backRight[sampleCount];

  for (size_t i = 0; i < sampleCount; i++)
  {
    frontLeft[i] = (float_t)i;
    frontRight[i] = mSin(-(float_t)i * .25f);
    backLeft[i] = mSin((float_t)i * .125f);
  }

  mTEST_ASSERT_SUCCESS(mAudio_InplaceMidLateralLongitudinalToQuadro(frontLeft, frontRight, backLeft, backRight, sampleCount));

  for (size_t i = 0; i < sampleCount; i++)
  {
    const float_t mid = (float_t)i;
    const float_t sideA = mSin(-(float_t)i * .25f) * .5f;
    const float_t sideB = mSin((float_t)i * .125f) * .5f;

    mTEST_ASSERT_EQUAL(mid + sideA + sideB, frontLeft[i]);
    mTEST_ASSERT_EQUAL(mid - sideA + sideB, frontRight[i]);
    mTEST_ASSERT_EQUAL(mid + sideA - sideB, backLeft[i]);
    mTEST_ASSERT_EQUAL(mid - sideA - sideB, backRight[i]);
  }

  if (mCpuExtensions::avxSupported)
  {
    for (size_t i = 0; i < sampleCount; i++)
    {
      frontLeft[i] = (float_t)i;
      frontRight[i] = mSin(-(float_t)i * .25f);
      backLeft[i] = mSin((float_t)i * .125f);
    }

    mResult result;

    {
      mDEFER(mCpuExtensions::avxSupported = true);
      mCpuExtensions::avxSupported = false;

      result = mAudio_InplaceMidLateralLongitudinalToQuadro(frontLeft, frontRight, backLeft, backRight, sampleCount);
    }

    mTEST_ASSERT_SUCCESS(result);

    for (size_t i = 0; i < sampleCount; i++)
    {
      const float_t mid = (float_t)i;
      const float_t sideA = mSin(-(float_t)i * .25f) * .5f;
      const float_t sideB = mSin((float_t)i * .125f) * .5f;

      mTEST_ASSERT_EQUAL(mid + sideA + sideB, frontLeft[i]);
      mTEST_ASSERT_EQUAL(mid - sideA + sideB, frontRight[i]);
      mTEST_ASSERT_EQUAL(mid + sideA - sideB, backLeft[i]);
      mTEST_ASSERT_EQUAL(mid - sideA - sideB, backRight[i]);
    }
  }

  mTEST_RETURN_SUCCESS();
}

mTEST(mAudio, AddWithVolumeFloat)
{
  constexpr size_t sampleCount = 1023;
  const float_t volume = 0.125f;

  float_t src[sampleCount];
  float_t dst[sampleCount];

  for (size_t i = 0; i < sampleCount; i++)
  {
    dst[i] = mSin(i * 0.1f);
    src[i] = mSin(-(float_t)i * 0.25f);
  }

  mTEST_ASSERT_SUCCESS(mAudio_AddWithVolumeFloat(dst, src, volume, sampleCount));

  for (size_t i = 0; i < sampleCount; i++)
    mTEST_ASSERT_EQUAL(dst[i], mSin(i * 0.1f) + volume * mSin(-(float_t)i * 0.25f));
  
  if (mCpuExtensions::avxSupported)
  {
    for (size_t i = 0; i < sampleCount; i++)
      dst[i] = mSin(i * 0.1f);

    mResult result;

    {
      mDEFER(mCpuExtensions::avxSupported = true);
      mCpuExtensions::avxSupported = false;

      result = mAudio_AddWithVolumeFloat(dst, src, volume, sampleCount);
    }

    mTEST_ASSERT_SUCCESS(result);

    for (size_t i = 0; i < sampleCount; i++)
      mTEST_ASSERT_EQUAL(dst[i], mSin(i * 0.1f) + volume * mSin(-(float_t)i * 0.25f));

  }

  mTEST_RETURN_SUCCESS();
}

mTEST(mAudio, ConvertFloatToInt16)
{
  constexpr size_t sampleCount = 1023;
  const float_t volume = 0.125f;

  float_t src[sampleCount];
  int16_t dst[sampleCount];

  for (size_t i = 0; i < sampleCount; i++)
    src[i] = mSin(i * 0.01f);

  mTEST_ASSERT_SUCCESS(mAudio_ConvertFloatToInt16WithDithering(dst, src, sampleCount));

  for (size_t i = 0; i < sampleCount; i++)
  {
    mTEST_ASSERT_EQUAL(src[i], mSin(i * 0.01f));
    mTEST_ASSERT_TRUE(mAbs(dst[i] - mClamp((int16_t)roundf(src[i] * mMaxValue<int16_t>()), mMinValue<int16_t>(), mMaxValue<int16_t>())) <= 1);
  }

  if (mCpuExtensions::avx2Supported)
  {
    mTEST_ASSERT_SUCCESS(mZeroMemory(dst, mARRAYSIZE(dst)));

    mResult result;

    {
      mDEFER(mCpuExtensions::avx2Supported = true);
      mCpuExtensions::avx2Supported = false;

      result = mAudio_ConvertFloatToInt16WithDithering(dst, src, sampleCount);
    }

    mTEST_ASSERT_SUCCESS(result);

    for (size_t i = 0; i < sampleCount; i++)
    {
      mTEST_ASSERT_EQUAL(src[i], mSin(i * 0.01f));
      mTEST_ASSERT_TRUE(mAbs(dst[i] - (int16_t)roundf(mSin(i * 0.01f) * mMaxValue<int16_t>())) <= 1);
    }
  }
  
  if (mCpuExtensions::sse41Supported)
  {
    mTEST_ASSERT_SUCCESS(mZeroMemory(dst, mARRAYSIZE(dst)));

    mResult result;

    {
      mDEFER(mCpuExtensions::avx2Supported = true);
      mDEFER(mCpuExtensions::sse41Supported = true);
      mCpuExtensions::avx2Supported = false;
      mCpuExtensions::sse41Supported = false;

      result = mAudio_ConvertFloatToInt16WithDithering(dst, src, sampleCount);
    }

    mTEST_ASSERT_SUCCESS(result);

    for (size_t i = 0; i < sampleCount; i++)
    {
      mTEST_ASSERT_EQUAL(src[i], mSin(i * 0.01f));
      mTEST_ASSERT_TRUE(mAbs(dst[i] - (int16_t)roundf(mSin(i * 0.01f) * mMaxValue<int16_t>())) <= 1);
    }
  }

  mTEST_RETURN_SUCCESS();
}

mTEST(mAudio, InterleavedQuadroMidLateralLFELongitudinalToDualInterleavedStereo)
{
  constexpr size_t sampleCount = 1023;

  float_t inSamples[sampleCount * 4];
  float_t front[sampleCount * 2];
  float_t back[sampleCount * 2];

  for (size_t i = 0; i < sampleCount; i++)
  {
    inSamples[i * 4 + 0] = mSin(0.00101f * i);
    inSamples[i * 4 + 1] = mCos(0.00142f * i);
    inSamples[i * 4 + 2] = mSin(0.00321f * i);
    inSamples[i * 4 + 3] = mCos(0.00462f * i);
  }

  mTEST_ASSERT_SUCCESS(mAudio_InterleavedQuadroMidLateralLFELongitudinalToDualInterleavedStereo(inSamples, front, back, sampleCount));

  for (size_t i = 0; i < sampleCount; i++)
  {
    const float_t mid = inSamples[i * 4 + 0];
    const float_t lat = inSamples[i * 4 + 1];
    const float_t lfe = inSamples[i * 4 + 2];
    const float_t longd = inSamples[i * 4 + 3];

    mTEST_ASSERT_EQUAL(mid, mSin(0.00101f * i));
    mTEST_ASSERT_EQUAL(lat, mCos(0.00142f * i));
    mTEST_ASSERT_EQUAL(lfe, mSin(0.00321f * i));
    mTEST_ASSERT_EQUAL(longd, mCos(0.00462f * i));

    mTEST_ASSERT_EQUAL(front[i * 2 + 0], mid + lfe + lat * .5f + longd * .5f);
    mTEST_ASSERT_EQUAL(front[i * 2 + 1], mid + lfe - lat * .5f + longd * .5f);
    mTEST_ASSERT_EQUAL(back[i * 2 + 0], mid + lfe + lat * .5f - longd * .5f);
    mTEST_ASSERT_EQUAL(back[i * 2 + 1], mid + lfe - lat * .5f - longd * .5f);
  }

  mTEST_RETURN_SUCCESS();
}
