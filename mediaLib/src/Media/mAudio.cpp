#include "mAudio.h"

#include "mCachedFileReader.h"
#include "mProfiler.h"

#include "samplerate.h"

#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable: 4752)
#endif

#ifdef GIT_BUILD // Define __M_FILE__
  #ifdef __M_FILE__
    #undef __M_FILE__
  #endif
  #define __M_FILE__ "YB7sS5KIxXOP9uQiEc357/Krmb8HcPP8fm5dXyPXgickcXLGZ6lez92bt71hXzGqP9KGuQ36EFUzh8Hq"
#endif

static void mAudio_ExtractFloatChannelFromInterleavedFloat_AVX2(size_t &sampleIndex, const size_t sampleCount, OUT float_t *pChannel, const size_t channelIndex, IN float_t *pInterleaved)
{
  for (; sampleIndex < sampleCount - 7; sampleIndex += 8)
  {
    const __m256 srcLo = _mm256_loadu_ps(pInterleaved + channelIndex + 2 * sampleIndex);
    const __m256 srcHi = _mm256_loadu_ps(pInterleaved + channelIndex + 2 * sampleIndex + 8);

    const __m256 pack0 = _mm256_shuffle_ps(srcLo, srcHi, _MM_SHUFFLE(2, 0, 2, 0));
    const __m256 pack1 = _mm256_castsi256_ps(_mm256_permute4x64_epi64(_mm256_castps_si256(pack0), _MM_SHUFFLE(3, 1, 2, 0)));

    _mm256_storeu_ps(pChannel + sampleIndex, pack1);
  }
}

mFUNCTION(mAudio_ExtractFloatChannelFromInterleavedFloat, OUT float_t *pChannel, const size_t channelIndex, IN float_t *pInterleaved, const size_t channelCount, const size_t sampleCount)
{
  mFUNCTION_SETUP();

  mERROR_IF(pChannel == nullptr || pInterleaved == nullptr, mR_ArgumentNull);
  mERROR_IF(channelIndex >= channelCount, mR_InvalidParameter);

  if (channelCount > 2 || sampleCount * sizeof(float_t) < sizeof(__m128))
  {
    for (size_t sampleIndex = 0; sampleIndex < sampleCount; sampleIndex++)
      pChannel[sampleIndex] = pInterleaved[sampleIndex * channelCount + channelIndex];
  }
  else if (channelCount == 1)
  {
    mERROR_CHECK(mMemmove(pChannel, pInterleaved, sampleCount));
  }
  else // if (channelCount == 2)
  {
    size_t sampleIndex = 0;

    mCpuExtensions::Detect();

    if (mCpuExtensions::avx2Supported && sampleCount > 7)
    {
      mAudio_ExtractFloatChannelFromInterleavedFloat_AVX2(sampleIndex, sampleCount, pChannel, channelIndex, pInterleaved);
    }
    else if (sampleCount > 3)
    {
      for (; sampleIndex < sampleCount - 3; sampleIndex += 4)
      {
        const __m128 srcLo = _mm_loadu_ps(pInterleaved + channelIndex + 2 * sampleIndex);
        const __m128 srcHi = _mm_loadu_ps(pInterleaved + channelIndex + 2 * sampleIndex + 4);

        const __m128 pack = _mm_shuffle_ps(srcLo, srcHi, _MM_SHUFFLE(2, 0, 2, 0));

        _mm_storeu_ps(pChannel + sampleIndex, pack);
      }
    }

    for (; sampleIndex < sampleCount; sampleIndex++)
      pChannel[sampleIndex] = pInterleaved[sampleIndex * 2 + channelIndex];
  }

  mRETURN_SUCCESS();
}

static void mAudio_ExtractFloatChannelFromInterleavedInt16_SSE41(size_t &sample, OUT float_t *pChannel, IN int16_t *pInterleaved, const size_t sampleCount)
{
  const __m128 div = _mm_set1_ps(1.f / (float_t)(INT16_MAX));
  constexpr size_t loopSize = (sizeof(__m128i) / sizeof(int16_t));

  for (; sample < sampleCount - (loopSize - 1); sample += loopSize)
  {
    const __m128i src = _mm_loadu_si128(reinterpret_cast<__m128i *>(&pInterleaved[sample]));
    const __m128 _0 = _mm_cvtepi32_ps(_mm_cvtepi16_epi32(src));
    const __m128 _1 = _mm_cvtepi32_ps(_mm_cvtepi16_epi32(_mm_srli_si128(src, 8)));

    _mm_storeu_ps(&pChannel[sample], _mm_mul_ps(_0, div));
    _mm_storeu_ps(&pChannel[sample + 4], _mm_mul_ps(_1, div));
  }
}

mFUNCTION(mAudio_ExtractFloatChannelFromInterleavedInt16, OUT float_t *pChannel, const size_t channelIndex, IN int16_t *pInterleaved, const size_t channelCount, const size_t sampleCount)
{
  mFUNCTION_SETUP();

  mERROR_IF(pChannel == nullptr || pInterleaved == nullptr, mR_ArgumentNull);

  mCpuExtensions::Detect();

  if (mCpuExtensions::sse41Supported && channelCount <= 2 && sampleCount * sizeof(float_t) > sizeof(__m128))
  {
    size_t sample = 0;

    if (channelCount == 1)
    {
      mAudio_ExtractFloatChannelFromInterleavedInt16_SSE41(sample, pChannel, pInterleaved, sampleCount);
    }
    else
    {
      constexpr size_t loopSize = ((sizeof(__m128i) / 2) / sizeof(int16_t));
      const __m128 div = _mm_set1_ps(1.f / (float_t)(INT16_MAX));

      for (; sample < sampleCount - (loopSize - 1); sample += loopSize)
      {
        const __m128i src = _mm_loadu_si128(reinterpret_cast<__m128i *>(&pInterleaved[sample * channelCount + channelIndex]));
        const __m128 _0 = _mm_cvtepi32_ps(_mm_cvtepi16_epi32(src));
        const __m128 _1 = _mm_cvtepi32_ps(_mm_cvtepi16_epi32(_mm_srli_si128(src, 8)));
        const __m128 pack = _mm_shuffle_ps(_0, _1, _MM_SHUFFLE(2, 0, 2, 0));

        _mm_storeu_ps(&pChannel[sample], _mm_mul_ps(pack, div));
      }
    }

    for (; sample < sampleCount; sample++)
      pChannel[sample] = (float_t)pInterleaved[sample * channelCount + channelIndex] * (1.f / (float_t)(INT16_MAX));
  }
  else
  {
    for (size_t sample = 0; sample < sampleCount; sample++)
      pChannel[sample] = (float_t)pInterleaved[sample * channelCount + channelIndex] * (1.f / (float_t)(INT16_MAX));
  }

  mRETURN_SUCCESS();
}

static void mAudio_ConvertInt16ToFloat_AVX2(OUT float_t *pDestination, IN const int16_t *pSource, const size_t sampleCount)
{
  const float_t div = 1.f / mMaxValue<int16_t>();
  const __m256 mmdiv = _mm256_set1_ps(div);
  size_t sampleIndex = 0;

  for (; sampleIndex < sampleCount - 15; sampleIndex += 16)
  {
    const __m256i src = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(&pSource[sampleIndex]));
    const __m256 _0 = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_extracti128_si256(src, 0))), mmdiv);
    const __m256 _1 = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_extracti128_si256(src, 1))), mmdiv);
    _mm256_storeu_ps(pDestination + sampleIndex, _0);
    _mm256_storeu_ps(pDestination + sampleIndex + 8, _1);
  }

  for (; sampleIndex < sampleCount; sampleIndex++)
    pDestination[sampleIndex] = (float_t)pSource[sampleIndex] * div;
}

static void mAudio_ConvertInt16ToFloat_SSE41(OUT float_t *pDestination, IN const int16_t *pSource, const size_t sampleCount)
{
  const float_t div = 1.f / mMaxValue<int16_t>();
  const __m128 mmdiv = _mm_set1_ps(div);
  size_t sampleIndex = 0;

  for (; sampleIndex < sampleCount - 7; sampleIndex += 8)
  {
    const __m128i src = _mm_loadu_si128(reinterpret_cast<const __m128i *>(&pSource[sampleIndex]));
    const __m128 _0 = _mm_mul_ps(_mm_cvtepi32_ps(_mm_cvtepi16_epi32(src)), mmdiv);
    const __m128 _1 = _mm_mul_ps(_mm_cvtepi32_ps(_mm_cvtepi16_epi32(_mm_srli_si128(src, 8))), mmdiv);
    _mm_storeu_ps(pDestination + sampleIndex, _0);
    _mm_storeu_ps(pDestination + sampleIndex + 4, _1);
  }

  for (; sampleIndex < sampleCount; sampleIndex++)
    pDestination[sampleIndex] = (float_t)pSource[sampleIndex] * div;
}

mFUNCTION(mAudio_ConvertInt16ToFloat, OUT float_t *pDestination, IN const int16_t *pSource, const size_t sampleCount)
{
  mFUNCTION_SETUP();

  mERROR_IF(pDestination == nullptr || pSource == nullptr, mR_ArgumentNull);

  mCpuExtensions::Detect();

  if (mCpuExtensions::avx2Supported && sampleCount > 15)
  {
    mAudio_ConvertInt16ToFloat_AVX2(pDestination, pSource, sampleCount);
  }
  else if (mCpuExtensions::sse41Supported && sampleCount > 7)
  {
    mAudio_ConvertInt16ToFloat_SSE41(pDestination, pSource, sampleCount);
  }
  else
  {
    const float_t div = 1.f / mMaxValue<int16_t>();

    for (size_t sampleIndex = 0; sampleIndex < sampleCount; sampleIndex++)
      pDestination[sampleIndex] = (float_t)pSource[sampleIndex] * div;
  }

  mRETURN_SUCCESS();
}

static void mAudio_ConvertFloatToInt16WithDithering_AVX2(size_t &sampleIndex, IN int16_t *pDestination, OUT const float_t *pSource, const size_t sampleCount)
{
  const float_t mul = mMaxValue<int16_t>();

  typedef __m256 simd_t;
  typedef __m256i isimd_t;

  constexpr size_t loopSize = (sizeof(simd_t) * 2) / sizeof(float_t);
  constexpr size_t loopSizeHalf = loopSize / 2;

  const simd_t dither = _mm256_set_ps(.25f, -.25f, .25f, -.25f, .25f, -.25f, .25f, -.25f); // Fixed pattern dithering.
  const simd_t mmmul = _mm256_set1_ps(mul);
  const isimd_t min = _mm256_set1_epi32(mMinValue<int16_t>());
  const isimd_t max = _mm256_set1_epi32(mMaxValue<int16_t>());

  if (sampleCount >= loopSize)
  {
    for (; sampleIndex < sampleCount - (loopSize - 1); sampleIndex += loopSize)
    {
      const simd_t srcHi = _mm256_add_ps(dither, _mm256_mul_ps(_mm256_loadu_ps(pSource + sampleIndex), mmmul));
      const simd_t srcLo = _mm256_add_ps(dither, _mm256_mul_ps(_mm256_loadu_ps(pSource + sampleIndex + loopSizeHalf), mmmul));
      const isimd_t hi = _mm256_cvtps_epi32(srcHi);
      const isimd_t lo = _mm256_cvtps_epi32(srcLo);
      const isimd_t clampHi = _mm256_min_epi32(max, _mm256_max_epi32(min, hi));
      const isimd_t clampLo = _mm256_min_epi32(max, _mm256_max_epi32(min, lo));
      const isimd_t prePack = _mm256_packs_epi32(clampHi, clampLo);
      const isimd_t pack = _mm256_permute4x64_epi64(prePack, _MM_SHUFFLE(3, 1, 2, 0));
      _mm256_storeu_si256(reinterpret_cast<isimd_t *>(pDestination + sampleIndex), pack);
    }
  }

  for (; sampleIndex < sampleCount; sampleIndex++)
    pDestination[sampleIndex] = mClamp((int16_t)roundf(pSource[sampleIndex] * mul), mMinValue<int16_t>(), mMaxValue<int16_t>());
}

static void mAudio_ConvertFloatToInt16WithDithering_SSE41(size_t &sampleIndex, IN int16_t *pDestination, OUT const float_t *pSource, const size_t sampleCount)
{
  typedef __m128 simd_t;
  typedef __m128i isimd_t;

  constexpr size_t loopSize = (sizeof(simd_t) * 2) / sizeof(float_t);
  constexpr size_t loopSizeHalf = loopSize / 2;

  const float_t mul = mMaxValue<int16_t>();
  const simd_t mmmul = _mm_set1_ps(mul);

  const simd_t dither = _mm_set_ps(.25f, -.25f, .25f, -.25f); // Fixed pattern dithering.

  const isimd_t min = _mm_set1_epi32(mMinValue<int16_t>());
  const isimd_t max = _mm_set1_epi32(mMaxValue<int16_t>());

  for (; sampleIndex < sampleCount - (loopSize - 1); sampleIndex += loopSize)
  {
    const simd_t srcHi = _mm_add_ps(dither, _mm_mul_ps(_mm_loadu_ps(pSource + sampleIndex), mmmul));
    const simd_t srcLo = _mm_add_ps(dither, _mm_mul_ps(_mm_loadu_ps(pSource + sampleIndex + loopSizeHalf), mmmul));
    const isimd_t hi = _mm_cvtps_epi32(srcHi);
    const isimd_t lo = _mm_cvtps_epi32(srcLo);
    const isimd_t clampHi = _mm_min_epi32(max, _mm_max_epi32(min, hi));
    const isimd_t clampLo = _mm_min_epi32(max, _mm_max_epi32(min, lo));
    const isimd_t pack = _mm_packs_epi32(clampHi, clampLo);
    _mm_storeu_si128(reinterpret_cast<isimd_t *>(pDestination + sampleIndex), pack);
  }
}

mFUNCTION(mAudio_ConvertFloatToInt16WithDithering, IN int16_t *pDestination, OUT const float_t *pSource, const size_t sampleCount)
{
  mFUNCTION_SETUP();

  mCpuExtensions::Detect();

  size_t sampleIndex = 0;
  const float_t mul = mMaxValue<int16_t>();

  if (mCpuExtensions::avx2Supported)
  {
    mAudio_ConvertFloatToInt16WithDithering_AVX2(sampleIndex, pDestination, pSource, sampleCount);
  }
  else
  {
    typedef __m128 simd_t;
    typedef __m128i isimd_t;

    constexpr size_t loopSize = (sizeof(simd_t) * 2) / sizeof(float_t);
    constexpr size_t loopSizeHalf = loopSize / 2;

    const isimd_t packhi = _mm_set_epi8(-1, -1, -1, -1, -1, -1, -1, -1, 13, 12, 9, 8, 5, 4, 1, 0);
    const isimd_t packlo = _mm_set_epi8(13, 12, 9, 8, 5, 4, 1, 0, -1, -1, -1, -1, -1, -1, -1, -1);

    if (sampleCount >= loopSize)
    {
      if (mCpuExtensions::sse41Supported)
      {
        mAudio_ConvertFloatToInt16WithDithering_SSE41(sampleIndex, pDestination, pSource, sampleCount);
      }
      else
      {
        const simd_t mmmul = _mm_set1_ps(mul);
        const simd_t dither = _mm_set_ps(.25f, -.25f, .25f, -.25f); // Fixed pattern dithering.

        const simd_t min = _mm_set1_ps(-1);
        const simd_t max = _mm_set1_ps(1);

        for (; sampleIndex < sampleCount - (loopSize - 1); sampleIndex += loopSize)
        {
          const simd_t srcHi = _mm_add_ps(dither, _mm_mul_ps(_mm_min_ps(max, _mm_max_ps(min, _mm_loadu_ps(pSource + sampleIndex))), mmmul));
          const simd_t srcLo = _mm_add_ps(dither, _mm_mul_ps(_mm_min_ps(max, _mm_max_ps(min, _mm_loadu_ps(pSource + sampleIndex + loopSizeHalf))), mmmul));
          const isimd_t pack = _mm_packs_epi32(_mm_cvtps_epi32(srcHi), _mm_cvtps_epi32(srcLo));
          _mm_storeu_si128(reinterpret_cast<isimd_t *>(pDestination + sampleIndex), pack);
        }
      }
    }

    for (; sampleIndex < sampleCount; sampleIndex++)
      pDestination[sampleIndex] = mClamp((int16_t)roundf(pSource[sampleIndex] * mul), mMinValue<int16_t>(), mMaxValue<int16_t>());
  }

  mRETURN_SUCCESS();
}

static void mAudio_ConvertFloatToInt16WithDitheringAndFactor_AVX2(size_t &sampleIndex, IN int16_t *pDestination, OUT const float_t *pSource, const size_t sampleCount, const float_t factor)
{
  const float_t mul = mMaxValue<int16_t>() * factor;

  typedef __m256 simd_t;
  typedef __m256i isimd_t;

  constexpr size_t loopSize = (sizeof(simd_t) * 2) / sizeof(float_t);
  constexpr size_t loopSizeHalf = loopSize / 2;

  const simd_t dither = _mm256_set_ps(.25f, -.25f, .25f, -.25f, .25f, -.25f, .25f, -.25f); // Fixed pattern dithering.
  const simd_t mmmul = _mm256_set1_ps(mul);
  const isimd_t min = _mm256_set1_epi32(mMinValue<int16_t>());
  const isimd_t max = _mm256_set1_epi32(mMaxValue<int16_t>());

  if (sampleCount >= loopSize)
  {
    for (; sampleIndex < sampleCount - (loopSize - 1); sampleIndex += loopSize)
    {
      const simd_t srcHi = _mm256_add_ps(dither, _mm256_mul_ps(_mm256_loadu_ps(pSource + sampleIndex), mmmul));
      const simd_t srcLo = _mm256_add_ps(dither, _mm256_mul_ps(_mm256_loadu_ps(pSource + sampleIndex + loopSizeHalf), mmmul));
      const isimd_t hi = _mm256_cvtps_epi32(srcHi);
      const isimd_t lo = _mm256_cvtps_epi32(srcLo);
      const isimd_t clampHi = _mm256_min_epi32(max, _mm256_max_epi32(min, hi));
      const isimd_t clampLo = _mm256_min_epi32(max, _mm256_max_epi32(min, lo));
      const isimd_t prePack = _mm256_packs_epi32(clampHi, clampLo);
      const isimd_t pack = _mm256_permute4x64_epi64(prePack, _MM_SHUFFLE(3, 1, 2, 0));
      _mm256_storeu_si256(reinterpret_cast<isimd_t *>(pDestination + sampleIndex), pack);
    }
  }

  for (; sampleIndex < sampleCount; sampleIndex++)
    pDestination[sampleIndex] = mClamp((int16_t)roundf(pSource[sampleIndex] * mul), mMinValue<int16_t>(), mMaxValue<int16_t>());
}

static void mAudio_ConvertFloatToInt16WithDitheringAndFactor_SSE41(size_t &sampleIndex, IN int16_t *pDestination, OUT const float_t *pSource, const size_t sampleCount, const float_t factor)
{
  typedef __m128 simd_t;
  typedef __m128i isimd_t;

  constexpr size_t loopSize = (sizeof(simd_t) * 2) / sizeof(float_t);
  constexpr size_t loopSizeHalf = loopSize / 2;

  const float_t mul = mMaxValue<int16_t>() * factor;
  const simd_t mmmul = _mm_set1_ps(mul);

  const simd_t dither = _mm_set_ps(.25f, -.25f, .25f, -.25f); // Fixed pattern dithering.

  const isimd_t min = _mm_set1_epi32(mMinValue<int16_t>());
  const isimd_t max = _mm_set1_epi32(mMaxValue<int16_t>());

  for (; sampleIndex < sampleCount - (loopSize - 1); sampleIndex += loopSize)
  {
    const simd_t srcHi = _mm_add_ps(dither, _mm_mul_ps(_mm_loadu_ps(pSource + sampleIndex), mmmul));
    const simd_t srcLo = _mm_add_ps(dither, _mm_mul_ps(_mm_loadu_ps(pSource + sampleIndex + loopSizeHalf), mmmul));
    const isimd_t hi = _mm_cvtps_epi32(srcHi);
    const isimd_t lo = _mm_cvtps_epi32(srcLo);
    const isimd_t clampHi = _mm_min_epi32(max, _mm_max_epi32(min, hi));
    const isimd_t clampLo = _mm_min_epi32(max, _mm_max_epi32(min, lo));
    const isimd_t pack = _mm_packs_epi32(clampHi, clampLo);
    _mm_storeu_si128(reinterpret_cast<isimd_t *>(pDestination + sampleIndex), pack);
  }
}

mFUNCTION(mAudio_ConvertFloatToInt16WithDitheringAndFactor, IN int16_t *pDestination, OUT const float_t *pSource, const size_t sampleCount, const float_t factor)
{
  mFUNCTION_SETUP();

  mCpuExtensions::Detect();

  size_t sampleIndex = 0;

  if (mCpuExtensions::avx2Supported)
  {
    mAudio_ConvertFloatToInt16WithDitheringAndFactor_AVX2(sampleIndex, pDestination, pSource, sampleCount, factor);
  }
  else
  {
    typedef __m128 simd_t;
    typedef __m128i isimd_t;

    constexpr size_t loopSize = (sizeof(simd_t) * 2) / sizeof(float_t);
    constexpr size_t loopSizeHalf = loopSize / 2;

    const float_t mul = mMaxValue<int16_t>() * factor;

    const isimd_t packhi = _mm_set_epi8(-1, -1, -1, -1, -1, -1, -1, -1, 13, 12, 9, 8, 5, 4, 1, 0);
    const isimd_t packlo = _mm_set_epi8(13, 12, 9, 8, 5, 4, 1, 0, -1, -1, -1, -1, -1, -1, -1, -1);

    if (sampleCount >= loopSize)
    {
      if (mCpuExtensions::sse41Supported)
      {
        mAudio_ConvertFloatToInt16WithDitheringAndFactor_SSE41(sampleIndex, pDestination, pSource, sampleCount, factor);
      }
      else
      {
        const simd_t mmmul = _mm_set1_ps(mul);
        const simd_t dither = _mm_set_ps(.25f, -.25f, .25f, -.25f); // Fixed pattern dithering.

        const simd_t min = _mm_set1_ps(-1);
        const simd_t max = _mm_set1_ps(1);

        for (; sampleIndex < sampleCount - (loopSize - 1); sampleIndex += loopSize)
        {
          const simd_t srcHi = _mm_add_ps(dither, _mm_mul_ps(_mm_min_ps(max, _mm_max_ps(min, _mm_loadu_ps(pSource + sampleIndex))), mmmul));
          const simd_t srcLo = _mm_add_ps(dither, _mm_mul_ps(_mm_min_ps(max, _mm_max_ps(min, _mm_loadu_ps(pSource + sampleIndex + loopSizeHalf))), mmmul));
          const isimd_t pack = _mm_packs_epi32(_mm_cvtps_epi32(srcHi), _mm_cvtps_epi32(srcLo));
          _mm_storeu_si128(reinterpret_cast<isimd_t *>(pDestination + sampleIndex), pack);
        }
      }
    }

    for (; sampleIndex < sampleCount; sampleIndex++)
      pDestination[sampleIndex] = mClamp((int16_t)roundf(pSource[sampleIndex] * mul), mMinValue<int16_t>(), mMaxValue<int16_t>());
  }

  mRETURN_SUCCESS();
}

mFUNCTION(mAudio_SetInterleavedChannelFloat, OUT float_t *pInterleaved, IN float_t *pChannel, const size_t channelIndex, const size_t channelCount, const size_t sampleCount)
{
  mFUNCTION_SETUP();

  mERROR_IF(pInterleaved == nullptr || pChannel == nullptr, mR_ArgumentNull);
  mERROR_IF(channelIndex >= channelCount, mR_InvalidParameter);
  mERROR_IF(channelCount == 0, mR_InvalidParameter);

  if (channelCount == 1)
  {
    mERROR_CHECK(mMemmove(pInterleaved, pChannel, sampleCount));
  }
  else if (channelCount == 2 && sampleCount * sizeof(float_t) > sizeof(__m128))
  {
    size_t sample = 0;

    const __m128 mask = _mm_castsi128_ps(_mm_set_epi32(0, -1, 0, -1));

    if (channelIndex == 0)
    {
      for (; sample < sampleCount - (4 - 1); sample += 4)
      {
        const __m128 src = _mm_loadu_ps(&pChannel[sample]);
        const __m128 lo = _mm_and_ps(mask, _mm_shuffle_ps(src, src, _MM_SHUFFLE(0, 1, 0, 0)));
        const __m128 hi = _mm_and_ps(mask, _mm_shuffle_ps(src, src, _MM_SHUFFLE(0, 3, 0, 2)));

        float_t *pLo = &pInterleaved[channelIndex + sample * 2];
        float_t *pHi = &pInterleaved[channelIndex + (sample + 2) * 2];

        _mm_storeu_ps(pLo, lo);
        _mm_storeu_ps(pHi, hi);
      }
    }
    else
    {
      for (; sample < sampleCount - (4 - 1); sample += 4)
      {
        const __m128 src = _mm_loadu_ps(&pChannel[sample]);
        const __m128 lo = _mm_and_ps(mask, _mm_shuffle_ps(src, src, _MM_SHUFFLE(0, 1, 0, 0)));
        const __m128 hi = _mm_and_ps(mask, _mm_shuffle_ps(src, src, _MM_SHUFFLE(0, 3, 0, 2)));

        float_t *pLo = &pInterleaved[channelIndex + sample * 2];
        float_t *pHi = &pInterleaved[channelIndex + (sample + 2) * 2];

        _mm_storeu_ps(pLo, _mm_or_ps(lo, _mm_loadu_ps(pLo)));
        _mm_storeu_ps(pHi, _mm_or_ps(hi, _mm_loadu_ps(pHi)));
      }
    }

    for (; sample < sampleCount; sample++)
      pInterleaved[channelIndex + sample * 2] = pChannel[sample];
  }
  else
  {
    for (size_t sample = 0; sample < sampleCount; sample++)
      pInterleaved[channelIndex + sample * channelCount] = pChannel[sample];
  }

  mRETURN_SUCCESS();
}

mFUNCTION(mAudio_AddToInterleavedFromChannelWithVolumeFloat, OUT float_t *pInterleaved, IN float_t *pChannel, const size_t channelIndex, const size_t channelCount, const size_t sampleCount, const float_t volume)
{
  mFUNCTION_SETUP();

  mERROR_IF(pInterleaved == nullptr || pChannel == nullptr, mR_ArgumentNull);
  mERROR_IF(channelIndex >= channelCount, mR_InvalidParameter);
  mERROR_IF(channelCount == 0, mR_InvalidParameter);

  if (channelCount == 1 && sampleCount * sizeof(float_t) > sizeof(__m128))
  {
    size_t sample = 0;

    const __m128 v = _mm_set1_ps(volume);

    for (; sample < sampleCount - (4 - 1); sample += 4)
    {
      const __m128 src = _mm_loadu_ps(&pChannel[sample]);
      
      float_t *pDst = &pInterleaved[sample];
      _mm_storeu_ps(pDst, _mm_add_ps(_mm_mul_ps(src, v), _mm_loadu_ps(pDst)));
    }

    for (; sample < sampleCount; sample++)
      pInterleaved[sample] += pChannel[sample] * volume;
  }
  else if (channelCount == 2 && sampleCount * sizeof(float_t) > sizeof(__m128))
  {
    size_t sample = 0;

    const __m128 mask = _mm_castsi128_ps(_mm_set_epi32(0, -1, 0, -1));
    const __m128 v = _mm_set1_ps(volume);

    for (; sample < sampleCount - (4 - 1); sample += 4)
    {
      const __m128 src = _mm_mul_ps(_mm_loadu_ps(&pChannel[sample]), v);
      const __m128 lo = _mm_and_ps(mask, _mm_shuffle_ps(src, src, _MM_SHUFFLE(0, 1, 0, 0)));
      const __m128 hi = _mm_and_ps(mask, _mm_shuffle_ps(src, src, _MM_SHUFFLE(0, 3, 0, 2)));

      float_t *pLo = &pInterleaved[channelIndex + sample * 2];
      float_t *pHi = &pInterleaved[channelIndex + (sample + 2) * 2];

      _mm_storeu_ps(pLo, _mm_add_ps(lo, _mm_loadu_ps(pLo)));
      _mm_storeu_ps(pHi, _mm_add_ps(hi, _mm_loadu_ps(pHi)));
    }

    for (; sample < sampleCount; sample++)
      pInterleaved[channelIndex + sample * 2] += pChannel[sample] * volume;
  }
  else
  {
    for (size_t sample = 0; sample < sampleCount; sample++)
      pInterleaved[channelIndex + sample * channelCount] += pChannel[sample] * volume;
  }

  mRETURN_SUCCESS();
}

mFUNCTION(mAudio_AddToInterleavedBufferFromMonoWithVolumeFloat, OUT float_t *pInterleaved, IN float_t *pChannel, const size_t interleavedChannelCount, const size_t sampleCount, const float_t volume)
{
  mFUNCTION_SETUP();

  mERROR_IF(pInterleaved == nullptr || pChannel == nullptr, mR_ArgumentNull);
  mERROR_IF(interleavedChannelCount == 0, mR_InvalidParameter);

  if (sizeof(float_t) * sampleCount < sizeof(__m128) || interleavedChannelCount > 2)
  {
    for (size_t sample = 0; sample < sampleCount; sample++)
    {
      float_t *pSample = &pInterleaved[sample * interleavedChannelCount];
      const float_t sampleData = pChannel[sample];

      for (size_t channel = 0; channel < interleavedChannelCount; channel++)
        pSample[channel] += sampleData * volume;
    }
  }
  else if (interleavedChannelCount == 1)
  {
    mERROR_CHECK(mAudio_AddToInterleavedFromChannelWithVolumeFloat(pInterleaved, pChannel, 0, 1, sampleCount, volume));
  }
  else // Mono to stereo.
  {
    __m128 v = _mm_set1_ps(volume);
    constexpr size_t loopsize = sizeof(v) / sizeof(float_t);

    size_t sample = 0;

    for (; sample < sampleCount - (loopsize - 1); sample += 4)
    {
      const __m128 src = _mm_mul_ps(_mm_loadu_ps(&pChannel[sample]), v);

      const __m128 lo = _mm_shuffle_ps(src, src, _MM_SHUFFLE(1, 1, 0, 0));
      const __m128 hi = _mm_shuffle_ps(src, src, _MM_SHUFFLE(3, 3, 2, 2));

      float_t *pLo = &pInterleaved[sample * 2];
      float_t *pHi = &pInterleaved[(sample + 2) * 2];

      _mm_storeu_ps(pLo, _mm_add_ps(lo, _mm_loadu_ps(pLo)));
      _mm_storeu_ps(pHi, _mm_add_ps(hi, _mm_loadu_ps(pHi)));
    }

    for (; sample < sampleCount; sample++)
    {
      float_t *pSample = &pInterleaved[sample * 2];
      const float_t sampleData = pChannel[sample];

      pSample[0] += sampleData * volume;
      pSample[1] += sampleData * volume;
    }
  }

  mRETURN_SUCCESS();
}

#ifndef _MSC_VER
__attribute__((target("avx")))
#endif
void mAudio_ApplyVolumeFloat_AVX(OUT float_t *pAudio, const float_t volume, const size_t sampleCount)
{
  __m256 v;

  // If too small.
  if (sampleCount * sizeof(float_t) < sizeof(v))
  {
    for (size_t i = 0; i < sampleCount; i++)
      pAudio[i] *= volume;

    return;
  }

  v = _mm256_set1_ps(volume);
  constexpr size_t loopSize = sizeof(v) / sizeof(float_t);

  size_t unaligned = ((size_t)pAudio & (sizeof(v) - 1));
  size_t i = 0;

  // If Unaligned and alignable: Align.
  if ((unaligned & (sizeof(float_t) - 1)) == 0)
  {
    if (unaligned != 0)
    {
      float_t *pPrevAudio = pAudio;
      pAudio = reinterpret_cast<float_t *>(reinterpret_cast<uint8_t *>((size_t)pAudio & ~(size_t)(sizeof(v) - 1)) + sizeof(v));
      const size_t step = (pAudio - pPrevAudio);

      for (size_t j = 0; j < step; j++)
        pPrevAudio[j] *= volume;

      i += step;
    }

    for (; i < sampleCount - (loopSize - 1); i += loopSize)
    {
      const __m256 src = _mm256_load_ps(pAudio);
      _mm256_store_ps(pAudio, _mm256_mul_ps(src, v));

      pAudio += loopSize;
    }

    for (; i < sampleCount; i++)
    {
      *pAudio *= volume;
      pAudio++;
    }
  }
  else
  {
    for (; i < sampleCount - (loopSize - 1); i += loopSize)
    {
      const __m256 src = _mm256_loadu_ps(pAudio);
      _mm256_storeu_ps(pAudio, _mm256_mul_ps(src, v));

      pAudio += loopSize;
    }

    for (; i < sampleCount; i++)
    {
      *pAudio *= volume;
      pAudio++;
    }
  }
}

mFUNCTION(mAudio_ApplyVolumeFloat, OUT float_t *pAudio, const float_t volume, const size_t sampleCount)
{
  mFUNCTION_SETUP();

  mERROR_IF(pAudio == nullptr, mR_ArgumentNull);

  mCpuExtensions::Detect();

  if (mCpuExtensions::avxSupported)
  {
    mAudio_ApplyVolumeFloat_AVX(pAudio, volume, sampleCount);
  }
  else
  {
    __m128 v;

    // If too small.
    if (sampleCount * sizeof(float_t) < sizeof(v))
    {
      for (size_t i = 0; i < sampleCount; i++)
        pAudio[i] *= volume;

      mRETURN_SUCCESS();
    }
     
    v = _mm_set1_ps(volume);
    constexpr size_t loopSize = sizeof(v) / sizeof(float_t);

    size_t unaligned = ((size_t)pAudio & (sizeof(v) - 1));
    size_t i = 0;

    // If Unaligned and alignable: Align.
    if ((unaligned & (sizeof(float_t) - 1)) == 0)
    {
      if (unaligned != 0)
      {
        float_t *pPrevAudio = pAudio;
        pAudio = reinterpret_cast<float_t *>(reinterpret_cast<uint8_t *>((size_t)pAudio & ~(size_t)(sizeof(v) - 1)) + sizeof(v));
        const size_t step = (pAudio - pPrevAudio);

        for (size_t j = 0; j < step; j++)
          pPrevAudio[j] *= volume;

        i += step;
      }

      for (; i < sampleCount - (loopSize - 1); i += loopSize)
      {
        const __m128 src = _mm_load_ps(pAudio);
        _mm_store_ps(pAudio, _mm_mul_ps(src, v));

        pAudio += loopSize;
      }

      for (; i < sampleCount; i++)
      {
        *pAudio *= volume;
        pAudio++;
      }
    }
    else
    {
      for (; i < sampleCount - (loopSize - 1); i += loopSize)
      {
        const __m128 src = _mm_loadu_ps(pAudio);
        _mm_storeu_ps(pAudio, _mm_mul_ps(src, v));

        pAudio += loopSize;
      }

      for (; i < sampleCount; i++)
      {
        *pAudio *= volume;
        pAudio++;
      }
    }
  }

  mRETURN_SUCCESS();
}

static void mAudio_AddWithVolumeFloat_AVX(OUT float_t *pDestination, IN float_t *pSource, const float_t volume, const size_t sampleCount)
{
  size_t sampleIndex = 0;

  typedef __m256 simd_t;
  constexpr size_t loopSize = sizeof(simd_t) / sizeof(float_t);

  if (sampleCount > loopSize)
  {
    const simd_t mmvolume = _mm256_set1_ps(volume);

    for (; sampleIndex < sampleCount - (loopSize - 1); sampleIndex += loopSize)
    {
      const simd_t s = _mm256_loadu_ps(pSource + sampleIndex);
      const simd_t d = _mm256_loadu_ps(pDestination + sampleIndex);

      _mm256_storeu_ps(pDestination + sampleIndex, _mm256_add_ps(d, _mm256_mul_ps(s, mmvolume)));
    }
  }

  for (; sampleIndex < sampleCount; sampleIndex++)
    pDestination[sampleIndex] += pSource[sampleIndex] * volume;
}

mFUNCTION(mAudio_AddWithVolumeFloat, OUT float_t *pDestination, IN float_t *pSource, const float_t volume, const size_t sampleCount)
{
  mFUNCTION_SETUP();

  mERROR_IF(pDestination == nullptr || pSource == nullptr, mR_ArgumentNull);

  mCpuExtensions::Detect();

  if (mCpuExtensions::avxSupported)
  {
    mAudio_AddWithVolumeFloat_AVX(pDestination, pSource, volume, sampleCount);
  }
  else
  {
    size_t sampleIndex = 0;

    typedef __m128 simd_t;
    constexpr size_t loopSize = sizeof(simd_t) / sizeof(float_t);

    if (sampleCount > loopSize)
    {
      const simd_t mmvolume = _mm_set1_ps(volume);

      for (; sampleIndex < sampleCount - (loopSize - 1); sampleIndex += loopSize)
      {
        const simd_t s = _mm_loadu_ps(pSource + sampleIndex);
        const simd_t d = _mm_loadu_ps(pDestination + sampleIndex);

        _mm_storeu_ps(pDestination + sampleIndex, _mm_add_ps(d, _mm_mul_ps(s, mmvolume)));
      }
    }

    for (; sampleIndex < sampleCount; sampleIndex++)
      pDestination[sampleIndex] += pSource[sampleIndex] * volume;
  }

  mRETURN_SUCCESS();
}

static void mAudio_AddWithVolumeFloatVariableLowpass_AVX(OUT float_t *pDestination, IN float_t *pSource, const float_t volume, const size_t sampleCount, const float_t lerpFactor)
{
  size_t sampleIndex = 0;

  typedef __m256 simd_t;
  constexpr size_t loopSize = sizeof(simd_t) / sizeof(float_t);

  const simd_t lerpFac = _mm256_set1_ps(lerpFactor);
  const simd_t iLerpFac = _mm256_set1_ps(1.f - lerpFactor);

  if (sampleCount > loopSize)
  {
    const simd_t mmvolume = _mm256_set1_ps(volume);

    for (; sampleIndex < sampleCount - loopSize; sampleIndex += loopSize)
    {
      const simd_t s0 = _mm256_loadu_ps(pSource + sampleIndex);
      const simd_t s1 = _mm256_loadu_ps(pSource + sampleIndex + 1);

      const simd_t s = _mm256_add_ps(_mm256_mul_ps(s0, iLerpFac), _mm256_mul_ps(s1, lerpFac));

      const simd_t d = _mm256_loadu_ps(pDestination + sampleIndex);

      _mm256_storeu_ps(pDestination + sampleIndex, _mm256_add_ps(d, _mm256_mul_ps(s, mmvolume)));
    }
  }

  for (; sampleIndex < sampleCount - 1; sampleIndex++)
    pDestination[sampleIndex] += mLerp(pSource[sampleIndex], pSource[sampleIndex + 1], lerpFactor) * volume;

  pDestination[sampleIndex] += pSource[sampleIndex] * volume;
}

mFUNCTION(mAudio_AddWithVolumeFloatVariableLowpass, OUT float_t *pDestination, IN float_t *pSource, const float_t volume, const size_t sampleCount, const float_t lowPassStrength)
{
  mFUNCTION_SETUP();

  mERROR_IF(pDestination == nullptr || pSource == nullptr, mR_ArgumentNull);

  const float_t lerpFactor = mClamp(lowPassStrength * 0.5f, 0.f, 0.5f);

  mCpuExtensions::Detect();

  if (mCpuExtensions::avxSupported)
  {
    mAudio_AddWithVolumeFloatVariableLowpass_AVX(pDestination, pSource, volume, sampleCount, lerpFactor);
  }
  else
  {
    size_t sampleIndex = 0;

    typedef __m128 simd_t;
    constexpr size_t loopSize = sizeof(simd_t) / sizeof(float_t);

    const simd_t lerpFac = _mm_set1_ps(lerpFactor);
    const simd_t iLerpFac = _mm_set1_ps(1.f - lerpFactor);

    if (sampleCount > loopSize)
    {
      const simd_t mmvolume = _mm_set1_ps(volume);

      for (; sampleIndex < sampleCount - loopSize; sampleIndex += loopSize)
      {
        const simd_t s0 = _mm_loadu_ps(pSource + sampleIndex);
        const simd_t s1 = _mm_loadu_ps(pSource + sampleIndex + 1);

        const simd_t s = _mm_add_ps(_mm_mul_ps(s0, iLerpFac), _mm_mul_ps(s1, lerpFac));

        const simd_t d = _mm_loadu_ps(pDestination + sampleIndex);

        _mm_storeu_ps(pDestination + sampleIndex, _mm_add_ps(d, _mm_mul_ps(s, mmvolume)));
      }
    }

    for (; sampleIndex < sampleCount - 1; sampleIndex++)
      pDestination[sampleIndex] += mLerp(pSource[sampleIndex], pSource[sampleIndex + 1], lerpFactor) * volume;

    pDestination[sampleIndex] += pSource[sampleIndex] * volume;
  }

  mRETURN_SUCCESS();
}

static void mAudio_AddWithVolumeFloatVariableOffAxisFilter_AVX(OUT float_t *pDestination, IN float_t *pSource, const float_t volume, const size_t sampleCount, const float_t filterStrength)
{
  size_t sampleIndex = 0;

  typedef __m256 simd_t;
  constexpr size_t loopSize = sizeof(simd_t) / sizeof(float_t);

  constexpr float_t scaleFactor = 1.f / 6.f;
  const float_t lerpFactor = mClamp(filterStrength * scaleFactor, 0.f, 0.25f);
  const float_t mainFactor = 1.f - lerpFactor * 5.f;

  if (sampleCount > loopSize)
  {
    const simd_t mainFac = _mm256_set1_ps(mainFactor);
    const simd_t facA = _mm256_set1_ps(lerpFactor);
    const simd_t mmvolume = _mm256_set1_ps(volume);

    for (; sampleIndex < sampleCount - (loopSize + 27); sampleIndex += loopSize)
    {
      const simd_t s0 = _mm256_loadu_ps(pSource + sampleIndex);
      const simd_t s1 = _mm256_loadu_ps(pSource + sampleIndex + 1);
      const simd_t s2 = _mm256_loadu_ps(pSource + sampleIndex + 2);
      const simd_t s3 = _mm256_loadu_ps(pSource + sampleIndex + 25);
      const simd_t s4 = _mm256_loadu_ps(pSource + sampleIndex + 26);
      const simd_t s5 = _mm256_loadu_ps(pSource + sampleIndex + 27);

      const simd_t s = _mm256_add_ps(_mm256_mul_ps(s0, mainFac), _mm256_mul_ps(facA, _mm256_add_ps(_mm256_add_ps(s1, s2), _mm256_add_ps(_mm256_add_ps(s3, s4), s5))));

      const simd_t d = _mm256_loadu_ps(pDestination + sampleIndex);

      _mm256_storeu_ps(pDestination + sampleIndex, _mm256_add_ps(d, _mm256_mul_ps(s, mmvolume)));
    }
  }

  for (; sampleIndex < sampleCount - 27; sampleIndex++)
    pDestination[sampleIndex] += (pSource[sampleIndex] * mainFactor + (pSource[sampleIndex + 1] + pSource[sampleIndex + 2] + pSource[sampleIndex + 25] + pSource[sampleIndex + 26] + pSource[sampleIndex + 27]) * lerpFactor) * volume;

  for (; sampleIndex < sampleCount - 1; sampleIndex++)
    pDestination[sampleIndex] += mLerp(pSource[sampleIndex], pSource[sampleIndex + 1], lerpFactor * 2.f) * volume;

  pDestination[sampleIndex] += pSource[sampleIndex] * volume;
}

mFUNCTION(mAudio_AddWithVolumeFloatVariableOffAxisFilter, OUT float_t *pDestination, IN float_t *pSource, const float_t volume, const size_t sampleCount, const float_t filterStrength)
{
  mFUNCTION_SETUP();

  mPROFILE_SCOPED("mAudio_AddWithVolumeFloatVariableOffAxisFilter");

  mERROR_IF(pDestination == nullptr || pSource == nullptr, mR_ArgumentNull);

  mCpuExtensions::Detect();

  if (mCpuExtensions::avxSupported)
  {
    mAudio_AddWithVolumeFloatVariableOffAxisFilter_AVX(pDestination, pSource, volume, sampleCount, filterStrength);
  }
  else
  {
    size_t sampleIndex = 0;

    typedef __m128 simd_t;
    constexpr size_t loopSize = sizeof(simd_t) / sizeof(float_t);

    constexpr float_t scaleFactor = 1.f / 6.f;
    const float_t lerpFactor = mClamp(filterStrength * scaleFactor, 0.f, 0.25f);
    const float_t mainFactor = 1.f - lerpFactor * 5.f;

    if (sampleCount > loopSize)
    {
      const simd_t mainFac = _mm_set1_ps(mainFactor);
      const simd_t facA = _mm_set1_ps(lerpFactor);
      const simd_t mmvolume = _mm_set1_ps(volume);

      for (; sampleIndex < sampleCount - (loopSize + 27); sampleIndex += loopSize)
      {
        const simd_t s0 = _mm_loadu_ps(pSource + sampleIndex);
        const simd_t s1 = _mm_loadu_ps(pSource + sampleIndex + 1);
        const simd_t s2 = _mm_loadu_ps(pSource + sampleIndex + 2);
        const simd_t s3 = _mm_loadu_ps(pSource + sampleIndex + 25);
        const simd_t s4 = _mm_loadu_ps(pSource + sampleIndex + 26);
        const simd_t s5 = _mm_loadu_ps(pSource + sampleIndex + 27);

        const simd_t s = _mm_add_ps(_mm_mul_ps(s0, mainFac), _mm_mul_ps(facA, _mm_add_ps(_mm_add_ps(s1, s2), _mm_add_ps(_mm_add_ps(s3, s4), s5))));

        const simd_t d = _mm_loadu_ps(pDestination + sampleIndex);

        _mm_storeu_ps(pDestination + sampleIndex, _mm_add_ps(d, _mm_mul_ps(s, mmvolume)));
      }
    }

    for (; sampleIndex < sampleCount - 27; sampleIndex++)
      pDestination[sampleIndex] += (pSource[sampleIndex] * mainFactor + (pSource[sampleIndex + 1] + pSource[sampleIndex + 2] + pSource[sampleIndex + 25] + pSource[sampleIndex + 26] + pSource[sampleIndex + 27]) * lerpFactor) * volume;

    for (; sampleIndex < sampleCount - 1; sampleIndex++)
      pDestination[sampleIndex] += mLerp(pSource[sampleIndex], pSource[sampleIndex + 1], lerpFactor * 2.f) * volume;

    pDestination[sampleIndex] += pSource[sampleIndex] * volume;
  }

  mRETURN_SUCCESS();
}

thread_local static float_t *_pResampled = nullptr;
thread_local static size_t _resampledSampleCount = 0;

mFUNCTION(mAudio_AddResampleToInterleavedFromChannelWithVolume, OUT float_t *pInterleaved, IN float_t *pChannel, const size_t channelIndex, const size_t channelCount, const size_t interleavedSampleCount, const size_t channelSampleCount, const float_t volume, const mAudio_ResampleQuality quality)
{
  mFUNCTION_SETUP();

  mPROFILE_SCOPED("mAudio_AddResampleToInterleavedFromChannelWithVolume");

  mERROR_IF(pChannel == nullptr || pInterleaved == nullptr, mR_ArgumentNull);
  mERROR_IF(channelIndex >= channelCount, mR_InvalidParameter);

  if (interleavedSampleCount > _resampledSampleCount)
  {
    mERROR_CHECK(mRealloc(&_pResampled, interleavedSampleCount));
    _resampledSampleCount = interleavedSampleCount;
  }

  SRC_DATA data;
  data.data_in = pChannel;
  data.data_out = _pResampled;
  data.end_of_input = FALSE;
  data.input_frames = (int32_t)channelSampleCount;
  data.output_frames = (int32_t)interleavedSampleCount;
  data.src_ratio = (double_t)interleavedSampleCount / (double_t)channelSampleCount;

  mERROR_IF(0 != src_simple(&data, quality, 1), mR_InternalError);

  mERROR_CHECK(mAudio_AddToInterleavedFromChannelWithVolumeFloat(pInterleaved, _pResampled, channelIndex, channelCount, interleavedSampleCount, volume));

  mRETURN_SUCCESS();
}

mFUNCTION(mAudio_AddResampleMonoToInterleavedFromChannelWithVolume, OUT float_t *pInterleaved, IN float_t *pChannel, const size_t channelCount, const size_t interleavedSampleCount, const size_t channelSampleCount, const float_t volume, const mAudio_ResampleQuality quality)
{
  mFUNCTION_SETUP();

  mPROFILE_SCOPED("mAudio_AddResampleMonoToInterleavedFromChannelWithVolume");

  mERROR_IF(pChannel == nullptr || pInterleaved == nullptr, mR_ArgumentNull);

  if (interleavedSampleCount > _resampledSampleCount)
  {
    mERROR_CHECK(mRealloc(&_pResampled, interleavedSampleCount));
    _resampledSampleCount = interleavedSampleCount;
  }

  SRC_DATA data;
  data.data_in = pChannel;
  data.data_out = _pResampled;
  data.end_of_input = FALSE;
  data.input_frames = (int32_t)channelSampleCount;
  data.output_frames = (int32_t)interleavedSampleCount;
  data.src_ratio = (double_t)interleavedSampleCount / (double_t)channelSampleCount;

  mERROR_IF(0 != src_simple(&data, quality, 1), mR_InternalError);

  mERROR_CHECK(mAudio_AddToInterleavedBufferFromMonoWithVolumeFloat(pInterleaved, _pResampled, channelCount, interleavedSampleCount, volume));

  mRETURN_SUCCESS();
}

mFUNCTION(mAudio_InplaceResampleMono, OUT float_t *pChannel, const size_t currentSampleCount, const size_t desiredSampleCount, const mAudio_ResampleQuality quality)
{
  mFUNCTION_SETUP();

  mPROFILE_SCOPED("mAudio_InplaceResampleMono");

  mERROR_IF(pChannel == nullptr, mR_ArgumentNull);

  if (desiredSampleCount > _resampledSampleCount)
  {
    mERROR_CHECK(mRealloc(&_pResampled, desiredSampleCount));
    _resampledSampleCount = desiredSampleCount;
  }

  SRC_DATA data;
  data.data_in = pChannel;
  data.data_out = _pResampled;
  data.end_of_input = FALSE;
  data.input_frames = (int32_t)currentSampleCount;
  data.output_frames = (int32_t)desiredSampleCount;
  data.src_ratio = (double_t)desiredSampleCount / (double_t)currentSampleCount;

  mERROR_IF(0 != src_simple(&data, quality, 1), mR_InternalError);

  mERROR_CHECK(mMemmove(pChannel, _pResampled, desiredSampleCount));

  mRETURN_SUCCESS();
}

mFUNCTION(mAudio_InplaceResampleMonoWithFade, OUT float_t *pChannel, const size_t currentSampleCount, const size_t desiredSampleCount, const size_t fadeSamples, const mAudio_ResampleQuality quality)
{
  mFUNCTION_SETUP();

  mPROFILE_SCOPED("mAudio_InplaceResampleMonoWithFade");

  mERROR_IF(pChannel == nullptr, mR_ArgumentNull);
  mERROR_IF(fadeSamples * 2 >= currentSampleCount || fadeSamples * 2 >= desiredSampleCount, mR_ArgumentOutOfBounds);

  if (desiredSampleCount > _resampledSampleCount)
  {
    mERROR_CHECK(mRealloc(&_pResampled, desiredSampleCount));
    _resampledSampleCount = desiredSampleCount;
  }

  SRC_DATA data;
  data.data_in = pChannel;
  data.data_out = _pResampled;
  data.end_of_input = FALSE;
  data.input_frames = (int32_t)currentSampleCount;
  data.output_frames = (int32_t)desiredSampleCount;
  data.src_ratio = (double_t)desiredSampleCount / (double_t)currentSampleCount;

  mERROR_IF(0 != src_simple(&data, quality, 1), mR_InternalError);

  const float_t factor = 1.f / fadeSamples;

  for (size_t i = 0; i < fadeSamples; i++)
  {
    _pResampled[i] = mLerp(pChannel[i], _pResampled[i], factor * i);
    _pResampled[desiredSampleCount - 1 - i] = mLerp(pChannel[currentSampleCount - 1 - i], _pResampled[desiredSampleCount - 1 - i], factor * i);
  }

  mERROR_CHECK(mMemmove(pChannel, _pResampled, desiredSampleCount));

  mRETURN_SUCCESS();
}

mFUNCTION(mAudio_ResampleMonoToMonoWithVolume, OUT float_t *pDestination, IN float_t *pSource, const size_t currentSampleCount, const size_t desiredSampleCount, const float_t volume, const mAudio_ResampleQuality quality)
{
  mFUNCTION_SETUP();

  mPROFILE_SCOPED("mAudio_ResampleMonoToMonoWithVolume");

  mERROR_IF(pDestination == nullptr || pSource == nullptr, mR_ArgumentNull);

  SRC_DATA data;
  data.data_in = pSource;
  data.data_out = pDestination;
  data.end_of_input = FALSE;
  data.input_frames = (int32_t)currentSampleCount;
  data.output_frames = (int32_t)desiredSampleCount;
  data.src_ratio = (double_t)desiredSampleCount / (double_t)currentSampleCount;

  mERROR_IF(0 != src_simple(&data, quality, 1), mR_InternalError);

  mERROR_CHECK(mAudio_ApplyVolumeFloat(pDestination, volume, desiredSampleCount));

  mRETURN_SUCCESS();
}

mFUNCTION(mAudio_AddResampleMonoToMonoWithVolume, OUT float_t *pDestination, IN float_t *pSource, const size_t currentSampleCount, const size_t desiredSampleCount, const float_t volume, const mAudio_ResampleQuality quality)
{
  mFUNCTION_SETUP();

  mPROFILE_SCOPED("mAudio_AddResampleMonoToMonoWithVolume");

  mERROR_IF(pDestination == nullptr || pSource == nullptr, mR_ArgumentNull);

  if (desiredSampleCount > _resampledSampleCount)
  {
    mERROR_CHECK(mRealloc(&_pResampled, desiredSampleCount));
    _resampledSampleCount = desiredSampleCount;
  }

  SRC_DATA data;
  data.data_in = pSource;
  data.data_out = _pResampled;
  data.end_of_input = FALSE;
  data.input_frames = (int32_t)currentSampleCount;
  data.output_frames = (int32_t)desiredSampleCount;
  data.src_ratio = (double_t)desiredSampleCount / (double_t)currentSampleCount;

  mERROR_IF(0 != src_simple(&data, quality, 1), mR_InternalError);

  mERROR_CHECK(mAudio_AddWithVolumeFloat(pDestination, _pResampled, volume, desiredSampleCount));

  mRETURN_SUCCESS();
}

mFUNCTION(mAudio_AddResampleMonoToMonoWithVolumeVariableLowpass, OUT float_t *pDestination, IN float_t *pSource, const size_t currentSampleCount, const size_t desiredSampleCount, const float_t volume, const float_t lowPassStrength, const mAudio_ResampleQuality quality)
{
  mFUNCTION_SETUP();

  mPROFILE_SCOPED("mAudio_AddResampleMonoToMonoWithVolumeVariableLowpass");

  mERROR_IF(pDestination == nullptr || pSource == nullptr, mR_ArgumentNull);

  if (desiredSampleCount > _resampledSampleCount)
  {
    mERROR_CHECK(mRealloc(&_pResampled, desiredSampleCount));
    _resampledSampleCount = desiredSampleCount;
  }

  SRC_DATA data;
  data.data_in = pSource;
  data.data_out = _pResampled;
  data.end_of_input = FALSE;
  data.input_frames = (int32_t)currentSampleCount;
  data.output_frames = (int32_t)desiredSampleCount;
  data.src_ratio = (double_t)desiredSampleCount / (double_t)currentSampleCount;

  mERROR_IF(0 != src_simple(&data, quality, 1), mR_InternalError);

  mERROR_CHECK(mAudio_AddWithVolumeFloatVariableLowpass(pDestination, _pResampled, volume, desiredSampleCount, lowPassStrength));

  mRETURN_SUCCESS();
}

mFUNCTION(mAudio_AddResampleMonoToMonoWithVolumeVariableOffAxisFilter, OUT float_t *pDestination, IN float_t *pSource, const size_t currentSampleCount, const size_t desiredSampleCount, const float_t volume, const float_t filterStrength, const mAudio_ResampleQuality quality)
{
  mFUNCTION_SETUP();

  mPROFILE_SCOPED("mAudio_AddResampleMonoToMonoWithVolumeVariableOffAxisFilter");

  mERROR_IF(pDestination == nullptr || pSource == nullptr, mR_ArgumentNull);

  if (desiredSampleCount > _resampledSampleCount)
  {
    mERROR_CHECK(mRealloc(&_pResampled, desiredSampleCount));
    _resampledSampleCount = desiredSampleCount;
  }

  SRC_DATA data;
  data.data_in = pSource;
  data.data_out = _pResampled;
  data.end_of_input = FALSE;
  data.input_frames = (int32_t)currentSampleCount;
  data.output_frames = (int32_t)desiredSampleCount;
  data.src_ratio = (double_t)desiredSampleCount / (double_t)currentSampleCount;

  mERROR_IF(0 != src_simple(&data, quality, 1), mR_InternalError);

  mERROR_CHECK(mAudio_AddWithVolumeFloatVariableOffAxisFilter(pDestination, _pResampled, volume, desiredSampleCount, filterStrength));

  mRETURN_SUCCESS();
}

static void mAudio_InplaceMidSideToStereo_AVX(IN_OUT float_t *pMidToLeft, IN_OUT float_t *pSideToRight, const size_t sampleCount)
{
  size_t sampleIndex = 0;

  float_t *pLeft = pMidToLeft;
  float_t *pRight = pSideToRight;

  typedef __m256 simd_t;
  constexpr size_t loopSize = (sizeof(simd_t) / sizeof(float_t));

  if (sampleCount > loopSize)
  {
    const simd_t factor = _mm256_set1_ps(.5f);

    for (; sampleIndex < sampleCount - (loopSize - 1); sampleIndex += loopSize)
    {
      float_t *pPos0 = pLeft + sampleIndex;
      float_t *pPos1 = pRight + sampleIndex;

      const simd_t mid = _mm256_loadu_ps(pPos0);
      const simd_t side = _mm256_mul_ps(factor, _mm256_loadu_ps(pPos1));

      const simd_t left = _mm256_add_ps(mid, side);
      const simd_t right = _mm256_sub_ps(mid, side);

      _mm256_storeu_ps(pPos0, left);
      _mm256_storeu_ps(pPos1, right);
    }
  }

  for (; sampleIndex < sampleCount; sampleIndex++)
  {
    const float_t mid = pLeft[sampleIndex];
    const float_t side = pRight[sampleIndex] * .5f;

    const float_t left = mid + side;
    const float_t right = mid - side;

    pLeft[sampleIndex] = left;
    pRight[sampleIndex] = right;
  }
}

mFUNCTION(mAudio_InplaceMidSideToStereo, IN_OUT float_t *pMidToLeft, IN_OUT float_t *pSideToRight, const size_t sampleCount)
{
  mFUNCTION_SETUP();

  mERROR_IF(pMidToLeft == nullptr || pSideToRight == nullptr, mR_ArgumentNull);

  mCpuExtensions::Detect();

  if (mCpuExtensions::avxSupported)
  {
    mAudio_InplaceMidSideToStereo_AVX(pMidToLeft, pSideToRight, sampleCount);
  }
  else
  {
    size_t sampleIndex = 0;

    float_t *pLeft = pMidToLeft;
    float_t *pRight = pSideToRight;

    typedef __m128 simd_t;
    constexpr size_t loopSize = (sizeof(simd_t) / sizeof(float_t));

    if (sampleCount > loopSize)
    {
      const simd_t factor = _mm_set1_ps(.5f);

      for (; sampleIndex < sampleCount - (loopSize - 1); sampleIndex += loopSize)
      {
        float_t *pPos0 = pLeft + sampleIndex;
        float_t *pPos1 = pRight + sampleIndex;

        const simd_t mid = _mm_loadu_ps(pPos0);
        const simd_t side = _mm_mul_ps(factor, _mm_loadu_ps(pPos1));

        const simd_t left = _mm_add_ps(mid, side);
        const simd_t right = _mm_sub_ps(mid, side);

        _mm_storeu_ps(pPos0, left);
        _mm_storeu_ps(pPos1, right);
      }
    }

    for (; sampleIndex < sampleCount; sampleIndex++)
    {
      const float_t mid = pLeft[sampleIndex];
      const float_t side = pRight[sampleIndex] * .5f;

      const float_t left = mid + side;
      const float_t right = mid - side;

      pLeft[sampleIndex] = left;
      pRight[sampleIndex] = right;
    }
  }

  mRETURN_SUCCESS();
}

static void mAudio_InplaceMidLateralLongitudinalToQuadro_AVX(IN_OUT float_t *pMidToFrontLeft, IN_OUT float_t *pLateralToFrontRight, IN_OUT float_t *pLongitudinalToBackLeft, OUT float_t *pBackRight, const size_t sampleCount)
{
  float_t *pFrontLeft = pMidToFrontLeft;
  float_t *pFrontRight = pLateralToFrontRight;
  float_t *pBackLeft = pLongitudinalToBackLeft;
  size_t sampleIndex = 0;

  typedef __m256 simd_t;
  constexpr size_t loopSize = (sizeof(simd_t) / sizeof(float_t));

  if (sampleCount > loopSize)
  {
    const simd_t factor = _mm256_set1_ps(.5f);

    for (; sampleIndex < sampleCount - (loopSize - 1); sampleIndex += loopSize)
    {
      float_t *pPos0 = pFrontLeft + sampleIndex;
      float_t *pPos1 = pFrontRight + sampleIndex;
      float_t *pPos2 = pBackLeft + sampleIndex;

      const simd_t mid = _mm256_loadu_ps(pPos0);
      const simd_t sideA = _mm256_mul_ps(factor, _mm256_loadu_ps(pPos1));
      const simd_t sideB = _mm256_mul_ps(factor, _mm256_loadu_ps(pPos2));

      const simd_t left = _mm256_add_ps(mid, sideA);
      const simd_t right = _mm256_sub_ps(mid, sideA);

      const simd_t fleft = _mm256_add_ps(left, sideB);
      const simd_t bleft = _mm256_sub_ps(left, sideB);
      const simd_t fright = _mm256_add_ps(right, sideB);
      const simd_t bright = _mm256_sub_ps(right, sideB);

      float_t *pPos3 = pBackRight + sampleIndex;

      _mm256_storeu_ps(pPos0, fleft);
      _mm256_storeu_ps(pPos1, fright);
      _mm256_storeu_ps(pPos2, bleft);
      _mm256_storeu_ps(pPos3, bright);
    }
  }

  for (; sampleIndex < sampleCount; sampleIndex++)
  {
    const float_t mid = pFrontLeft[sampleIndex];
    const float_t sideA = pFrontRight[sampleIndex] * .5f;
    const float_t sideB = pBackLeft[sampleIndex] * .5f;

    pFrontLeft[sampleIndex] = mid + sideA + sideB;
    pFrontRight[sampleIndex] = mid - sideA + sideB;
    pBackLeft[sampleIndex] = mid + sideA - sideB;
    pBackRight[sampleIndex] = mid - sideA - sideB;
  }
}

mFUNCTION(mAudio_InplaceMidLateralLongitudinalToQuadro, IN_OUT float_t *pMidToFrontLeft, IN_OUT float_t *pLateralToFrontRight, IN_OUT float_t *pLongitudinalToBackLeft, OUT float_t *pBackRight, const size_t sampleCount)
{
  mFUNCTION_SETUP();

  mERROR_IF(pMidToFrontLeft == nullptr || pLateralToFrontRight == nullptr || pLongitudinalToBackLeft == nullptr || pBackRight == nullptr, mR_ArgumentNull);

  mCpuExtensions::Detect();

  if (mCpuExtensions::avxSupported)
  {
    mAudio_InplaceMidLateralLongitudinalToQuadro_AVX(pMidToFrontLeft, pLateralToFrontRight, pLongitudinalToBackLeft, pBackRight, sampleCount);
  }
  else
  {
    float_t *pFrontLeft = pMidToFrontLeft;
    float_t *pFrontRight = pLateralToFrontRight;
    float_t *pBackLeft = pLongitudinalToBackLeft;
    size_t sampleIndex = 0;

    typedef __m128 simd_t;
    constexpr size_t loopSize = (sizeof(simd_t) / sizeof(float_t));

    if (sampleCount > loopSize)
    {
      const simd_t factor = _mm_set1_ps(.5f);

      for (; sampleIndex < sampleCount - (loopSize - 1); sampleIndex += loopSize)
      {
        float_t *pPos0 = pFrontLeft + sampleIndex;
        float_t *pPos1 = pFrontRight + sampleIndex;
        float_t *pPos2 = pBackLeft + sampleIndex;

        const simd_t mid = _mm_loadu_ps(pPos0);
        const simd_t sideA = _mm_mul_ps(factor, _mm_loadu_ps(pPos1));
        const simd_t sideB = _mm_mul_ps(factor, _mm_loadu_ps(pPos2));

        const simd_t left = _mm_add_ps(mid, sideA);
        const simd_t right = _mm_sub_ps(mid, sideA);

        const simd_t fleft = _mm_add_ps(left, sideB);
        const simd_t bleft = _mm_sub_ps(left, sideB);
        const simd_t fright = _mm_add_ps(right, sideB);
        const simd_t bright = _mm_sub_ps(right, sideB);

        float_t *pPos3 = pBackRight + sampleIndex;

        _mm_storeu_ps(pPos0, fleft);
        _mm_storeu_ps(pPos1, fright);
        _mm_storeu_ps(pPos2, bleft);
        _mm_storeu_ps(pPos3, bright);
      }
    }

    for (; sampleIndex < sampleCount; sampleIndex++)
    {
      const float_t mid = pFrontLeft[sampleIndex];
      const float_t sideA = pFrontRight[sampleIndex] * .5f;
      const float_t sideB = pBackLeft[sampleIndex] * .5f;

      pFrontLeft[sampleIndex] = mid + sideA + sideB;
      pFrontRight[sampleIndex] = mid - sideA + sideB;
      pBackLeft[sampleIndex] = mid + sideA - sideB;
      pBackRight[sampleIndex] = mid - sideA - sideB;
    }
  }

  mRETURN_SUCCESS();
}

mFUNCTION(mAudio_InterleavedQuadroMidLateralLFELongitudinalToDualInterleavedStereo, IN const float_t *pQuadroMidLateralLFELongitudinal, OUT float_t *pStereoChannelFront, OUT float_t *pStereoChannelBack, const size_t sampleCount)
{
  mFUNCTION_SETUP();

  mERROR_IF(pQuadroMidLateralLFELongitudinal == nullptr || pStereoChannelFront == nullptr || pStereoChannelBack == nullptr, mR_ArgumentNull);

  size_t sampleIndex = 0;

  {
    typedef __m128 simd_t;

    if (sampleCount >= 4)
    {
      const simd_t half = _mm_set1_ps(.5f);

      for (; sampleIndex < sampleCount - 3; sampleIndex += 4)
      {
        const simd_t src0 = _mm_loadu_ps(pQuadroMidLateralLFELongitudinal + 4 * (sampleIndex + 0));
        const simd_t src1 = _mm_loadu_ps(pQuadroMidLateralLFELongitudinal + 4 * (sampleIndex + 1));
        const simd_t src2 = _mm_loadu_ps(pQuadroMidLateralLFELongitudinal + 4 * (sampleIndex + 2));
        const simd_t src3 = _mm_loadu_ps(pQuadroMidLateralLFELongitudinal + 4 * (sampleIndex + 3));

        const simd_t lo02 = _mm_unpacklo_ps(src0, src2);
        const simd_t hi02 = _mm_unpackhi_ps(src0, src2);
        const simd_t lo13 = _mm_unpacklo_ps(src1, src3);
        const simd_t hi13 = _mm_unpackhi_ps(src1, src3);

        const simd_t mid = _mm_unpacklo_ps(lo02, lo13);
        const simd_t lat = _mm_mul_ps(half, _mm_unpackhi_ps(lo02, lo13));
        const simd_t lfe = _mm_unpacklo_ps(hi02, hi13);
        const simd_t longd = _mm_mul_ps(half, _mm_unpackhi_ps(hi02, hi13));

        const simd_t midlfe = _mm_add_ps(lfe, mid);

        const simd_t left = _mm_add_ps(midlfe, lat);
        const simd_t right = _mm_sub_ps(midlfe, lat);

        const simd_t fleft = _mm_add_ps(left, longd);
        const simd_t bleft = _mm_sub_ps(left, longd);
        const simd_t fright = _mm_add_ps(right, longd);
        const simd_t bright = _mm_sub_ps(right, longd);

        const simd_t fpack0 = _mm_unpacklo_ps(fleft, fright);
        const simd_t fpack1 = _mm_unpackhi_ps(fleft, fright);

        const simd_t bpack0 = _mm_unpacklo_ps(bleft, bright);
        const simd_t bpack1 = _mm_unpackhi_ps(bleft, bright);

        _mm_storeu_ps(pStereoChannelFront + sampleIndex * 2, fpack0);
        _mm_storeu_ps(pStereoChannelFront + sampleIndex * 2 + 4, fpack1);

        _mm_storeu_ps(pStereoChannelBack + sampleIndex * 2, bpack0);
        _mm_storeu_ps(pStereoChannelBack + sampleIndex * 2 + 4, bpack1);
      }
    }

    for (; sampleIndex < sampleCount; sampleIndex++)
    {
      const float_t mid = pQuadroMidLateralLFELongitudinal[4 * sampleIndex + 0];
      const float_t lat = pQuadroMidLateralLFELongitudinal[4 * sampleIndex + 1] * .5f;
      const float_t lfe = pQuadroMidLateralLFELongitudinal[4 * sampleIndex + 2];
      const float_t longd = pQuadroMidLateralLFELongitudinal[4 * sampleIndex + 3] * .5f;

      const float_t midlfe = mid + lfe;
      const float_t left = midlfe + lat;
      const float_t right = midlfe - lat;

      pStereoChannelFront[sampleIndex * 2 + 0] = left + longd;
      pStereoChannelFront[sampleIndex * 2 + 1] = right + longd;

      pStereoChannelBack[sampleIndex * 2 + 0] = left - longd;
      pStereoChannelBack[sampleIndex * 2 + 1] = right - longd;
    }
  }

  mRETURN_SUCCESS();
}

static void mAudio_GetAbsMax_AVX(IN float_t *pBuffer, const size_t count, OUT float_t *pMax)
{
  if (count * sizeof(float_t) < 64)
  {
    float_t max = 0;

    for (size_t sampleIndex = 0; sampleIndex < count; sampleIndex++)
      max = mMax(max, mAbs(pBuffer[sampleIndex]));

    *pMax = max;
  }
  else
  {
    const __m256 signBit = _mm256_castsi256_ps(_mm256_set1_epi32((int32_t)0b01111111111111111111111111111111));

    constexpr size_t loopSize = (sizeof(__m256)) / sizeof(float_t);

    size_t sampleIndex = loopSize;

    __m256 mmax = _mm256_and_ps(signBit, _mm256_loadu_ps(pBuffer));

    for (; sampleIndex < count - (loopSize - 1); sampleIndex += loopSize)
    {
      const __m256 src = _mm256_and_ps(signBit, _mm256_loadu_ps(pBuffer + sampleIndex));
      mmax = _mm256_max_ps(src, mmax);
    }

    if (sampleIndex != count)
    {
      const __m256 src = _mm256_and_ps(signBit, _mm256_loadu_ps(pBuffer + count - loopSize));
      mmax = _mm256_max_ps(src, mmax);
    }

    mALIGN(16) float_t store[4];
    _mm_store_ps(store, _mm_max_ps(_mm256_extractf128_ps(mmax, 0), _mm256_extractf128_ps(mmax, 1)));

    float_t max = 0;

    for (size_t i = 0; i < 4; i++)
      max = mMax(max, mAbs(store[i]));

    *pMax = max;
  }
}

mFUNCTION(mAudio_GetAbsMax, IN float_t *pBuffer, const size_t count, OUT float_t *pMax)
{
  mFUNCTION_SETUP();

  mERROR_IF(pMax == nullptr || pBuffer == nullptr, mR_ArgumentNull);
  mERROR_IF(count == 0, mR_InvalidParameter);

  mCpuExtensions::Detect();

  if (mCpuExtensions::avxSupported)
  {
    mAudio_GetAbsMax_AVX(pBuffer, count, pMax);
  }
  else
  {
    if (count * sizeof(float_t) < 32)
    {
      float_t max = 0;

      for (size_t sampleIndex = 0; sampleIndex < count; sampleIndex++)
        max = mMax(max, mAbs(pBuffer[sampleIndex]));

      *pMax = max;
    }
    else
    {
      const __m128 signBit = _mm_castsi128_ps(_mm_set1_epi32((int32_t)0b01111111111111111111111111111111));

      constexpr size_t loopSize = (sizeof(__m128)) / sizeof(float_t);

      size_t sampleIndex = loopSize;

      __m128 mmax = _mm_and_ps(signBit, _mm_loadu_ps(pBuffer));

      for (; sampleIndex < count - (loopSize - 1); sampleIndex += loopSize)
      {
        const __m128 src = _mm_and_ps(signBit, _mm_loadu_ps(pBuffer + sampleIndex));
        mmax = _mm_max_ps(src, mmax);
      }

      if (sampleIndex != count)
      {
        const __m128 src = _mm_and_ps(signBit, _mm_loadu_ps(pBuffer + count - loopSize));
        mmax = _mm_max_ps(src, mmax);
      }

      mALIGN(16) float_t store[4];
      _mm_store_ps(store, mmax);

      float_t max = 0;

      for (size_t i = 0; i < 4; i++)
        max = mMax(max, mAbs(store[i]));

      *pMax = max;
    }
  }

  mRETURN_SUCCESS();
}

#ifdef _MSC_VER
#pragma warning(pop)
#endif

//////////////////////////////////////////////////////////////////////////

struct mAudioSourceWav : mAudioSource
{
  mPtr<mCachedFileReader> fileReader;

  struct
  {
    uint8_t riffHeader[4];
    uint32_t chunkSize; // RIFF Chunk Size.
    uint8_t waveHeader[4];

    // fmt sub chunk.
    uint8_t fmtSubchunkID[4]; // "fmt ".
    uint32_t fmtSubchunkSize; // Size of the fmt chunk.
    uint16_t fmtAudioFormat; // Audio format 1=PCM, 6=mulaw, 7=alaw, 257=IBM Mu-Law, 258=IBM A-Law, 259=ADPCM.
    uint16_t fmtChannelCount;
    uint32_t fmtSampleRate;
    uint32_t fmtBytesPerSec;
    uint16_t fmtBlockAlign; // 2=16-bit mono, 4=16-bit stereo.
    uint16_t fmtBitsPerSample;

    // data sub chunk.
    uint8_t dataSubchunkID[4]; // "data".
    uint32_t dataSubchunkSize; // Data length.
  } riffWaveHeader;

  size_t readPosition;
  size_t startOffset;
  bool endReached;
};

static mFUNCTION(mAudioSourceWav_Destroy_Internal, IN_OUT mAudioSourceWav *pAudioSource);
static mFUNCTION(mAudioSourceWav_GetBuffer_Internal, mPtr<mAudioSource> &audioSource, OUT float_t *pBuffer, const size_t bufferLength, const size_t channelIndex, OUT size_t *pBufferCount);
static mFUNCTION(mAudioSourceWav_MoveToNextBuffer_Internal, mPtr<mAudioSource> &audioSource, const size_t samples);

//////////////////////////////////////////////////////////////////////////

mFUNCTION(mAudioSourceWav_Create, OUT mPtr<mAudioSource> *pAudioSource, IN mAllocator *pAllocator, const mString &filename)
{
  mFUNCTION_SETUP();

  mAudioSourceWav *pAudioSourceWav = nullptr;
  mDEFER_CALL_ON_ERROR(pAudioSource, mSharedPointer_Destroy);
  mERROR_CHECK((mSharedPointer_AllocateInherited<mAudioSource, mAudioSourceWav>(pAudioSource, pAllocator, (std::function<void(mAudioSourceWav *)>)[](mAudioSourceWav *pData) {mAudioSourceWav_Destroy_Internal(pData); }, &pAudioSourceWav)));

  mDEFER_CALL_ON_ERROR(pAudioSource, mSharedPointer_Destroy);

  pAudioSourceWav->volume = 1.0f;
  pAudioSourceWav->seekable = false;

  static_assert(sizeof(decltype(mAudioSourceWav::riffWaveHeader)) == (sizeof(uint32_t) * 5 + sizeof(uint16_t) * 4 + sizeof(uint8_t) * 4 * 4), "struct packing invalid");

  mERROR_CHECK(mCachedFileReader_Create(&pAudioSourceWav->fileReader, pAllocator, filename, 1024 * 1024 * 4));
  mERROR_CHECK(mCachedFileReader_ReadAt(pAudioSourceWav->fileReader, 0, sizeof(decltype(mAudioSourceWav::riffWaveHeader)), (uint8_t *)&pAudioSourceWav->riffWaveHeader));

  mERROR_IF(pAudioSourceWav->riffWaveHeader.riffHeader[0] != 'R' || pAudioSourceWav->riffWaveHeader.riffHeader[1] != 'I' || pAudioSourceWav->riffWaveHeader.riffHeader[2] != 'F' || pAudioSourceWav->riffWaveHeader.riffHeader[3] != 'F', mR_ResourceInvalid);
  mERROR_IF(pAudioSourceWav->riffWaveHeader.waveHeader[0] != 'W' || pAudioSourceWav->riffWaveHeader.waveHeader[1] != 'A' || pAudioSourceWav->riffWaveHeader.waveHeader[2] != 'V' || pAudioSourceWav->riffWaveHeader.waveHeader[3] != 'E', mR_ResourceInvalid);
  mERROR_IF(pAudioSourceWav->riffWaveHeader.fmtSubchunkID[0] != 'f' || pAudioSourceWav->riffWaveHeader.fmtSubchunkID[1] != 'm' || pAudioSourceWav->riffWaveHeader.fmtSubchunkID[2] != 't' || pAudioSourceWav->riffWaveHeader.fmtSubchunkID[3] != ' ', mR_ResourceInvalid);
  mERROR_IF(pAudioSourceWav->riffWaveHeader.dataSubchunkID[0] != 'd' || pAudioSourceWav->riffWaveHeader.dataSubchunkID[1] != 'a' || pAudioSourceWav->riffWaveHeader.dataSubchunkID[2] != 't' || pAudioSourceWav->riffWaveHeader.dataSubchunkID[3] != 'a', mR_ResourceInvalid);
  mERROR_IF(pAudioSourceWav->riffWaveHeader.fmtBitsPerSample != 16, mR_ResourceIncompatible);
  mERROR_IF(pAudioSourceWav->riffWaveHeader.fmtAudioFormat != 1, mR_ResourceIncompatible);
  mERROR_IF(pAudioSourceWav->riffWaveHeader.fmtBlockAlign != 2 && pAudioSourceWav->riffWaveHeader.fmtBlockAlign != 4, mR_ResourceIncompatible);
  mERROR_IF(pAudioSourceWav->riffWaveHeader.fmtChannelCount != 1 && pAudioSourceWav->riffWaveHeader.fmtChannelCount != 2, mR_ResourceIncompatible);

  pAudioSourceWav->readPosition = pAudioSourceWav->startOffset = sizeof(decltype(mAudioSourceWav::riffWaveHeader));
  pAudioSourceWav->sampleRate = (size_t)pAudioSourceWav->riffWaveHeader.fmtSampleRate;
  pAudioSourceWav->channelCount = (size_t)pAudioSourceWav->riffWaveHeader.fmtChannelCount;

  pAudioSourceWav->pGetBufferFunc = mAudioSourceWav_GetBuffer_Internal;
  pAudioSourceWav->pMoveToNextBufferFunc = mAudioSourceWav_MoveToNextBuffer_Internal;

  mRETURN_SUCCESS();
}

mFUNCTION(mAudioSourceWav_Destroy, IN_OUT mPtr<mAudioSource> *pAudioSource)
{
  mFUNCTION_SETUP();

  mERROR_IF(pAudioSource == nullptr, mR_ArgumentNull);

  mERROR_CHECK(mSharedPointer_Destroy(pAudioSource));

  mRETURN_SUCCESS();
}

//////////////////////////////////////////////////////////////////////////

static mFUNCTION(mAudioSourceWav_Destroy_Internal, IN_OUT mAudioSourceWav *pAudioSource)
{
  mFUNCTION_SETUP();

  mERROR_IF(pAudioSource == nullptr, mR_ArgumentNull);

  mERROR_CHECK(mCachedFileReader_Destroy(&pAudioSource->fileReader));

  mRETURN_SUCCESS();
}

static mFUNCTION(mAudioSourceWav_GetBuffer_Internal, mPtr<mAudioSource> &audioSource, OUT float_t *pBuffer, const size_t bufferLength, const size_t channelIndex, OUT size_t *pBufferCount)
{
  mFUNCTION_SETUP();

  mPROFILE_SCOPED("mAudioSourceWav_GetBuffer_Internal");

  mERROR_IF(audioSource == nullptr || pBuffer == nullptr || pBufferCount == nullptr, mR_ArgumentNull);
  mERROR_IF(channelIndex >= audioSource->channelCount, mR_IndexOutOfBounds);
  mERROR_IF(audioSource->pGetBufferFunc != mAudioSourceWav_GetBuffer_Internal, mR_ResourceIncompatible);

  mAudioSourceWav *pAudioSourceWav = static_cast<mAudioSourceWav *>(audioSource.GetPointer());

  const size_t readSize = mMin((bufferLength + pAudioSourceWav->riffWaveHeader.fmtBlockAlign) * sizeof(int16_t) * pAudioSourceWav->channelCount, pAudioSourceWav->riffWaveHeader.dataSubchunkSize - (pAudioSourceWav->readPosition - pAudioSourceWav->startOffset));
  const size_t readItems = mMin(bufferLength * sizeof(int16_t) * pAudioSourceWav->channelCount, pAudioSourceWav->riffWaveHeader.dataSubchunkSize - (pAudioSourceWav->readPosition - pAudioSourceWav->startOffset)) / (sizeof(int16_t) * pAudioSourceWav->channelCount);

  *pBufferCount = readItems;

  int16_t *pData = nullptr;
  mERROR_CHECK(mCachedFileReader_PointerAt(pAudioSourceWav->fileReader, pAudioSourceWav->readPosition, readSize, (uint8_t **)&pData));

  mERROR_CHECK(mAudio_ExtractFloatChannelFromInterleavedInt16(pBuffer, channelIndex, pData, pAudioSourceWav->channelCount, bufferLength));

  if (readItems < bufferLength)
  {
    mERROR_CHECK(mZeroMemory(pBuffer + readItems, bufferLength - readItems));
    pAudioSourceWav->stopPlayback = true;
  }

  mRETURN_SUCCESS();
}

static mFUNCTION(mAudioSourceWav_MoveToNextBuffer_Internal, mPtr<mAudioSource> &audioSource, const size_t samples)
{
  mFUNCTION_SETUP();

  mERROR_IF(audioSource == nullptr, mR_ArgumentNull);
  mERROR_IF(audioSource->pMoveToNextBufferFunc != mAudioSourceWav_MoveToNextBuffer_Internal, mR_ResourceIncompatible);

  mAudioSourceWav *pAudioSourceWav = static_cast<mAudioSourceWav *>(audioSource.GetPointer());

  pAudioSourceWav->readPosition += ((samples * pAudioSourceWav->channelCount) / pAudioSourceWav->riffWaveHeader.fmtBlockAlign) * pAudioSourceWav->riffWaveHeader.fmtBlockAlign * sizeof(int16_t);

  mRETURN_SUCCESS();
}

//////////////////////////////////////////////////////////////////////////

struct mAudioSourceResampler : mAudioSource
{
  float_t *pData;
  size_t dataCapacity;
  SRC_STATE **ppStates;
  mAllocator *pAllocator;
  mPtr<mAudioSource> audioSource;
  mAudio_ResampleQuality resampleQuality;
};

static mFUNCTION(mAudioSourceResampler_Destroy_Internal, mAudioSourceResampler *pResampler);
static mFUNCTION(mAudioSourceResampler_GetBuffer_Internal, mPtr<mAudioSource> &audioSource, OUT float_t *pBuffer, const size_t bufferLength, const size_t channelIndex, OUT size_t *pBufferCount);
static mFUNCTION(mAudioSourceResampler_MoveToNextBuffer_Internal, mPtr<mAudioSource> &audioSource, const size_t samples);
static mFUNCTION(mAudioSourceResampler_SeekSample_Internal, mPtr<mAudioSource> &audioSource, const size_t sample);

//////////////////////////////////////////////////////////////////////////

mFUNCTION(mAudioSourceResampler_Create, OUT mPtr<mAudioSource> *pResampler, IN mAllocator *pAllocator, mPtr<mAudioSource> &sourceAudioSource, const size_t targetSampleRate, const mAudio_ResampleQuality quality /* = mA_RQ_BestQuality */)
{
  mFUNCTION_SETUP();

  mERROR_IF(pResampler == nullptr || sourceAudioSource == nullptr, mR_ArgumentNull);
  mERROR_IF(sourceAudioSource->isBeingConsumed, mR_ResourceStateInvalid);

  mAudioSourceResampler *pInstance = nullptr;
  mDEFER_CALL_ON_ERROR(pResampler, mSharedPointer_Destroy);
  mERROR_CHECK((mSharedPointer_AllocateInherited<mAudioSource, mAudioSourceResampler>(pResampler, pAllocator, [](mAudioSourceResampler *pData) { mAudioSourceResampler_Destroy_Internal(pData); }, &pInstance)));

  pInstance->audioSource = sourceAudioSource;
  pInstance->channelCount = pInstance->audioSource->channelCount;
  pInstance->sampleRate = targetSampleRate;
  pInstance->volume = pInstance->audioSource->volume;
  pInstance->pAllocator = pAllocator;
  pInstance->resampleQuality = quality;

  if (pInstance->sampleRate != pInstance->audioSource->sampleRate)
  {
    const double_t ratio = (double_t)targetSampleRate / (double_t)pInstance->audioSource->sampleRate;
    SRC_STATE **ppSourceStates = nullptr;

    mDEFER_ON_ERROR(mAllocator_FreePtr(pInstance->pAllocator, &ppSourceStates));
    mERROR_CHECK(mAllocator_AllocateZero(pInstance->pAllocator, &ppSourceStates, pInstance->channelCount));

    for (size_t i = 0; i < pInstance->channelCount; i++)
    {
      int error = 0;
      ppSourceStates[i] = src_new((int32_t)quality, 1, &error);
      error |= src_set_ratio(ppSourceStates[i], ratio);

      if (ppSourceStates[i] == nullptr || error != 0)
      {
        for (size_t j = 0; j <= i; j++)
        {
          if (ppSourceStates[j] != nullptr)
          {
            src_delete(ppSourceStates[j]);
            ppSourceStates[j] = nullptr;
          }
        }

        mRETURN_RESULT(mR_InternalError);
      }
    }

    pInstance->ppStates = ppSourceStates;
  }

  pInstance->seekable = pInstance->audioSource->seekable;

  pInstance->pGetBufferFunc = mAudioSourceResampler_GetBuffer_Internal;
  pInstance->pMoveToNextBufferFunc = mAudioSourceResampler_MoveToNextBuffer_Internal;
  
  if (pInstance->seekable)
    pInstance->pSeekSampleFunc = mAudioSourceResampler_SeekSample_Internal;

  pInstance->audioSource->isBeingConsumed = true;

  mRETURN_SUCCESS();
}

//////////////////////////////////////////////////////////////////////////

static mFUNCTION(mAudioSourceResampler_Destroy_Internal, mAudioSourceResampler *pResampler)
{
  mFUNCTION_SETUP();

  mERROR_IF(pResampler == nullptr, mR_ArgumentNull);

  mERROR_CHECK(mSharedPointer_Destroy(&pResampler->audioSource));

  if (pResampler->pData != nullptr)
    mERROR_CHECK(mAllocator_FreePtr(pResampler->pAllocator, &pResampler->pData));

  pResampler->dataCapacity = 0;

  if (pResampler->ppStates != nullptr)
  {
    for (size_t i = 0; i < pResampler->channelCount; i++)
    {
      if (pResampler->ppStates[i] != nullptr)
      {
        src_delete(pResampler->ppStates[i]);
        pResampler->ppStates[i] = nullptr;
      }
    }

    mERROR_CHECK(mAllocator_FreePtr(pResampler->pAllocator, &pResampler->ppStates));
  }

  mRETURN_SUCCESS();
}

static mFUNCTION(mAudioSourceResampler_GetBuffer_Internal, mPtr<mAudioSource> &audioSource, OUT float_t *pBuffer, const size_t bufferLength, const size_t channelIndex, OUT size_t *pBufferCount)
{
  mFUNCTION_SETUP();

  mPROFILE_SCOPED("mAudioSourceResampler_GetBuffer_Internal");

  mERROR_IF(audioSource == nullptr || pBuffer == nullptr || pBufferCount == nullptr, mR_ArgumentNull);
  mERROR_IF(audioSource->pGetBufferFunc != mAudioSourceResampler_GetBuffer_Internal, mR_ResourceIncompatible);

  mAudioSourceResampler *pResampler = static_cast<mAudioSourceResampler *>(audioSource.GetPointer());

  mERROR_IF(pResampler->channelCount <= channelIndex, mR_IndexOutOfBounds);
  mERROR_IF(pResampler->audioSource->pGetBufferFunc == nullptr, mR_NotSupported);

  pResampler->volume = pResampler->audioSource->volume;
  pResampler->stopPlayback |= pResampler->audioSource->stopPlayback;
  pResampler->hasBeenConsumed |= pResampler->audioSource->hasBeenConsumed;

  if (pResampler->audioSource->sampleRate == pResampler->sampleRate)
  {
    mDEFER_ON_ERROR(pResampler->audioSource->hasBeenConsumed = true);
    mERROR_CHECK(pResampler->audioSource->pGetBufferFunc(pResampler->audioSource, pBuffer, bufferLength, channelIndex, pBufferCount));
  }
  else
  {
    if (pResampler->ppStates == nullptr)
    {
      SRC_STATE **ppSourceStates = nullptr;

      mDEFER_ON_ERROR(mAllocator_FreePtr(pResampler->pAllocator, &ppSourceStates));
      mERROR_CHECK(mAllocator_AllocateZero(pResampler->pAllocator, &ppSourceStates, pResampler->channelCount));

      for (size_t i = 0; pResampler->channelCount; i++)
      {
        int error = 0;
        ppSourceStates[i] = src_new((int32_t)pResampler->resampleQuality, 1, &error);
        error |= src_set_ratio(ppSourceStates[i], (double_t)pResampler->sampleRate / (double_t)pResampler->audioSource->sampleRate);

        if (ppSourceStates[i] == nullptr || error != 0)
        {
          for (size_t j = 0; j <= i; j++)
          {
            if (ppSourceStates[j] != nullptr)
            {
              src_delete(ppSourceStates[j]);
              ppSourceStates[j] = nullptr;
            }
          }

          mRETURN_RESULT(mR_InternalError);
        }
      }

      pResampler->ppStates = ppSourceStates;
    }

    const size_t sampleCount = (size_t)round((double_t)bufferLength * (double_t)pResampler->audioSource->sampleRate / (double_t)pResampler->sampleRate);

    if (sampleCount > pResampler->dataCapacity)
    {
      const size_t newDataCapacity = mMax(pResampler->dataCapacity * 2, pResampler->dataCapacity + (sampleCount - pResampler->dataCapacity) * 2);
      
      mERROR_CHECK(mAllocator_Reallocate(pResampler->pAllocator, &pResampler->pData, newDataCapacity));
      pResampler->dataCapacity = newDataCapacity;
    }

    size_t bufferCount = 0;
    
    // GetBuffer
    {
      mDEFER_ON_ERROR(pResampler->audioSource->hasBeenConsumed = true);
      mERROR_CHECK(pResampler->audioSource->pGetBufferFunc(pResampler->audioSource, pResampler->pData, sampleCount, channelIndex, &bufferCount));
    }

    *pBufferCount = (size_t)round((double_t)bufferCount * (double_t)pResampler->sampleRate / (double_t)pResampler->audioSource->sampleRate);

    SRC_DATA data;
    data.data_in = pResampler->pData;
    data.data_out = pBuffer;
    data.end_of_input = sampleCount == bufferCount ? FALSE : TRUE;
    data.input_frames = (int32_t)bufferCount;
    data.output_frames = (int32_t)*pBufferCount;
    data.src_ratio = (double_t)*pBufferCount / (double_t)bufferCount;
    
    mERROR_IF(0 != src_process(pResampler->ppStates[channelIndex], &data), mR_InternalError);
  }

  mRETURN_SUCCESS();
}

static mFUNCTION(mAudioSourceResampler_MoveToNextBuffer_Internal, mPtr<mAudioSource> &audioSource, const size_t samples)
{
  mFUNCTION_SETUP();

  mPROFILE_SCOPED("mAudioSourceResampler_MoveToNextBuffer_Internal");

  mERROR_IF(audioSource == nullptr, mR_ArgumentNull);
  mERROR_IF(audioSource->pMoveToNextBufferFunc != mAudioSourceResampler_MoveToNextBuffer_Internal, mR_ResourceIncompatible);

  mAudioSourceResampler *pResampler = static_cast<mAudioSourceResampler *>(audioSource.GetPointer());

  pResampler->volume = pResampler->audioSource->volume;
  pResampler->stopPlayback |= pResampler->audioSource->stopPlayback;
  pResampler->hasBeenConsumed |= pResampler->audioSource->hasBeenConsumed;

  if (pResampler->audioSource->pMoveToNextBufferFunc != nullptr)
  {
    mDEFER_ON_ERROR(pResampler->audioSource->hasBeenConsumed = true);
    mERROR_CHECK(pResampler->audioSource->pMoveToNextBufferFunc(pResampler->audioSource, samples));
  }

  mRETURN_SUCCESS();
}

static mFUNCTION(mAudioSourceResampler_SeekSample_Internal, mPtr<mAudioSource> &audioSource, const size_t sample)
{
  mFUNCTION_SETUP();

  mPROFILE_SCOPED("mAudioSourceResampler_SeekSample_Internal");

  mERROR_IF(audioSource == nullptr, mR_ArgumentNull);
  mERROR_IF(audioSource->pSeekSampleFunc != mAudioSourceResampler_SeekSample_Internal, mR_ResourceIncompatible);

  mAudioSourceResampler *pResampler = static_cast<mAudioSourceResampler *>(audioSource.GetPointer());

  mERROR_IF(!pResampler->audioSource->seekable || pResampler->audioSource->pSeekSampleFunc == nullptr, mR_NotSupported);

  size_t sourceSample;
  
  if (pResampler->audioSource->sampleRate == pResampler->sampleRate)
  {
    sourceSample = sample;
  }
  else
  {
    sourceSample = (size_t)roundf((float_t)sample * (float_t)pResampler->audioSource->seekable / (float_t)pResampler->sampleRate);

    if (pResampler->ppStates != nullptr)
    {
      mResult result = mR_Success;

      for (size_t i = 0; i < pResampler->channelCount; i++)
        if (0 != src_reset(pResampler->ppStates[i]))
          result = mR_InternalError;

      mERROR_IF(mFAILED(result), result);
    }
  }

  mERROR_CHECK(pResampler->audioSource->pSeekSampleFunc(pResampler->audioSource, sourceSample));

  mRETURN_SUCCESS();
}

//////////////////////////////////////////////////////////////////////////

struct mMidSideStereoDecoder : mAudioSource
{
  mPtr<mAudioSource> audioSource;
  mAllocator *pAllocator;
  float_t *pData;
  size_t dataCapacity, currentDataSize;
  bool retrievedNewBuffersThisFrame;
};

static mFUNCTION(mMidSideStereoDecoder_Destroy_Internal, IN_OUT mMidSideStereoDecoder *pDecoder);
static mFUNCTION(mMidSideStereoDecoder_GetBuffer_Internal, mPtr<mAudioSource> &audioSource, OUT float_t *pBuffer, const size_t bufferLength, const size_t channelIndex, OUT size_t *pBufferCount);
static mFUNCTION(mMidSideStereoDecoder_MoveToNextBuffer_Internal, mPtr<mAudioSource> &audioSource, const size_t samples);
static mFUNCTION(mMidSideStereoDecoder_SeekSample_Internal, mPtr<mAudioSource> &audioSource, const size_t sample);

//////////////////////////////////////////////////////////////////////////

mFUNCTION(mMidSideStereoDecoder_Create, OUT mPtr<mAudioSource> *pMidSideStereoDecoder, IN mAllocator *pAllocator, mPtr<mAudioSource> &stereoAudioSource)
{
  mFUNCTION_SETUP();

  mERROR_IF(pMidSideStereoDecoder == nullptr || stereoAudioSource == nullptr, mR_ArgumentNull);
  mERROR_IF(stereoAudioSource->isBeingConsumed, mR_InvalidParameter);
  mERROR_IF(stereoAudioSource->channelCount != 2, mR_InvalidParameter);

  mMidSideStereoDecoder *pDecoder = nullptr;
  mDEFER_CALL_ON_ERROR(pMidSideStereoDecoder, mSharedPointer_Destroy);
  mERROR_CHECK((mSharedPointer_AllocateInherited<mAudioSource, mMidSideStereoDecoder>(pMidSideStereoDecoder, pAllocator, [](mMidSideStereoDecoder *pData) { mMidSideStereoDecoder_Destroy_Internal(pData); }, &pDecoder)));

  pDecoder->audioSource = stereoAudioSource;

  pDecoder->channelCount = 2;
  pDecoder->seekable = pDecoder->audioSource->seekable;
  pDecoder->sampleRate = pDecoder->audioSource->sampleRate;
  pDecoder->volume = pDecoder->audioSource->volume;
  pDecoder->pAllocator = pAllocator;

  pDecoder->pGetBufferFunc = mMidSideStereoDecoder_GetBuffer_Internal;
  pDecoder->pMoveToNextBufferFunc = mMidSideStereoDecoder_MoveToNextBuffer_Internal;

  if (pDecoder->seekable)
    pDecoder->pSeekSampleFunc = mMidSideStereoDecoder_SeekSample_Internal;

  pDecoder->audioSource->isBeingConsumed = true;

  mRETURN_SUCCESS();
}

//////////////////////////////////////////////////////////////////////////

static mFUNCTION(mMidSideStereoDecoder_Destroy_Internal, IN_OUT mMidSideStereoDecoder *pDecoder)
{
  mFUNCTION_SETUP();

  mERROR_IF(pDecoder == nullptr, mR_ArgumentNull);

  mERROR_CHECK(mSharedPointer_Destroy(&pDecoder->audioSource));

  if (pDecoder->pData != nullptr)
    mERROR_CHECK(mAllocator_FreePtr(pDecoder->pAllocator, &pDecoder->pData));

  mRETURN_SUCCESS();
}

static mFUNCTION(mMidSideStereoDecoder_GetBuffer_Internal, mPtr<mAudioSource> &audioSource, OUT float_t *pBuffer, const size_t bufferLength, const size_t channelIndex, OUT size_t *pBufferCount)
{
  mFUNCTION_SETUP();

  mPROFILE_SCOPED("mMidSideStereoDecoder_GetBuffer_Internal");

  mERROR_IF(audioSource == nullptr || pBufferCount == nullptr || pBuffer == nullptr, mR_ArgumentNull);
  mERROR_IF(channelIndex >= audioSource->channelCount, mR_InvalidParameter);
  mERROR_IF(audioSource->pGetBufferFunc != mMidSideStereoDecoder_GetBuffer_Internal, mR_ResourceIncompatible);

  mMidSideStereoDecoder *pDecoder = static_cast<mMidSideStereoDecoder *>(audioSource.GetPointer());

  pDecoder->volume = pDecoder->audioSource->volume;
  pDecoder->stopPlayback |= pDecoder->audioSource->stopPlayback;
  pDecoder->hasBeenConsumed |= pDecoder->audioSource->hasBeenConsumed;

  if (!pDecoder->retrievedNewBuffersThisFrame)
  {
    pDecoder->retrievedNewBuffersThisFrame = true;

    mERROR_IF(pDecoder->channelCount != 2, mR_ResourceStateInvalid);

    if (pDecoder->dataCapacity < pDecoder->channelCount * bufferLength)
    {
      const size_t newCapacity = pDecoder->channelCount * bufferLength;
      mERROR_CHECK(mAllocator_Reallocate(pDecoder->pAllocator, &pDecoder->pData, newCapacity));
      pDecoder->dataCapacity = newCapacity;
    }

    for (size_t i = 0; i < pDecoder->channelCount; i++)
      mERROR_CHECK(pDecoder->audioSource->pGetBufferFunc(pDecoder->audioSource, pDecoder->pData + i * bufferLength, bufferLength, i, &pDecoder->currentDataSize));

    mERROR_CHECK(mAudio_InplaceMidSideToStereo(pDecoder->pData, pDecoder->pData + bufferLength, pDecoder->currentDataSize));
  }

  mERROR_CHECK(mMemcpy(pBuffer, pDecoder->pData + channelIndex * bufferLength, pDecoder->currentDataSize));

  *pBufferCount = pDecoder->currentDataSize;

  mRETURN_SUCCESS();
}

static mFUNCTION(mMidSideStereoDecoder_MoveToNextBuffer_Internal, mPtr<mAudioSource> &audioSource, const size_t samples)
{
  mFUNCTION_SETUP();

  mPROFILE_SCOPED("mMidSideStereoDecoder_MoveToNextBuffer_Internal");

  mERROR_IF(audioSource == nullptr, mR_ArgumentNull);
  mERROR_IF(audioSource->pMoveToNextBufferFunc != mMidSideStereoDecoder_MoveToNextBuffer_Internal, mR_ResourceIncompatible);

  mMidSideStereoDecoder *pDecoder = static_cast<mMidSideStereoDecoder *>(audioSource.GetPointer());

  pDecoder->volume = pDecoder->audioSource->volume;
  pDecoder->stopPlayback |= pDecoder->audioSource->stopPlayback;
  pDecoder->hasBeenConsumed |= pDecoder->audioSource->hasBeenConsumed;
  pDecoder->sampleRate = pDecoder->audioSource->sampleRate;

  if (pDecoder->audioSource->pMoveToNextBufferFunc != nullptr)
    mERROR_CHECK(pDecoder->audioSource->pMoveToNextBufferFunc(pDecoder->audioSource, samples));

  pDecoder->retrievedNewBuffersThisFrame = false;

  mRETURN_SUCCESS();
}

static mFUNCTION(mMidSideStereoDecoder_SeekSample_Internal, mPtr<mAudioSource> &audioSource, const size_t sample)
{
  mFUNCTION_SETUP();

  mPROFILE_SCOPED("mMidSideStereoDecoder_SeekSample_Internal");

  mERROR_IF(audioSource == nullptr, mR_ArgumentNull);
  mERROR_IF(audioSource->pSeekSampleFunc != mMidSideStereoDecoder_SeekSample_Internal, mR_ResourceIncompatible);

  mMidSideStereoDecoder *pDecoder = static_cast<mMidSideStereoDecoder *>(audioSource.GetPointer());

  pDecoder->volume = pDecoder->audioSource->volume;
  pDecoder->stopPlayback |= pDecoder->audioSource->stopPlayback;
  pDecoder->hasBeenConsumed |= pDecoder->audioSource->hasBeenConsumed;

  if (pDecoder->audioSource->pSeekSampleFunc != nullptr)
    mERROR_CHECK(pDecoder->audioSource->pSeekSampleFunc(pDecoder->audioSource, sample));

  pDecoder->sampleRate = pDecoder->audioSource->sampleRate;

  mRETURN_SUCCESS();
}

//////////////////////////////////////////////////////////////////////////

struct mMidSideSideQuadroDecoder : mAudioSource
{
  mPtr<mAudioSource> mid, leftRight, frontBack;
  size_t midChannelIndex, leftRightChannelIndex, frontBackChannelIndex;
  mAllocator *pAllocator;
  float_t *pData;
  size_t dataCapacity, currentDataSize;
  bool retrievedNewBuffersThisFrame;
};

static mFUNCTION(mMidSideSideQuadroDecoder_Destroy_Internal, IN_OUT mMidSideSideQuadroDecoder *pDecoder);
static mFUNCTION(mMidSideSideQuadroDecoder_GetBuffer_Internal, mPtr<mAudioSource> &audioSource, OUT float_t *pBuffer, const size_t bufferLength, const size_t channelIndex, OUT size_t *pBufferCount);
static mFUNCTION(mMidSideSideQuadroDecoder_MoveToNextBuffer_Internal, mPtr<mAudioSource> &audioSource, const size_t samples);

//////////////////////////////////////////////////////////////////////////

mFUNCTION(mMidSideSideQuadroDecoder_Create, OUT mPtr<mAudioSource> *pMidSideStereoDecoder, IN mAllocator *pAllocator, mPtr<mAudioSource> &midAudioSource, const size_t midAudioSourceChannelIndex, mPtr<mAudioSource> &leftRightAudioSource, const size_t leftRightAudioSourceChannelIndex, mPtr<mAudioSource> &frontBackAudioSource, const size_t frontBackAudioSourceChannelIndex)
{
  mFUNCTION_SETUP();

  mERROR_IF(pMidSideStereoDecoder == nullptr || midAudioSource == nullptr || leftRightAudioSource == nullptr || frontBackAudioSource == nullptr, mR_ArgumentNull);
  mERROR_IF(midAudioSource->isBeingConsumed || leftRightAudioSource->isBeingConsumed || frontBackAudioSource->isBeingConsumed, mR_InvalidParameter);
  mERROR_IF(midAudioSource->channelCount <= midAudioSourceChannelIndex || leftRightAudioSource->channelCount <= leftRightAudioSourceChannelIndex || frontBackAudioSource->channelCount <= frontBackAudioSourceChannelIndex, mR_InvalidParameter);
  mERROR_IF(midAudioSource->sampleRate != leftRightAudioSource->sampleRate || leftRightAudioSource->sampleRate != frontBackAudioSource->channelCount != 1, mR_InvalidParameter);

  mMidSideSideQuadroDecoder *pDecoder = nullptr;
  mDEFER_CALL_ON_ERROR(pMidSideStereoDecoder, mSharedPointer_Destroy);
  mERROR_CHECK((mSharedPointer_AllocateInherited<mAudioSource, mMidSideSideQuadroDecoder>(pMidSideStereoDecoder, pAllocator, [](mMidSideSideQuadroDecoder *pData) { mMidSideSideQuadroDecoder_Destroy_Internal(pData); }, &pDecoder)));

  pDecoder->mid = midAudioSource;
  pDecoder->midChannelIndex = midAudioSourceChannelIndex;
  pDecoder->leftRight = leftRightAudioSource;
  pDecoder->leftRightChannelIndex = leftRightAudioSourceChannelIndex;
  pDecoder->frontBack = frontBackAudioSource;
  pDecoder->frontBackChannelIndex = frontBackAudioSourceChannelIndex;

  pDecoder->channelCount = 4;
  pDecoder->seekable = false;
  pDecoder->sampleRate = pDecoder->mid->sampleRate;
  pDecoder->volume = 1.0f;
  pDecoder->pAllocator = pAllocator;

  pDecoder->pGetBufferFunc = mMidSideStereoDecoder_GetBuffer_Internal;
  pDecoder->pMoveToNextBufferFunc = mMidSideStereoDecoder_MoveToNextBuffer_Internal;

  pDecoder->mid->isBeingConsumed = true;
  pDecoder->leftRight->isBeingConsumed = true;
  pDecoder->frontBack->isBeingConsumed = true;

  mRETURN_SUCCESS();
}

//////////////////////////////////////////////////////////////////////////

static mFUNCTION(mMidSideSideQuadroDecoder_Destroy_Internal, IN_OUT mMidSideSideQuadroDecoder *pDecoder)
{
  mFUNCTION_SETUP();

  mERROR_IF(pDecoder == nullptr, mR_ArgumentNull);

  mERROR_CHECK(mSharedPointer_Destroy(&pDecoder->mid));
  mERROR_CHECK(mSharedPointer_Destroy(&pDecoder->leftRight));
  mERROR_CHECK(mSharedPointer_Destroy(&pDecoder->frontBack));

  if (pDecoder->pData != nullptr)
    mERROR_CHECK(mAllocator_FreePtr(pDecoder->pAllocator, &pDecoder->pData));

  mRETURN_SUCCESS();
}

static mFUNCTION(mMidSideSideQuadroDecoder_GetBuffer_Internal, mPtr<mAudioSource> &audioSource, OUT float_t *pBuffer, const size_t bufferLength, const size_t channelIndex, OUT size_t *pBufferCount)
{
  mFUNCTION_SETUP();

  mPROFILE_SCOPED("mMidSideSideQuadroDecoder_GetBuffer_Internal");

  mERROR_IF(audioSource == nullptr || pBufferCount == nullptr || pBuffer == nullptr, mR_ArgumentNull);
  mERROR_IF(channelIndex >= audioSource->channelCount, mR_InvalidParameter);
  mERROR_IF(audioSource->pGetBufferFunc != mMidSideSideQuadroDecoder_GetBuffer_Internal, mR_ResourceIncompatible);

  mMidSideSideQuadroDecoder *pDecoder = static_cast<mMidSideSideQuadroDecoder *>(audioSource.GetPointer());

  if (!pDecoder->retrievedNewBuffersThisFrame)
  {
    pDecoder->retrievedNewBuffersThisFrame = true;

    mERROR_IF(pDecoder->channelCount != 4, mR_ResourceStateInvalid);

    if (pDecoder->dataCapacity < pDecoder->channelCount * bufferLength)
    {
      const size_t newCapacity = pDecoder->channelCount * bufferLength;
      mERROR_CHECK(mAllocator_Reallocate(pDecoder->pAllocator, &pDecoder->pData, newCapacity));
      pDecoder->dataCapacity = newCapacity;
    }

    size_t midSampleCount, leftRightSampleCount, frontBackSampleCount, minimumSampleCount;

    mERROR_CHECK(pDecoder->mid->pGetBufferFunc(pDecoder->mid, pDecoder->pData + 0 * bufferLength, bufferLength, pDecoder->midChannelIndex, &midSampleCount));

    minimumSampleCount = midSampleCount;

    if (pDecoder->leftRight == pDecoder->mid)
      mERROR_CHECK(pDecoder->leftRight->pGetBufferFunc(pDecoder->leftRight, pDecoder->pData + 1 * bufferLength, bufferLength, pDecoder->leftRightChannelIndex, &leftRightSampleCount));
    else
      mERROR_CHECK(pDecoder->leftRight->pGetBufferFunc(pDecoder->leftRight, pDecoder->pData + 1 * bufferLength, midSampleCount, pDecoder->leftRightChannelIndex, &leftRightSampleCount));

    minimumSampleCount = mMin(midSampleCount, leftRightSampleCount);

    if (pDecoder->frontBack == pDecoder->mid)
     mERROR_CHECK(pDecoder->frontBack->pGetBufferFunc(pDecoder->frontBack, pDecoder->pData + 2 * bufferLength, bufferLength, pDecoder->frontBackChannelIndex, &frontBackSampleCount));
    else if (pDecoder->frontBack == pDecoder->leftRight)
     mERROR_CHECK(pDecoder->frontBack->pGetBufferFunc(pDecoder->frontBack, pDecoder->pData + 2 * bufferLength, midSampleCount, pDecoder->frontBackChannelIndex, &frontBackSampleCount));
    else
      mERROR_CHECK(pDecoder->frontBack->pGetBufferFunc(pDecoder->frontBack, pDecoder->pData + 2 * bufferLength, minimumSampleCount, pDecoder->frontBackChannelIndex, &frontBackSampleCount));

    minimumSampleCount = mMin(midSampleCount, frontBackSampleCount);
    pDecoder->currentDataSize = minimumSampleCount;

    mERROR_CHECK(mAudio_InplaceMidLateralLongitudinalToQuadro(pDecoder->pData, pDecoder->pData + bufferLength, pDecoder->pData + bufferLength * 2, pDecoder->pData + bufferLength * 3, pDecoder->currentDataSize));
  }

  mERROR_CHECK(mMemcpy(pBuffer, pDecoder->pData + channelIndex * bufferLength, pDecoder->currentDataSize));

  *pBufferCount = pDecoder->currentDataSize;

  mRETURN_SUCCESS();
}

static mFUNCTION(mMidSideSideQuadroDecoder_MoveToNextBuffer_Internal, mPtr<mAudioSource> &audioSource, const size_t samples)
{
  mFUNCTION_SETUP();

  mPROFILE_SCOPED("mMidSideSideQuadroDecoder_MoveToNextBuffer_Internal");

  mERROR_IF(audioSource == nullptr, mR_ArgumentNull);
  mERROR_IF(audioSource->pMoveToNextBufferFunc != mMidSideSideQuadroDecoder_MoveToNextBuffer_Internal, mR_ResourceIncompatible);

  mMidSideSideQuadroDecoder *pDecoder = static_cast<mMidSideSideQuadroDecoder *>(audioSource.GetPointer());

  pDecoder->stopPlayback |= pDecoder->mid->stopPlayback | pDecoder->leftRight->stopPlayback | pDecoder->frontBack->stopPlayback;
  pDecoder->hasBeenConsumed |= pDecoder->mid->hasBeenConsumed | pDecoder->leftRight->hasBeenConsumed | pDecoder->frontBack->hasBeenConsumed;

  if (pDecoder->mid->pMoveToNextBufferFunc != nullptr)
    mERROR_CHECK(pDecoder->mid->pMoveToNextBufferFunc(pDecoder->mid, samples));

  if (pDecoder->leftRight != pDecoder->mid)
    if (pDecoder->leftRight->pMoveToNextBufferFunc != nullptr)
      mERROR_CHECK(pDecoder->leftRight->pMoveToNextBufferFunc(pDecoder->leftRight, samples));

  if (pDecoder->frontBack != pDecoder->mid && pDecoder->frontBack != pDecoder->leftRight)
    if (pDecoder->frontBack->pMoveToNextBufferFunc != nullptr)
      mERROR_CHECK(pDecoder->frontBack->pMoveToNextBufferFunc(pDecoder->frontBack, samples));

  pDecoder->retrievedNewBuffersThisFrame = false;

  mRETURN_SUCCESS();
}

//////////////////////////////////////////////////////////////////////////

struct mSinOscillator : mAudioSource
{
  float_t frequency;
  size_t consumedSamples, samplePosition;
};

static mFUNCTION(mSinOscillator_GetBuffer_Internal, mPtr<mAudioSource> &audioSource, OUT float_t *pBuffer, const size_t bufferLength, const size_t channelIndex, OUT size_t *pBufferCount);
static mFUNCTION(mSinOscillator_MoveToNextBuffer_Internal, mPtr<mAudioSource> &audioSource, const size_t samples);

//////////////////////////////////////////////////////////////////////////

mFUNCTION(mSinOscillator_Create, OUT mPtr<mAudioSource> *pOscillator, IN mAllocator *pAllocator, const float_t frequency, const size_t sampleRate)
{
  mFUNCTION_SETUP();

  mERROR_IF(pOscillator == nullptr, mR_ArgumentNull);

  mSinOscillator *pSinOscillator = nullptr;
  mDEFER_CALL_ON_ERROR(pOscillator, mSharedPointer_Destroy);
  mERROR_CHECK((mSharedPointer_AllocateInherited<mAudioSource, mSinOscillator>(pOscillator, pAllocator, nullptr, &pSinOscillator)));

  pSinOscillator->sampleRate = sampleRate;
  pSinOscillator->channelCount = 1;
  pSinOscillator->frequency = frequency / pSinOscillator->sampleRate * mTWOPIf;
  pSinOscillator->volume = 1.f;

  pSinOscillator->pGetBufferFunc = mSinOscillator_GetBuffer_Internal;
  pSinOscillator->pMoveToNextBufferFunc = mSinOscillator_MoveToNextBuffer_Internal;

  mRETURN_SUCCESS();
}

//////////////////////////////////////////////////////////////////////////

static mFUNCTION(mSinOscillator_GetBuffer_Internal, mPtr<mAudioSource> &audioSource, OUT float_t *pBuffer, const size_t bufferLength, const size_t channelIndex, OUT size_t *pBufferCount)
{
  mFUNCTION_SETUP();

  mERROR_IF(audioSource == nullptr || pBuffer == nullptr || pBufferCount == nullptr, mR_ArgumentNull);
  mERROR_IF(channelIndex >= audioSource->channelCount, mR_InvalidParameter);
  mERROR_IF(audioSource->pGetBufferFunc != mSinOscillator_GetBuffer_Internal, mR_ResourceIncompatible);

  mSinOscillator *pSinOscillator = static_cast<mSinOscillator *>(audioSource.GetPointer());

  for (size_t i = 0; i < bufferLength; i++)
    pBuffer[i] = mSin((pSinOscillator->samplePosition + i) * pSinOscillator->frequency);

  pSinOscillator->consumedSamples = *pBufferCount = bufferLength;

  mRETURN_SUCCESS();
}

static mFUNCTION(mSinOscillator_MoveToNextBuffer_Internal, mPtr<mAudioSource> &audioSource, const size_t /* samples */)
{
  mFUNCTION_SETUP();

  mERROR_IF(audioSource == nullptr, mR_ArgumentNull);
  mERROR_IF(audioSource->pMoveToNextBufferFunc != mSinOscillator_MoveToNextBuffer_Internal, mR_ResourceIncompatible);

  mSinOscillator *pSinOscillator = static_cast<mSinOscillator *>(audioSource.GetPointer());

  pSinOscillator->samplePosition += pSinOscillator->consumedSamples;

  mRETURN_SUCCESS();
}

//////////////////////////////////////////////////////////////////////////

struct mLoopingAudioSource : mAudioSource
{
  size_t samplePosition;
  CONST_FIELD size_t startSample, endSample;
  CONST_FIELD mPtr<mAudioSource> innerAudioSource;
  CONST_FIELD mAllocator *pAllocator;
  float_t *pBuffer;
  size_t bufferCount, bufferCapacity;
  bool firstBuffer;
};

static mFUNCTION(mLoopingAudioSource_Destroy_Internal, IN_OUT mLoopingAudioSource *pAudioSource);
static mFUNCTION(mLoopingAudioSource_GetBuffer_Internal, mPtr<mAudioSource> &audioSource, OUT float_t *pBuffer, const size_t bufferLength, const size_t channelIndex, OUT size_t *pBufferCount);
static mFUNCTION(mLoopingAudioSource_MoveToNextBuffer_Internal, mPtr<mAudioSource> &audioSource, const size_t samples);

//////////////////////////////////////////////////////////////////////////

mFUNCTION(mLoopingAudioSource_Create, OUT mPtr<mAudioSource> *pAudioSource, IN mAllocator *pAllocator, mPtr<mAudioSource> &innerAudioSource, const size_t currentSampleIndex, const size_t maxSampleIndex)
{
  mFUNCTION_SETUP();

  mERROR_IF(pAudioSource == nullptr || innerAudioSource == nullptr, mR_ArgumentNull);
  mERROR_IF(currentSampleIndex >= maxSampleIndex, mR_InvalidParameter);
  mERROR_IF(innerAudioSource->isBeingConsumed, mR_ResourceStateInvalid);
  mERROR_IF(innerAudioSource->pGetBufferFunc == nullptr || innerAudioSource->pMoveToNextBufferFunc == nullptr, mR_ResourceStateInvalid);
  mERROR_IF(!innerAudioSource->seekable || innerAudioSource->pSeekSampleFunc == nullptr, mR_ResourceStateInvalid);

  mLoopingAudioSource *pLoopingAudioSource = nullptr;
  mDEFER_CALL_ON_ERROR(pAudioSource, mSharedPointer_Destroy);
  mERROR_CHECK((mSharedPointer_AllocateInherited<mAudioSource, mLoopingAudioSource>(pAudioSource, pAllocator, [](mLoopingAudioSource *pData) { mLoopingAudioSource_Destroy_Internal(pData); }, &pLoopingAudioSource)));

  pLoopingAudioSource->innerAudioSource = innerAudioSource;
  pLoopingAudioSource->seekable = false;
  pLoopingAudioSource->channelCount = innerAudioSource->channelCount;
  pLoopingAudioSource->volume = innerAudioSource->volume;
  pLoopingAudioSource->sampleRate = innerAudioSource->sampleRate;

  pLoopingAudioSource->pAllocator = pAllocator;

  pLoopingAudioSource->startSample = pLoopingAudioSource->samplePosition = currentSampleIndex;
  pLoopingAudioSource->endSample = maxSampleIndex;

  pLoopingAudioSource->pGetBufferFunc = mLoopingAudioSource_GetBuffer_Internal;
  pLoopingAudioSource->pMoveToNextBufferFunc = mLoopingAudioSource_MoveToNextBuffer_Internal;

  pLoopingAudioSource->innerAudioSource->isBeingConsumed = true;

  mRETURN_SUCCESS();
}

//////////////////////////////////////////////////////////////////////////

static mFUNCTION(mLoopingAudioSource_Destroy_Internal, IN_OUT mLoopingAudioSource *pAudioSource)
{
  mFUNCTION_SETUP();

  mERROR_IF(pAudioSource == nullptr, mR_ArgumentNull);

  mERROR_CHECK(mSharedPointer_Destroy(&pAudioSource->innerAudioSource));

  if (pAudioSource->pBuffer != nullptr)
  {
    pAudioSource->bufferCapacity = 0;
    pAudioSource->bufferCount = 0;

    mERROR_CHECK(mAllocator_FreePtr(pAudioSource->pAllocator, &pAudioSource->pBuffer));
  }

  mRETURN_SUCCESS();
}

static mFUNCTION(mLoopingAudioSource_GetBuffer_Internal, mPtr<mAudioSource> &audioSource, OUT float_t *pBuffer, const size_t bufferLength, const size_t channelIndex, OUT size_t *pBufferCount)
{
  mFUNCTION_SETUP();

  mPROFILE_SCOPED("mLoopingAudioSource_GetBuffer_Internal");

  mERROR_IF(audioSource == nullptr || pBufferCount == nullptr || pBuffer == nullptr, mR_ArgumentNull);
  mERROR_IF(channelIndex >= audioSource->channelCount, mR_InvalidParameter);
  mERROR_IF(audioSource->pGetBufferFunc != mLoopingAudioSource_GetBuffer_Internal, mR_ResourceIncompatible);

  mLoopingAudioSource *pAudioSource = static_cast<mLoopingAudioSource *>(audioSource.GetPointer());

  pAudioSource->stopPlayback |= pAudioSource->innerAudioSource->stopPlayback;
  pAudioSource->hasBeenConsumed |= pAudioSource->innerAudioSource->hasBeenConsumed;

  if (pAudioSource->firstBuffer)
  {
    pAudioSource->firstBuffer = false;

    if (pAudioSource->samplePosition + bufferLength > pAudioSource->endSample)
    {
      const size_t cachedSamples = (pAudioSource->samplePosition + bufferLength) - pAudioSource->endSample;

      if (pAudioSource->bufferCapacity < cachedSamples * pAudioSource->channelCount)
      {
        const size_t newBufferCapacity = (cachedSamples * 2 + 1) * pAudioSource->channelCount;
        mERROR_CHECK(mAllocator_Reallocate(pAudioSource->pAllocator, &pAudioSource->pBuffer, newBufferCapacity));
        pAudioSource->bufferCapacity = newBufferCapacity;
      }

      if (cachedSamples != 0)
      {
        size_t sampleCount = 0;

        for (size_t i = 0; i < pAudioSource->channelCount; i++)
        {
          mDEFER_ON_ERROR(pAudioSource->innerAudioSource->hasBeenConsumed = true);
          mERROR_CHECK(pAudioSource->innerAudioSource->pGetBufferFunc(pAudioSource->innerAudioSource, pAudioSource->pBuffer + i * cachedSamples, cachedSamples, i, &sampleCount));
          mERROR_IF(sampleCount != cachedSamples, mR_EndOfStream);
        }

        /* mERROR_CHECK */(pAudioSource->innerAudioSource->pMoveToNextBufferFunc(pAudioSource->innerAudioSource, 0));
        
        pAudioSource->bufferCount = cachedSamples;
      }

      mERROR_CHECK(pAudioSource->innerAudioSource->pSeekSampleFunc(pAudioSource->innerAudioSource, pAudioSource->startSample));
      
      pAudioSource->samplePosition = pAudioSource->startSample - pAudioSource->samplePosition + pAudioSource->endSample;
    }
    else
    {
      pAudioSource->samplePosition += bufferLength;
    }
  }

  if (pAudioSource->bufferCount > 0)
  {
    mERROR_CHECK(mMemmove(pBuffer, pAudioSource->pBuffer + channelIndex * pAudioSource->bufferCount, pAudioSource->bufferCount));
    
    *pBufferCount = pAudioSource->bufferCount;

    if (pAudioSource->bufferCount < bufferLength)
    {
      size_t sampleCount = 0;
      mDEFER_ON_ERROR(pAudioSource->innerAudioSource->hasBeenConsumed = true);
      mERROR_CHECK(pAudioSource->innerAudioSource->pGetBufferFunc(pAudioSource->innerAudioSource, pBuffer + pAudioSource->bufferCount, bufferLength - pAudioSource->bufferCount, channelIndex, &sampleCount));
      *pBufferCount += sampleCount;
    }
  }
  else
  {
    size_t sampleCount = 0;
    mDEFER_ON_ERROR(pAudioSource->innerAudioSource->hasBeenConsumed = true);
    mERROR_CHECK(pAudioSource->innerAudioSource->pGetBufferFunc(pAudioSource->innerAudioSource, pBuffer, bufferLength, channelIndex, &sampleCount));
    *pBufferCount = sampleCount;
  }

  mRETURN_SUCCESS();
}

static mFUNCTION(mLoopingAudioSource_MoveToNextBuffer_Internal, mPtr<mAudioSource> &audioSource, const size_t samples)
{
  mFUNCTION_SETUP();

  mPROFILE_SCOPED("mLoopingAudioSource_MoveToNextBuffer_Internal");

  mERROR_IF(audioSource == nullptr, mR_ArgumentNull);
  mERROR_IF(audioSource->pMoveToNextBufferFunc != mLoopingAudioSource_MoveToNextBuffer_Internal, mR_ResourceIncompatible);

  mLoopingAudioSource *pAudioSource = static_cast<mLoopingAudioSource *>(audioSource.GetPointer());

  pAudioSource->stopPlayback |= pAudioSource->innerAudioSource->stopPlayback;
  pAudioSource->hasBeenConsumed |= pAudioSource->innerAudioSource->hasBeenConsumed;

  pAudioSource->bufferCount = 0;
  pAudioSource->firstBuffer = true;

  // Move to next buffer.
  {
    mDEFER_ON_ERROR(pAudioSource->innerAudioSource->hasBeenConsumed = true);
    mERROR_CHECK(pAudioSource->innerAudioSource->pMoveToNextBufferFunc(pAudioSource->innerAudioSource, samples));
  }

  mRETURN_SUCCESS();
}
