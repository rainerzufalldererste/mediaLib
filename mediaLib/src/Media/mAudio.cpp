#include "mAudio.h"

#include "mCachedFileReader.h"

#include "samplerate.h"

#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable: 4752)
#endif

#ifndef _MSC_VER
__attribute__((target("sse4.1")))
#endif

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

    if (mCpuExtensions::avx2Supported && sampleCount * sizeof(float_t) < sizeof(__m256))
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
    else
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

mFUNCTION(mAudio_ExtractFloatChannelFromInterleavedInt16, OUT float_t *pChannel, const size_t channelIndex, IN int16_t *pInterleaved, const size_t channelCount, const size_t sampleCount)
{
  mFUNCTION_SETUP();

  mERROR_IF(pChannel == nullptr || pInterleaved == nullptr, mR_ArgumentNull);

  mCpuExtensions::Detect();

  if (mCpuExtensions::sse41Supported && channelCount <= 2 && sampleCount * sizeof(float_t) > sizeof(__m128))
  {
    size_t sample = 0;

    const __m128 div = _mm_set1_ps(1.f / (float_t)(INT16_MAX));

    if (channelCount == 1)
    {
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
    else
    {
      constexpr size_t loopSize = ((sizeof(__m128i) / 2) / sizeof(int16_t));

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

mFUNCTION(mAudio_ConvertInt16ToFloat, OUT float_t *pDestination, IN const int16_t *pSource, const size_t sampleCount)
{
  mFUNCTION_SETUP();

  mERROR_IF(pDestination == nullptr || pSource == nullptr, mR_ArgumentNull);

  mCpuExtensions::Detect();

  const float_t div = 1.f / mMaxValue<int16_t>();

  if (mCpuExtensions::avx2Supported)
  {
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
  else if (mCpuExtensions::sse41Supported)
  {
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
  else
  {
    for (size_t sampleIndex = 0; sampleIndex < sampleCount; sampleIndex++)
      pDestination[sampleIndex] = (float_t)pSource[sampleIndex] * div;
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
mFUNCTION(mAudio_ApplyVolumeFloat, OUT float_t *pAudio, const float_t volume, const size_t sampleCount)
{
  mFUNCTION_SETUP();

  mERROR_IF(pAudio == nullptr, mR_ArgumentNull);

  mCpuExtensions::Detect();

  if (mCpuExtensions::avxSupported)
  {
    __m256 v;

    // If too small.
    if (sampleCount * sizeof(float_t) < sizeof(v))
    {
      for (size_t i = 0; i < sampleCount; i++)
        pAudio[i] *= volume;

      mRETURN_SUCCESS();
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

thread_local static float_t *_pResampled = nullptr;
thread_local static size_t _resampledSampleCount = 0;

mFUNCTION(mAudio_AddResampleToInterleavedFromChannelWithVolume, OUT float_t *pInterleaved, IN float_t *pChannel, const size_t channelIndex, const size_t channelCount, const size_t interleavedSampleCount, const size_t channelSampleCount, const float_t volume)
{
  mFUNCTION_SETUP();

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

  mERROR_IF(0 != src_simple(&data, SRC_SINC_BEST_QUALITY, 1), mR_InternalError);

  mERROR_CHECK(mAudio_AddToInterleavedFromChannelWithVolumeFloat(pInterleaved, _pResampled, channelIndex, channelCount, interleavedSampleCount, volume));

  mRETURN_SUCCESS();
}

mFUNCTION(mAudio_AddResampleMonoToInterleavedFromChannelWithVolume, OUT float_t *pInterleaved, IN float_t *pChannel, const size_t channelCount, const size_t interleavedSampleCount, const size_t channelSampleCount, const float_t volume)
{
  mFUNCTION_SETUP();

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

  mERROR_IF(0 != src_simple(&data, SRC_SINC_BEST_QUALITY, 1), mR_InternalError);

  mERROR_CHECK(mAudio_AddToInterleavedBufferFromMonoWithVolumeFloat(pInterleaved, _pResampled, channelCount, interleavedSampleCount, volume));

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

mFUNCTION(mAudioSourceWav_Destroy_Internal, IN_OUT mAudioSourceWav *pAudioSource);
mFUNCTION(mAudioSourceWav_GetBuffer_Internal, mPtr<mAudioSource> &audioSource, OUT float_t *pBuffer, const size_t bufferLength, const size_t channelIndex, OUT size_t *pBufferCount);
mFUNCTION(mAudioSourceWav_MoveToNextBuffer_Internal, mPtr<mAudioSource> &audioSource, const size_t samples);

//////////////////////////////////////////////////////////////////////////

mFUNCTION(mAudioSourceWav_Create, OUT mPtr<mAudioSource> *pAudioSource, IN mAllocator *pAllocator, const mString &filename)
{
  mFUNCTION_SETUP();

  mAudioSourceWav *pAudioSourceWav = nullptr;
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

mFUNCTION(mAudioSourceWav_Destroy_Internal, IN_OUT mAudioSourceWav *pAudioSource)
{
  mFUNCTION_SETUP();

  mERROR_IF(pAudioSource == nullptr, mR_ArgumentNull);

  mERROR_CHECK(mCachedFileReader_Destroy(&pAudioSource->fileReader));

  mRETURN_SUCCESS();
}

mFUNCTION(mAudioSourceWav_GetBuffer_Internal, mPtr<mAudioSource> &audioSource, OUT float_t *pBuffer, const size_t bufferLength, const size_t channelIndex, OUT size_t *pBufferCount)
{
  mFUNCTION_SETUP();

  mERROR_IF(audioSource == nullptr || pBuffer == nullptr || pBufferCount == nullptr, mR_ArgumentNull);
  mERROR_IF(channelIndex >= audioSource->channelCount, mR_IndexOutOfBounds);

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

mFUNCTION(mAudioSourceWav_MoveToNextBuffer_Internal, mPtr<mAudioSource> &audioSource, const size_t samples)
{
  mFUNCTION_SETUP();

  mERROR_IF(audioSource == nullptr, mR_ArgumentNull);

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
  mAudioSourceResampler_Quality resampleQuality;
};

mFUNCTION(mAudioSourceResampler_Destroy_Internal, mAudioSourceResampler *pResampler);
mFUNCTION(mAudioSourceResampler_GetBuffer_Internal, mPtr<mAudioSource> &audioSource, OUT float_t *pBuffer, const size_t bufferLength, const size_t channelIndex, OUT size_t *pBufferCount);
mFUNCTION(mAudioSourceResampler_MoveToNextBuffer_Internal, mPtr<mAudioSource> &audioSource, const size_t samples);
mFUNCTION(mAudioSourceResampler_SeekSample_Internal, mPtr<mAudioSource> &audioSource, const size_t sample);

//////////////////////////////////////////////////////////////////////////

mFUNCTION(mAudioSourceResampler_Create, OUT mPtr<mAudioSource> *pResampler, IN mAllocator *pAllocator, mPtr<mAudioSource> &sourceAudioSource, const size_t targetSampleRate, const mAudioSourceResampler_Quality quality /* = mASR_Q_BestQuality */)
{
  mFUNCTION_SETUP();

  mERROR_IF(pResampler == nullptr || sourceAudioSource == nullptr, mR_ArgumentNull);
  mERROR_IF(sourceAudioSource->isBeingConsumed, mR_ResourceStateInvalid);

  mAudioSourceResampler *pInstance = nullptr;
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

mFUNCTION(mAudioSourceResampler_Destroy_Internal, mAudioSourceResampler *pResampler)
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

mFUNCTION(mAudioSourceResampler_GetBuffer_Internal, mPtr<mAudioSource> &audioSource, OUT float_t *pBuffer, const size_t bufferLength, const size_t channelIndex, OUT size_t *pBufferCount)
{
  mFUNCTION_SETUP();

  mERROR_IF(audioSource == nullptr || pBuffer == nullptr || pBufferCount == nullptr, mR_ArgumentNull);

  mAudioSourceResampler *pResampler = static_cast<mAudioSourceResampler *>(audioSource.GetPointer());

  mERROR_IF(pResampler->channelCount <= channelIndex, mR_IndexOutOfBounds);
  mERROR_IF(pResampler->audioSource->pGetBufferFunc == nullptr, mR_NotSupported);

  pResampler->volume = pResampler->audioSource->volume;
  pResampler->stopPlayback |= pResampler->audioSource->stopPlayback;
  pResampler->hasBeenConsumed |= pResampler->audioSource->hasBeenConsumed;

  if (pResampler->audioSource->sampleRate == pResampler->sampleRate)
  {
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
    mERROR_CHECK(pResampler->audioSource->pGetBufferFunc(pResampler->audioSource, pResampler->pData, sampleCount, channelIndex, &bufferCount));

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

mFUNCTION(mAudioSourceResampler_MoveToNextBuffer_Internal, mPtr<mAudioSource> &audioSource, const size_t samples)
{
  mFUNCTION_SETUP();

  mERROR_IF(audioSource == nullptr, mR_ArgumentNull);

  mAudioSourceResampler *pResampler = static_cast<mAudioSourceResampler *>(audioSource.GetPointer());

  pResampler->volume = pResampler->audioSource->volume;
  pResampler->stopPlayback |= pResampler->audioSource->stopPlayback;
  pResampler->hasBeenConsumed |= pResampler->audioSource->hasBeenConsumed;

  if (pResampler->audioSource->pMoveToNextBufferFunc != nullptr)
    mERROR_CHECK(pResampler->audioSource->pMoveToNextBufferFunc(pResampler->audioSource, samples));

  mRETURN_SUCCESS();
}

mFUNCTION(mAudioSourceResampler_SeekSample_Internal, mPtr<mAudioSource> &audioSource, const size_t sample)
{
  mFUNCTION_SETUP();

  mERROR_IF(audioSource == nullptr, mR_ArgumentNull);

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
