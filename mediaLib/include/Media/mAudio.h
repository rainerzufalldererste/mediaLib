#ifndef mAudio_h__
#define mAudio_h__

#include "mediaLib.h"

#ifdef GIT_BUILD // Define __M_FILE__
  #ifdef __M_FILE__
    #undef __M_FILE__
  #endif
  #define __M_FILE__ "w+GVUgE6PqC/G296f+6XL0mgKl7viykMlk1XCUlwBBh4X06l+LPg/8VYeRjlW8GJBS7iJzwDhq1TGnsn"
#endif

// From 'libresample.h'.
enum mAudio_ResampleQuality
{
  mA_RQ_BestQuality = 0,
  mA_RQ_MediumQuality = 1,
  mA_RQ_Fastest = 2,
  mA_RQ_ZeroOrderHold = 3,
  mA_RQ_Linear = 4,
};

mFUNCTION(mAudio_ExtractFloatChannelFromInterleavedFloat, OUT float_t *pChannel, const size_t channelIndex, IN float_t *pInterleaved, const size_t channelCount, const size_t sampleCount);
mFUNCTION(mAudio_ExtractFloatChannelFromInterleavedInt16, OUT float_t *pChannel, const size_t channelIndex, IN int16_t *pInterleaved, const size_t channelCount, const size_t sampleCount);

mFUNCTION(mAudio_ConvertInt16ToFloat, OUT float_t *pDestination, IN const int16_t *pSource, const size_t sampleCount);
mFUNCTION(mAudio_ConvertFloatToInt16WithDithering, IN int16_t *pDestination, OUT const float_t *pSource, const size_t sampleCount);
mFUNCTION(mAudio_ConvertFloatToInt16WithDitheringAndFactor, IN int16_t *pDestination, OUT const float_t *pSource, const size_t sampleCount, const float_t factor);

mFUNCTION(mAudio_SetInterleavedChannelFloat, OUT float_t *pInterleaved, IN float_t *pChannel, const size_t channelIndex, const size_t channelCount, const size_t sampleCount);
mFUNCTION(mAudio_AddToInterleavedFromChannelWithVolumeFloat, OUT float_t *pInterleaved, IN float_t *pChannel, const size_t channelIndex, const size_t channelCount, const size_t sampleCount, const float_t volume);
mFUNCTION(mAudio_AddToInterleavedBufferFromMonoWithVolumeFloat, OUT float_t *pInterleaved, IN float_t *pChannel, const size_t interleavedChannelCount, const size_t sampleCount, const float_t volume);

mFUNCTION(mAudio_ApplyVolumeFloat, OUT float_t *pAudio, const float_t volume, const size_t sampleCount);
mFUNCTION(mAudio_AddWithVolumeFloat, OUT float_t *pDestination, IN float_t *pSource, const float_t volume, const size_t sampleCount);
mFUNCTION(mAudio_AddWithVolumeFloatVariableLowpass, OUT float_t *pDestination, IN float_t *pSource, const float_t volume, const size_t sampleCount, const float_t lowPassStrength);
mFUNCTION(mAudio_AddWithVolumeFloatVariableOffAxisFilter, OUT float_t *pDestination, IN float_t *pSource, const float_t volume, const size_t sampleCount, const float_t filterStrength);

mFUNCTION(mAudio_AddResampleToInterleavedFromChannelWithVolume, OUT float_t *pInterleaved, IN float_t *pChannel, const size_t channelIndex, const size_t channelCount, const size_t interleavedSampleCount, const size_t channelSampleCount, const float_t volume, const mAudio_ResampleQuality quality);
mFUNCTION(mAudio_AddResampleMonoToInterleavedFromChannelWithVolume, OUT float_t *pInterleaved, IN float_t *pChannel, const size_t channelCount, const size_t interleavedSampleCount, const size_t channelSampleCount, const float_t volume, const mAudio_ResampleQuality quality);
mFUNCTION(mAudio_InplaceResampleMono, OUT float_t *pChannel, const size_t currentSampleCount, const size_t desiredSampleCount, const mAudio_ResampleQuality quality);

// Fades to the original signal for `fadeSamples` at the beginning and end of the block.
mFUNCTION(mAudio_InplaceResampleMonoWithFade, OUT float_t *pChannel, const size_t currentSampleCount, const size_t desiredSampleCount, const size_t fadeSamples, const mAudio_ResampleQuality quality);

mFUNCTION(mAudio_ResampleMonoToMonoWithVolume, OUT float_t *pDestination, IN float_t *pSource, const size_t currentSampleCount, const size_t desiredSampleCount, const float_t volume, const mAudio_ResampleQuality quality);
mFUNCTION(mAudio_AddResampleMonoToMonoWithVolume, OUT float_t *pDestination, IN float_t *pSource, const size_t currentSampleCount, const size_t desiredSampleCount, const float_t volume, const mAudio_ResampleQuality quality);
mFUNCTION(mAudio_AddResampleMonoToMonoWithVolumeVariableLowpass, OUT float_t *pDestination, IN float_t *pSource, const size_t currentSampleCount, const size_t desiredSampleCount, const float_t volume, const float_t lowPassStrength, const mAudio_ResampleQuality quality);
mFUNCTION(mAudio_AddResampleMonoToMonoWithVolumeVariableOffAxisFilter, OUT float_t *pDestination, IN float_t *pSource, const size_t currentSampleCount, const size_t desiredSampleCount, const float_t volume, const float_t filterStrength, const mAudio_ResampleQuality quality);

mFUNCTION(mAudio_InplaceMidSideToStereo, IN_OUT float_t *pMidToLeft, IN_OUT float_t *pSideToRight, const size_t sampleCount);
mFUNCTION(mAudio_InplaceMidLateralLongitudinalToQuadro, IN_OUT float_t *pMidToFrontLeft, IN_OUT float_t *pLateralToFrontRight, IN_OUT float_t *pLongitudinalToBackLeft, OUT float_t *pBackRight, const size_t sampleCount);
mFUNCTION(mAudio_InterleavedQuadroMidLateralLFELongitudinalToDualInterleavedStereo, IN const float_t *pQuadroMidLateralLFELongitudinal, OUT float_t *pStereoChannelFront, OUT float_t *pStereoChannelBack, const size_t sampleCount);

mFUNCTION(mAudio_GetAbsMax, IN float_t *pBuffer, const size_t count, OUT float_t *pMax);

//////////////////////////////////////////////////////////////////////////

inline float_t mAudio_FactorToDecibel(const float_t factor)
{
  return mLog10(factor) * 20.f;
}

inline float_t mAudio_DecibelToFactor(const float_t decibel)
{
  return mPow(10.f, decibel * 0.05f);
}

// See: http://www.sengpielaudio.com/calculator-levelchange.htm
inline float_t mAudio_LoudnessFactorToDecibel(const float_t factor)
{
  return 10.f * mLog2(factor);
}

inline float_t mAudio_DecibelToLoudnessFactor(const float_t decibel)
{
  return mPow(2.f, decibel * 0.1f);
}

inline float_t mAudio_LoudnessFactorToFactor(const float_t factor)
{
  return mAudio_DecibelToFactor(mAudio_LoudnessFactorToDecibel(factor));
}

inline float_t mAudio_FactorToLoudnessFactor(const float_t factor)
{
  return mAudio_DecibelToLoudnessFactor(mAudio_FactorToDecibel(factor));
}

//////////////////////////////////////////////////////////////////////////

struct mAudioSource_PerformanceInfo
{
  float_t processingTimeMs;
};

struct mAudioSource
{
  PUBLIC_FIELD volatile float_t volume; // this is a factor, not in decibels.
  CONST_FIELD size_t sampleRate;
  CONST_FIELD size_t channelCount;
  bool hasBeenConsumed;
  bool isBeingConsumed;
  PUBLIC_FIELD volatile bool stopPlayback;
  CONST_FIELD bool seekable;
  mAudioSource_PerformanceInfo performanceInfo; // may or may not be unused by consumer.

  typedef mFUNCTION(mAudioSource_GetBuffer, mPtr<mAudioSource> &audioSource, OUT float_t *pBuffer, const size_t bufferLength, const size_t channelIndex, OUT size_t *pBufferCount);
  typedef mFUNCTION(mAudioSource_MoveToNextBuffer, mPtr<mAudioSource> &audioSource, const size_t samples);
  typedef mFUNCTION(mAudioSource_SeekSample, mPtr<mAudioSource> &audioSource, const size_t sampleIndex);
  typedef mFUNCTION(mAudioSource_BroadcastDelay, mPtr<mAudioSource> &audioSource, const size_t samples);
  typedef mFUNCTION(mAudioSource_BroadcastBottleneck, mPtr<mAudioSource> &audioSource, const float_t referenceTimeMs);

  // In order to retrieve data at first `pGetBufferFunc` will be called for the channels (in any order), afterwards `pMoveToNextBufferFunc` will be called once per object.
  // All channels have to be asked for the same `bufferLength`.
  // Should return `mR_EndOfStream` when the stream has ended.
  mAudioSource_GetBuffer *pGetBufferFunc; // Should never be nullptr.

  // In order to retrieve data at first `pGetBufferFunc` will be called for the channels (in any order), afterwards `pMoveToNextBufferFunc` will be called once per object.
  // `samples` can vary from the `bufferLength` that was passed to `pGetBufferFunc`.
  // Should return `mR_EndOfStream` when the stream has ended.
  mAudioSource_MoveToNextBuffer *pMoveToNextBufferFunc; // Should never be nullptr.

  // `pSeekSampleFunc` should be called before `pGetBufferFunc` is called and (if the audio source was previously iterated) after `pMoveToNextBufferFunc` was called.
  mAudioSource_SeekSample *pSeekSampleFunc; // Can be nullptr if not seekable.

  mAudioSource_BroadcastDelay *pBroadcastDelayFunc; // Can be nullptr.

  mAudioSource_BroadcastBottleneck *pBroadcastBottleneckFunc; // Can be nullptr.
};

//////////////////////////////////////////////////////////////////////////

mFUNCTION(mAudioSourceWav_Create, OUT mPtr<mAudioSource> *pAudioSource, IN mAllocator *pAllocator, const mString &filename);
mFUNCTION(mAudioSourceWav_Destroy, IN_OUT mPtr<mAudioSource> *pAudioSource);

//////////////////////////////////////////////////////////////////////////

// Attention: mAudioSoure.volume will constantly be set to the volume of the internal sourceAudioSource.
mFUNCTION(mAudioSourceResampler_Create, OUT mPtr<mAudioSource> *pResampler, IN mAllocator *pAllocator, mPtr<mAudioSource> &sourceAudioSource, const size_t targetSampleRate, const mAudio_ResampleQuality quality = mA_RQ_BestQuality);

//////////////////////////////////////////////////////////////////////////

// Attention: mAudioSoure.volume will constantly be set to the volume of the internal sourceAudioSource.
mFUNCTION(mMidSideStereoDecoder_Create, OUT mPtr<mAudioSource> *pMidSideStereoDecoder, IN mAllocator *pAllocator, mPtr<mAudioSource> &stereoAudioSource);

//////////////////////////////////////////////////////////////////////////

// Attention: volume of the different audio sources will be ignored.
mFUNCTION(mMidSideSideQuadroDecoder_Create, OUT mPtr<mAudioSource> *pMidSideStereoDecoder, IN mAllocator *pAllocator, mPtr<mAudioSource> &midAudioSource, const size_t midAudioSourceChannelIndex, mPtr<mAudioSource> &leftRightAudioSource, const size_t leftRightAudioSourceChannelIndex, mPtr<mAudioSource> &frontBackAudioSource, const size_t frontBackAudioSourceChannelIndex);

//////////////////////////////////////////////////////////////////////////

mFUNCTION(mSinOscillator_Create, OUT mPtr<mAudioSource> *pOscillator, IN mAllocator *pAllocator, const float_t frequency, const size_t sampleRate);

//////////////////////////////////////////////////////////////////////////

mFUNCTION(mLoopingAudioSource_Create, OUT mPtr<mAudioSource> *pAudioSource, IN mAllocator *pAllocator, mPtr<mAudioSource> &innerAudioSource, const size_t currentSampleIndex, const size_t maxSampleIndex);

#endif // mAudio_h__
