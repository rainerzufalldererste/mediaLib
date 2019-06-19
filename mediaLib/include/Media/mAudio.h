#ifndef mAudio_h__
#define mAudio_h__

#include "mediaLib.h"

mFUNCTION(mAudio_ExtractFloatChannelFromInterleavedFloat, OUT float_t *pChannel, const size_t channelIndex, IN float_t *pInterleaved, const size_t channelCount, const size_t sampleCount);
mFUNCTION(mAudio_ExtractFloatChannelFromInterleavedInt16, OUT float_t *pChannel, const size_t channelIndex, IN int16_t *pInterleaved, const size_t channelCount, const size_t sampleCount);

mFUNCTION(mAudio_ConvertInt16ToFloat, OUT float_t *pDestination, IN const int16_t *pSource, const size_t sampleCount);
mFUNCTION(mAudio_ConvertFloatToInt16WithDithering, IN int16_t *pDestination, OUT const float_t *pSource, const size_t sampleCount);

mFUNCTION(mAudio_SetInterleavedChannelFloat, OUT float_t *pInterleaved, IN float_t *pChannel, const size_t channelIndex, const size_t channelCount, const size_t sampleCount);
mFUNCTION(mAudio_AddToInterleavedFromChannelWithVolumeFloat, OUT float_t *pInterleaved, IN float_t *pChannel, const size_t channelIndex, const size_t channelCount, const size_t sampleCount, const float_t volume);
mFUNCTION(mAudio_AddToInterleavedBufferFromMonoWithVolumeFloat, OUT float_t *pInterleaved, IN float_t *pChannel, const size_t interleavedChannelCount, const size_t sampleCount, const float_t volume);

mFUNCTION(mAudio_ApplyVolumeFloat, OUT float_t *pAudio, const float_t volume, const size_t sampleCount);
mFUNCTION(mAudio_AddWithVolumeFloat, OUT float_t *pDestination, IN float_t *pSource, const float_t volume, const size_t sampleCount);

mFUNCTION(mAudio_AddResampleToInterleavedFromChannelWithVolume, OUT float_t *pInterleaved, IN float_t *pChannel, const size_t channelIndex, const size_t channelCount, const size_t interleavedSampleCount, const size_t channelSampleCount, const float_t volume);
mFUNCTION(mAudio_AddResampleMonoToInterleavedFromChannelWithVolume, OUT float_t *pInterleaved, IN float_t *pChannel, const size_t channelCount, const size_t interleavedSampleCount, const size_t channelSampleCount, const float_t volume);
mFUNCTION(mAudio_InplaceResampleMono, OUT float_t *pChannel, const size_t currentSampleCount, const size_t desiredSampleCount);

// Fades to the original signal for `fadeSamples` at the beginning and end of the block.
mFUNCTION(mAudio_InplaceResampleMonoWithFade, OUT float_t *pChannel, const size_t currentSampleCount, const size_t desiredSampleCount, const size_t fadeSamples);

mFUNCTION(mAudio_AddResampleMonoToMonoWithVolume, OUT float_t *pDestination, IN float_t *pSource, const size_t currentSampleCount, const size_t desiredSampleCount, const float_t volume);
mFUNCTION(mAudio_ResampleMonoToMonoWithVolume, OUT float_t *pDestination, IN float_t *pSource, const size_t currentSampleCount, const size_t desiredSampleCount, const float_t volume);

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

//////////////////////////////////////////////////////////////////////////

struct mAudioSource
{
  PUBLIC_FIELD volatile float_t volume; // this is a factor, not in decibels.
  CONST_FIELD size_t sampleRate;
  CONST_FIELD size_t channelCount;
  bool hasBeenConsumed;
  bool isBeingConsumed;
  PUBLIC_FIELD volatile bool stopPlayback;
  CONST_FIELD bool seekable;

  typedef mFUNCTION(mAudioSource_GetBuffer, mPtr<mAudioSource> &audioSource, OUT float_t *pBuffer, const size_t bufferLength, const size_t channelIndex, OUT size_t *pBufferCount);
  typedef mFUNCTION(mAudioSource_MoveToNextBuffer, mPtr<mAudioSource> &audioSource, const size_t samples);
  typedef mFUNCTION(mAudioSource_SeekSample, mPtr<mAudioSource> &audioSource, const size_t sampleIndex);
  typedef mFUNCTION(mAudioSource_BroadcastDelay, mPtr<mAudioSource> &audioSource, const size_t samples);

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

  mAudioSource_BroadcastDelay *pBroadcastDelayFunc; // Can be nullptr
};

//////////////////////////////////////////////////////////////////////////

mFUNCTION(mAudioSourceWav_Create, OUT mPtr<mAudioSource> *pAudioSource, IN mAllocator *pAllocator, const mString &filename);
mFUNCTION(mAudioSourceWav_Destroy, IN_OUT mPtr<mAudioSource> *pAudioSource);

//////////////////////////////////////////////////////////////////////////

// From 'libresample.h'.
enum mAudioSourceResampler_Quality
{
  mASR_Q_BestQuality = 0,
  mASR_Q_MediumQuality = 1,
  mASR_Q_Fastest = 2,
  mASR_Q_ZeroOrderHold = 3,
  mASR_Q_Linear = 4,
};

// Attention: mAudioSoure.volume will constantly be set to the volume of the internal sourceAudioSource.
mFUNCTION(mAudioSourceResampler_Create, OUT mPtr<mAudioSource> *pResampler, IN mAllocator *pAllocator, mPtr<mAudioSource> &sourceAudioSource, const size_t targetSampleRate, const mAudioSourceResampler_Quality quality = mASR_Q_BestQuality);

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
