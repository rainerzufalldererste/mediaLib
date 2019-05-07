#ifndef mAudio_h__
#define mAudio_h__

#include "mediaLib.h"

mFUNCTION(mAudio_ExtractFloatChannelFromInterleavedFloat, OUT float_t *pChannel, const size_t channelIndex, IN float_t *pInterleaved, const size_t channelCount, const size_t sampleCount);
mFUNCTION(mAudio_ExtractFloatChannelFromInterleavedInt16, OUT float_t *pChannel, const size_t channelIndex, IN int16_t *pInterleaved, const size_t channelCount, const size_t sampleCount);

mFUNCTION(mAudio_ConvertInt16ToFloat, OUT float_t *pDestination, IN const int16_t *pSource, const size_t sampleCount);

mFUNCTION(mAudio_SetInterleavedChannelFloat, OUT float_t *pInterleaved, IN float_t *pChannel, const size_t channelIndex, const size_t channelCount, const size_t sampleCount);
mFUNCTION(mAudio_AddToInterleavedFromChannelWithVolumeFloat, OUT float_t *pInterleaved, IN float_t *pChannel, const size_t channelIndex, const size_t channelCount, const size_t sampleCount, const float_t volume);
mFUNCTION(mAudio_AddToInterleavedBufferFromMonoWithVolumeFloat, OUT float_t *pInterleaved, IN float_t *pChannel, const size_t interleavedChannelCount, const size_t sampleCount, const float_t volume);

mFUNCTION(mAudio_ApplyVolumeFloat, OUT float_t *pAudio, const float_t volume, const size_t sampleCount);

mFUNCTION(mAudio_AddResampleToInterleavedFromChannelWithVolume, OUT float_t *pInterleaved, IN float_t *pChannel, const size_t channelIndex, const size_t channelCount, const size_t interleavedSampleCount, const size_t channelSampleCount, const float_t volume);
mFUNCTION(mAudio_AddResampleMonoToInterleavedFromChannelWithVolume, OUT float_t *pInterleaved, IN float_t *pChannel, const size_t channelCount, const size_t interleavedSampleCount, const size_t channelSampleCount, const float_t volume);

//////////////////////////////////////////////////////////////////////////

struct mAudioSource
{
  float_t volume;
  size_t sampleRate;
  size_t channelCount;
  bool hasBeenConsumed;
  bool isBeingConsumed;
  bool stopPlayback;
  bool seekable;

  typedef mFUNCTION(mAudioSource_GetBuffer, mPtr<mAudioSource> &audioSource, OUT float_t *pBuffer, const size_t bufferLength, const size_t channelIndex, OUT size_t *pBufferCount);
  typedef mFUNCTION(mAudioSource_MoveToNextBuffer, mPtr<mAudioSource> &audioSource, const size_t samples);
  typedef mFUNCTION(mAudioSource_SeekSample, mPtr<mAudioSource> &audioSource, const size_t sampleIndex);

  mAudioSource_GetBuffer *pGetBufferFunc;
  mAudioSource_MoveToNextBuffer *pMoveToNextBufferFunc;
  mAudioSource_SeekSample *pSeekSampleFunc;
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

#endif // mAudio_h__
