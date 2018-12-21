#ifndef mAudioEngine_h__
#define mAudioEngine_h__

#include "mediaLib.h"
#include "mThreading.h"
#include "mPool.h"
#include "mQueue.h"

enum
{
  mAudioEngine_PreferredSampleRate = 44100,
  mAudioEngine_MaxSupportedAudioSourceSamepleRate = 48000,
  mAudioEngine_MaxSupportedChannelCount = 2,
  mAudioEngine_BufferSize = 1024 * 2,
};

struct mAudioSource
{
  float_t volume;
  size_t sampleRate;
  size_t channelCount;
  bool hasBeenConsumed;
  bool stopPlayback;

  typedef mFUNCTION(mAudioSource_GetBuffer, mPtr<mAudioSource> &audioSource, OUT float_t *pBuffer, const size_t bufferLength, const size_t channelIndex);
  typedef mFUNCTION(mAudioSource_MoveToNextBuffer, mPtr<mAudioSource> &audioSource, const size_t samples);

  mAudioSource_GetBuffer *pGetBufferFunc;
  mAudioSource_MoveToNextBuffer *pMoveToNextBufferFunc;
};

mFUNCTION(mAudioSourceWav_Create, OUT mPtr<mAudioSource> *pAudioSource, IN mAllocator *pAllocator, const mString &filename);
mFUNCTION(mAudioSourceWav_Destroy, IN_OUT mPtr<mAudioSource> *pAudioSource);

struct mAudioEngine
{
  mMutex *pMutex;
  size_t sampleRate;
  size_t channelCount;
  size_t bufferSize;
  mPtr<mPool<mPtr<mAudioSource>>> audioSources;
  mPtr<mQueue<size_t>> unusedAudioSources;
  uint32_t deviceId;
  float_t audioCallbackBuffer[(size_t)mAudioEngine_MaxSupportedChannelCount * (size_t)mAudioEngine_MaxSupportedAudioSourceSamepleRate * (size_t)mAudioEngine_MaxSupportedAudioSourceSamepleRate / (size_t)mAudioEngine_PreferredSampleRate];
};

mFUNCTION(mAudioEngine_Create, OUT mPtr<mAudioEngine> *pAudioEngine, IN mAllocator *pAllocator);
mFUNCTION(mAudioEngine_Destroy, IN_OUT mPtr<mAudioEngine> *pAudioEngine);

mFUNCTION(mAudioEngine_SetPaused, mPtr<mAudioEngine> &audioEngine, const bool paused);

mFUNCTION(mAudioEngine_AddAudioSource, mPtr<mAudioEngine> &audioEngine, mPtr<mAudioSource> &audioSource);

#endif // mAudioEngine_h__
