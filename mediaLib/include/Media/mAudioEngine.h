#ifndef mAudioEngine_h__
#define mAudioEngine_h__

#include "mediaLib.h"
#include "mThreading.h"
#include "mPool.h"
#include "mQueue.h"
#include "mAudio.h"

enum
{
  mAudioEngine_PreferredSampleRate = 44100,
  mAudioEngine_MaxSupportedAudioSourceSamepleRate = 48000,
  mAudioEngine_MaxSupportedChannelCount = 2,
  mAudioEngine_BufferSize = 1024 * 2,
};

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
  mAllocator *pAllocator;
};

mFUNCTION(mAudioEngine_Create, OUT mPtr<mAudioEngine> *pAudioEngine, IN mAllocator *pAllocator);
mFUNCTION(mAudioEngine_Destroy, IN_OUT mPtr<mAudioEngine> *pAudioEngine);

mFUNCTION(mAudioEngine_SetPaused, mPtr<mAudioEngine> &audioEngine, const bool paused);

mFUNCTION(mAudioEngine_AddAudioSource, mPtr<mAudioEngine> &audioEngine, mPtr<mAudioSource> &audioSource);

#endif // mAudioEngine_h__
