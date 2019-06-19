#ifndef mAudioEngine_h__
#define mAudioEngine_h__

#include "mediaLib.h"
#include "mAudio.h"

constexpr size_t mAudioEngine_PreferredSampleRate = 44100;
constexpr size_t mAudioEngine_MaxSupportedAudioSourceSamepleRate = 48000;
constexpr size_t mAudioEngine_MaxSupportedChannelCount = 2;
constexpr size_t mAudioEngine_BufferSize = 1024;

struct mAudioEngine;

mFUNCTION(mAudioEngine_Create, OUT mPtr<mAudioEngine> *pAudioEngine, IN mAllocator *pAllocator);
mFUNCTION(mAudioEngine_Destroy, IN_OUT mPtr<mAudioEngine> *pAudioEngine);

mFUNCTION(mAudioEngine_SetPaused, mPtr<mAudioEngine> &audioEngine, const bool paused);

mFUNCTION(mAudioEngine_AddAudioSource, mPtr<mAudioEngine> &audioEngine, mPtr<mAudioSource> &audioSource);

#endif // mAudioEngine_h__
