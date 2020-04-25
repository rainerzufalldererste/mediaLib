#ifndef mAudioEngine_h__
#define mAudioEngine_h__

#include "mediaLib.h"
#include "mAudio.h"

#ifdef GIT_BUILD // Define __M_FILE__
  #ifdef __M_FILE__
    #undef __M_FILE__
  #endif
  #define __M_FILE__ "U6nU261yzgQqbyvV6YEVLoJNVUisYNFBtP8JtVJO2PTl1DvOWGviEdpVXWT3/udbo0lj/L9iM6EYUp+f"
#endif

constexpr size_t mAudioEngine_PreferredSampleRate = 44100;
constexpr size_t mAudioEngine_MaxSupportedAudioSourceSamepleRate = 48000;
constexpr size_t mAudioEngine_MaxSupportedChannelCount = 2;
constexpr size_t mAudioEngine_BufferSize = 1024;

struct mAudioEngine;

mFUNCTION(mAudioEngine_Create, OUT mPtr<mAudioEngine> *pAudioEngine, IN mAllocator *pAllocator);
mFUNCTION(mAudioEngine_Destroy, IN_OUT mPtr<mAudioEngine> *pAudioEngine);

mFUNCTION(mAudioEngine_SetPaused, mPtr<mAudioEngine> &audioEngine, const bool paused);

// `volume`: Factor that'll be applied to the final buffer.
mFUNCTION(mAudioEngine_SetVolume, mPtr<mAudioEngine> &audioEngine, const float_t volume);
mFUNCTION(mAudioEngine_GetVolume, mPtr<mAudioEngine> &audioEngine, OUT float_t *pVolume);

mFUNCTION(mAudioEngine_AddAudioSource, mPtr<mAudioEngine> &audioEngine, mPtr<mAudioSource> &audioSource);

#endif // mAudioEngine_h__
