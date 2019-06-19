#ifndef mAudioScene_h__
#define mAudioScene_h__

#include "mediaLib.h"
#include "mAudio.h"

struct mSpacialAudioSource
{
  CONST_FIELD mPtr<mAudioSource> monoAudioSource;
  CONST_FIELD size_t channelIndex;
  PUBLIC_FIELD volatile mVec3f nextPosition; // in Meters.
  CONST_FIELD std::function<mResult(mSpacialAudioSource *pAudioSource, const size_t samples)> updateCallback;
};

struct mVirtualMicrophone
{
  PUBLIC_FIELD volatile mVec3f nextPosition; // in Meters.
  PUBLIC_FIELD volatile mVec3f nextForward;
  
  // < 1: Wider than dotp of audio source position and forward.
  // > 1: Narrower than dotp of audio source position and forward.
  //
  // Examples:
  // 0 = Omni directional with 1 >= `polarPatternRaise` > 0.
  // 1 = Cardioid with `polarPatternRaise` = 0.
  PUBLIC_FIELD volatile float_t polarPatternWidth;
  
  // Minimum Volume Factor of a spacial audio source based on its angle to the microphone.
  // Should be 0 <= x <= 1.
  PUBLIC_FIELD volatile float_t polarPatternRaise;

  PUBLIC_FIELD volatile float_t behindFilterStrength;
  CONST_FIELD std::function<mResult(mVirtualMicrophone *pMicrophone, const size_t samples)> updateCallback;
};

mFUNCTION(mAudioScene_Create, OUT mPtr<mAudioSource> *pAudioScene, IN mAllocator *pAllocator, const size_t outputCount, const size_t sampleRate = 44100);

// `behindFilterStrength` is not supported yet.
mFUNCTION(mAudioScene_AddVirtualMicrophone, 
  OUT mPtr<mAudioSource> &audioScene, 
  OUT OPTIONAL mPtr<mVirtualMicrophone> *pMicrophone,
  const mVec3f position, 
  const mVec3f forward, 
  OPTIONAL const std::function<mResult (mVirtualMicrophone *pMicrophone, const size_t samples)> &updateFunc, 
  const float_t polarPatternWidth, 
  const float_t polarPatternRaise, 
  const float_t behindFilterStrength);

// This function does not fail if `isBeingConsumed` is already true, because you might have already fed the audioScene a different channel of this audio source.
mFUNCTION(mAudioScene_AddSpacialMonoAudioSource, 
  OUT mPtr<mAudioSource> &audioScene, 
  OUT OPTIONAL mPtr<mSpacialAudioSource> *pSpacialAudioSource,
  IN mAllocator *pAllocator,
  mPtr<mAudioSource> &audioSource, 
  const size_t channelIndex, 
  const mVec3f position, 
  OPTIONAL const std::function<mResult(mSpacialAudioSource *pAudioSource, const size_t samples)> &updateFunc);

#endif // mAudioScene_h__
