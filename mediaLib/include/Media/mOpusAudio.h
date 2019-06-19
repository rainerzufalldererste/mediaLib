#ifndef mOpusAudio_h__
#define mOpusAudio_h__

#include "mediaLib.h"
#include "mAudio.h"

struct mOpusEncoder;

mFUNCTION(mOpusEncoder_Create, OUT mPtr<mOpusEncoder> *pEncoder, IN mAllocator *pAllocator, const mString &filename, mPtr<mAudioSource> &audioSource, const size_t bitrate = 320 * 1024);
mFUNCTION(mOpusEncoder_Run, mPtr<mOpusEncoder> &encoder);
mFUNCTION(mOpusEncoder_Destroy, IN_OUT mPtr<mOpusEncoder> *pEncoder);

struct mOpusEncoderPassive;

mFUNCTION(mOpusEncoderPassive_Create, OUT mPtr<mOpusEncoderPassive> *pEncoder, IN mAllocator *pAllocator, const mString &filename, const size_t channelCount, const size_t sampleRate, const size_t bitrate = 320 * 1024);
mFUNCTION(mOpusEncoderPassive_AddSamples, mPtr<mOpusEncoderPassive> &encoder, IN const float_t *pChannelInterleavedData, const size_t sampleCountPerChannel);

// Finalizes & Flushes the stream and destroys the encoder.
mFUNCTION(mOpusEncoderPassive_Destroy, IN_OUT mPtr<mOpusEncoderPassive> *pEncoder);

mFUNCTION(mOpusFileAudioSource_Create, OUT mPtr<mAudioSource> *pAudioSource, IN mAllocator *pAllocator, const mString &filename);
mFUNCTION(mOpusFileAudioSource_Destroy, OUT mPtr<mAudioSource> *pAudioSource);

// threadsafe.
mFUNCTION(mOpusFileAudioSource_GetSampleCount, OUT mPtr<mAudioSource> &audioSource, OUT size_t *pSampleCount);

// threadsafe.
mFUNCTION(mOpusFileAudioSource_GetSamplePosition, OUT mPtr<mAudioSource> &audioSource, OUT size_t *pSamplePosition);

// threadsafe.
mFUNCTION(mOpusFileAudioSource_SetPaused, OUT mPtr<mAudioSource> &audioSource, OUT bool paused);

// threadsafe.
mFUNCTION(mOpusFileAudioSource_IsPaused, OUT mPtr<mAudioSource> &audioSource, OUT bool *pIsPaused);

#endif // mOpusAudio_h__
