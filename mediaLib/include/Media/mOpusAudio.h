#ifndef mOpusAudio_h__
#define mOpusAudio_h__

#include "mediaLib.h"
#include "mAudio.h"

struct mOpusEncoder;

mFUNCTION(mOpusEncoder_CreateFromAudioSource, OUT mPtr<mOpusEncoder> *pEncoder, IN mAllocator *pAllocator, const mString &filename, mPtr<mAudioSource> &&audioSource, const size_t bitrate = 320 * 1024);
mFUNCTION(mOpusEncoder_Run, mPtr<mOpusEncoder> &encoder);
mFUNCTION(mOpusEncoder_Destroy, IN_OUT mPtr<mOpusEncoder> *pEncoder);

mFUNCTION(mOpusFileAudioSource_Create, OUT mPtr<mAudioSource> *pAudioSource, IN mAllocator *pAllocator, const mString &filename);
mFUNCTION(mOpusFileAudioSource_Destroy, OUT mPtr<mAudioSource> *pAudioSource);

#endif // mOpusAudio_h__
