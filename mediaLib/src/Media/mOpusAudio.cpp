#include "mOpusAudio.h"

#include "ogg/ogg.h"
#include "opusfile.h"
#include "opusenc.h"

constexpr size_t mOpusEncoder_MaxChannelCount = 2;
constexpr size_t mOpusEncoder_ChannelBufferSize = 1024;

static_assert((mOpusEncoder_ChannelBufferSize & (32 - 1)) == 0, "Invalid Channel Buffer Size.");

struct mOpusEncoder
{
  OggOpusComments *pComments;
  OggOpusEnc *pEncoder;
  mPtr<mAudioSource> audioSource;
};

mFUNCTION(mOpusEncoder_Destroy_Internal, IN_OUT mOpusEncoder *pEncoder);

//////////////////////////////////////////////////////////////////////////

mFUNCTION(mOpusEncoder_CreateFromAudioSource, OUT mPtr<mOpusEncoder> *pEncoder, IN mAllocator *pAllocator, const mString &filename, mPtr<mAudioSource> &&audioSource, const size_t bitrate /* = 320 * 1024 */)
{
  mFUNCTION_SETUP();

  mERROR_IF(pEncoder == nullptr || audioSource == nullptr, mR_ArgumentNull);
  mERROR_IF(audioSource->pGetBufferFunc == nullptr || audioSource->pMoveToNextBufferFunc == nullptr, mR_NotInitialized);
  mERROR_IF(filename.bytes <= 1 || filename.hasFailed, mR_InvalidParameter);
  mERROR_IF(audioSource->channelCount > mOpusEncoder_MaxChannelCount || audioSource->channelCount == 0, mR_NotSupported);
  mERROR_IF(audioSource->isBeingConsumed, mR_ResourceStateInvalid);

  mERROR_CHECK(mSharedPointer_Allocate<mOpusEncoder>(pEncoder, pAllocator, [](mOpusEncoder *pData) { mOpusEncoder_Destroy_Internal(pData); }, 1));

  (*pEncoder)->audioSource = std::move(audioSource);

  (*pEncoder)->pComments = ope_comments_create();
  mERROR_IF((*pEncoder)->pComments == nullptr, mR_InternalError);

  int32_t error = 0;

  (*pEncoder)->pEncoder = ope_encoder_create_file(filename.c_str(), (*pEncoder)->pComments, (opus_int32)((*pEncoder)->audioSource->sampleRate), (int32_t)((*pEncoder)->audioSource->channelCount), 0, &error);
  mERROR_IF((*pEncoder)->pEncoder == nullptr || error != OPE_OK, mR_InternalError);

  (*pEncoder)->audioSource->isBeingConsumed = true;

  error = ope_encoder_ctl((*pEncoder)->pEncoder, OPUS_SET_BITRATE((int32_t)bitrate));
  mERROR_IF(error != OPE_OK, mR_InvalidParameter);

  mRETURN_SUCCESS();
}

mFUNCTION(mOpusEncoder_Run, mPtr<mOpusEncoder> &encoder)
{
  mFUNCTION_SETUP();

  mERROR_IF(encoder == nullptr, mR_ArgumentNull);

  bool endOfStream = false;
  size_t bufferCount = mOpusEncoder_ChannelBufferSize;
  mALIGN(32) float_t buffer[mOpusEncoder_ChannelBufferSize * mOpusEncoder_MaxChannelCount];

  while (!encoder->audioSource->stopPlayback && !endOfStream)
  {
    mERROR_CHECK(mZeroMemory(buffer, mARRAYSIZE(buffer)));

    for (size_t channel = 0; channel < mMin(encoder->audioSource->channelCount, mOpusEncoder_MaxChannelCount); channel++)
    {
      float_t channelBuffer[mOpusEncoder_ChannelBufferSize];
      mResult result;

      if (mFAILED(mSILENCE_ERROR(result = encoder->audioSource->pGetBufferFunc(encoder->audioSource, channelBuffer, mARRAYSIZE(channelBuffer), channel, &bufferCount))))
      {
        mERROR_IF(result != mR_EndOfStream, result);
        endOfStream = true;
        break;
      }

      mERROR_CHECK(mAudio_SetInterleavedChannelFloat(buffer, channelBuffer, channel, mMin(encoder->audioSource->channelCount, mOpusEncoder_MaxChannelCount), bufferCount));
    }

    mERROR_CHECK(mAudio_ApplyVolumeFloat(buffer, encoder->audioSource->volume, bufferCount));

    const int error = ope_encoder_write_float(encoder->pEncoder, buffer, (int)bufferCount);

    mERROR_IF(error != OPE_OK, mR_InternalError);

    if (endOfStream)
      break;

    if (encoder->audioSource->pMoveToNextBufferFunc != nullptr)
    {
      mResult result;

      if (mFAILED(mSILENCE_ERROR(result = encoder->audioSource->pMoveToNextBufferFunc(encoder->audioSource, mOpusEncoder_ChannelBufferSize))))
      {
        mERROR_IF(result != mR_EndOfStream, result);
        break;
      }
    }
  }

  const int error = ope_encoder_drain(encoder->pEncoder);
  mERROR_IF(error != OPE_OK, mR_InternalError);

  mRETURN_SUCCESS();
}

mFUNCTION(mOpusEncoder_Destroy, IN_OUT mPtr<mOpusEncoder> *pEncoder)
{
  mFUNCTION_SETUP();

  mERROR_IF(pEncoder == nullptr, mR_ArgumentNull);

  mERROR_CHECK(mSharedPointer_Destroy(pEncoder));

  mRETURN_SUCCESS();
}

//////////////////////////////////////////////////////////////////////////

mFUNCTION(mOpusEncoder_Destroy_Internal, IN_OUT mOpusEncoder *pEncoder)
{
  mFUNCTION_SETUP();

  mERROR_IF(pEncoder == nullptr, mR_ArgumentNull);

  if (pEncoder->pEncoder != nullptr)
  {
    /* int error = */ope_encoder_drain(pEncoder->pEncoder);
    ope_encoder_destroy(pEncoder->pEncoder);
    pEncoder->pEncoder = nullptr;
  }

  if (pEncoder->pComments != nullptr)
  {
    ope_comments_destroy(pEncoder->pComments);
    pEncoder->pComments = nullptr;
  }

  mERROR_CHECK(mSharedPointer_Destroy(&pEncoder->audioSource));

  mRETURN_SUCCESS();
}

//////////////////////////////////////////////////////////////////////////

struct mOpusFileAudioSource : mAudioSource
{
  OggOpusFile *pFile;
  mAllocator *pAllocator;
  float_t *pData;
  size_t dataSize, dataCapacity, lastConsumedSamples;
  bool endReached;
};

mFUNCTION(mOpusFileAudioSource_Destroy_Internal, mOpusFileAudioSource *pAudioSource);
mFUNCTION(mOpusFileAudioSource_GetBuffer_Internal, mPtr<mAudioSource> &audioSource, OUT float_t *pBuffer, const size_t bufferLength, const size_t channelIndex, OUT size_t *pBufferCount);
mFUNCTION(mOpusFileAudioSource_MoveToNextBuffer_Internal, mPtr<mAudioSource> &audioSource, const size_t samples);
mFUNCTION(mOpusFileAudioSource_SeekSample_Internal, mPtr<mAudioSource> &audioSource, const size_t sample);
mFUNCTION(mOpusFileAudioSource_ConsumeBuffer_Internal, IN mOpusFileAudioSource *pAudioSource, const size_t samples);
mFUNCTION(mOpusFileAudioSource_ReadBuffer_Internal, IN mOpusFileAudioSource *pAudioSource, const size_t samplesToLoad);

//////////////////////////////////////////////////////////////////////////

mFUNCTION(mOpusFileAudioSource_Create, OUT mPtr<mAudioSource> *pAudioSource, IN mAllocator *pAllocator, const mString &filename)
{
  mFUNCTION_SETUP();

  mERROR_IF(pAudioSource == nullptr, mR_ArgumentNull);
  mERROR_IF(filename.hasFailed || filename.count <= 1, mR_InvalidParameter);

  mOpusFileAudioSource *pInstance = nullptr;
  mERROR_CHECK((mSharedPointer_AllocateInherited<mAudioSource, mOpusFileAudioSource>(pAudioSource, pAllocator, [](mOpusFileAudioSource *pData) { mOpusFileAudioSource_Destroy_Internal(pData); }, &pInstance)));

  pInstance->pAllocator = pAllocator;

  int32_t error = 0;

  pInstance->pFile = op_open_file(filename.c_str(), &error);

  mERROR_IF(error != 0 || pInstance->pFile == nullptr, mR_ResourceIncompatible);

  pInstance->seekable = op_seekable(pInstance->pFile) != 0;
  pInstance->channelCount = op_channel_count(pInstance->pFile, 0);
  pInstance->sampleRate = 48000;
  pInstance->volume = 1.f;

  pInstance->pGetBufferFunc = mOpusFileAudioSource_GetBuffer_Internal;
  pInstance->pMoveToNextBufferFunc = mOpusFileAudioSource_MoveToNextBuffer_Internal;
  
  if (pInstance->seekable)
    pInstance->pSeekSampleFunc = mOpusFileAudioSource_SeekSample_Internal;

  mRETURN_SUCCESS();
}

mFUNCTION(mOpusFileAudioSource_Destroy, OUT mPtr<mAudioSource> *pAudioSource)
{
  mFUNCTION_SETUP();

  mERROR_IF(pAudioSource == nullptr, mR_ArgumentNull);

  mERROR_CHECK(mSharedPointer_Destroy(pAudioSource));
  
  mRETURN_SUCCESS();
}

//////////////////////////////////////////////////////////////////////////

mFUNCTION(mOpusFileAudioSource_Destroy_Internal, mOpusFileAudioSource *pAudioSource)
{
  mFUNCTION_SETUP();

  mERROR_IF(pAudioSource == nullptr, mR_ArgumentNull);

  if (pAudioSource->pFile != nullptr)
  {
    op_free(pAudioSource->pFile);
    pAudioSource->pFile = nullptr;
  }

  if (pAudioSource->pData != nullptr)
  {
    mERROR_CHECK(mAllocator_FreePtr(pAudioSource->pAllocator, &pAudioSource->pData));
    pAudioSource->dataSize = 0;
  }

  mRETURN_SUCCESS();
}

mFUNCTION(mOpusFileAudioSource_GetBuffer_Internal, mPtr<mAudioSource> &audioSource, OUT float_t *pBuffer, const size_t bufferLength, const size_t channelIndex, OUT size_t *pBufferCount)
{
  mFUNCTION_SETUP();

  mERROR_IF(audioSource == nullptr || pBuffer == nullptr || pBufferCount == nullptr, mR_ArgumentNull);
  mERROR_IF(channelIndex >= audioSource->channelCount, mR_InvalidParameter);

  mOpusFileAudioSource *pAudioSource = static_cast<mOpusFileAudioSource *>(audioSource.GetPointer());

  if (pAudioSource->dataSize < bufferLength * pAudioSource->channelCount)
  {
    const mResult result = mSILENCE_ERROR(mOpusFileAudioSource_ReadBuffer_Internal(pAudioSource, bufferLength * pAudioSource->channelCount - pAudioSource->dataSize));

    if (mFAILED(result))
    {
      mERROR_IF(result != mR_EndOfStream, result);
      pAudioSource->endReached = true;
    }
  }

  *pBufferCount = mMin(bufferLength, pAudioSource->dataSize);

  mERROR_CHECK(mAudio_ExtractFloatChannelFromInterleavedFloat(pBuffer, channelIndex, pAudioSource->pData, pAudioSource->channelCount, *pBufferCount));

  pAudioSource->lastConsumedSamples += bufferLength;

  mRETURN_SUCCESS();
}

mFUNCTION(mOpusFileAudioSource_MoveToNextBuffer_Internal, mPtr<mAudioSource> &audioSource, const size_t samples)
{
  mFUNCTION_SETUP();

  mERROR_IF(audioSource == nullptr, mR_ArgumentNull);

  mOpusFileAudioSource *pAudioSource = static_cast<mOpusFileAudioSource *>(audioSource.GetPointer());

  mERROR_IF(pAudioSource->endReached, mR_EndOfStream);

  mERROR_CHECK(mOpusFileAudioSource_ConsumeBuffer_Internal(pAudioSource, pAudioSource->lastConsumedSamples));

  pAudioSource->lastConsumedSamples = 0;

  if (samples * pAudioSource->channelCount > pAudioSource->dataSize)
  {
    const mResult result = mSILENCE_ERROR(mOpusFileAudioSource_ReadBuffer_Internal(pAudioSource, samples * pAudioSource->channelCount));

    if (mFAILED(result))
    {
      mERROR_IF(result != mR_EndOfStream, result);
      pAudioSource->endReached = true;
    }
  }

  mRETURN_SUCCESS();
}

mFUNCTION(mOpusFileAudioSource_SeekSample_Internal, mPtr<mAudioSource> &audioSource, const size_t sample)
{
  mFUNCTION_SETUP();

  mERROR_IF(audioSource == nullptr, mR_ArgumentNull);

  mOpusFileAudioSource *pAudioSource = static_cast<mOpusFileAudioSource *>(audioSource.GetPointer());

  mERROR_IF(0 != op_pcm_seek(pAudioSource->pFile, (int64_t)sample), mR_InternalError);

  mRETURN_SUCCESS();
}

mFUNCTION(mOpusFileAudioSource_ConsumeBuffer_Internal, IN mOpusFileAudioSource *pAudioSource, const size_t samples)
{
  mFUNCTION_SETUP();

  pAudioSource->dataSize -= samples;
  mERROR_CHECK(mMemmove(pAudioSource->pData, pAudioSource->pData + samples, pAudioSource->dataSize));

  mRETURN_SUCCESS();
}

mFUNCTION(mOpusFileAudioSource_ReadBuffer_Internal, IN mOpusFileAudioSource *pAudioSource, const size_t samplesToLoad)
{
  mFUNCTION_SETUP();

  if (pAudioSource->dataSize + samplesToLoad > pAudioSource->dataCapacity)
  {
    const size_t newCapacity = mMax(pAudioSource->dataSize * 2, pAudioSource->dataSize + samplesToLoad);
    mERROR_CHECK(mAllocator_Reallocate(pAudioSource->pAllocator, &pAudioSource->pData, newCapacity));
    pAudioSource->dataCapacity = newCapacity;
  }

  size_t samplesRemaining = samplesToLoad;

  while (samplesRemaining > 0)
  {
    int32_t streamIndex = -1;
    int32_t samplesLoaded = -1;

    while (streamIndex != 0)
      samplesLoaded = op_read_float(pAudioSource->pFile, pAudioSource->pData + pAudioSource->dataSize, (int32_t)samplesRemaining, &streamIndex);

    mERROR_IF(samplesLoaded == 0, mR_EndOfStream);
    mERROR_IF(samplesLoaded < 0, mR_InternalError);

    const size_t sampleCount = (size_t)samplesLoaded * pAudioSource->channelCount;

    pAudioSource->dataSize += sampleCount;
    
    if (samplesLoaded >= samplesRemaining)
      break;
    else
      samplesRemaining -= sampleCount;
  }

  mRETURN_SUCCESS();
}
