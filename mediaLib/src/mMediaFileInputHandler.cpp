// Copyright 2018 Christoph Stiller
// 
// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files(the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions :
// 
// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
// 
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include <mfapi.h>
#include <mfidl.h>
#include <Mferror.h>
#include <mfreadwrite.h>
#include "mMediaFileInputHandler.h"
#include "mMediaFoundation.h"

struct mMediaTypeLookup
{
  size_t streamIndex;
  mMediaMajorType mediaType;
};

struct mMediaFileInputHandler
{
  mAllocator *pAllocator;
  IMFSourceReader *pSourceReader;
  ProcessVideoBufferFunction *pProcessVideoDataCallback;
  ProcessAudioBufferFunction *pProcessAudioDataCallback;

  size_t streamCount;
  mMediaTypeLookup *pStreamTypeLookup;

  size_t videoStreamCount;
  mVideoStreamType *pVideoStreams;

  size_t audioStreamCount;
  mAudioStreamType *pAudioStreams;

  bool iteratorCreated;
};

struct mMediaFileInputIterator
{
  mPtr<mMediaFileInputHandler> mediaFileInputHandler;
  mMediaMajorType mediaType;
  size_t wmf_streamIndex;
  bool hasFinished;
};

mFUNCTION(mMediaFileInputHandler_Create_Internal, IN mMediaFileInputHandler *pData, IN mAllocator *pAllocator, const std::wstring &fileName, const bool enableVideoProcessing, const bool enableAudioProcessing);
mFUNCTION(mMediaFileInputHandler_Destroy_Internal, IN mMediaFileInputHandler *pData);
mFUNCTION(mMediaFileInputHandler_RunSession_Internal, IN mMediaFileInputHandler *pData);
mFUNCTION(mMediaFileInputHandler_InitializeExtenalDependencies);
mFUNCTION(mMediaFileInputHandler_CleanupExtenalDependencies);
mFUNCTION(mMediaFileInputHandler_AddStream_Internal, IN mMediaFileInputHandler *pInputHandler, IN mMediaTypeLookup *pMediaType);
mFUNCTION(mMediaFileInputHandler_AddVideoStream_Internal, IN mMediaFileInputHandler *pInputHandler, IN mVideoStreamType *pVideoStreamType);
mFUNCTION(mMediaFileInputHandler_AddAudioStream_Internal, IN mMediaFileInputHandler *pInputHandler, IN mAudioStreamType *pAudioStreamType);

mFUNCTION(mAudioStreamType_Create, IN IMFMediaType *pMediaType, IN mMediaTypeLookup *pMediaTypeLookup, const size_t streamIndex, OUT mAudioStreamType *pAudioStreamType);
mFUNCTION(mVideoStreamType_Create, IN IMFMediaType *pMediaType, IN mMediaTypeLookup *pMediaTypeLookup, const size_t streamIndex, OUT mVideoStreamType *pVideoStreamType);

mFUNCTION(mMediaFileInputIterator_Create, OUT mPtr<mMediaFileInputIterator> *pIterator, mPtr<mMediaFileInputHandler> &mediaFileInputHandlerRef, const size_t wmf_streamIndex, const mMediaMajorType mediaType);
mFUNCTION(mMediaFileInputIterator_Create_Internal, mMediaFileInputIterator *pIterator, mPtr<mMediaFileInputHandler> &mediaFileInputHandlerRef, const size_t wmf_streamIndex, const mMediaMajorType mediaType);
mFUNCTION(mMediaFileInputIterator_Destroy_Internal, mMediaFileInputIterator *pIterator);
mFUNCTION(mMediaFileInputIterator_IterateToStreamIndex, mPtr<mMediaFileInputIterator> &iterator, OUT IMFSample **ppSample, OUT LONGLONG *pTimeStamp);

template <typename T>
static mINLINE void _ReleaseReference(T **pData)
{
  if (pData && *pData)
  {
    (*pData)->Release();
    *pData = nullptr;
  }
}

//////////////////////////////////////////////////////////////////////////

mFUNCTION(mMediaFileInputHandler_Create, OUT mPtr<mMediaFileInputHandler> *pPtr, IN OPTIONAL mAllocator *pAllocator, const std::wstring &fileName, const mMediaFileInputHandler_CreateFlags createFlags)
{
  mFUNCTION_SETUP();

  mERROR_IF(pPtr == nullptr, mR_ArgumentNull);

  if (*pPtr != nullptr)
  {
    mERROR_CHECK(mSharedPointer_Destroy(pPtr));
    *pPtr = nullptr;
  }

  mMediaFileInputHandler *pInputHandler = nullptr;
  mDEFER_ON_ERROR(mAllocator_FreePtr(pAllocator, &pInputHandler));
  mERROR_CHECK(mAllocator_AllocateZero(pAllocator, &pInputHandler, 1));

  mDEFER_DESTRUCTION_ON_ERROR(pPtr, mSharedPointer_Destroy);
  mERROR_CHECK(mSharedPointer_Create<mMediaFileInputHandler>(pPtr, pInputHandler, [](mMediaFileInputHandler *pData) { mMediaFileInputHandler_Destroy_Internal(pData); }, pAllocator));
  pInputHandler = nullptr; // to not be destroyed on error.

  mERROR_CHECK(mMediaFileInputHandler_Create_Internal(pPtr->GetPointer(), pAllocator, fileName, (createFlags & mMediaFileInputHandler_CreateFlags::mMMFIH_CF_VideoEnabled) != 0, (createFlags & mMediaFileInputHandler_CreateFlags::mMMFIH_CF_AudioEnabled) != 0));

  mRETURN_SUCCESS();
}

mFUNCTION(mMediaFileInputHandler_Destroy, IN_OUT mPtr<mMediaFileInputHandler> *pPtr)
{
  mFUNCTION_SETUP();

  mERROR_IF(pPtr == nullptr || *pPtr == nullptr, mR_ArgumentNull);
  mERROR_CHECK(mSharedPointer_Destroy(pPtr));

  mRETURN_SUCCESS();
}

mFUNCTION(mMediaFileInputHandler_Play, mPtr<mMediaFileInputHandler> &ptr)
{
  mFUNCTION_SETUP();

  mERROR_IF(ptr == nullptr, mR_ArgumentNull);
  mERROR_IF(ptr->iteratorCreated, mR_ResourceStateInvalid);

  ptr->iteratorCreated = true;

  mERROR_CHECK(mMediaFileInputHandler_RunSession_Internal(ptr.GetPointer()));

  mRETURN_SUCCESS();
}

mFUNCTION(mMediaFileInputHandler_GetVideoStreamResolution, mPtr<mMediaFileInputHandler> &ptr, OUT mVec2s *pResolution, const size_t videoStreamIndex /* = 0 */)
{
  mFUNCTION_SETUP();

  mERROR_IF(ptr == nullptr, mR_ArgumentNull);

  mERROR_IF(ptr->videoStreamCount < videoStreamIndex, mR_IndexOutOfBounds);

  if (pResolution)
    *pResolution = ptr->pVideoStreams[videoStreamIndex].resolution;

  mRETURN_SUCCESS();
}

mFUNCTION(mMediaFileInputHandler_SetVideoCallback, mPtr<mMediaFileInputHandler> &ptr, IN ProcessVideoBufferFunction *pCallback)
{
  mFUNCTION_SETUP();

  mERROR_IF(ptr == nullptr, mR_ArgumentNull);
  ptr->pProcessVideoDataCallback = pCallback;

  mRETURN_SUCCESS();
}

mFUNCTION(mMediaFileInputHandler_SetAudioCallback, mPtr<mMediaFileInputHandler> &ptr, IN ProcessAudioBufferFunction *pCallback)
{
  mFUNCTION_SETUP();

  mERROR_IF(ptr == nullptr, mR_ArgumentNull);
  ptr->pProcessAudioDataCallback = pCallback;

  mRETURN_SUCCESS();
}

mFUNCTION(mMediaFileInputHandler_GetVideoStreamCount, mPtr<mMediaFileInputHandler> &ptr, OUT size_t *pCount)
{
  mFUNCTION_SETUP();

  mERROR_IF(ptr == nullptr || pCount == nullptr, mR_ArgumentNull);

  *pCount = ptr->videoStreamCount;

  mRETURN_SUCCESS();
}

mFUNCTION(mMediaFileInputHandler_GetAudioStreamCount, mPtr<mMediaFileInputHandler>& ptr, OUT size_t * pCount)
{
  mFUNCTION_SETUP();

  mERROR_IF(ptr == nullptr || pCount == nullptr, mR_ArgumentNull);

  *pCount = ptr->audioStreamCount;

  mRETURN_SUCCESS();
}

mFUNCTION(mMediaFileInputHandler_GetVideoStreamType, mPtr<mMediaFileInputHandler>& ptr, const size_t index, OUT mVideoStreamType * pStreamType)
{
  mFUNCTION_SETUP();

  mERROR_IF(ptr == nullptr || pStreamType == nullptr, mR_ArgumentNull);
  mERROR_IF(ptr->videoStreamCount <= index, mR_IndexOutOfBounds);

  *pStreamType = ptr->pVideoStreams[index];

  mRETURN_SUCCESS();
}

mFUNCTION(mMediaFileInputHandler_GetAudioStreamType, mPtr<mMediaFileInputHandler>& ptr, const size_t index, OUT mAudioStreamType * pStreamType)
{
  mFUNCTION_SETUP();

  mERROR_IF(ptr == nullptr || pStreamType == nullptr, mR_ArgumentNull);
  mERROR_IF(ptr->audioStreamCount <= index, mR_IndexOutOfBounds);

  *pStreamType = ptr->pAudioStreams[index];

  mRETURN_SUCCESS();
}

mFUNCTION(mMediaFileInputHandler_GetIterator, mPtr<mMediaFileInputHandler> &ptr, OUT mPtr<mMediaFileInputIterator> *pIterator, const mMediaMajorType mediaType, const size_t streamIndex /* = 0 */)
{
  mFUNCTION_SETUP();

  mERROR_IF(ptr == nullptr, mR_ArgumentNull);
  mERROR_IF(ptr->iteratorCreated, mR_ResourceStateInvalid);

  ptr->iteratorCreated = true;

  size_t wmf_streamIndex = (size_t)-1;

  switch (mediaType)
  {
  case mMMT_Video:
    mERROR_IF(ptr->videoStreamCount <= streamIndex, mR_IndexOutOfBounds);
    wmf_streamIndex = ptr->pVideoStreams[streamIndex].wmf_streamIndex;
    break;

  case mMMT_Audio:
    mERROR_IF(ptr->audioStreamCount <= streamIndex, mR_IndexOutOfBounds);
    wmf_streamIndex = ptr->pAudioStreams[streamIndex].wmf_streamIndex;
    break;

  default:
    mRETURN_RESULT(mR_ResourceIncompatible);
    break;
  }

  mERROR_IF(wmf_streamIndex == (size_t)-1, mR_ResourceNotFound);

  mPtr<mMediaFileInputIterator> iterator;

  mERROR_CHECK(mMediaFileInputIterator_Create(&iterator, ptr, wmf_streamIndex, mediaType));

  *pIterator = iterator;

  mRETURN_SUCCESS();
}

mFUNCTION(mMediaFileInputIterator_Destroy, IN_OUT mPtr<mMediaFileInputIterator> *pIterator)
{
  mFUNCTION_SETUP();

  mERROR_IF(pIterator == nullptr, mR_ArgumentNull);

  mERROR_CHECK(mSharedPointer_Destroy(pIterator));

  mRETURN_SUCCESS();
}

mFUNCTION(mMediaFileInputIterator_GetNextVideoFrame, mPtr<mMediaFileInputIterator> &iterator, OUT mPtr<mImageBuffer> *pImageBuffer, OUT OPTIONAL mVideoStreamType *pVideoStreamType)
{
  mFUNCTION_SETUP();

  mERROR_IF(iterator == nullptr || pImageBuffer == nullptr, mR_ArgumentNull);
  mERROR_IF(iterator->mediaType != mMediaMajorType::mMMT_Video, mR_ResourceIncompatible);
  mERROR_IF(iterator->hasFinished, mR_EndOfStream);

  IMFSample *pSample = nullptr;
  mDEFER_DESTRUCTION(&pSample, _ReleaseReference);

  LONGLONG timestamp;

  mERROR_CHECK(mMediaFileInputIterator_IterateToStreamIndex(iterator, &pSample, &timestamp));

  {
    HRESULT hr = S_OK;
    mUnused(hr);

    IMFMediaBuffer *pMediaBuffer = nullptr;
    mDEFER_DESTRUCTION(&pMediaBuffer, _ReleaseReference);
    mERROR_IF(FAILED(hr = pSample->ConvertToContiguousBuffer(&pMediaBuffer)), mR_InternalError);

    mERROR_IF(iterator->wmf_streamIndex >= iterator->mediaFileInputHandler->streamCount, mR_IndexOutOfBounds);
    mMediaTypeLookup mediaTypeLookup = iterator->mediaFileInputHandler->pStreamTypeLookup[iterator->wmf_streamIndex];

    if (mediaTypeLookup.mediaType == mMMT_Undefined)
      mRETURN_RESULT(mR_ResourceIncompatible);

    uint8_t *pSampleData;
    DWORD sampleDataCurrentLength;
    DWORD sampleDataMaxLength;

    mERROR_IF(FAILED(hr = pMediaBuffer->Lock(&pSampleData, &sampleDataCurrentLength, &sampleDataMaxLength)), mR_InternalError);
    mDEFER(pMediaBuffer->Unlock());

    switch (mediaTypeLookup.mediaType)
    {
    case mMMT_Video:
    {
      mERROR_IF(iterator->mediaFileInputHandler->videoStreamCount < mediaTypeLookup.streamIndex, mR_IndexOutOfBounds);
      mVideoStreamType videoStreamType = iterator->mediaFileInputHandler->pVideoStreams[mediaTypeLookup.streamIndex];
      mERROR_CHECK(mTimeStamp_FromSeconds(&videoStreamType.timePoint, timestamp / (double_t)(10 * 1000 * 1000)));

      mPtr<mImageBuffer> sourceBuffer;
      mDEFER_DESTRUCTION(&sourceBuffer, mImageBuffer_Destroy);
      mERROR_CHECK(mImageBuffer_Create(&sourceBuffer, &mDefaultAllocator, pSampleData, videoStreamType.resolution, videoStreamType.pixelFormat));

      mERROR_CHECK(mImageBuffer_Create(pImageBuffer, &mDefaultAllocator, videoStreamType.resolution, videoStreamType.pixelFormat));
      mERROR_CHECK(mPixelFormat_TransformBuffer(sourceBuffer, *pImageBuffer));

      if (pVideoStreamType != nullptr)
        *pVideoStreamType = videoStreamType;

      break;
    }

    default:
      mERROR_IF(true, mR_InternalError);
      break;
    }}

  mRETURN_SUCCESS();
}

mFUNCTION(mMediaFileInputIterator_GetNextAudioFrame, mPtr<mMediaFileInputIterator> &iterator, OUT mPtr<uint8_t> *pData, OUT OPTIONAL mAudioStreamType *pAudioStreamType)
{
  mFUNCTION_SETUP();

  mERROR_IF(iterator == nullptr || pData == nullptr, mR_ArgumentNull);
  mERROR_IF(iterator->mediaType != mMediaMajorType::mMMT_Audio, mR_ResourceIncompatible);
  mERROR_IF(iterator->hasFinished, mR_EndOfStream);

  IMFSample *pSample = nullptr;
  mDEFER_DESTRUCTION(&pSample, _ReleaseReference);

  LONGLONG timestamp;

  mERROR_CHECK(mMediaFileInputIterator_IterateToStreamIndex(iterator, &pSample, &timestamp));

  {
    HRESULT hr = S_OK;
    mUnused(hr);

    IMFMediaBuffer *pMediaBuffer = nullptr;
    mDEFER_DESTRUCTION(&pMediaBuffer, _ReleaseReference);
    mERROR_IF(FAILED(hr = pSample->ConvertToContiguousBuffer(&pMediaBuffer)), mR_InternalError);

    mERROR_IF(iterator->wmf_streamIndex >= iterator->mediaFileInputHandler->streamCount, mR_IndexOutOfBounds);
    mMediaTypeLookup mediaTypeLookup = iterator->mediaFileInputHandler->pStreamTypeLookup[iterator->wmf_streamIndex];

    if (mediaTypeLookup.mediaType == mMMT_Undefined)
      mRETURN_RESULT(mR_ResourceIncompatible);

    uint8_t *pSampleData;
    DWORD sampleDataCurrentLength;
    DWORD sampleDataMaxLength;

    mERROR_IF(FAILED(hr = pMediaBuffer->Lock(&pSampleData, &sampleDataCurrentLength, &sampleDataMaxLength)), mR_InternalError);
    mDEFER(pMediaBuffer->Unlock());

    switch (mediaTypeLookup.mediaType)
    {
    case mMMT_Audio:
    {
      mERROR_IF(iterator->mediaFileInputHandler->audioStreamCount < mediaTypeLookup.streamIndex, mR_IndexOutOfBounds);
      mAudioStreamType audioStreamType = iterator->mediaFileInputHandler->pAudioStreams[mediaTypeLookup.streamIndex];
      audioStreamType.bufferSize = sampleDataCurrentLength;
      mERROR_CHECK(mTimeStamp_FromSeconds(&audioStreamType.timePoint, timestamp / (double_t)(10 * 1000 * 1000)));

      mPtr<uint8_t> buffer;
      mDEFER_DESTRUCTION(&buffer, mSharedPointer_Destroy);
      mERROR_CHECK(mSharedPointer_Allocate(&buffer, iterator->mediaFileInputHandler->pAllocator, sampleDataCurrentLength));
      mERROR_CHECK(mMemcpy(buffer.GetPointer(), pSampleData, sampleDataCurrentLength));

      *pData = buffer;

      if (pAudioStreamType != nullptr)
        *pAudioStreamType = audioStreamType;

      break;
    }

    default:
      mERROR_IF(true, mR_InternalError);
      break;
    }}

  mRETURN_SUCCESS();
}

mFUNCTION(mMediaFileInputIterator_SkipFrame, mPtr<mMediaFileInputIterator> &iterator)
{
  mFUNCTION_SETUP();

  mERROR_IF(iterator == nullptr, mR_ArgumentNull);
  mERROR_IF(iterator->hasFinished, mR_EndOfStream);

  IMFSample *pSample = nullptr;
  mDEFER_DESTRUCTION(&pSample, _ReleaseReference);

  LONGLONG timestamp;

  mERROR_CHECK(mMediaFileInputIterator_IterateToStreamIndex(iterator, &pSample, &timestamp));

  mRETURN_SUCCESS();
}

mFUNCTION(mMediaFileInputIterator_SeekTo, mPtr<mMediaFileInputIterator> &iterator, const mTimeStamp &timeStamp)
{
  mFUNCTION_SETUP();

  mERROR_IF(iterator == nullptr, mR_ArgumentNull);
  mERROR_IF(timeStamp.timePoint < 0.0, mR_IndexOutOfBounds);
  iterator->hasFinished = false;

  HRESULT hr = S_OK;
  mUnused(hr);

  PROPVARIANT var;
  PropVariantInit(&var);

  var.vt = VT_I8;
  var.hVal.QuadPart = (LONGLONG)(timeStamp.timePoint * 1000 * 1000 * 10);

  mERROR_IF(FAILED(hr = iterator->mediaFileInputHandler->pSourceReader->SetCurrentPosition(GUID_NULL, var)), mR_InternalError);

  mRETURN_SUCCESS();
}

mFUNCTION(mMediaFileInputHandler_AddStream_Internal, IN mMediaFileInputHandler *pInputHandler, IN mMediaTypeLookup *pMediaType)
{
  mFUNCTION_SETUP();

  mERROR_IF(pInputHandler == nullptr || pMediaType == nullptr, mR_ArgumentNull);

  ++pInputHandler->streamCount;

  if (pInputHandler->pStreamTypeLookup == nullptr)
    mERROR_CHECK(mAllocator_Allocate(pInputHandler->pAllocator, &pInputHandler->pStreamTypeLookup, pInputHandler->streamCount));
  else
    mERROR_CHECK(mAllocator_Reallocate(pInputHandler->pAllocator, &pInputHandler->pStreamTypeLookup, pInputHandler->streamCount));

  mERROR_CHECK(mAllocator_Copy(pInputHandler->pAllocator, &pInputHandler->pStreamTypeLookup[pInputHandler->streamCount - 1], pMediaType, 1));

  mRETURN_SUCCESS();
}

mFUNCTION(mMediaFileInputHandler_AddVideoStream_Internal, IN mMediaFileInputHandler *pInputHandler, IN mVideoStreamType *pVideoStreamType)
{
  mFUNCTION_SETUP();

  mERROR_IF(pInputHandler == nullptr || pVideoStreamType == nullptr, mR_ArgumentNull);

  ++pInputHandler->videoStreamCount;

  if (pInputHandler->pVideoStreams == nullptr)
    mERROR_CHECK(mAllocator_Allocate(pInputHandler->pAllocator, &pInputHandler->pVideoStreams, pInputHandler->videoStreamCount));
  else
    mERROR_CHECK(mAllocator_Reallocate(pInputHandler->pAllocator, &pInputHandler->pVideoStreams, pInputHandler->videoStreamCount));

  mERROR_CHECK(mAllocator_Copy(pInputHandler->pAllocator, &pInputHandler->pVideoStreams[pInputHandler->videoStreamCount - 1], pVideoStreamType, 1));

  mRETURN_SUCCESS();
}

mFUNCTION(mMediaFileInputHandler_AddAudioStream_Internal, IN mMediaFileInputHandler *pInputHandler, IN mAudioStreamType *pAudioStreamType)
{
  mFUNCTION_SETUP();

  mERROR_IF(pInputHandler == nullptr || pAudioStreamType == nullptr, mR_ArgumentNull);

  ++pInputHandler->audioStreamCount;

  if (pInputHandler->pVideoStreams == nullptr)
    mERROR_CHECK(mAllocator_Allocate(pInputHandler->pAllocator, &pInputHandler->pAudioStreams, pInputHandler->audioStreamCount));
  else
    mERROR_CHECK(mAllocator_Reallocate(pInputHandler->pAllocator, &pInputHandler->pAudioStreams, pInputHandler->audioStreamCount));

  mERROR_CHECK(mAllocator_Copy(pInputHandler->pAllocator, &pInputHandler->pAudioStreams[pInputHandler->audioStreamCount - 1], pAudioStreamType, 1));

  mRETURN_SUCCESS();
}

mFUNCTION(mMediaFileInputHandler_Create_Internal, IN mMediaFileInputHandler *pInputHandler, IN mAllocator *pAllocator, const std::wstring &fileName, const bool enableVideoProcessing, const bool enableAudioProcessing)
{
  mFUNCTION_SETUP();
  mERROR_IF(pInputHandler == nullptr, mR_ArgumentNull);

  pInputHandler->pAllocator = pAllocator;
  pInputHandler->iteratorCreated = false;

  HRESULT hr = S_OK;
  mUnused(hr);

  mERROR_CHECK(mMediaFoundation_AddReference());

  IMFAttributes *pAttributes = nullptr;
  mDEFER_DESTRUCTION(&pAttributes, _ReleaseReference);
  mERROR_IF(FAILED(MFCreateAttributes(&pAttributes, 1)), mR_InternalError);
  //mERROR_IF(FAILED(pAttributes->SetUINT32(MF_SOURCE_READER_ENABLE_VIDEO_PROCESSING, TRUE)), mR_InternalError);

  IMFMediaType *pVideoMediaTypeYUV = nullptr;
  mDEFER_DESTRUCTION(&pVideoMediaTypeYUV, _ReleaseReference);

  IMFMediaType *pVideoMediaTypeRGB = nullptr;
  mDEFER_DESTRUCTION(&pVideoMediaTypeRGB, _ReleaseReference);

  if (enableVideoProcessing)
  {
    mERROR_IF(FAILED(MFCreateMediaType(&pVideoMediaTypeYUV)), mR_InternalError);
    mERROR_IF(FAILED(pVideoMediaTypeYUV->SetGUID(MF_MT_MAJOR_TYPE, MFMediaType_Video)), mR_InternalError);
    mERROR_IF(FAILED(pVideoMediaTypeYUV->SetGUID(MF_MT_SUBTYPE, MFVideoFormat_I420)), mR_InternalError);

    mERROR_IF(FAILED(MFCreateMediaType(&pVideoMediaTypeRGB)), mR_InternalError);
    mERROR_IF(FAILED(pVideoMediaTypeRGB->SetGUID(MF_MT_MAJOR_TYPE, MFMediaType_Video)), mR_InternalError);
    mERROR_IF(FAILED(pVideoMediaTypeRGB->SetGUID(MF_MT_SUBTYPE, MFVideoFormat_RGB32)), mR_InternalError);
  }

  IMFMediaType *pAudioMediaType = nullptr;
  mDEFER_DESTRUCTION(&pAudioMediaType, _ReleaseReference);

  if (enableAudioProcessing)
  {
    mERROR_IF(FAILED(MFCreateMediaType(&pAudioMediaType)), mR_InternalError);
    mERROR_IF(FAILED(pAudioMediaType->SetGUID(MF_MT_MAJOR_TYPE, MFMediaType_Audio)), mR_InternalError);
    mERROR_IF(FAILED(pAudioMediaType->SetGUID(MF_MT_SUBTYPE, MFAudioFormat_PCM)), mR_InternalError);
  }

  hr = MFCreateSourceReaderFromURL(fileName.c_str(), pAttributes, &pInputHandler->pSourceReader);
  mERROR_IF(hr == HRESULT_FROM_WIN32(ERROR_PATH_NOT_FOUND), mR_ResourceNotFound);
  mERROR_IF(FAILED(hr), mR_ResourceInvalid);

  GUID majorType;
  GUID minorType;
  DWORD streamIndex = 0;
  DWORD mediaTypeIndex = 0;
  bool isValid = false;

  while (true)
  {
    bool usableMediaTypeFoundInStream = false;
    mediaTypeIndex = 0;

    while (true)
    {
      IMFMediaType *pType = nullptr;
      mDEFER_DESTRUCTION(&pType, _ReleaseReference);
      hr = pInputHandler->pSourceReader->GetNativeMediaType(streamIndex, mediaTypeIndex, &pType);

      if (hr == MF_E_NO_MORE_TYPES)
      {
        hr = S_OK;
        break;
      }
      else if (hr == MF_E_INVALIDSTREAMNUMBER)
      {
        break;
      }
      else if (SUCCEEDED(hr))
      {
        mERROR_IF(FAILED(hr = pType->GetMajorType(&majorType)), mR_InternalError);
        mERROR_IF(FAILED(hr = pType->GetGUID(MF_MT_SUBTYPE, &minorType)), mR_InternalError);
        mUnused(minorType);

        if (enableVideoProcessing && majorType == MFMediaType_Video)
        {
          if (!usableMediaTypeFoundInStream)
          {
            usableMediaTypeFoundInStream = true;

            mERROR_IF(FAILED(hr = pInputHandler->pSourceReader->SetStreamSelection(streamIndex, true)), mR_InternalError);
            mERROR_IF(FAILED(hr = pInputHandler->pSourceReader->GetCurrentMediaType(streamIndex, &pType)), mR_InternalError);
            
            bool isYUV = true;

            if (SUCCEEDED(hr = pInputHandler->pSourceReader->SetCurrentMediaType(streamIndex, nullptr, pVideoMediaTypeYUV)))
              isYUV = true;
            else if (SUCCEEDED(hr = pInputHandler->pSourceReader->SetCurrentMediaType(streamIndex, nullptr, pVideoMediaTypeRGB)))
              isYUV = false;
            else
              mERROR_IF(true, mR_InternalError);

            isValid = true;

            mMediaTypeLookup mediaTypeLookup;
            mERROR_CHECK(mMemset(&mediaTypeLookup, 1));
            mediaTypeLookup.mediaType = mMMT_Video;
            mediaTypeLookup.streamIndex = pInputHandler->videoStreamCount;

            mERROR_CHECK(mMediaFileInputHandler_AddStream_Internal(pInputHandler, &mediaTypeLookup));

            mVideoStreamType videoStreamType;
            mERROR_CHECK(mVideoStreamType_Create(pType, &mediaTypeLookup, streamIndex, &videoStreamType));

            videoStreamType.pixelFormat = isYUV ? mPF_YUV420 : mPF_B8G8R8A8;

            mERROR_CHECK(mMediaFileInputHandler_AddVideoStream_Internal(pInputHandler, &videoStreamType));
          }
        }
        else if (enableAudioProcessing && majorType == MFMediaType_Audio)
        {
          if (!usableMediaTypeFoundInStream)
          {
            usableMediaTypeFoundInStream = true;

            mERROR_IF(FAILED(hr = pInputHandler->pSourceReader->SetStreamSelection(streamIndex, true)), mR_InternalError);
            mERROR_IF(FAILED(hr = pInputHandler->pSourceReader->GetCurrentMediaType(streamIndex, &pType)), mR_InternalError);
            mERROR_IF(FAILED(hr = pInputHandler->pSourceReader->SetCurrentMediaType(streamIndex, nullptr, pAudioMediaType)), mR_InternalError);

            isValid = true;

            mMediaTypeLookup mediaTypeLookup;
            mERROR_CHECK(mMemset(&mediaTypeLookup, 1));
            mediaTypeLookup.mediaType = mMMT_Audio;
            mediaTypeLookup.streamIndex = pInputHandler->audioStreamCount;

            mERROR_CHECK(mMediaFileInputHandler_AddStream_Internal(pInputHandler, &mediaTypeLookup));

            mAudioStreamType audioStreamType;
            mERROR_CHECK(mAudioStreamType_Create(pType, &mediaTypeLookup, streamIndex, &audioStreamType));
            mERROR_CHECK(mMediaFileInputHandler_AddAudioStream_Internal(pInputHandler, &audioStreamType));
          }
        }
      }

      ++mediaTypeIndex;
    }

    if (hr == MF_E_INVALIDSTREAMNUMBER)
      break;

    if (!usableMediaTypeFoundInStream)
    {
      mMediaTypeLookup mediaTypeLookup;
      mERROR_CHECK(mMemset(&mediaTypeLookup, 1));
      mediaTypeLookup.mediaType = mMMT_Undefined;
      mediaTypeLookup.streamIndex = 0;

      mERROR_CHECK(mMediaFileInputHandler_AddStream_Internal(pInputHandler, &mediaTypeLookup));
    }

    ++streamIndex;
  }

  mERROR_IF(!isValid, mR_ResourceIncompatible);

  mRETURN_SUCCESS();
}

mFUNCTION(mMediaFileInputHandler_Destroy_Internal, IN mMediaFileInputHandler *pData)
{
  mFUNCTION_SETUP();

  mERROR_IF(pData == nullptr, mR_ArgumentNull);

  if (pData->pSourceReader)
  {
    pData->pSourceReader->Release();
    pData->pSourceReader = nullptr;
  }

  if (pData->pProcessVideoDataCallback)
    pData->pProcessVideoDataCallback = nullptr;

  if (pData->pProcessAudioDataCallback)
    pData->pProcessAudioDataCallback = nullptr;

  if (pData->pStreamTypeLookup)
    mERROR_CHECK(mAllocator_FreePtr(pData->pAllocator, &pData->pStreamTypeLookup));

  pData->streamCount = 0;

  if (pData->pVideoStreams)
    mERROR_CHECK(mAllocator_FreePtr(pData->pAllocator, &pData->pVideoStreams));

  pData->videoStreamCount = 0;

  if (pData->pAudioStreams)
    mERROR_CHECK(mAllocator_FreePtr(pData->pAllocator, &pData->pAudioStreams));

  pData->audioStreamCount = 0;

  mERROR_CHECK(mMediaFoundation_RemoveReference());

  mRETURN_SUCCESS();
}

mFUNCTION(mMediaFileInputHandler_RunSession_Internal, IN mMediaFileInputHandler *pData)
{
  mFUNCTION_SETUP();

  size_t sampleCount = 0;
  HRESULT hr = S_OK;
  mUnused(hr);

  IMFSample *pSample = nullptr;
  mDEFER_DESTRUCTION(&pSample, _ReleaseReference);

  bool quit = false;

  while (!quit)
  {
    DWORD streamIndex, flags;
    LONGLONG timeStamp;

    mDEFER_DESTRUCTION(&pSample, _ReleaseReference);
    mERROR_IF(FAILED(hr = pData->pSourceReader->ReadSample((DWORD)MF_SOURCE_READER_ANY_STREAM, 0, &streamIndex, &flags, &timeStamp, &pSample)), mR_InternalError);
    
    if (flags & MF_SOURCE_READERF_ENDOFSTREAM)
    {
      break;
    }

    mERROR_IF(flags & MF_SOURCE_READERF_ERROR, mR_InternalError);
    mERROR_IF(flags & MF_SOURCE_READERF_NEWSTREAM, mR_InvalidParameter);

    if (flags & MF_SOURCE_READERF_CURRENTMEDIATYPECHANGED)
    {
      mERROR_IF(streamIndex >= pData->streamCount, mR_IndexOutOfBounds);
      mMediaTypeLookup *pMediaTypeLookup = &pData->pStreamTypeLookup[streamIndex];

      IMFMediaType *pMediaType;
      mDEFER_DESTRUCTION(&pMediaType, _ReleaseReference);
      mERROR_IF(FAILED(pData->pSourceReader->GetCurrentMediaType(streamIndex, &pMediaType)), mR_InternalError);

      GUID majorType;
      mERROR_IF(FAILED(hr = pMediaType->GetGUID(MF_MT_MAJOR_TYPE, &majorType)), mR_InternalError);

      if (majorType == MFMediaType_Audio)
      {
        pMediaTypeLookup->mediaType = mMMT_Audio;

        if (pMediaTypeLookup->mediaType != mMMT_Audio)
          pMediaTypeLookup->streamIndex = pData->audioStreamCount;

        mAudioStreamType audioStreamType;
        mERROR_CHECK(mAudioStreamType_Create(pMediaType, pMediaTypeLookup, streamIndex, &audioStreamType));

        if (pMediaTypeLookup->mediaType != mMMT_Audio)
          mERROR_CHECK(mMediaFileInputHandler_AddAudioStream_Internal(pData, &audioStreamType));
        else
          pData->pAudioStreams[pMediaTypeLookup->streamIndex] = audioStreamType;
      }
      else if (majorType == MFMediaType_Video)
      {
        pMediaTypeLookup->mediaType = mMMT_Video;

        if (pMediaTypeLookup->mediaType != mMMT_Video)
          pMediaTypeLookup->streamIndex = pData->videoStreamCount;

        mVideoStreamType videoStreamType;

        mERROR_CHECK(mVideoStreamType_Create(pMediaType, pMediaTypeLookup, streamIndex, &videoStreamType));

        if (pMediaTypeLookup->mediaType != mMMT_Video)
        {
          mERROR_CHECK(mMediaFileInputHandler_AddVideoStream_Internal(pData, &videoStreamType));
        }
        else
        {
          videoStreamType.pixelFormat = pData->pVideoStreams[pMediaTypeLookup->streamIndex].pixelFormat;

          pData->pVideoStreams[pMediaTypeLookup->streamIndex] = videoStreamType;
        }
      }
      else
      {
        pMediaTypeLookup[streamIndex].mediaType = mMMT_Undefined;
      }
    }

    if (pSample)
    {
      ++sampleCount;

      IMFMediaBuffer *pMediaBuffer = nullptr;
      mDEFER_DESTRUCTION(&pMediaBuffer, _ReleaseReference);
      mERROR_IF(FAILED(hr = pSample->ConvertToContiguousBuffer(&pMediaBuffer)), mR_InternalError);

      mERROR_IF(streamIndex >= pData->streamCount, mR_IndexOutOfBounds);
      mMediaTypeLookup mediaTypeLookup = pData->pStreamTypeLookup[streamIndex];

      if (mediaTypeLookup.mediaType == mMMT_Undefined)
        continue;

      uint8_t *pSampleData;
      DWORD sampleDataCurrentLength;
      DWORD sampleDataMaxLength;

      mERROR_IF(FAILED(hr = pMediaBuffer->Lock(&pSampleData, &sampleDataCurrentLength, &sampleDataMaxLength)), mR_InternalError);
      mDEFER(pMediaBuffer->Unlock());
      
      switch (mediaTypeLookup.mediaType)
      {
      case mMMT_Video:
      {
        if (pData->pProcessVideoDataCallback)
        {
          mERROR_IF(pData->videoStreamCount < mediaTypeLookup.streamIndex, mR_IndexOutOfBounds);
          mVideoStreamType videoStreamType = pData->pVideoStreams[mediaTypeLookup.streamIndex];
          mERROR_CHECK(mTimeStamp_FromSeconds(&videoStreamType.timePoint, timeStamp / (double_t)(10 * 1000 * 1000)));

          mPtr<mImageBuffer> imageBuffer;
          mDEFER_DESTRUCTION(&imageBuffer, mImageBuffer_Destroy);
          mERROR_CHECK(mImageBuffer_Create(&imageBuffer, &mDefaultAllocator, pSampleData, videoStreamType.resolution, videoStreamType.pixelFormat));
          mERROR_CHECK((*pData->pProcessVideoDataCallback)(imageBuffer, videoStreamType));
        }

        break;
      }

      case mMMT_Audio:
      {
        if (pData->pProcessAudioDataCallback)
        {
          mERROR_IF(pData->audioStreamCount < mediaTypeLookup.streamIndex, mR_IndexOutOfBounds);
          mAudioStreamType audioStreamType = pData->pAudioStreams[mediaTypeLookup.streamIndex];
          audioStreamType.bufferSize = sampleDataCurrentLength;
          mERROR_CHECK(mTimeStamp_FromSeconds(&audioStreamType.timePoint, timeStamp / (double_t)(10 * 1000 * 1000)));

          mERROR_CHECK((*pData->pProcessAudioDataCallback)(pSampleData, audioStreamType));
        }

        break;
      }

      default:
        break;
      }

      //IMF2DBuffer2 *p2DBuffer = nullptr;
      //mDEFER_DESTRUCTION(&p2DBuffer, _ReleaseReference);
      //mERROR_IF(FAILED(hr = pMediaBuffer->QueryInterface(__uuidof(IMF2DBuffer2), (void **)&p2DBuffer)), mR_InternalError);
    }
  }

  mRETURN_SUCCESS();
}

mFUNCTION(mAudioStreamType_Create, IN IMFMediaType *pMediaType, IN mMediaTypeLookup *pMediaTypeLookup, const size_t streamIndex, OUT mAudioStreamType *pAudioStreamType)
{
  mFUNCTION_SETUP();

  mERROR_IF(pMediaType == nullptr || pMediaTypeLookup == nullptr || pAudioStreamType == nullptr, mR_ArgumentNull);

  uint32_t samplesPerSecond, bitsPerSample, channelCount;
  HRESULT hr = S_OK;
  mUnused(hr);

  mERROR_IF(FAILED(hr = pMediaType->GetUINT32(MF_MT_AUDIO_SAMPLES_PER_SECOND, &samplesPerSecond)), mR_InternalError);
  mERROR_IF(FAILED(hr = pMediaType->GetUINT32(MF_MT_AUDIO_BITS_PER_SAMPLE, &bitsPerSample)), mR_InternalError);
  mERROR_IF(FAILED(hr = pMediaType->GetUINT32(MF_MT_AUDIO_NUM_CHANNELS, &channelCount)), mR_InternalError);

  mERROR_CHECK(mMemset(pAudioStreamType, 1));
  pAudioStreamType->mediaType = pMediaTypeLookup->mediaType;
  pAudioStreamType->wmf_streamIndex = streamIndex;
  pAudioStreamType->streamIndex = pMediaTypeLookup->streamIndex;
  pAudioStreamType->bitsPerSample = bitsPerSample;
  pAudioStreamType->samplesPerSecond = samplesPerSecond;
  pAudioStreamType->channelCount = channelCount;
  pAudioStreamType->bufferSize = 0;

  mRETURN_SUCCESS();
}

mFUNCTION(mVideoStreamType_Create, IN IMFMediaType *pMediaType, IN mMediaTypeLookup *pMediaTypeLookup, const size_t streamIndex, OUT mVideoStreamType *pVideoStreamType)
{
  mFUNCTION_SETUP();

  mERROR_IF(pMediaType == nullptr || pMediaTypeLookup == nullptr || pVideoStreamType == nullptr, mR_ArgumentNull);

  uint32_t resolutionX = 0, resolutionY = 0, stride = 0, frameRateRatioX, frameRateRatioY;

  HRESULT hr = S_OK;
  mUnused(hr);

  mERROR_IF(FAILED(hr = MFGetAttributeSize(pMediaType, MF_MT_FRAME_SIZE, &resolutionX, &resolutionY)), mR_InternalError);

  if (FAILED(hr = pMediaType->GetUINT32(MF_MT_DEFAULT_STRIDE, &stride)))
    stride = resolutionX;

  if (FAILED(hr = MFGetAttributeRatio(pMediaType, MF_MT_FRAME_RATE, &frameRateRatioX, &frameRateRatioY)), mR_InternalError)
  {
    frameRateRatioX = 30;
    frameRateRatioY = 1;
  }

  mERROR_CHECK(mMemset(pVideoStreamType, 1));
  pVideoStreamType->mediaType = pMediaTypeLookup->mediaType;
  pVideoStreamType->wmf_streamIndex = streamIndex;
  pVideoStreamType->streamIndex = pMediaTypeLookup->streamIndex;
  pVideoStreamType->pixelFormat = mPixelFormat::mPF_B8G8R8A8;
  pVideoStreamType->resolution.x = resolutionX;
  pVideoStreamType->resolution.y = resolutionY;
  pVideoStreamType->stride = stride;
  pVideoStreamType->frameRate = frameRateRatioX / (double_t)frameRateRatioY;

  mRETURN_SUCCESS();
}

mFUNCTION(mMediaFileInputIterator_Create, OUT mPtr<mMediaFileInputIterator> *pIterator, mPtr<mMediaFileInputHandler> &mediaFileInputHandlerRef, const size_t wmf_streamIndex, const mMediaMajorType mediaType)
{
  mFUNCTION_SETUP();

  mERROR_IF(pIterator == nullptr || mediaFileInputHandlerRef == nullptr, mR_ArgumentNull);

  if (*pIterator != nullptr)
  {
    mERROR_CHECK(mSharedPointer_Destroy(pIterator));
    *pIterator = nullptr;
  }

  mMediaFileInputIterator *pIteratorRaw = nullptr;
  mDEFER_ON_ERROR(mAllocator_FreePtr(mediaFileInputHandlerRef->pAllocator, &pIteratorRaw));
  mERROR_CHECK(mAllocator_AllocateZero(mediaFileInputHandlerRef->pAllocator, &pIteratorRaw, 1));

  mDEFER_DESTRUCTION_ON_ERROR(pIterator, mSharedPointer_Destroy);
  mERROR_CHECK(mSharedPointer_Create<mMediaFileInputIterator>(pIterator, pIteratorRaw, [](mMediaFileInputIterator *pData) { mMediaFileInputIterator_Destroy_Internal(pData); }, mediaFileInputHandlerRef->pAllocator));
  pIteratorRaw = nullptr; // to not be destroyed on error.

  mERROR_CHECK(mMediaFileInputIterator_Create_Internal(pIterator->GetPointer(), mediaFileInputHandlerRef, wmf_streamIndex, mediaType));

  mRETURN_SUCCESS();
}

mFUNCTION(mMediaFileInputIterator_Create_Internal, mMediaFileInputIterator *pIterator, mPtr<mMediaFileInputHandler> &mediaFileInputHandlerRef, const size_t wmf_streamIndex, const mMediaMajorType mediaType)
{
  mFUNCTION_SETUP();

  mERROR_IF(pIterator == nullptr || mediaFileInputHandlerRef == nullptr, mR_ArgumentNull);

  pIterator->mediaFileInputHandler = mediaFileInputHandlerRef;
  pIterator->wmf_streamIndex = wmf_streamIndex;
  pIterator->mediaType = mediaType;
  pIterator->hasFinished = false;

  mRETURN_SUCCESS();
}

mFUNCTION(mMediaFileInputIterator_Destroy_Internal, mMediaFileInputIterator *pIterator)
{
  mFUNCTION_SETUP();

  mERROR_IF(pIterator == nullptr, mR_ArgumentNull);

  if (pIterator->mediaFileInputHandler != nullptr)
    mERROR_CHECK(mSharedPointer_Destroy(&pIterator->mediaFileInputHandler));

  pIterator->wmf_streamIndex = 0;
  pIterator->mediaType = mMMT_Undefined;
  pIterator->hasFinished = false;

  mRETURN_SUCCESS();
}

mFUNCTION(mMediaFileInputIterator_IterateToStreamIndex, mPtr<mMediaFileInputIterator> &iterator, OUT IMFSample **ppSample, OUT LONGLONG *pTimeStamp)
{
  mFUNCTION_SETUP();

  mERROR_IF(iterator == nullptr || ppSample == nullptr || pTimeStamp == nullptr, mR_ArgumentNull);

  HRESULT hr = S_OK;
  mUnused(hr);

  bool quit = false;

  while (!quit)
  {
    DWORD streamIndex, flags;

    mERROR_IF(FAILED(hr = iterator->mediaFileInputHandler->pSourceReader->ReadSample((DWORD)iterator->wmf_streamIndex, 0, &streamIndex, &flags, pTimeStamp, ppSample)), mR_InternalError);

    if(streamIndex != iterator->wmf_streamIndex)
      continue;

    if (flags & MF_SOURCE_READERF_ENDOFSTREAM)
    {
      iterator->hasFinished = true;
      mRETURN_RESULT(mR_EndOfStream);
      break;
    }

    mERROR_IF(flags & MF_SOURCE_READERF_ERROR, mR_InternalError);
    mERROR_IF(flags & MF_SOURCE_READERF_NEWSTREAM, mR_InvalidParameter);

    if (flags & MF_SOURCE_READERF_CURRENTMEDIATYPECHANGED)
    {
      mERROR_IF(streamIndex >= iterator->mediaFileInputHandler->streamCount, mR_IndexOutOfBounds);
      mMediaTypeLookup *pMediaTypeLookup = &iterator->mediaFileInputHandler->pStreamTypeLookup[streamIndex];

      IMFMediaType *pMediaType;
      mDEFER_DESTRUCTION(&pMediaType, _ReleaseReference);
      mERROR_IF(FAILED(iterator->mediaFileInputHandler->pSourceReader->GetCurrentMediaType(streamIndex, &pMediaType)), mR_InternalError);

      GUID majorType;
      mERROR_IF(FAILED(hr = pMediaType->GetGUID(MF_MT_MAJOR_TYPE, &majorType)), mR_InternalError);

      if (majorType == MFMediaType_Audio)
      {
        pMediaTypeLookup->mediaType = mMMT_Audio;

        if (pMediaTypeLookup->mediaType != mMMT_Audio)
          pMediaTypeLookup->streamIndex = iterator->mediaFileInputHandler->audioStreamCount;

        mAudioStreamType audioStreamType;
        mERROR_CHECK(mAudioStreamType_Create(pMediaType, pMediaTypeLookup, streamIndex, &audioStreamType));

        if (pMediaTypeLookup->mediaType != mMMT_Audio)
          mERROR_CHECK(mMediaFileInputHandler_AddAudioStream_Internal(iterator->mediaFileInputHandler.GetPointer(), &audioStreamType));
        else
          iterator->mediaFileInputHandler->pAudioStreams[pMediaTypeLookup->streamIndex] = audioStreamType;
      }
      else if (majorType == MFMediaType_Video)
      {
        pMediaTypeLookup->mediaType = mMMT_Video;

        if (pMediaTypeLookup->mediaType != mMMT_Video)
          pMediaTypeLookup->streamIndex = iterator->mediaFileInputHandler->videoStreamCount;

        mVideoStreamType videoStreamType;

        mERROR_CHECK(mVideoStreamType_Create(pMediaType, pMediaTypeLookup, streamIndex, &videoStreamType));

        if (pMediaTypeLookup->mediaType != mMMT_Video)
        {
          mERROR_CHECK(mMediaFileInputHandler_AddVideoStream_Internal(iterator->mediaFileInputHandler.GetPointer(), &videoStreamType));
        }
        else
        {
          videoStreamType.pixelFormat = iterator->mediaFileInputHandler->pVideoStreams[pMediaTypeLookup->streamIndex].pixelFormat;

          iterator->mediaFileInputHandler->pVideoStreams[pMediaTypeLookup->streamIndex] = videoStreamType;
        }
      }
      else
      {
        pMediaTypeLookup[streamIndex].mediaType = mMMT_Undefined;
      }
    }

    if (*ppSample)
      mRETURN_SUCCESS();
  }

  mRETURN_SUCCESS();
}
