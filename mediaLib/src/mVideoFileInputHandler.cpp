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
#include "mVideoFileInputHandler.h"

static volatile size_t _referenceCount = 0;

struct mVideoFileInputHandler
{
  IMFSourceReader *pSourceReader;
  DWORD streamIndex;
  LONG stride;
  uint32_t resolutionX;
  uint32_t resolutionY;
  uint8_t *pFrameData;
  DWORD frameDataCurrentLength;
  DWORD frameDataMaxLength;
  ProcessBufferFunction *pProcessDataCallback;
};

mFUNCTION(mVideoFileInputHandler_Create_Internal, IN mVideoFileInputHandler *pData, const std::wstring &fileName, const bool runAsFastAsPossible);
mFUNCTION(mVideoFileInputHandler_Destroy_Internal, IN mVideoFileInputHandler *pData);
mFUNCTION(mVideoFileInputHandler_RunSession_Internal, IN mVideoFileInputHandler *pData);
mFUNCTION(mVideoFileInputHandler_InitializeExtenalDependencies);
mFUNCTION(mVideoFileInputHandler_CleanupExtenalDependencies);

template <typename T>
static void _ReleaseReference(T **pData)
{
  if (pData && *pData)
  {
    (*pData)->Release();
    *pData = nullptr;
  }
}

//////////////////////////////////////////////////////////////////////////

mFUNCTION(mVideoFileInputHandler_Create, OUT mPtr<mVideoFileInputHandler> *pPtr, const std::wstring &fileName)
{
  mFUNCTION_SETUP();

  mERROR_IF(pPtr == nullptr, mR_ArgumentNull);

  if (pPtr != nullptr)
  {
    mERROR_CHECK(mSharedPointer_Destroy(pPtr));
    *pPtr = nullptr;
  }

  mVideoFileInputHandler *pInputHandler = nullptr;
  mDEFER_DESTRUCTION_ON_ERROR(&pInputHandler, mFreePtr);
  mERROR_CHECK(mAllocZero(&pInputHandler, 1));

  mDEFER_DESTRUCTION_ON_ERROR(pPtr, mSharedPointer_Destroy);
  mERROR_CHECK(mSharedPointer_Create<mVideoFileInputHandler>(pPtr, pInputHandler, [](mVideoFileInputHandler *pData) {mVideoFileInputHandler_Destroy_Internal(pData); }, mAT_mAlloc));
  pInputHandler = nullptr; // to not be destroyed on error.

  mERROR_CHECK(mVideoFileInputHandler_Create_Internal(pPtr->GetPointer(), fileName, false));
  mRETURN_SUCCESS();
}

mFUNCTION(mVideoFileInputHandler_Destroy, IN_OUT mPtr<mVideoFileInputHandler> *pPtr)
{
  mFUNCTION_SETUP();

  mERROR_IF(pPtr == nullptr || *pPtr == nullptr, mR_ArgumentNull);
  mERROR_CHECK(mSharedPointer_Destroy(pPtr));

  mRETURN_SUCCESS();
}

mFUNCTION(mVideoFileInputHandler_Play, mPtr<mVideoFileInputHandler> &ptr)
{
  mFUNCTION_SETUP();

  mERROR_IF(ptr == nullptr, mR_ArgumentNull);
  mERROR_CHECK(mVideoFileInputHandler_RunSession_Internal(ptr.GetPointer()));

  mRETURN_SUCCESS();
}

mFUNCTION(mVideoFileInputHandler_GetSize, mPtr<mVideoFileInputHandler> &ptr, OUT size_t *pSizeX, OUT size_t *pSizeY)
{
  mFUNCTION_SETUP();

  mERROR_IF(ptr == nullptr, mR_ArgumentNull);

  if (pSizeX)
    *pSizeX = (size_t)ptr->resolutionX;

  if (pSizeY)
    *pSizeY = (size_t)ptr->resolutionY;

  mRETURN_SUCCESS();
}

mFUNCTION(mVideoFileInputHandler_SetCallback, mPtr<mVideoFileInputHandler> &ptr, IN ProcessBufferFunction *pCallback)
{
  mFUNCTION_SETUP();

  mERROR_IF(ptr == nullptr, mR_ArgumentNull);
  ptr->pProcessDataCallback = pCallback;

  mRETURN_SUCCESS();
}

mFUNCTION(mVideoFileInputHandler_Create_Internal, IN mVideoFileInputHandler *pInputHandler, const std::wstring &fileName, const bool runAsFastAsPossible)
{
  mFUNCTION_SETUP();
  mERROR_IF(pInputHandler == nullptr, mR_ArgumentNull);

  HRESULT hr = S_OK;
  mUnused(hr);

  const int64_t referenceCount = ++_referenceCount;

  if (referenceCount == 1)
    mERROR_CHECK(mVideoFileInputHandler_InitializeExtenalDependencies());

  IMFAttributes *pAttributes = nullptr;
  mDEFER_DESTRUCTION(&pAttributes, _ReleaseReference);
  mERROR_IF(FAILED(MFCreateAttributes(&pAttributes, 1)), mR_InternalError);
  mERROR_IF(FAILED(pAttributes->SetUINT32(MF_SOURCE_READER_ENABLE_VIDEO_PROCESSING, TRUE)), mR_InternalError);

  IMFMediaType *pVideoMediaType = nullptr;
  mDEFER_DESTRUCTION(&pVideoMediaType, _ReleaseReference);
  mERROR_IF(FAILED(MFCreateMediaType(&pVideoMediaType)), mR_InternalError);
  mERROR_IF(FAILED(pVideoMediaType->SetGUID(MF_MT_MAJOR_TYPE, MFMediaType_Video)), mR_InternalError);
  mERROR_IF(FAILED(pVideoMediaType->SetGUID(MF_MT_SUBTYPE, MFVideoFormat_RGB32)), mR_InternalError);

  mERROR_IF(FAILED(hr = MFCreateSourceReaderFromURL(fileName.c_str(), pAttributes, &pInputHandler->pSourceReader)), mR_InvalidParameter);

  GUID majorType;
  GUID minorType;
  DWORD streamIndex = 0;
  DWORD mediaTypeIndex = 0;
  bool isValid = false;

  while (true)
  {
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
        pType->GetMajorType(&majorType);
        pType->GetGUID(MF_MT_SUBTYPE, &minorType);
        mUnused(minorType);

        if (majorType == MFMediaType_Video)
        {
          mERROR_IF(FAILED(hr = pInputHandler->pSourceReader->SetStreamSelection(streamIndex, true)), mR_InternalError);
          mERROR_IF(FAILED(hr = pInputHandler->pSourceReader->GetCurrentMediaType(streamIndex, &pType)), mR_InternalError);
          mERROR_IF(FAILED(hr = MFGetAttributeSize(pType, MF_MT_FRAME_SIZE, &pInputHandler->resolutionX, &pInputHandler->resolutionY)), mR_InternalError);
          mERROR_IF(FAILED(hr = pInputHandler->pSourceReader->SetCurrentMediaType(streamIndex, nullptr, pVideoMediaType)), mR_InternalError);
          isValid = true;
          break;
        }
      }

      ++mediaTypeIndex;
    }

    if (hr == MF_E_INVALIDSTREAMNUMBER || isValid)
      break;

    ++streamIndex;
  }

  mERROR_IF(!isValid, mR_InvalidParameter);

  pInputHandler->streamIndex = streamIndex;

  mRETURN_SUCCESS();
}

mFUNCTION(mVideoFileInputHandler_Destroy_Internal, IN mVideoFileInputHandler *pData)
{
  mFUNCTION_SETUP();

  mERROR_IF(pData == nullptr, mR_ArgumentNull);

  if (pData->pSourceReader)
  {
    pData->pSourceReader->Release();
    pData->pSourceReader = nullptr;
  }

  if (pData->pFrameData)
  {
    mERROR_CHECK(mFreePtr(&pData->pFrameData));
    pData->frameDataCurrentLength = 0;
    pData->frameDataMaxLength = 0;
  }

  const size_t referenceCount = --_referenceCount;

  if (referenceCount == 0)
    mERROR_CHECK(mVideoFileInputHandler_CleanupExtenalDependencies());

  mRETURN_SUCCESS();
}

mFUNCTION(mVideoFileInputHandler_InitializeExtenalDependencies)
{
  mFUNCTION_SETUP();
  
  HRESULT hr = S_OK;
  mUnused(hr);

  mERROR_IF(FAILED(hr = MFStartup(MF_VERSION, MFSTARTUP_FULL)), mR_InternalError);
  mRETURN_SUCCESS();
}

mFUNCTION(mVideoFileInputHandler_CleanupExtenalDependencies)
{
  mFUNCTION_SETUP();

  HRESULT hr = S_OK;
  mUnused(hr);

  mERROR_IF(FAILED(hr = MFShutdown()), mR_InternalError);
  mRETURN_SUCCESS();
}

mFUNCTION(mVideoFileInputHandler_RunSession_Internal, IN mVideoFileInputHandler *pData)
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
    mERROR_IF(FAILED(hr = pData->pSourceReader->ReadSample(MF_SOURCE_READER_ANY_STREAM, 0, &streamIndex, &flags, &timeStamp, &pSample)), mR_InternalError);

    if(streamIndex != pData->streamIndex)
      continue;

    mPRINT("Stream %d (%I64d)\n", streamIndex, timeStamp);
    
    if (flags & MF_SOURCE_READERF_ENDOFSTREAM)
    {
      mPRINT("End of stream\n");
      break;
    }

    mERROR_IF(flags & MF_SOURCE_READERF_NEWSTREAM, mR_InvalidParameter);
    mERROR_IF(flags & MF_SOURCE_READERF_CURRENTMEDIATYPECHANGED, mR_InvalidParameter);
    mERROR_IF(flags & MF_SOURCE_READERF_NATIVEMEDIATYPECHANGED, mR_InvalidParameter);

    if (flags & MF_SOURCE_READERF_STREAMTICK)
      mPRINT("Stream tick\n");

    if (pSample)
    {
      ++sampleCount;

      IMFMediaBuffer *pMediaBuffer = nullptr;
      mDEFER_DESTRUCTION(&pMediaBuffer, _ReleaseReference);
      mERROR_IF(FAILED(hr = pSample->ConvertToContiguousBuffer(&pMediaBuffer)), mR_InternalError);

      DWORD currentLength = 0;
      mUnused(currentLength);
      mERROR_IF(FAILED(hr = pMediaBuffer->GetCurrentLength(&currentLength)), mR_InternalError);

      if (pData->pFrameData == nullptr)
      {
        mERROR_CHECK(mAlloc(&pData->pFrameData, (size_t)currentLength));
        pData->frameDataCurrentLength = currentLength;
        pData->frameDataMaxLength = currentLength;
      }

      if (pData->frameDataMaxLength < currentLength)
      {
        mERROR_CHECK(mRealloc(&pData->pFrameData, (size_t)currentLength));
        pData->frameDataMaxLength = currentLength;
      }

      pData->frameDataCurrentLength = currentLength;

      mERROR_IF(FAILED(hr = pMediaBuffer->Lock(&pData->pFrameData, &pData->frameDataCurrentLength, &pData->frameDataMaxLength)), mR_InternalError);
      
      if (pData->pProcessDataCallback)
        mERROR_CHECK((*pData->pProcessDataCallback)(pData->pFrameData));

      mDEFER(pMediaBuffer->Unlock());

      //IMF2DBuffer2 *p2DBuffer = nullptr;
      //mDEFER_DESTRUCTION(&p2DBuffer, _ReleaseReference);
      //mERROR_IF(FAILED(hr = pMediaBuffer->QueryInterface(__uuidof(IMF2DBuffer2), (void **)&p2DBuffer)), mR_InternalError);
    }
  }

  mPRINT("Processed %" PRIu64 " samples\n", (uint64_t)sampleCount);

  mRETURN_SUCCESS();
}
