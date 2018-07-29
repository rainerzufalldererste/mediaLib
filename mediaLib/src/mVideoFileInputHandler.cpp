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
};

mFUNCTION(mVideoFileInputHandler_Create_Internal, mVideoFileInputHandler *pData, const std::wstring &fileName, const bool runAsFastAsPossible);
mFUNCTION(mVideoFileInputHandler_Destroy_Internal, mVideoFileInputHandler *pData);
mFUNCTION(mVideoFileInputHandler_RunSession_Internal, mVideoFileInputHandler *pData);
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

mFUNCTION(mVideoFileInputHandler_RunSession, IN_OUT mPtr<mVideoFileInputHandler> ptr)
{
  mFUNCTION_SETUP();

  mERROR_IF(ptr == nullptr, mR_ArgumentNull);
  mERROR_CHECK(mVideoFileInputHandler_RunSession_Internal(ptr.GetPointer()));

  mRETURN_SUCCESS();
}

mFUNCTION(mVideoFileInputHandler_Create_Internal, mVideoFileInputHandler *pInputHandler, const std::wstring &fileName, const bool runAsFastAsPossible)
{
  mFUNCTION_SETUP();
  mERROR_IF(pInputHandler == nullptr, mR_ArgumentNull);

  HRESULT hr = S_OK;
  mUnused(hr);

  const int64_t referenceCount = ++_referenceCount;

  if (referenceCount == 1)
    mERROR_CHECK(mVideoFileInputHandler_InitializeExtenalDependencies());

  mERROR_IF(FAILED(hr = MFCreateSourceReaderFromURL(fileName.c_str(), NULL, &pInputHandler->pSourceReader)), mR_InvalidParameter);

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
          mERROR_IF(FAILED(hr = pInputHandler->pSourceReader->SetCurrentMediaType(streamIndex, nullptr, pType)), mR_InternalError);
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

mFUNCTION(mVideoFileInputHandler_Destroy_Internal, mVideoFileInputHandler *pData)
{
  mFUNCTION_SETUP();

  mERROR_IF(pData == nullptr, mR_ArgumentNull);

  if (pData->pSourceReader)
  {
    pData->pSourceReader->Release();
    pData->pSourceReader = nullptr;
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

mFUNCTION(mVideoFileInputHandler_RunSession_Internal, mVideoFileInputHandler *pData)
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
    mERROR_IF(FAILED(hr = pData->pSourceReader->ReadSample(pData->streamIndex, 0, &streamIndex, &flags, &timeStamp, &pSample)), mR_InternalError);

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

      IMF2DBuffer *p2DBuffer = nullptr;
      mDEFER_DESTRUCTION(&p2DBuffer, _ReleaseReference);
      mERROR_IF(FAILED(hr = pMediaBuffer->QueryInterface(IID_IMF2DBuffer, (void **)&p2DBuffer)), mR_InternalError);
    }
  }

  mPRINT("Processed %" PRIu64 " samples\n", (uint64_t)sampleCount);

  mRETURN_SUCCESS();
}
