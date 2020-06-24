#include "mHttpRequest.h"

#define CURL_STATICLIB 1
#include "curl/curl.h"

#ifdef GIT_BUILD // Define __M_FILE__
  #ifdef __M_FILE__
    #undef __M_FILE__
  #endif
  #define __M_FILE__ "gOnPVflmXccwLhrcSuKQE3Rf9FY8OoFmMot3c1+gvgfZIX30GMcj7O3k11NLPCR23w+CIUCkC502oURI"
#endif

//////////////////////////////////////////////////////////////////////////

struct mHttpRequest
{
  CURL *pCurl;
  curl_slist *pChunk;
  mString url;
  size_t responseSize, responseCapacity;
  uint8_t *pResponseBytes;
  bool responseRequested;
  int32_t timeout;
};

static volatile bool mHttpRequest_Initialized = false;
constexpr size_t mHttpRequest_DefaultTimeout = 5000;

//////////////////////////////////////////////////////////////////////////

mFUNCTION(mHttpRequest_Destroy_Internal, mHttpRequest *pHttpRequest);
size_t mHttpRequest_WriteMemoryCallback(const void *pContents, const size_t size, const size_t count, void *pUserData);

//////////////////////////////////////////////////////////////////////////

mFUNCTION(mHttpRequest_Create, OUT mPtr<mHttpRequest> *pHttpRequest, IN mAllocator *pAllocator, const mString &url)
{
  mFUNCTION_SETUP();

  mERROR_IF(pHttpRequest == nullptr, mR_ArgumentNull);
  mERROR_IF(url.hasFailed || url.count < 2, mR_InvalidParameter);

  mERROR_CHECK(mSharedPointer_Allocate<mHttpRequest>(pHttpRequest, pAllocator, [](mHttpRequest *pData) { mHttpRequest_Destroy_Internal(pData); }, 1));

  if (!mHttpRequest_Initialized)
  {
    mDEFER_ON_ERROR(mHttpRequest_Initialized = false);
    mHttpRequest_Initialized = true;

    mERROR_IF(0 != curl_global_init(CURL_GLOBAL_DEFAULT), mR_InternalError);
  }

  (*pHttpRequest)->pCurl = curl_easy_init();
  mERROR_IF((*pHttpRequest)->pCurl == nullptr, mR_InternalError);

  new (&(*pHttpRequest)->url) mString(url);
  
  (*pHttpRequest)->timeout = mHttpRequest_DefaultTimeout;

  mRETURN_SUCCESS();
}

mFUNCTION(mHttpRequest_Destroy, IN_OUT mPtr<mHttpRequest> *pHttpRequest)
{
  return mSharedPointer_Destroy(pHttpRequest);
}

mFUNCTION(mHttpRequest_AddHeader, mPtr<mHttpRequest> &httpRequest, const mString &header)
{
  mFUNCTION_SETUP();

  mERROR_IF(httpRequest == nullptr, mR_ArgumentNull);
  mERROR_IF(header.hasFailed || header.count < 2, mR_InvalidParameter);

  curl_slist *pNewList = curl_slist_append(httpRequest->pChunk, header.c_str());

  mERROR_IF(pNewList == nullptr, mR_InternalError);

  httpRequest->pChunk = pNewList;

  mRETURN_SUCCESS();
}

mFUNCTION(mHttpRequest_SetTimeout, mPtr<mHttpRequest> &httpRequest, const size_t timeout)
{
  mFUNCTION_SETUP();

  mERROR_IF(httpRequest == nullptr, mR_ArgumentNull);
  mERROR_IF(timeout > INT32_MAX, mR_ArgumentOutOfBounds);

  httpRequest->timeout = (int32_t)timeout;

  mRETURN_SUCCESS();
}

mFUNCTION(mHttpRequest_Send, mPtr<mHttpRequest> &httpRequest)
{
  mFUNCTION_SETUP();

  mERROR_IF(httpRequest == nullptr, mR_ArgumentNull);

  mERROR_IF(CURLE_OK != curl_easy_setopt(httpRequest->pCurl, CURLOPT_URL, httpRequest->url.c_str()), mR_InternalError);
  mERROR_IF(CURLE_OK != curl_easy_setopt(httpRequest->pCurl, CURLOPT_WRITEFUNCTION, mHttpRequest_WriteMemoryCallback), mR_InternalError);
  mERROR_IF(CURLE_OK != curl_easy_setopt(httpRequest->pCurl, CURLOPT_WRITEDATA, reinterpret_cast<void *>(httpRequest.GetPointer())), mR_InternalError);

  if (httpRequest->pChunk != nullptr)
    mERROR_IF(CURLE_OK != curl_easy_setopt(httpRequest->pCurl, CURLOPT_HTTPHEADER, httpRequest->pChunk), mR_InternalError);

  if (httpRequest->timeout)
    mERROR_IF(CURLE_OK != curl_easy_setopt(httpRequest->pCurl, CURLOPT_TIMEOUT_MS, httpRequest->timeout), mR_InternalError);
  
  const CURLcode result = curl_easy_perform(httpRequest->pCurl);

  switch (result)
  {
  case CURLE_OK:
  {
    int32_t response_code;
    mERROR_IF(CURLE_OK != curl_easy_getinfo(httpRequest->pCurl, CURLINFO_RESPONSE_CODE, &response_code), mR_InternalError);
    mERROR_IF(response_code >= 400, mR_IOFailure);

    break;
  }

  case CURLE_OPERATION_TIMEDOUT:
    mRETURN_RESULT(mR_Timeout);

  case CURLE_COULDNT_CONNECT:
    mRETURN_RESULT(mR_ResourceNotFound);

  default:
    mRETURN_RESULT(mR_Failure);
  }

  httpRequest->responseRequested = true;

  mRETURN_SUCCESS();
}

mFUNCTION(mHttpRequest_GetResponseSize, mPtr<mHttpRequest> &httpRequest, OUT size_t *pBytes)
{
  mFUNCTION_SETUP();

  mERROR_IF(httpRequest == nullptr || pBytes == nullptr, mR_ArgumentNull);
  mERROR_IF(!httpRequest->responseRequested, mR_ResourceStateInvalid);

  *pBytes = httpRequest->responseSize;

  mRETURN_SUCCESS();
}

mFUNCTION(mHttpRequest_GetResponseBytes, mPtr<mHttpRequest> &httpRequest, OUT uint8_t *pBuffer, const size_t bufferSize)
{
  mFUNCTION_SETUP();

  mERROR_IF(httpRequest == nullptr || pBuffer == nullptr, mR_ArgumentNull);
  mERROR_IF(!httpRequest->responseRequested, mR_ResourceStateInvalid);
  mERROR_IF(httpRequest->responseSize > bufferSize, mR_ArgumentOutOfBounds);

  mERROR_CHECK(mMemcpy(pBuffer, httpRequest->pResponseBytes, httpRequest->responseSize));

  mRETURN_SUCCESS();
}

mFUNCTION(mHttpRequest_GetResponseString, mPtr<mHttpRequest> &httpRequest, OUT mString *pResponse)
{
  mFUNCTION_SETUP();

  mERROR_IF(httpRequest == nullptr || pResponse == nullptr, mR_ArgumentNull);
  mERROR_IF(!httpRequest->responseRequested, mR_ResourceStateInvalid);

  *pResponse = reinterpret_cast<const char *>(httpRequest->pResponseBytes);

  mRETURN_SUCCESS();
}

//////////////////////////////////////////////////////////////////////////

mFUNCTION(mHttpRequest_Destroy_Internal, mHttpRequest *pHttpRequest)
{
  mFUNCTION_SETUP();

  mERROR_IF(pHttpRequest == nullptr, mR_ArgumentNull);

  if (pHttpRequest->pCurl != nullptr)
    curl_easy_cleanup(pHttpRequest->pCurl);

  if (pHttpRequest->pChunk != nullptr)
    curl_slist_free_all(pHttpRequest->pChunk);

  pHttpRequest->url.~mString();

  if (pHttpRequest->pResponseBytes != nullptr)
    mERROR_CHECK(mFreePtr(&pHttpRequest->pResponseBytes));

  mRETURN_SUCCESS();
}

size_t mHttpRequest_WriteMemoryCallback(const void *pContents, const size_t size, const size_t count, void *pUserData)
{
  mHttpRequest *pHttpRequest = reinterpret_cast<mHttpRequest *>(pUserData);

  const size_t additionalSize = size * count;
  size_t newSize = pHttpRequest->responseSize + additionalSize + 1;

  if (newSize > pHttpRequest->responseCapacity)
  {
    newSize = mMax(pHttpRequest->responseCapacity * 2 + 1, newSize);

    if (mFAILED(mRealloc(&pHttpRequest->pResponseBytes, newSize)))
      return 0;

    pHttpRequest->responseCapacity = newSize;
  }

  mMemcpy(pHttpRequest->pResponseBytes + pHttpRequest->responseSize, reinterpret_cast<const uint8_t *>(pContents), additionalSize);

  pHttpRequest->responseSize += additionalSize;
  pHttpRequest->pResponseBytes[pHttpRequest->responseSize] = 0;

  return additionalSize;
}
