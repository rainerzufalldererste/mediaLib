#include "mHttpServer.h"

#include "mTcpSocket.h"
#include "mThread.h"
#include "mThreadPool.h"

#include "http_parser/src/http_parser.h"

//////////////////////////////////////////////////////////////////////////

struct mHttpServer
{
  mAllocator *pAllocator;
  mPtr<mTcpServer> tcpServer;
  http_parser_settings settings;
  mThread *pListenerThread;
  mPtr<mTasklessThreadPool> threadPool;
  volatile bool keepRunning;
};

struct mHttpRequest_Parser : mHttpRequest
{
  mString *pLastAttributeValue;
  mResult result;
  mString body;
};

mFUNCTION(mHttpServer_Destroy_Internal, IN_OUT mHttpServer *pHttpServer);
mFUNCTION(mHttpServer_ThreadInternal, mPtr<mHttpServer> &server);

int32_t _OnUrl(http_parser *, const char *at, size_t length);
int32_t _OnHeaderField(http_parser *, const char *at, size_t length);
int32_t _OnHeaderValue(http_parser *, const char *at, size_t length);
int32_t _OnBody(http_parser *, const char *at, size_t length);

//////////////////////////////////////////////////////////////////////////

mFUNCTION(mHttpServer_Create, OUT mPtr<mHttpServer> *pHttpServer, IN mAllocator *pAllocator, const uint16_t port /* = 80 */)
{
  mFUNCTION_SETUP();

  mERROR_IF(pHttpServer == nullptr, mR_ArgumentNull);

  mERROR_CHECK(mSharedPointer_Allocate<mHttpServer>(pHttpServer, pAllocator, [](mHttpServer *pData) { mHttpServer_Destroy_Internal(pData); }, 1));

  mERROR_CHECK(mTcpServer_Create(&(*pHttpServer)->tcpServer, pAllocator, port));

  http_parser_settings_init(&(*pHttpServer)->settings);

  (*pHttpServer)->settings.on_url = _OnUrl;
  (*pHttpServer)->settings.on_header_field = _OnHeaderField;
  (*pHttpServer)->settings.on_header_value = _OnHeaderValue;
  (*pHttpServer)->settings.on_body = _OnBody;

  (*pHttpServer)->keepRunning = true;
  (*pHttpServer)->pAllocator = pAllocator;

  mERROR_CHECK(mThread_Create(&(*pHttpServer)->pListenerThread, pAllocator, mHttpServer_ThreadInternal, *pHttpServer));

  mRETURN_SUCCESS();
}

//////////////////////////////////////////////////////////////////////////

mFUNCTION(mHttpServer_Destroy_Internal, IN_OUT mHttpServer *pHttpServer)
{
  mFUNCTION_SETUP();

  mERROR_IF(pHttpServer == nullptr, mR_ArgumentNull);

  pHttpServer->keepRunning = false;

  mERROR_CHECK(mSharedPointer_Destroy(&pHttpServer->tcpServer));
  mERROR_CHECK(mThread_Destroy(&pHttpServer->pListenerThread));
  mERROR_CHECK(mTasklessThreadPool_Destroy(&pHttpServer->threadPool));

  mRETURN_SUCCESS();
}

mFUNCTION(mHttpServer_ThreadInternal, mPtr<mHttpServer> &server)
{
  mFUNCTION_SETUP();

  while (server->keepRunning)
  {
    mPtr<mTcpClient> client;

    if (mSUCCEEDED(mTcpServer_Listen(server->tcpServer, &client, server->pAllocator)))
    {
      std::function<void ()> task = [server, client]() mutable
      {
        char data[8 * 1024];
        size_t bytesReceived = 0;

        if (mSUCCEEDED(mTcpClient_Receive(client, data, sizeof(data), &bytesReceived)))
        {
          data[mMin(bytesReceived + 1, sizeof(data) - 1)] = '\0';

          http_parser parser;
          http_parser_init(&parser, HTTP_REQUEST);

          mUniqueContainer<mHttpRequest_Parser> request;

          parser.data = request.GetPointer();

          http_parser_execute(&parser, &server->settings, data, bytesReceived);

          if (mFAILED(request->result))
            RETURN;

          request->pLastAttributeValue = nullptr;
        }

        RETURN;
      };

      mERROR_CHECK(mTasklessThreadPool_EnqueueTask(server->threadPool, task));
    }
  }

  mRETURN_SUCCESS();
}

int32_t _OnUrl(http_parser *pParser, const char *at, size_t length)
{
  mHttpRequest_Parser *pRequest = reinterpret_cast<mHttpRequest_Parser *>(pParser->data);

  if (mFAILED(pRequest->result))
    return 0;

  if (pRequest->url.bytes == 0)
    pRequest->result = mString_Create(&pRequest->url, at, length);
  else
    pRequest->result = mString_Append(pRequest->url, at, length);

  return 0;
}

int32_t _OnHeaderField(http_parser *pParser, const char *at, size_t length)
{
  mHttpRequest_Parser *pRequest = reinterpret_cast<mHttpRequest_Parser *>(pParser->data);

  if (mFAILED(pRequest->result))
    return 0;

  mString key, value;

  pRequest->result = mString_Create(&key, at, length);

  if (mFAILED(pRequest->result))
    return 0;

  pRequest->result = mHashMap_Add(pRequest->attributes, key, &value);

  if (mFAILED(pRequest->result))
    return 0;

  bool contained = false;

  pRequest->result = mHashMap_ContainsGetPointer(pRequest->attributes, key, &contained, &pRequest->pLastAttributeValue);

  if (mFAILED(pRequest->result))
    return 0;

  if (!contained)
  {
    pRequest->pLastAttributeValue = nullptr;
    pRequest->result = mR_ResourceNotFound;
    return 0;
  }

  return 0;
}

int32_t _OnHeaderValue(http_parser *pParser, const char *at, size_t length)
{
  mHttpRequest_Parser *pRequest = reinterpret_cast<mHttpRequest_Parser *>(pParser->data);

  if (mFAILED(pRequest->result))
    return 0;

  if (pRequest->pLastAttributeValue == nullptr)
  {
    pRequest->result = mR_ResourceStateInvalid;
    return 0;
  }

  if (pRequest->pLastAttributeValue->bytes == 0)
    pRequest->result = mString_Create(pRequest->pLastAttributeValue, at, length);
  else
    pRequest->result = mString_Append(*pRequest->pLastAttributeValue, at, length);

  return 0;
}

int32_t _OnBody(http_parser *pParser, const char *at, size_t length)
{
  mHttpRequest_Parser *pRequest = reinterpret_cast<mHttpRequest_Parser *>(pParser->data);

  if (mFAILED(pRequest->result))
    return 0;

  if (pRequest->body.bytes == 0)
    pRequest->result = mString_Create(&pRequest->body, at, length);
  else
    pRequest->result = mString_Append(pRequest->body, at, length);

  return 0;
}

//////////////////////////////////////////////////////////////////////////

#undef RETURN

#pragma warning(push, 0)
#include "http_parser/src/http_parser.c"
#pragma warning(pop)
