#include "mHttpServer.h"

#include "mTcpSocket.h"
#include "mThread.h"
#include "mThreadPool.h"

#include "http_parser/src/http_parser.h"

#ifdef GIT_BUILD // Define __M_FILE__
  #ifdef __M_FILE__
    #undef __M_FILE__
  #endif
  #define __M_FILE__ "H/L+uBO77NtNTeWaUgmywwRSPoj74Gv6U+ejvXjCr/Q2XXeSy0CFuKfXlxm/Y6anWlybOlGLZ9N70rOK"
#endif

//////////////////////////////////////////////////////////////////////////

struct mHttpServer
{
  mAllocator *pAllocator;
  mPtr<mTcpServer> tcpServer;
  http_parser_settings settings;
  mThread *pListenerThread;
  mThread *pStaleHandlerThread;
  mPtr<mTasklessThreadPool> threadPool;
  volatile bool keepRunning;

  mUniqueContainer<mQueue<mPtr<mHttpRequestHandler>>> requestHandlers;
  mPtr<mHttpErrorRequestHandler> errorRequestHandler;
  mUniqueContainer<mQueue<mPtr<mTcpClient>>> staleTcpClients;
  mMutex *pStaleTcpClientMutex;
};

struct mHttpRequest_Parser : mHttpRequest
{
  mKeyValuePair<mString, mString> *pLastAttributeValue;
  mResult result;
  mString body;
};

static mFUNCTION(mHttpServer_Destroy_Internal, IN_OUT mHttpServer *pHttpServer);
static mFUNCTION(mHttpServer_Thread_Internal, mPtr<mHttpServer> &server);
static mFUNCTION(mHttpServer_StaleTcpHandlerThread_Internal, mPtr<mHttpServer> &server);
static mFUNCTION(mHttpServer_SendResponsePacket_Internal, mPtr<mTcpClient> &client, const mPtr<mHttpResponse> &response, IN mAllocator *pAllocator);
static mFUNCTION(mHttpServer_RespondWithError_Internal, mPtr<mHttpServer> &server, mPtr<mTcpClient> &client, const mHttpResponseStatusCode statusCode, const mString &errorString, IN mAllocator *pAllocator);

static int32_t mHttpServer_OnUrl_Internal(http_parser *, const char *at, size_t length);
static int32_t mHttpServer_OnHeaderField_Internal(http_parser *, const char *at, size_t length);
static int32_t mHttpServer_OnHeaderValue_Internal(http_parser *, const char *at, size_t length);
static int32_t mHttpServer_OnBody_Internal(http_parser *, const char *at, size_t length);

static void mHttpServer_HandleTcpClient_Internal(mPtr<mHttpServer> &server, mPtr<mTcpClient> &client);

static mFUNCTION(mHttpRequest_Parser_Init_Internal, mPtr<mHttpRequest_Parser> &request, IN mAllocator *pAllocator, mPtr<mTcpClient> &client);
static void mHttpRequest_Parser_Destroy_Internal(IN_OUT mHttpRequest_Parser *pRequest);

static mFUNCTION(mHttpRequest_ParseArguments_Internal, const char *params, OUT mPtr<mQueue<mKeyValuePair<mString, mString>>> &args, IN mAllocator *pAllocator, const bool ignoreFragment);

static mFUNCTION(mHttpResponse_Init_Internal, mPtr<mHttpResponse> &response, IN mAllocator *pAllocator);
static void mHttpResponse_Destroy_Internal(IN_OUT mHttpResponse *pResponse);

static const char *mHttpResponse_AsString_100[] = { "100 Continue", "101 Switching Protocols", "", "103 Early Hints" };
static const char *mHttpResponse_AsString_200[] = { "200 OK", "201 Created", "202 Accepted", "203 Non-Authoritative Information", "204 No Content", "205 Reset Content", "206 Partial Content" };
static const char *mHttpResponse_AsString_300[] = { "300 Multiple Choices", "301 Moved Permanently", "302 Found", "303 See Other", "304 Not Modified", "", "", "307 Temporary Redirect", "308 Permanent Redirect" };
static const char *mHttpResponse_AsString_400[] = { "400 Bad Request", "401 Unauthorized", "402 Payment Required", "403 Forbidden", "404 Not Found", "405 Method Not Allowed", "406 Not Acceptable", "407 Proxy Authentication Required", "408 Request Timeout", "409 Conflict", "410 Gone", "411 Length Required", "412 Precondition Failed", "413 Payload Too Large", "414 URI Too Long", "415 Unsupported Media Type", "416 Range Not Satisfiable", "417 Expectation Failed", "418 I'm a teapot", "", "", "", "422 Unprocessable Entity", "", "", "425 Too Early", "426 Upgrade Required", "", "428 Precondition Required", "429 Too Many Requests", "", "431 Request Header Fields Too Large", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "451 Unavailable For Legal Reasons" };
static const char *mHttpResponse_AsString_500[] = { "500 Internal Server Error", "501 Not Implemented", "502 Bad Gateway", "503 Service Unavailable", "504 Gateway Timeout", "505 HTTP Version Not Supported", "506 Variant Also Negotiates", "507 Insufficient Storage", "508 Loop Detected", "", "510 Not Extended", "511 Network Authentication Required" };

//////////////////////////////////////////////////////////////////////////

inline mFUNCTION(mHttpRequest_UnHexSingle_Internal, IN const char *hex, OUT uint8_t &byte)
{
  mFUNCTION_SETUP();

  if (*hex >= '0' && *hex <= '9')
    byte |= (*hex - '0');
  else if (*hex >= 'a' && *hex <= 'f')
    byte |= (*hex - 'a' + 0xA);
  else if (*hex >= 'A' && *hex <= 'F')
    byte |= (*hex - 'A' + 0xA);
  else
    mRETURN_RESULT(mR_ResourceInvalid);

  mRETURN_SUCCESS();
}

inline mFUNCTION(mHttpRequest_UnHex_Internal, IN const char *hex, OUT uint8_t &byte)
{
  mFUNCTION_SETUP();
  
  byte = 0;

  mERROR_CHECK(mHttpRequest_UnHexSingle_Internal(hex, byte));

  byte <<= 4;

  mERROR_CHECK(mHttpRequest_UnHexSingle_Internal(hex + 1, byte));

  mRETURN_SUCCESS();
}

//////////////////////////////////////////////////////////////////////////

mFUNCTION(mHttpServer_Create, OUT mPtr<mHttpServer> *pHttpServer, IN mAllocator *pAllocator, const uint16_t port /* = 80 */, const size_t threadCount /* = 8 */)
{
  mFUNCTION_SETUP();

  mERROR_IF(pHttpServer == nullptr, mR_ArgumentNull);

  mPtr<mTasklessThreadPool> threadPool;
  mERROR_CHECK(mTasklessThreadPool_Create(&threadPool, pAllocator, threadCount));

  mERROR_CHECK(mHttpServer_Create(pHttpServer, pAllocator, threadPool, port));

  mRETURN_SUCCESS();
}

mFUNCTION(mHttpServer_Create, OUT mPtr<mHttpServer> *pHttpServer, IN mAllocator *pAllocator, mPtr<mTasklessThreadPool> &threadPool, const uint16_t port /* = 80 */)
{
  mFUNCTION_SETUP();

  mERROR_IF(pHttpServer == nullptr, mR_ArgumentNull);

  mERROR_CHECK(mSharedPointer_Allocate<mHttpServer>(pHttpServer, pAllocator, [](mHttpServer *pData) { mHttpServer_Destroy_Internal(pData); }, 1));

  mERROR_CHECK(mTcpServer_Create(&(*pHttpServer)->tcpServer, pAllocator, port));

  http_parser_settings_init(&(*pHttpServer)->settings);

  (*pHttpServer)->settings.on_url = mHttpServer_OnUrl_Internal;
  (*pHttpServer)->settings.on_header_field = mHttpServer_OnHeaderField_Internal;
  (*pHttpServer)->settings.on_header_value = mHttpServer_OnHeaderValue_Internal;
  (*pHttpServer)->settings.on_body = mHttpServer_OnBody_Internal;

  (*pHttpServer)->keepRunning = true;
  (*pHttpServer)->pAllocator = pAllocator;
  (*pHttpServer)->threadPool = threadPool;

  mERROR_CHECK(mQueue_Create(&(*pHttpServer)->requestHandlers, pAllocator));
  mERROR_CHECK(mQueue_Create(&(*pHttpServer)->staleTcpClients, pAllocator));
  mERROR_CHECK(mMutex_Create(&(*pHttpServer)->pStaleTcpClientMutex, pAllocator));
  
  mRETURN_SUCCESS();
}

mFUNCTION(mHttpServer_Destroy, IN_OUT mPtr<mHttpServer> *pHttpServer)
{
  return mSharedPointer_Destroy(pHttpServer);
}

mFUNCTION(mHttpServer_Start, mPtr<mHttpServer> &httpServer)
{
  mFUNCTION_SETUP();

  mERROR_IF(httpServer == nullptr, mR_ArgumentNull);
  mERROR_IF(httpServer->pListenerThread != nullptr, mR_ResourceStateInvalid);

  mERROR_CHECK(mThread_Create(&httpServer->pListenerThread, httpServer->pAllocator, mHttpServer_Thread_Internal, httpServer));
  mERROR_CHECK(mThread_Create(&httpServer->pStaleHandlerThread, httpServer->pAllocator, mHttpServer_StaleTcpHandlerThread_Internal, httpServer));

  mRETURN_SUCCESS();
}

mFUNCTION(mHttpServer_AddRequestHandler, mPtr<mHttpServer> &httpServer, mPtr<mHttpRequestHandler> &requestHandler)
{
  mFUNCTION_SETUP();

  mERROR_IF(httpServer == nullptr || requestHandler == nullptr, mR_ArgumentNull);
  mERROR_IF(httpServer->pListenerThread != nullptr, mR_ResourceStateInvalid);

  mERROR_CHECK(mQueue_PushBack(httpServer->requestHandlers, requestHandler));

  mRETURN_SUCCESS();
}

mFUNCTION(mHttpServer_SetErrorRequestHandler, mPtr<mHttpServer> &httpServer, mPtr<mHttpErrorRequestHandler> &requestHandler)
{
  mFUNCTION_SETUP();

  mERROR_IF(httpServer == nullptr || requestHandler == nullptr, mR_ArgumentNull);
  mERROR_IF(httpServer->pListenerThread != nullptr, mR_ResourceStateInvalid);

  httpServer->errorRequestHandler = requestHandler;

  mRETURN_SUCCESS();
}

//////////////////////////////////////////////////////////////////////////

static mFUNCTION(mHttpServer_Destroy_Internal, IN_OUT mHttpServer *pHttpServer)
{
  mFUNCTION_SETUP();

  mERROR_IF(pHttpServer == nullptr, mR_ArgumentNull);

  pHttpServer->keepRunning = false;

  mERROR_CHECK(mTasklessThreadPool_Destroy(&pHttpServer->threadPool));
  mERROR_CHECK(mThread_Destroy(&pHttpServer->pListenerThread));
  mERROR_CHECK(mThread_Destroy(&pHttpServer->pStaleHandlerThread));
  mERROR_CHECK(mSharedPointer_Destroy(&pHttpServer->tcpServer));
  mERROR_CHECK(mMutex_Destroy(&pHttpServer->pStaleTcpClientMutex));

  mERROR_CHECK(mSharedPointer_Destroy(&pHttpServer->errorRequestHandler));
  mERROR_CHECK(mQueue_Destroy(&pHttpServer->requestHandlers));

  mERROR_CHECK(mQueue_Destroy(&pHttpServer->staleTcpClients));

  mRETURN_SUCCESS();
}

static mFUNCTION(mHttpServer_Thread_Internal, mPtr<mHttpServer> &server)
{
  mFUNCTION_SETUP();

  while (server->keepRunning)
  {
    mPtr<mTcpClient> client;

    if (mSUCCEEDED(mTcpServer_Listen(server->tcpServer, &client, server->pAllocator)))
      mERROR_CHECK(mTasklessThreadPool_EnqueueTask(server->threadPool, [server, client]() mutable { mHttpServer_HandleTcpClient_Internal(server, client); }));
  }

  mRETURN_SUCCESS();
}

static mFUNCTION(mHttpServer_StaleTcpHandlerThread_Internal, mPtr<mHttpServer> &server)
{
  mFUNCTION_SETUP();

  while (server->keepRunning)
  {
    size_t index = 0;
    size_t count = 0;

    while (server->keepRunning)
    {
      mPtr<mTcpClient> client = nullptr;

      // Get Next Client.
      {
        mERROR_CHECK(mMutex_Lock(server->pStaleTcpClientMutex));
        mDEFER_CALL(server->pStaleTcpClientMutex, mMutex_Unlock);

        mERROR_CHECK(mQueue_GetCount(server->staleTcpClients, &count));

        ++index;

        if (index >= count)
          break;

        mERROR_CHECK(mQueue_PeekAt(server->staleTcpClients, index, &client));
      }

      uint8_t _unused[1];
      size_t bytesReadable = 0;

      // if the client disconnected.
      if (mR_IOFailure == mSILENCE_ERROR(mTcpClient_Receive(client, &_unused, 0, nullptr)) || mFAILED(mTcpClient_GetReadableBytes(client, &bytesReadable)))
      {
        // Remove from queue.
        {
          mERROR_CHECK(mMutex_Lock(server->pStaleTcpClientMutex));
          mDEFER_CALL(server->pStaleTcpClientMutex, mMutex_Unlock);

          mERROR_CHECK(mQueue_PopAt(server->staleTcpClients, index, &client));
        }
        
        client = nullptr;

        --index;
      }
      // if the client has sent some more data.
      else if (bytesReadable > 0)
      {
        mERROR_CHECK(mTasklessThreadPool_EnqueueTask(server->threadPool, [server, client]() mutable { mHttpServer_HandleTcpClient_Internal(server, client); }));

        // Remove from queue.
        {
          mERROR_CHECK(mMutex_Lock(server->pStaleTcpClientMutex));
          mDEFER_CALL(server->pStaleTcpClientMutex, mMutex_Unlock);

          mERROR_CHECK(mQueue_PopAt(server->staleTcpClients, index, &client));
        }

        client = nullptr;

        --index;
      }
    }

    mSleep(1);
  }

  mRETURN_SUCCESS();
}

static void mHttpServer_HandleTcpClient_Internal(mPtr<mHttpServer> &server, mPtr<mTcpClient> &client)
{
  // TODO: Use Arena Allocator.

  char data[8 * 1024];
  size_t bytesReceived = 0;

  while (mSUCCEEDED(mSILENCE_ERROR(mTcpClient_Receive(client, data, sizeof(data), &bytesReceived))))
  {
    data[mMin(bytesReceived + 1, sizeof(data) - 1)] = '\0';

    http_parser parser;
    http_parser_init(&parser, HTTP_REQUEST);

    mUniqueContainer<mHttpRequest_Parser> request;
    mUniqueContainer<mHttpRequest_Parser>::CreateWithCleanupFunction(&request, mHttpRequest_Parser_Destroy_Internal);

    if (mFAILED(mHttpRequest_Parser_Init_Internal(request, server->pAllocator, client)))
      return;

    parser.data = request.GetPointer();

    http_parser_execute(&parser, &server->settings, data, bytesReceived);

    if (mFAILED(request->result))
    {
      mHttpServer_RespondWithError_Internal(server, client, mHRSC_BadRequest, "Failed to parse HTTP Header.", server->pAllocator);

      return;
    }

    request->pLastAttributeValue = nullptr;

    if (parser.type != HTTP_REQUEST)
    {
      mHttpServer_RespondWithError_Internal(server, client, mHRSC_BadRequest, "Expected HTTP Request.", server->pAllocator);

      return;
    }

    request->requestMethod = (mHttpRequestMethod)parser.method;

    if (request->requestMethod == mHRM_Post && request->body.bytes > 1)
    {
      if (mFAILED(mHttpRequest_ParseArguments_Internal(request->body.c_str(), request->postParameters, server->pAllocator, false)))
      {
        mHttpServer_RespondWithError_Internal(server, client, mHRSC_BadRequest, "Failed to parse POST parameters.", server->pAllocator);

        return;
      }
    }

    mUniqueContainer<mHttpResponse> response;
    mUniqueContainer<mHttpResponse>::ConstructWithCleanupFunction(&response, mHttpResponse_Destroy_Internal);

    if (mFAILED(mHttpResponse_Init_Internal(response, server->pAllocator)))
      return;

    bool handled = false;

    mPtr<mHttpRequest> requestWrap = (mPtr<mHttpRequest>)((mPtr<mHttpRequest_Parser>)request);

    for (auto &_handler : server->requestHandlers->Iterate())
    {
      handled = false;

      if (_handler->pHandleRequest && mSUCCEEDED(_handler->pHandleRequest(_handler, requestWrap, &handled, response)) && handled)
      {
        if (mFAILED(mHttpServer_SendResponsePacket_Internal(client, response, server->pAllocator)))
        {
          mHttpServer_RespondWithError_Internal(server, client, mHRSC_InternalServerError, "Failed to send response packet.", server->pAllocator);

          return;
        }

        break;
      }
    }

    if (!handled)
    {
      mHttpServer_RespondWithError_Internal(server, client, mHRSC_InternalServerError, "No Response Handler found for this request.", server->pAllocator);

      return;
    }

    size_t readableBytes = 0;

    if (mFAILED(mTcpClient_GetReadableBytes(client, &readableBytes)))
      return;

    if (readableBytes == 0)
    {
      if (mSUCCEEDED(mMutex_Lock(server->pStaleTcpClientMutex)))
      {
        mQueue_PushBack(server->staleTcpClients, std::move(client));
        mMutex_Unlock(server->pStaleTcpClientMutex);
      }

      return;
    }
  }

  return;
}

static mFUNCTION(mHttpServer_RespondWithError_Internal, mPtr<mHttpServer> &server, mPtr<mTcpClient> &client, const mHttpResponseStatusCode statusCode, const mString &errorString, IN mAllocator *pAllocator)
{
  mFUNCTION_SETUP();

  mERROR_IF(client == nullptr || server == nullptr, mR_ArgumentNull);
  mERROR_IF(server->errorRequestHandler == nullptr, mR_Success);
  mERROR_IF(server->errorRequestHandler->pHandleRequest == nullptr, mR_ResourceStateInvalid);

  mUniqueContainer<mHttpResponse> response;
  mUniqueContainer<mHttpResponse>::ConstructWithCleanupFunction(&response, mHttpResponse_Destroy_Internal);

  mERROR_CHECK(mHttpResponse_Init_Internal(response, pAllocator));
  mERROR_CHECK(mBinaryChunk_GrowBack(response->responseStream, (errorString.bytes + 1023) & ~(uint64_t)1023));

  response->statusCode = statusCode;

  mERROR_CHECK(server->errorRequestHandler->pHandleRequest(server->errorRequestHandler, errorString, response));

  mERROR_CHECK(mHttpServer_SendResponsePacket_Internal(client, response, pAllocator));

  mRETURN_SUCCESS();
}

static mFUNCTION(mHttpServer_SendResponsePacket_Internal, mPtr<mTcpClient> &client, const mPtr<mHttpResponse> &response, IN mAllocator *pAllocator)
{
  mFUNCTION_SETUP();

  mERROR_IF(client == nullptr || response == nullptr, mR_ArgumentNull);

  mPtr<mBinaryChunk> responsePacket;
  mERROR_CHECK(mBinaryChunk_Create(&responsePacket, pAllocator));

  const char http[] = "HTTP/1.1 ";
  mERROR_CHECK(mBinaryChunk_WriteBytes(responsePacket, reinterpret_cast<const uint8_t *>(http), sizeof(http) - 1));

  switch (response->statusCode / 100)
  {
  case 1:
  {
    const size_t statusCode = response->statusCode - 100;
    mERROR_IF(statusCode >= mARRAYSIZE(mHttpResponse_AsString_100), mR_ResourceInvalid);

    const char *statusText = mHttpResponse_AsString_100[statusCode];
    mERROR_CHECK(mBinaryChunk_WriteBytes(responsePacket, reinterpret_cast<const uint8_t *>(statusText), strlen(statusText)));

    break;
  }

  case 2:
  {
    const size_t statusCode = response->statusCode - 200;
    mERROR_IF(statusCode >= mARRAYSIZE(mHttpResponse_AsString_200), mR_ResourceInvalid);

    const char *statusText = mHttpResponse_AsString_200[statusCode];
    mERROR_CHECK(mBinaryChunk_WriteBytes(responsePacket, reinterpret_cast<const uint8_t *>(statusText), strlen(statusText)));

    break;
  }

  case 3:
  {
    const size_t statusCode = response->statusCode - 300;
    mERROR_IF(statusCode >= mARRAYSIZE(mHttpResponse_AsString_300), mR_ResourceInvalid);

    const char *statusText = mHttpResponse_AsString_300[statusCode];
    mERROR_CHECK(mBinaryChunk_WriteBytes(responsePacket, reinterpret_cast<const uint8_t *>(statusText), strlen(statusText)));

    break;
  }

  case 4:
  {
    const size_t statusCode = response->statusCode - 400;
    mERROR_IF(statusCode >= mARRAYSIZE(mHttpResponse_AsString_400), mR_ResourceInvalid);

    const char *statusText = mHttpResponse_AsString_400[statusCode];
    mERROR_IF(strlen(statusText) == 0, mR_ResourceInvalid);

    mERROR_CHECK(mBinaryChunk_WriteBytes(responsePacket, reinterpret_cast<const uint8_t *>(statusText), strlen(statusText)));

    break;
  }

  case 5:
  {
    const size_t statusCode = response->statusCode - 500;
    mERROR_IF(statusCode >= mARRAYSIZE(mHttpResponse_AsString_500), mR_ResourceInvalid);

    const char *statusText = mHttpResponse_AsString_500[statusCode];
    mERROR_IF(strlen(statusText) == 0, mR_ResourceInvalid);

    mERROR_CHECK(mBinaryChunk_WriteBytes(responsePacket, reinterpret_cast<const uint8_t *>(statusText), strlen(statusText)));

    break;
  }

  default:
  {
    mRETURN_RESULT(mR_ResourceInvalid);
  }
  }

  const char contentType[] = "\r\nContent-Type: ";
  const char charSet[] = ";charset=";
  const char contentLength[] = "\r\nConnection: Keep-Alive\r\nContent-Length: ";
  const char endOfParams[] = "\r\n\r\n";

  mERROR_IF(response->contentType.bytes <= 1, mR_ResourceInvalid);

  mERROR_CHECK(mBinaryChunk_WriteBytes(responsePacket, reinterpret_cast<const uint8_t *>(contentType), sizeof(contentType) - 1));
  mERROR_CHECK(mBinaryChunk_WriteBytes(responsePacket, reinterpret_cast<const uint8_t *>(response->contentType.c_str()), response->contentType.bytes - 1));

  if (response->charSet.bytes > 1)
  {
    mERROR_CHECK(mBinaryChunk_WriteBytes(responsePacket, reinterpret_cast<const uint8_t *>(charSet), sizeof(charSet) - 1));
    mERROR_CHECK(mBinaryChunk_WriteBytes(responsePacket, reinterpret_cast<const uint8_t *>(response->charSet.c_str()), response->charSet.bytes - 1));
  }

  mERROR_CHECK(mBinaryChunk_WriteBytes(responsePacket, reinterpret_cast<const uint8_t *>(contentLength), sizeof(contentLength) - 1));

  size_t streamBytes = 0;
  mERROR_CHECK(mBinaryChunk_GetWriteBytes(response->responseStream, &streamBytes));

  char length[64];
  _ui64toa(streamBytes, length, 10);

  mERROR_CHECK(mBinaryChunk_WriteBytes(responsePacket, reinterpret_cast<const uint8_t *>(length), strlen(length)));
  mERROR_CHECK(mBinaryChunk_WriteBytes(responsePacket, reinterpret_cast<const uint8_t *>(endOfParams), sizeof(endOfParams) - 1));

  // TODO: Additional, Cookies, ...

  if (streamBytes)
    mERROR_CHECK(mBinaryChunk_WriteBytes(responsePacket, response->responseStream->pData, streamBytes));

  mERROR_CHECK(mBinaryChunk_GetWriteBytes(responsePacket, &streamBytes));

  size_t bytesSent = 0;
  mERROR_CHECK(mTcpClient_Send(client, responsePacket->pData, streamBytes, &bytesSent));

  mRETURN_SUCCESS();
}

static int32_t mHttpServer_OnUrl_Internal(http_parser *pParser, const char *at, size_t length)
{
  mHttpRequest_Parser *pRequest = reinterpret_cast<mHttpRequest_Parser *>(pParser->data);

  if (mFAILED(pRequest->result))
    return 0;
  
  if (pRequest->url.bytes == 0)
    if (mFAILED(pRequest->result = mString_Create(&pRequest->url, "", 1, pRequest->pAllocator)))
      return 0;

  uint8_t nextEncodedChar[8];
  size_t encodedCharLenght = 0;

  size_t character = 0;

  while (character < length)
  {
    switch (*at)
    {
    case '%':
    {
      if (encodedCharLenght == 8)
      {
        pRequest->result = mR_ResourceInvalid;
        return 0;
      }

      if (mFAILED(pRequest->result = mHttpRequest_UnHex_Internal(at + 1, nextEncodedChar[encodedCharLenght])))
        return 0;

      at += 3;
      encodedCharLenght++;
      character += 3;

      size_t charSize = 0;

      if (mString_IsValidChar(reinterpret_cast<const char *>(nextEncodedChar), encodedCharLenght, nullptr, &charSize))
      {
        if (charSize != encodedCharLenght)
        {
          pRequest->result = mR_InternalError;
          return 0;
        }

        if (mFAILED(pRequest->result = mString_Append(pRequest->url, reinterpret_cast<const char *>(nextEncodedChar), encodedCharLenght)))
          return 0;

        encodedCharLenght = 0;
      }

      break;
    }

    case '?':
    {
      if (encodedCharLenght != 0)
      {
        pRequest->result = mR_ResourceInvalid;
        return 0;
      }

      // Swap after last char for '\0' in order to allow for string parsing expecting a zero terminator. Value will be restored afterwards.
      const char previousLastChar = *(at - character + length);
      *const_cast<char *>(at - character + length) = '\0';

      pRequest->result = mHttpRequest_ParseArguments_Internal(at + 1, pRequest->headParameters, pRequest->pAllocator, true);

      *const_cast<char *>(at - character + length) = previousLastChar;

      return 0;
    }

    default:
    {
      if (encodedCharLenght != 0)
      {
        pRequest->result = mR_ResourceInvalid;
        return 0;
      }

      if (mFAILED(pRequest->result = mString_Append(pRequest->url, at, 1)))
        return 0;

      at++;
      character++;

      break;
    }
    }
  }

  return 0;
}

static int32_t mHttpServer_OnHeaderField_Internal(http_parser *pParser, const char *at, size_t length)
{
  mHttpRequest_Parser *pRequest = reinterpret_cast<mHttpRequest_Parser *>(pParser->data);

  if (mFAILED(pRequest->result))
    return 0;

  mKeyValuePair<mString, mString> pair;

  if (mFAILED(pRequest->result = mString_Create(&pair.key, at, length)))
    return 0;

  if (mFAILED(pRequest->result = mQueue_PushBack(pRequest->attributes, pair)))
    return 0;

  if (mFAILED(pRequest->result = mQueue_PointerAt(pRequest->attributes, pRequest->attributes->count - 1, &pRequest->pLastAttributeValue)))
    return 0;

  return 0;
}

static int32_t mHttpServer_OnHeaderValue_Internal(http_parser *pParser, const char *at, size_t length)
{
  mHttpRequest_Parser *pRequest = reinterpret_cast<mHttpRequest_Parser *>(pParser->data);

  if (mFAILED(pRequest->result))
    return 0;

  if (pRequest->pLastAttributeValue == nullptr)
  {
    pRequest->result = mR_ResourceStateInvalid;
    return 0;
  }

  if (pRequest->pLastAttributeValue->value.bytes == 0)
    pRequest->result = mString_Create(&pRequest->pLastAttributeValue->value, at, length);
  else
    pRequest->result = mString_Append(pRequest->pLastAttributeValue->value, at, length);

  return 0;
}

static int32_t mHttpServer_OnBody_Internal(http_parser *pParser, const char *at, size_t length)
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

static mFUNCTION(mHttpRequest_Parser_Init_Internal, mPtr<mHttpRequest_Parser> &request, IN mAllocator *pAllocator, mPtr<mTcpClient> &client)
{
  mFUNCTION_SETUP();

  mERROR_IF(request == nullptr, mR_ArgumentNull);

  mERROR_CHECK(mQueue_Create(&request->attributes, pAllocator));
  mERROR_CHECK(mQueue_Create(&request->headParameters, pAllocator));
  mERROR_CHECK(mQueue_Create(&request->postParameters, pAllocator));

  request->pAllocator = pAllocator;
  request->client = client;

  mRETURN_SUCCESS();
}

static void mHttpRequest_Parser_Destroy_Internal(IN_OUT mHttpRequest_Parser *pRequest)
{
  if (pRequest == nullptr)
    return;

  mQueue_Destroy(&pRequest->attributes);
  mQueue_Destroy(&pRequest->headParameters);
  mQueue_Destroy(&pRequest->headParameters);
  mString_Destroy(&pRequest->url);
  mString_Destroy(&pRequest->body);
  mSharedPointer_Destroy(&pRequest->client);
}

static mFUNCTION(mHttpRequest_ParseArguments_Internal, const char *params, OUT mPtr<mQueue<mKeyValuePair<mString, mString>>> &args, IN mAllocator *pAllocator, const bool ignoreFragment)
{
  mFUNCTION_SETUP();

  mKeyValuePair<mString, mString> currentValue;

  enum ParserState
  {
    ExpectKey,
    InKey,
    ExpectValue,
    InValue
  } state = ExpectKey;

  uint8_t nextEncodedChar[8];
  size_t encodedCharLenght = 0;

  mERROR_CHECK(mString_Create(&currentValue.key, "", 1, pAllocator));
  mERROR_CHECK(mString_Create(&currentValue.value, "", 1, pAllocator));

  while (true)
  {
    switch (*params)
    {
    case '\0':
    {
      goto no_more_params;
    }

    case '%':
    {
      if (state == ExpectKey || state == ExpectValue)
        state = (ParserState)(state + 1);

      mERROR_IF(encodedCharLenght == 8, mR_ResourceInvalid);
      mERROR_CHECK(mHttpRequest_UnHex_Internal(params + 1, nextEncodedChar[encodedCharLenght]));

      params += 3;
      encodedCharLenght++;

      size_t charSize = 0;
      
      if (mString_IsValidChar(reinterpret_cast<const char *>(nextEncodedChar), encodedCharLenght, nullptr, &charSize))
      {
        mERROR_IF(charSize != encodedCharLenght, mR_InternalError);
        
        if (state == InKey)
          mERROR_CHECK(mString_Append(currentValue.key, reinterpret_cast<const char *>(nextEncodedChar), encodedCharLenght));
        else if (state == InValue)
          mERROR_CHECK(mString_Append(currentValue.value, reinterpret_cast<const char *>(nextEncodedChar), encodedCharLenght));

        encodedCharLenght = 0;
      }

      break;
    }

    case '&':
    {
      mERROR_IF(encodedCharLenght != 0, mR_ResourceInvalid);

      if (state != ExpectKey)
      {
        mERROR_CHECK(mQueue_PushBack(args, currentValue));
        
        mERROR_CHECK(mString_Create(&currentValue.key, "", 1, pAllocator));
        mERROR_CHECK(mString_Create(&currentValue.value, "", 1, pAllocator));
      }

      state = ExpectKey;

      params++;

      break;
    }

    case '=':
    {
      mERROR_IF(encodedCharLenght != 0, mR_ResourceInvalid);
      mERROR_IF(state != InKey, mR_ResourceInvalid);

      state = ExpectValue;

      params++;

      break;
    }

    case '#':
    {
      if (ignoreFragment)
        goto no_more_params;

      // Fall Through.
    }

    default:
    {
      mERROR_IF(encodedCharLenght != 0, mR_ResourceInvalid);

      if (state == ExpectKey || state == ExpectValue)
        state = (ParserState)(state + 1);

      if (state == InKey)
        mERROR_CHECK(mString_Append(currentValue.key, reinterpret_cast<const char *>(params), 1));
      else if (state == InValue)
        mERROR_CHECK(mString_Append(currentValue.value, reinterpret_cast<const char *>(params), 1));

      params++;

      break;
    }
    }
  }

no_more_params:
  mERROR_IF(encodedCharLenght != 0, mR_ResourceInvalid);

  if (InKey || InValue || ExpectValue)
    mERROR_CHECK(mQueue_PushBack(args, currentValue));

  mRETURN_SUCCESS();
}

//////////////////////////////////////////////////////////////////////////

static mFUNCTION(mHttpResponse_Init_Internal, mPtr<mHttpResponse> &response, IN mAllocator *pAllocator)
{
  mFUNCTION_SETUP();

  mERROR_IF(response == nullptr, mR_ArgumentNull);

  mERROR_CHECK(mBinaryChunk_Create(&response->responseStream, pAllocator));
  mERROR_CHECK(mQueue_Create(&response->setCookies, pAllocator));
  mERROR_CHECK(mQueue_Create(&response->attributes, pAllocator));
  
  mERROR_CHECK(mString_Create(&response->charSet, "UTF-8", pAllocator));
  response->contentType.pAllocator = pAllocator;

  mRETURN_SUCCESS();
}

static void mHttpResponse_Destroy_Internal(IN_OUT mHttpResponse *pResponse)
{
  if (pResponse == nullptr)
    return;

  mString_Destroy(&pResponse->contentType);
  mString_Destroy(&pResponse->charSet);
  mBinaryChunk_Destroy(&pResponse->responseStream);
  mQueue_Destroy(&pResponse->setCookies);
  mQueue_Destroy(&pResponse->attributes);
}

//////////////////////////////////////////////////////////////////////////

struct mDefaultHttpErrorRequestHandler : mHttpErrorRequestHandler
{
  bool respondWithJson;
};

mFUNCTION(mDefaultHttpErrorRequestHandler_HandleRequest, mPtr<mHttpErrorRequestHandler> &handler, const mString &errorString, IN_OUT mPtr<mHttpResponse> &response);

//////////////////////////////////////////////////////////////////////////

mFUNCTION(mHttpErrorRequestHandler_CreateDefaultHandler, OUT mPtr<mHttpErrorRequestHandler> *pRequestHandler, IN mAllocator *pAllocator, const bool respondWithJson /* = false */)
{
  mFUNCTION_SETUP();

  mERROR_IF(pRequestHandler == nullptr, mR_ArgumentNull);

  mDefaultHttpErrorRequestHandler *pHandler = nullptr;

  mERROR_CHECK((mSharedPointer_AllocateInherited<mHttpErrorRequestHandler, mDefaultHttpErrorRequestHandler>(pRequestHandler, pAllocator, nullptr, &pHandler)));

  pHandler->pHandleRequest = mDefaultHttpErrorRequestHandler_HandleRequest;
  pHandler->respondWithJson = respondWithJson;

  mRETURN_SUCCESS();
}

//////////////////////////////////////////////////////////////////////////

mFUNCTION(mDefaultHttpErrorRequestHandler_HandleRequest, mPtr<mHttpErrorRequestHandler> &handler, const mString &errorString, IN_OUT mPtr<mHttpResponse> &response)
{
  mFUNCTION_SETUP();

  const mDefaultHttpErrorRequestHandler *pHandler = static_cast<mDefaultHttpErrorRequestHandler *>(handler.GetPointer());

  if (!pHandler->respondWithJson)
  {
    mERROR_CHECK(mString_Create(&response->contentType, "text/html", response->contentType.pAllocator));

    const char start[] = "<html><head><title>Error</title></head><body><h1>";
    const char afterHeadline[] = "</h1>";
    const char end[] = "</body></html>";

    mERROR_CHECK(mBinaryChunk_WriteBytes(response->responseStream, reinterpret_cast<const uint8_t *>(start), sizeof(start) - 1));

    switch (response->statusCode / 100)
    {
    case 1:
    {
      const size_t statusCode = response->statusCode - 100;
      mERROR_IF(statusCode >= mARRAYSIZE(mHttpResponse_AsString_100), mR_ResourceInvalid);

      const char *statusText = mHttpResponse_AsString_100[statusCode];
      mERROR_IF(statusText[0] == '\0', mR_ResourceInvalid);

      mERROR_CHECK(mBinaryChunk_WriteBytes(response->responseStream, reinterpret_cast<const uint8_t *>(statusText), strlen(statusText)));

      break;
    }

    case 2:
    {
      const size_t statusCode = response->statusCode - 200;
      mERROR_IF(statusCode >= mARRAYSIZE(mHttpResponse_AsString_200), mR_ResourceInvalid);

      const char *statusText = mHttpResponse_AsString_200[statusCode];
      mERROR_IF(statusText[0] == '\0', mR_ResourceInvalid);

      mERROR_CHECK(mBinaryChunk_WriteBytes(response->responseStream, reinterpret_cast<const uint8_t *>(statusText), strlen(statusText)));

      break;
    }

    case 3:
    {
      const size_t statusCode = response->statusCode - 300;
      mERROR_IF(statusCode >= mARRAYSIZE(mHttpResponse_AsString_300), mR_ResourceInvalid);

      const char *statusText = mHttpResponse_AsString_300[statusCode];
      mERROR_IF(statusText[0] == '\0', mR_ResourceInvalid);

      mERROR_CHECK(mBinaryChunk_WriteBytes(response->responseStream, reinterpret_cast<const uint8_t *>(statusText), strlen(statusText)));

      break;
    }

    case 4:
    {
      const size_t statusCode = response->statusCode - 400;
      mERROR_IF(statusCode >= mARRAYSIZE(mHttpResponse_AsString_400), mR_ResourceInvalid);

      const char *statusText = mHttpResponse_AsString_400[statusCode];
      mERROR_IF(statusText[0] == '\0', mR_ResourceInvalid);

      mERROR_CHECK(mBinaryChunk_WriteBytes(response->responseStream, reinterpret_cast<const uint8_t *>(statusText), strlen(statusText)));

      break;
    }

    case 5:
    {
      const size_t statusCode = response->statusCode - 500;
      mERROR_IF(statusCode >= mARRAYSIZE(mHttpResponse_AsString_500), mR_ResourceInvalid);

      const char *statusText = mHttpResponse_AsString_500[statusCode];
      mERROR_IF(statusText[0] == '\0', mR_ResourceInvalid);

      mERROR_CHECK(mBinaryChunk_WriteBytes(response->responseStream, reinterpret_cast<const uint8_t *>(statusText), strlen(statusText)));

      break;
    }

    default:
    {
      mRETURN_RESULT(mR_ResourceInvalid);
    }
    }

    mERROR_CHECK(mBinaryChunk_WriteBytes(response->responseStream, reinterpret_cast<const uint8_t *>(afterHeadline), sizeof(afterHeadline) - 1));
    mERROR_CHECK(mBinaryChunk_WriteBytes(response->responseStream, reinterpret_cast<const uint8_t *>(errorString.c_str()), errorString.bytes - 1));
    mERROR_CHECK(mBinaryChunk_WriteBytes(response->responseStream, reinterpret_cast<const uint8_t *>(end), sizeof(end) - 1));
  }
  else
  {
    mERROR_CHECK(mString_Create(&response->contentType, "application/json", response->contentType.pAllocator));

    const char before[] = "{ \"error\": \"";
    const char after[] = "\" }";

    mERROR_CHECK(mBinaryChunk_WriteBytes(response->responseStream, reinterpret_cast<const uint8_t *>(before), sizeof(before) - 1));

    for (const auto &&_char : errorString)
    {
      if (_char.characterSize == 1)
      {
        if ((*_char.character >= 35 && *_char.character < 127) || *_char.character == ' ' || *_char.character == '\t')
        {
          mERROR_CHECK(mBinaryChunk_WriteData(response->responseStream, *_char.character));
        }
        else
        {
          char bytes[4] = "\\x";

          bytes[2] = ((_char.character[0] & 0xF0) >> 4) + '0';
          bytes[3] = (_char.character[0] & 0x0F) + '0';

          if (bytes[2] > '9')
            bytes[2] = bytes[2] - '0' - 10 + 'A';

          if (bytes[3] > '9')
            bytes[3] = bytes[3] - '0' - 10 + 'A';

          mERROR_CHECK(mBinaryChunk_WriteBytes(response->responseStream, reinterpret_cast<const uint8_t *>(bytes), sizeof(bytes)));
        }
      }
      else
      {
        for (size_t i = 0; i < _char.characterSize; i++)
        {
          char bytes[4] = "\\x";

          bytes[2] = ((_char.character[i] & 0xF0) >> 4) + '0';
          bytes[3] = (_char.character[i] & 0x0F) + '0';

          if (bytes[2] > '9')
            bytes[2] = bytes[2] - '0' - 10 + 'A';

          if (bytes[3] > '9')
            bytes[3] = bytes[3] - '0' - 10 + 'A';

          mERROR_CHECK(mBinaryChunk_WriteBytes(response->responseStream, reinterpret_cast<const uint8_t *>(bytes), sizeof(bytes)));
        }
      }
    }

    mERROR_CHECK(mBinaryChunk_WriteBytes(response->responseStream, reinterpret_cast<const uint8_t *>(after), sizeof(after) - 1));
  }

  mRETURN_SUCCESS();
}

//////////////////////////////////////////////////////////////////////////

#undef RETURN

#pragma warning(push, 0)
#include "http_parser/src/http_parser.c"
#pragma warning(pop)

#ifdef GIT_BUILD // Define __M_FILE__
  #ifdef __M_FILE__
    #undef __M_FILE__
  #endif
  #define __M_FILE__ "H/L+uBO77NtNTeWaUgmywwRSPoj74Gv6U+ejvXjCr/Q2XXeSy0CFuKfXlxm/Y6anWlybOlGLZ9N70rOK"
#endif
