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
  mKeyValuePair<mString, mString> *pLastAttributeValue;
  mResult result;
  mString body;
};

mFUNCTION(mHttpServer_Destroy_Internal, IN_OUT mHttpServer *pHttpServer);
mFUNCTION(mHttpServer_ThreadInternal, mPtr<mHttpServer> &server);

int32_t _OnUrl(http_parser *, const char *at, size_t length);
int32_t _OnHeaderField(http_parser *, const char *at, size_t length);
int32_t _OnHeaderValue(http_parser *, const char *at, size_t length);
int32_t _OnBody(http_parser *, const char *at, size_t length);

mFUNCTION(mHttpRequest_Parser_Init_Internal, mPtr<mHttpRequest_Parser> &request, IN mAllocator *pAllocator);
void mHttpRequest_Parser_Destroy_Internal(IN_OUT mHttpRequest_Parser *pRequest);

mFUNCTION(mHttpRequest_ParseArguments, const char *params, OUT mPtr<mQueue<mKeyValuePair<mString, mString>>> &args, IN mAllocator *pAllocator);

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

mFUNCTION(mHttpServer_Create, OUT mPtr<mHttpServer> *pHttpServer, IN mAllocator *pAllocator, const uint16_t port /* = 80 */, const size_t threadCount /* = 1 */)
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
  mERROR_CHECK(mTasklessThreadPool_Create(&(*pHttpServer)->threadPool, pAllocator, threadCount));

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
        // TODO: Use Arena Allocator.

        char data[8 * 1024];
        size_t bytesReceived = 0;

        if (mSUCCEEDED(mSILENCE_ERROR(mTcpClient_Receive(client, data, sizeof(data), &bytesReceived))))
        {
          data[mMin(bytesReceived + 1, sizeof(data) - 1)] = '\0';

          http_parser parser;
          http_parser_init(&parser, HTTP_REQUEST);

          mUniqueContainer<mHttpRequest_Parser> request;
          mUniqueContainer<mHttpRequest_Parser>::CreateWithCleanupFunction(&request, mHttpRequest_Parser_Destroy_Internal);

          if (mFAILED(mHttpRequest_Parser_Init_Internal(request, server->pAllocator)))
            RETURN;

          parser.data = request.GetPointer();
          
          http_parser_execute(&parser, &server->settings, data, bytesReceived);

          if (mFAILED(request->result))
            RETURN; // TODO: Reply with error.

          request->pLastAttributeValue = nullptr;
          
          if (parser.type != HTTP_REQUEST)
            RETURN; // TODO: Reply with error.

          request->requestMethod = (mHttpRequestMethod)parser.method;

          if (request->requestMethod == mHRM_Post && request->body.bytes > 1)
            if (mFAILED(mHttpRequest_ParseArguments(request->body.c_str(), request->postParameters, server->pAllocator)))
              RETURN; // TODO: Reply with error.
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

      pRequest->result = mHttpRequest_ParseArguments(at + 1, pRequest->headParameters, pRequest->pAllocator);

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

int32_t _OnHeaderField(http_parser *pParser, const char *at, size_t length)
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

  if (pRequest->pLastAttributeValue->value.bytes == 0)
    pRequest->result = mString_Create(&pRequest->pLastAttributeValue->value, at, length);
  else
    pRequest->result = mString_Append(pRequest->pLastAttributeValue->value, at, length);

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

mFUNCTION(mHttpRequest_Parser_Init_Internal, mPtr<mHttpRequest_Parser> &request, IN mAllocator *pAllocator)
{
  mFUNCTION_SETUP();

  mERROR_IF(request == nullptr, mR_ArgumentNull);

  mERROR_CHECK(mQueue_Create(&request->attributes, pAllocator));
  mERROR_CHECK(mQueue_Create(&request->headParameters, pAllocator));
  mERROR_CHECK(mQueue_Create(&request->postParameters, pAllocator));

  request->pAllocator = pAllocator;

  mRETURN_SUCCESS();
}

void mHttpRequest_Parser_Destroy_Internal(IN_OUT mHttpRequest_Parser *pRequest)
{
  if (pRequest == nullptr)
    return;

  mQueue_Destroy(&pRequest->attributes);
  mQueue_Destroy(&pRequest->headParameters);
  mQueue_Destroy(&pRequest->headParameters);
  mString_Destroy(&pRequest->url);
  mString_Destroy(&pRequest->body);
}

mFUNCTION(mHttpRequest_ParseArguments, const char *params, OUT mPtr<mQueue<mKeyValuePair<mString, mString>>> &args, IN mAllocator *pAllocator)
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

#undef RETURN

#pragma warning(push, 0)
#include "http_parser/src/http_parser.c"
#pragma warning(pop)
