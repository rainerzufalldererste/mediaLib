#pragma warning(push, 0)
#include <WinSock2.h>
#include <WS2tcpip.h>
#pragma warning(pop)

#include "mTcpSocket.h"

#pragma comment(lib, "Ws2_32.lib")

//////////////////////////////////////////////////////////////////////////

struct mTcpSocket
{

};

bool _StaticallyInitiailized = false;

#ifdef mPLATFORM_WINDOWS
WSADATA _WinsockInfo;
#endif

struct mTcpServer
{
  SOCKET socket;
};

struct mTcpClient
{
  SOCKET socket;
};

mFUNCTION(_Initialize);
mFUNCTION(mTcpServer_Destroy_Internal, IN_OUT mTcpServer *pTcpServer);
mFUNCTION(mTcpClient_Destroy_Internal, IN_OUT mTcpClient *pTcpClient);
mFUNCTION(mTcpClient_Create_Internal, OUT mPtr<mTcpClient> *pClient, IN mAllocator *pAllocator, const char *address, const uint16_t port);

//////////////////////////////////////////////////////////////////////////

mFUNCTION(mTcpServer_Create, OUT mPtr<mTcpServer> *pTcpServer, IN mAllocator *pAllocator, const uint16_t port)
{
  mFUNCTION_SETUP();

  mERROR_IF(pTcpServer == nullptr, mR_ArgumentNull);
  mERROR_IF(port == 0, mR_InvalidParameter);

  *pTcpServer = nullptr;

  mERROR_CHECK(_Initialize());

  struct addrinfo *pResult = nullptr;
  struct addrinfo hints = { 0 };

  hints.ai_family = AF_INET;
  hints.ai_socktype = SOCK_STREAM;
  hints.ai_protocol = IPPROTO_TCP;
  hints.ai_flags = AI_PASSIVE;

  char portString[sizeof("65535")];
  mERROR_CHECK(mSprintf(portString, mARRAYSIZE(portString), "%" PRIu16, port));

  int32_t error = getaddrinfo(nullptr, portString, &hints, &pResult);
  mERROR_IF(error != 0, mR_ResourceAlreadyExists);
  mDEFER(freeaddrinfo(pResult));

  mDEFER_CALL_ON_ERROR(pTcpServer, mSharedPointer_Destroy);
  mERROR_CHECK(mSharedPointer_Allocate<mTcpServer>(pTcpServer, pAllocator, [](mTcpServer *pData) { mTcpServer_Destroy_Internal(pData); }, 1));

  (*pTcpServer)->socket = INVALID_SOCKET;

  (*pTcpServer)->socket = socket(pResult->ai_family, pResult->ai_socktype, pResult->ai_protocol);
  
  if ((*pTcpServer)->socket == INVALID_SOCKET)
  {
    error = WSAGetLastError();

    mRETURN_RESULT(mR_InternalError);
  }

  error = bind((*pTcpServer)->socket, pResult->ai_addr, (int)pResult->ai_addrlen);

  if (error == SOCKET_ERROR)
  {
    error = WSAGetLastError();

    mRETURN_RESULT(mR_InternalError);
  }

  mRETURN_SUCCESS();
}

mFUNCTION(mTcpServer_Listen, mPtr<mTcpServer> &tcpServer, OUT mPtr<mTcpClient> *pClient, IN mAllocator *pAllocator)
{
  mFUNCTION_SETUP();

  mERROR_IF(tcpServer == nullptr || pClient == nullptr, mR_ArgumentNull);
  *pClient = nullptr;

  int32_t error = listen(tcpServer->socket, SOMAXCONN);
  
  if (error == SOCKET_ERROR)
  {
    error = WSAGetLastError();

    mRETURN_RESULT(mR_ResourceNotFound);
  }

  mDEFER_CALL_ON_ERROR(pClient, mSharedPointer_Destroy);
  mERROR_CHECK(mSharedPointer_Allocate<mTcpClient>(pClient, pAllocator, [](mTcpClient *pData) { mTcpClient_Destroy_Internal(pData); }, 1));
  
  (*pClient)->socket = INVALID_SOCKET;

  (*pClient)->socket = accept(tcpServer->socket, (sockaddr *)nullptr, (int32_t *)nullptr);

  if ((*pClient)->socket == INVALID_SOCKET)
  {
    error = WSAGetLastError();
   
    mRETURN_RESULT(mR_InternalError);
  }

  mRETURN_SUCCESS();
}

//////////////////////////////////////////////////////////////////////////

mFUNCTION(mTcpClient_Create, OUT mPtr<mTcpClient> *pClient, IN mAllocator *pAllocator, const mIPAddress_v4 &ipv4, const uint16_t port)
{
  mFUNCTION_SETUP();

  char ipAddrString[sizeof("255.255.255.255")];
  mERROR_CHECK(ipv4.ToString(ipAddrString, mARRAYSIZE(ipAddrString)));

  mERROR_CHECK(mTcpClient_Create_Internal(pClient, pAllocator, ipAddrString, port));

  mRETURN_SUCCESS();
}

mFUNCTION(mTcpClient_Create, OUT mPtr<mTcpClient> *pClient, IN mAllocator *pAllocator, const mIPAddress_v6 &ipv6, const uint16_t port)
{
  mFUNCTION_SETUP();

  char ipAddrString[sizeof("ffff:ffff:ffff:ffff:ffff:ffff:ffff:ffff")];
  mERROR_CHECK(ipv6.ToString(ipAddrString, mARRAYSIZE(ipAddrString)));

  mERROR_CHECK(mTcpClient_Create_Internal(pClient, pAllocator, ipAddrString, port));

  mRETURN_SUCCESS();
}

mFUNCTION(mTcpClient_Send, mPtr<mTcpClient> &tcpClient, IN const void *pData, const size_t length, OUT OPTIONAL size_t *pBytesSent)
{
  mFUNCTION_SETUP();

  mERROR_IF(tcpClient == nullptr || pData == nullptr, mR_ArgumentNull);
  mERROR_IF(length > INT32_MAX, mR_ArgumentOutOfBounds);

  const int32_t bytesSent = send(tcpClient->socket, reinterpret_cast<const char *>(pData), (int32_t)length, 0);

  if (bytesSent == SOCKET_ERROR || bytesSent < 0)
  {
    const int32_t error = WSAGetLastError();
    mUnused(error);

    if (pBytesSent != nullptr)
      *pBytesSent = (size_t)bytesSent;

    mRETURN_RESULT(mR_IOFailure);
  }

  if (pBytesSent != nullptr)
    *pBytesSent = (size_t)bytesSent;

  mRETURN_SUCCESS();
}

mFUNCTION(mTcpClient_Receive, mPtr<mTcpClient> &tcpClient, OUT void *pData, const size_t maxLength, OUT OPTIONAL size_t *pBytesReceived)
{
  mFUNCTION_SETUP();

  mERROR_IF(tcpClient == nullptr || pData == nullptr, mR_ArgumentNull);
  mERROR_IF(maxLength > INT32_MAX, mR_ArgumentOutOfBounds);

  const int32_t bytesReceived = recv(tcpClient->socket, reinterpret_cast<char *>(pData), (int32_t)maxLength, 0);

  if (bytesReceived < 0 || bytesReceived == SOCKET_ERROR)
  {
    const int32_t error = WSAGetLastError();
    mUnused(error);

    if (pBytesReceived != nullptr)
      *pBytesReceived = (size_t)bytesReceived;

    mRETURN_RESULT(mR_IOFailure);
  }
  else if (bytesReceived == 0)
  {
    if (pBytesReceived != nullptr)
      *pBytesReceived = (size_t)bytesReceived;

    mRETURN_RESULT(mR_EndOfStream);
  }
  else
  {
    if (pBytesReceived != nullptr)
      *pBytesReceived = (size_t)bytesReceived;
  }

  mRETURN_SUCCESS();
}

//////////////////////////////////////////////////////////////////////////

int32_t WSACleanupWrapper()
{
  return WSACleanup();
}

mFUNCTION(_Initialize)
{
  mFUNCTION_SETUP();

  if (_StaticallyInitiailized)
    mRETURN_SUCCESS();

  const int32_t result = WSAStartup(MAKEWORD(2, 2), &_WinsockInfo);

  if (result != 0)
  {
    mPRINT_ERROR("Failed to initialize WinSock with error code 0x%" PRIx32 ".\n", result);
    mRETURN_RESULT(mR_OperationNotSupported);
  }

  onexit(WSACleanupWrapper);
  _StaticallyInitiailized = true;

  mRETURN_SUCCESS();
}

mFUNCTION(mTcpServer_Destroy_Internal, IN_OUT mTcpServer *pTcpServer)
{
  mFUNCTION_SETUP();

  mERROR_IF(pTcpServer == nullptr, mR_ArgumentNull);

  if (pTcpServer->socket != INVALID_SOCKET)
  {
    closesocket(pTcpServer->socket);
    pTcpServer->socket = INVALID_SOCKET;
  }

  mRETURN_SUCCESS();
}

mFUNCTION(mTcpClient_Destroy_Internal, IN_OUT mTcpClient *pTcpClient)
{
  mFUNCTION_SETUP();

  mERROR_IF(pTcpClient == nullptr, mR_ArgumentNull);

  if (pTcpClient->socket != INVALID_SOCKET)
  {
    shutdown(pTcpClient->socket, SD_SEND);
    closesocket(pTcpClient->socket);
    pTcpClient->socket = INVALID_SOCKET;
  }

  mRETURN_SUCCESS();
}

mFUNCTION(mTcpClient_Create_Internal, OUT mPtr<mTcpClient> *pClient, IN mAllocator *pAllocator, const char *address, const uint16_t port)
{
  mFUNCTION_SETUP();

  mERROR_IF(pClient == nullptr, mR_ArgumentNull);
  mERROR_IF(port == 0, mR_InvalidParameter);

  struct addrinfo *pResult = nullptr;
  struct addrinfo hints = { 0 };

  hints.ai_family = AF_UNSPEC;
  hints.ai_socktype = SOCK_STREAM;
  hints.ai_protocol = IPPROTO_TCP;

  char portString[sizeof("65535")];
  mERROR_CHECK(mSprintf(portString, mARRAYSIZE(portString), "%" PRIu16, port));

  int32_t error = getaddrinfo(address, portString, &hints, &pResult);
  mERROR_IF(error != 0, mR_ResourceInvalid);
  mDEFER(freeaddrinfo(pResult));

  mDEFER_CALL_ON_ERROR(pClient, mSharedPointer_Destroy);
  mERROR_CHECK(mSharedPointer_Allocate<mTcpClient>(pClient, pAllocator, [](mTcpClient *pData) { mTcpClient_Destroy_Internal(pData); }, 1));

  (*pClient)->socket = INVALID_SOCKET;

  (*pClient)->socket = socket(pResult->ai_family, pResult->ai_socktype, pResult->ai_protocol);

  if ((*pClient)->socket == INVALID_SOCKET)
  {
    error = WSAGetLastError();

    mRETURN_RESULT(mR_InternalError);
  }

  error = connect((*pClient)->socket, pResult->ai_addr, (int32_t)pResult->ai_addrlen);

  if (error == SOCKET_ERROR)
  {
    error = WSAGetLastError();

    mRETURN_RESULT(mR_ResourceNotFound);
  }

  mRETURN_SUCCESS();
}
