#pragma warning(push, 0)
#include <WinSock2.h>
#include <WS2tcpip.h>
#pragma warning(pop)

#include "mTcpSocket.h"
#include "mProfiler.h"

#pragma comment(lib, "Ws2_32.lib")

#ifdef GIT_BUILD // Define __M_FILE__
  #ifdef __M_FILE__
    #undef __M_FILE__
  #endif
  #define __M_FILE__ "uCzF+6pzV/ptvx8Jh5ZOHkDqfkK63Xd0fTa9kupppSjjlRkXhl1Qh9Wqe04J8ejYmyzKaEqpYv1uS2hp"
#endif

//////////////////////////////////////////////////////////////////////////

struct mTcpServer
{
  SOCKET socket;
};

struct mTcpClient
{
  SOCKET socket;
};

static mFUNCTION(mTcpServer_Destroy_Internal, IN_OUT mTcpServer *pTcpServer);
static mFUNCTION(mTcpClient_Destroy_Internal, IN_OUT mTcpClient *pTcpClient);
static mFUNCTION(mTcpClient_Create_Internal, OUT mPtr<mTcpClient> *pClient, IN mAllocator *pAllocator, const char *address, const uint16_t port);
static mFUNCTION(mTcpSocket_GetConnectionInfoFromSocket, IN SOCKADDR_STORAGE_LH *pInfo, OUT mTcpEndPoint *pConnectionInfo);

//////////////////////////////////////////////////////////////////////////

mFUNCTION(mTcpServer_Create, OUT mPtr<mTcpServer> *pTcpServer, IN mAllocator *pAllocator, const uint16_t port, OPTIONAL IN const mIPAddress *pAddress /* = nullptr */)
{
  mFUNCTION_SETUP();

  mERROR_IF(pTcpServer == nullptr, mR_ArgumentNull);

  mPROFILE_SCOPED("mTcpServer_Create");

  *pTcpServer = nullptr;

  mERROR_CHECK(mNetwork_Init());

  struct addrinfo *pResult = nullptr;
  struct addrinfo hints = { 0 };

  hints.ai_family = AF_INET;
  hints.ai_socktype = SOCK_STREAM;
  hints.ai_protocol = IPPROTO_TCP;
  hints.ai_flags = AI_PASSIVE;

  char portString[sizeof("65535")];
  mERROR_CHECK(mFormatTo(portString, mARRAYSIZE(portString), port));

  char addressString[sizeof("0000:0000:0000:0000:0000:0000:0000:0000")];
  
  if (pAddress != nullptr)
    mERROR_CHECK(mIPAddress_ToString(*pAddress, addressString, sizeof(addressString)));

  int32_t error = getaddrinfo(pAddress == nullptr ? nullptr : addressString, portString, &hints, &pResult);
  mERROR_IF(error != 0, mR_ResourceAlreadyExists);
  mDEFER_CALL(pResult, freeaddrinfo);

  mDEFER_CALL_ON_ERROR(pTcpServer, mSharedPointer_Destroy);
  mERROR_CHECK(mSharedPointer_Allocate<mTcpServer>(pTcpServer, pAllocator, [](mTcpServer *pData) { mTcpServer_Destroy_Internal(pData); }, 1));

  (*pTcpServer)->socket = socket(pResult->ai_family, pResult->ai_socktype, pResult->ai_protocol);
  
  if ((*pTcpServer)->socket == INVALID_SOCKET)
  {
    error = WSAGetLastError();

    mRETURN_RESULT(mR_InternalError);
  }

  error = bind((*pTcpServer)->socket, pResult->ai_addr, (int32_t)pResult->ai_addrlen);

  if (error == SOCKET_ERROR)
  {
    error = WSAGetLastError();

    switch (error)
    {
    case WSAEADDRINUSE:
      mRETURN_RESULT(mR_ResourceAlreadyExists);

    default:
      mRETURN_RESULT(mR_InternalError);
    }
  }

  mRETURN_SUCCESS();
}

mFUNCTION(mTcpServer_Listen, mPtr<mTcpServer> &tcpServer, OUT mPtr<mTcpClient> *pClient, IN mAllocator *pAllocator, OPTIONAL size_t timeoutMs /* = (size_t)-1 */)
{
  mFUNCTION_SETUP();

  mERROR_IF(tcpServer == nullptr || pClient == nullptr, mR_ArgumentNull);
  mERROR_IF(timeoutMs != (size_t)-1 && timeoutMs > INT_MAX, mR_ArgumentOutOfBounds);
  *pClient = nullptr;

  mPROFILE_SCOPED("mTcpServer_Listen");

  int32_t error = listen(tcpServer->socket, SOMAXCONN);
  
  if (error == SOCKET_ERROR)
  {
    error = WSAGetLastError();

    mRETURN_RESULT(mR_ResourceNotFound);
  }

  if (timeoutMs != (size_t)-1)
  {
    mPROFILE_SCOPED("mTcpServer_Listen Poll");

    WSAPOLLFD connection;
    mZeroMemory(&connection);

    connection.fd = tcpServer->socket;
    connection.events = POLLRDNORM;
    connection.revents = POLLRDNORM;

    error = WSAPoll(&connection, 1, (INT)timeoutMs);

    mERROR_IF(error == 0, mR_Success);
    mERROR_IF(error < 0, mR_InternalError);
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

mFUNCTION(mTcpServer_GetLocalEndPointInfo, const mPtr<mTcpServer> &tcpServer, OUT mTcpEndPoint *pConnectionInfo)
{
  mFUNCTION_SETUP();

  mERROR_IF(tcpServer == nullptr || pConnectionInfo == nullptr, mR_ArgumentNull);

  SOCKADDR_STORAGE_LH address;
  int32_t nameLength = sizeof(address);

  const int32_t result = getsockname(tcpServer->socket, reinterpret_cast<SOCKADDR *>(&address), &nameLength);
  mERROR_IF(result != 0, mR_InternalError);

  mERROR_CHECK(mTcpSocket_GetConnectionInfoFromSocket(&address, pConnectionInfo));

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

mFUNCTION(mTcpClient_Send, mPtr<mTcpClient> &tcpClient, IN const void *pData, const size_t length, OUT OPTIONAL size_t *pBytesSent /* = nullptr */)
{
  mFUNCTION_SETUP();

  mERROR_IF(tcpClient == nullptr || pData == nullptr, mR_ArgumentNull);
  mERROR_IF(length > INT32_MAX, mR_ArgumentOutOfBounds);

  mPROFILE_SCOPED("mTcpClient_Send");

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

mFUNCTION(mTcpClient_Receive, mPtr<mTcpClient> &tcpClient, OUT void *pData, const size_t maxLength, OUT OPTIONAL size_t *pBytesReceived /* = nullptr */)
{
  mFUNCTION_SETUP();

  mERROR_IF(tcpClient == nullptr || pData == nullptr, mR_ArgumentNull);

  mPROFILE_SCOPED("mTcpClient_Receive");

  const int32_t bytesReceived = recv(tcpClient->socket, reinterpret_cast<char *>(pData), (int32_t)mMin((size_t)INT32_MAX, maxLength), 0);

  if (bytesReceived < 0 || bytesReceived == SOCKET_ERROR)
  {
    const int32_t error = WSAGetLastError();
    mUnused(error);

    if (pBytesReceived != nullptr)
      *pBytesReceived = 0;

    mRETURN_RESULT(mR_IOFailure);
  }
  else if (bytesReceived == 0)
  {
    if (pBytesReceived != nullptr)
      *pBytesReceived = 0;

    mRETURN_RESULT(mR_EndOfStream);
  }
  else
  {
    if (pBytesReceived != nullptr)
      *pBytesReceived = (size_t)bytesReceived;
  }

  mRETURN_SUCCESS();
}

mFUNCTION(mTcpClient_GetReadableBytes, mPtr<mTcpClient> &tcpClient, OUT size_t *pReadableBytes, OPTIONAL size_t timeoutMs /* = 0 */)
{
  mFUNCTION_SETUP();

  mERROR_IF(tcpClient == nullptr || pReadableBytes == nullptr, mR_ArgumentNull);

  mPROFILE_SCOPED("mTcpClient_GetReadableBytes");

  WSAPOLLFD pollInfo;
  pollInfo.fd = tcpClient->socket;
  pollInfo.events = POLLRDNORM;

  const int32_t result = WSAPoll(&pollInfo, 1, timeoutMs >= INT_MAX ? -1 : (INT)timeoutMs);

  if (result < 0 || pollInfo.revents < 0)
  {
    const int32_t error = WSAGetLastError();
    mUnused(error);

    *pReadableBytes = 0;

    mRETURN_RESULT(mR_IOFailure);
  }

  *pReadableBytes = (size_t)pollInfo.revents;

  mRETURN_SUCCESS();
}

mFUNCTION(mTcpClient_GetWriteableBytes, mPtr<mTcpClient> &tcpClient, OUT size_t *pWriteableBytes, OPTIONAL size_t timeoutMs /* = 0 */)
{
  mFUNCTION_SETUP();

  mERROR_IF(tcpClient == nullptr || pWriteableBytes == nullptr, mR_ArgumentNull);

  mPROFILE_SCOPED("mTcpClient_GetWriteableBytes");

  WSAPOLLFD pollInfo;
  pollInfo.fd = tcpClient->socket;
  pollInfo.events = POLLWRNORM;

  const int32_t result = WSAPoll(&pollInfo, 1, timeoutMs >= INT_MAX ? -1 : (INT)timeoutMs);

  if (result < 0 || pollInfo.revents < 0)
  {
    const int32_t error = WSAGetLastError();
    mUnused(error);

    *pWriteableBytes = 0;

    mRETURN_RESULT(mR_IOFailure);
  }

  *pWriteableBytes = (size_t)pollInfo.revents;

  mRETURN_SUCCESS();
}

mFUNCTION(mTcpClient_GetRemoteEndPointInfo, const mPtr<mTcpClient> &tcpClient, OUT mTcpEndPoint *pConnectionInfo)
{
  mFUNCTION_SETUP();

  mERROR_IF(tcpClient == nullptr || pConnectionInfo == nullptr, mR_ArgumentNull);

  SOCKADDR_STORAGE_LH address;
  int32_t nameLength = sizeof(address);

  const int32_t result = getpeername(tcpClient->socket, reinterpret_cast<SOCKADDR *>(&address), &nameLength);
  mERROR_IF(result != 0, mR_InternalError);

  mERROR_CHECK(mTcpSocket_GetConnectionInfoFromSocket(&address, pConnectionInfo));

  mRETURN_SUCCESS();
}

mFUNCTION(mTcpClient_GetLocalEndPointInfo, const mPtr<mTcpClient> &tcpClient, OUT mTcpEndPoint *pConnectionInfo)
{
  mFUNCTION_SETUP();

  mERROR_IF(tcpClient == nullptr || pConnectionInfo == nullptr, mR_ArgumentNull);

  SOCKADDR_STORAGE_LH address;
  int32_t nameLength = sizeof(address);

  const int32_t result = getsockname(tcpClient->socket, reinterpret_cast<SOCKADDR *>(&address), &nameLength);
  mERROR_IF(result != 0, mR_InternalError);

  mERROR_CHECK(mTcpSocket_GetConnectionInfoFromSocket(&address, pConnectionInfo));

  mRETURN_SUCCESS();
}

mFUNCTION(mTcpClient_DisableSendDelay, mPtr<mTcpClient> &tcpClient)
{
  mFUNCTION_SETUP();

  mERROR_IF(tcpClient == nullptr, mR_ArgumentNull);

  const BOOL option = TRUE;
  const int32_t result = setsockopt(tcpClient->socket, SOL_SOCKET, SO_DONTLINGER, reinterpret_cast<const char *>(&option), (int32_t)sizeof(option));

  mERROR_IF(result != 0, mR_InternalError);

  mRETURN_SUCCESS();
}

mFUNCTION(mTcpClient_SetSendTimeout, mPtr<mTcpClient> &tcpClient, const size_t milliseconds)
{
  mFUNCTION_SETUP();

  mERROR_IF(tcpClient == nullptr, mR_ArgumentNull);
  mERROR_IF(milliseconds > MAXDWORD, mR_ArgumentOutOfBounds);

  const DWORD option = (DWORD)milliseconds;
  const int32_t result = setsockopt(tcpClient->socket, SOL_SOCKET, SO_SNDTIMEO, reinterpret_cast<const char *>(&option), (int32_t)sizeof(option));

  mERROR_IF(result != 0, mR_InternalError);

  mRETURN_SUCCESS();
}

mFUNCTION(mTcpClient_SetReceiveTimeout, mPtr<mTcpClient> &tcpClient, const size_t milliseconds)
{
  mFUNCTION_SETUP();

  mERROR_IF(tcpClient == nullptr, mR_ArgumentNull);
  mERROR_IF(milliseconds > MAXDWORD, mR_ArgumentOutOfBounds);

  const DWORD option = (DWORD)milliseconds;
  const int32_t result = setsockopt(tcpClient->socket, SOL_SOCKET, SO_RCVTIMEO, reinterpret_cast<const char *>(&option), (int32_t)sizeof(option));

  mERROR_IF(result != 0, mR_InternalError);

  mRETURN_SUCCESS();
}

//////////////////////////////////////////////////////////////////////////

static mFUNCTION(mTcpServer_Destroy_Internal, IN_OUT mTcpServer *pTcpServer)
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

static mFUNCTION(mTcpClient_Destroy_Internal, IN_OUT mTcpClient *pTcpClient)
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

static mFUNCTION(mTcpClient_Create_Internal, OUT mPtr<mTcpClient> *pClient, IN mAllocator *pAllocator, const char *address, const uint16_t port)
{
  mFUNCTION_SETUP();

  mERROR_IF(pClient == nullptr, mR_ArgumentNull);
  mERROR_IF(port == 0, mR_InvalidParameter);

  mPROFILE_SCOPED("mTcpClient_Create_Internal");

  struct addrinfo *pResult = nullptr;
  struct addrinfo hints = { 0 };

  hints.ai_family = AF_UNSPEC;
  hints.ai_socktype = SOCK_STREAM;
  hints.ai_protocol = IPPROTO_TCP;

  char portString[sizeof("65535")];
  mERROR_CHECK(mFormatTo(portString, mARRAYSIZE(portString), port));

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

  // Connect.
  {
    mPROFILE_SCOPED("mTcpClient_Create_Internal Connect");

    error = connect((*pClient)->socket, pResult->ai_addr, (int32_t)pResult->ai_addrlen);

    if (error == SOCKET_ERROR)
    {
      error = WSAGetLastError();

      mRETURN_RESULT(mR_ResourceNotFound);
    }
  }

  mRETURN_SUCCESS();
}

static mFUNCTION(mTcpSocket_GetConnectionInfoFromSocket, IN SOCKADDR_STORAGE_LH *pInfo, OUT mTcpEndPoint *pConnectionInfo)
{
  mFUNCTION_SETUP();

  mERROR_IF(pInfo == nullptr || pConnectionInfo == nullptr, mR_ArgumentNull);

  switch (pInfo->ss_family)
  {
  case AF_INET:
  {
    const SOCKADDR_IN *pIPv4 = reinterpret_cast<const SOCKADDR_IN *>(pInfo);

    pConnectionInfo->isIPv6 = false;
    pConnectionInfo->port = htons(pIPv4->sin_port);
    memcpy(pConnectionInfo->ipv4, &pIPv4->sin_addr, sizeof(pConnectionInfo->ipv4));

    break;
  }

  case AF_INET6:
  {
    const SOCKADDR_IN6 *pIPv6 = reinterpret_cast<const SOCKADDR_IN6 *>(pInfo);

    pConnectionInfo->isIPv6 = true;
    pConnectionInfo->port = htons(pIPv6->sin6_port);
    memcpy(pConnectionInfo->ipv6, &pIPv6->sin6_addr, sizeof(pConnectionInfo->ipv6));

    break;
  }

  default:
  {
    mRETURN_RESULT(mR_NotSupported);
  }
  }

  mRETURN_SUCCESS();
}
