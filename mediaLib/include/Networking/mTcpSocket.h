#ifndef mTCPSocket_h__
#define mTCPSocket_h__

#include "mediaLib.h"
#include "mNetwork.h"

#ifdef GIT_BUILD // Define __M_FILE__
  #ifdef __M_FILE__
    #undef __M_FILE__
  #endif
  #define __M_FILE__ "c+hgJ/qsL0rAbbtA4d41yPRr1RJobiLDjDAY6G7QaMKtHpKZvGHBxedWT5dCuFXNX4y/WKbvjlx3nspX"
#endif

struct mTcpConnectionInfo : mIPAddress
{
  uint16_t port;

  inline mFUNCTION(ToString, OUT char *string, const size_t maxLength) const
  {
    if (isIPv6)
      return mSprintf(string, maxLength, "[%02" PRIx8 "%02" PRIx8 ":%02" PRIx8 "%02" PRIx8 ":%02" PRIx8 "%02" PRIx8 ":%02" PRIx8 "%02" PRIx8 ":%02" PRIx8 "%02" PRIx8 ":%02" PRIx8 "%02" PRIx8 ":%02" PRIx8 "%02" PRIx8 ":%02" PRIx8 "%02" PRIx8 "]:" PRIu16, ipv6[0], ipv6[1], ipv6[2], ipv6[3], ipv6[4], ipv6[5], ipv6[6], ipv6[7], ipv6[8], ipv6[9], ipv6[10], ipv6[11], ipv6[12], ipv6[13], ipv6[14], ipv6[15], port);
    else
      return mSprintf(string, maxLength, "%" PRIu8 ".%" PRIu8 ".%" PRIu8 ".%" PRIu8 ":" PRIu16, ipv4[0], ipv4[1], ipv4[2], ipv4[3], port);
  }
};

inline mFUNCTION(mTcpConnectionInfo_ToString, const mTcpConnectionInfo &connectionInfo, OUT char *string, const size_t maxLength)
{
  return connectionInfo.ToString(string, maxLength);
}

struct mTcpServer;
struct mTcpClient;

mFUNCTION(mTcpServer_Create, OUT mPtr<mTcpServer> *pTcpServer, IN mAllocator *pAllocator, const uint16_t port);
mFUNCTION(mTcpServer_Listen, mPtr<mTcpServer> &tcpServer, OUT mPtr<mTcpClient> *pClient, IN mAllocator *pAllocator);

mFUNCTION(mTcpClient_Create, OUT mPtr<mTcpClient> *pClient, IN mAllocator *pAllocator, const mIPAddress_v4 &ipv4, const uint16_t port);
mFUNCTION(mTcpClient_Create, OUT mPtr<mTcpClient> *pClient, IN mAllocator *pAllocator, const mIPAddress_v6 &ipv6, const uint16_t port);

mFUNCTION(mTcpClient_Send, mPtr<mTcpClient> &tcpClient, IN const void *pData, const size_t length, OUT OPTIONAL size_t *pBytesSent = nullptr);
mFUNCTION(mTcpClient_Receive, mPtr<mTcpClient> &tcpClient, OUT void *pData, const size_t maxLength, OUT OPTIONAL size_t *pBytesReceived = nullptr);
mFUNCTION(mTcpClient_GetReadableBytes, mPtr<mTcpClient> &tcpClient, OUT size_t *pReadableBytes, OPTIONAL size_t timeoutMs = 0);
mFUNCTION(mTcpClient_GetWriteableBytes, mPtr<mTcpClient> &tcpClient, OUT size_t *pWriteableBytes, OPTIONAL size_t timeoutMs = 0);

mFUNCTION(mTcpClient_GetConnectionInfo, mPtr<mTcpClient> &tcpClient, OUT mTcpConnectionInfo *pConnectionInfo);

#endif // mTCPSocket_h__
