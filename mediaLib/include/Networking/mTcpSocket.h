#ifndef mTCPSocket_h__
#define mTCPSocket_h__

#include "mediaLib.h"

#ifdef GIT_BUILD // Define __M_FILE__
  #ifdef __M_FILE__
    #undef __M_FILE__
  #endif
  #define __M_FILE__ "c+hgJ/qsL0rAbbtA4d41yPRr1RJobiLDjDAY6G7QaMKtHpKZvGHBxedWT5dCuFXNX4y/WKbvjlx3nspX"
#endif

struct mTcpServer;
struct mTcpClient;

mFUNCTION(mTcpServer_Create, OUT mPtr<mTcpServer> *pTcpServer, IN mAllocator *pAllocator, const uint16_t port);
mFUNCTION(mTcpServer_Listen, mPtr<mTcpServer> &tcpServer, OUT mPtr<mTcpClient> *pClient, IN mAllocator *pAllocator);

struct mIPAddress_v4
{
  uint8_t _0, _1, _2, _3;

  inline mIPAddress_v4(const uint8_t _0, const uint8_t _1, const uint8_t _2, const uint8_t _3) :
    _0(_0),
    _1(_1),
    _2(_2),
    _3(_3)
  { }

  inline mFUNCTION(ToString, OUT char *string, const size_t maxLength) const
  {
    return mSprintf(string, maxLength, "%" PRIu8 ".%" PRIu8 ".%" PRIu8 ".%" PRIu8, _0, _1, _2, _3);
  }
};

struct mIPAddress_v6
{
  uint16_t _0, _1, _2, _3, _4, _5, _6, _7;

  inline mIPAddress_v6(const uint16_t _0, const uint16_t _1, const uint16_t _2, const uint16_t _3, const uint16_t _4, const uint16_t _5, const uint16_t _6, const uint16_t _7) :
    _0(_0),
    _1(_1),
    _2(_2),
    _3(_3),
    _4(_4),
    _5(_5),
    _6(_6),
    _7(_7)
  { }

  inline mFUNCTION(ToString, OUT char *string, const size_t maxLength) const
  {
    return mSprintf(string, maxLength, "%" PRIx16 ":%" PRIx16 ":%" PRIx16 ":%" PRIx16 ":%" PRIx16 ":%" PRIx16 ":%" PRIx16 ":%" PRIx16, _0, _1, _2, _3, _4, _5, _6, _7);
  }
};

mFUNCTION(mTcpClient_Create, OUT mPtr<mTcpClient> *pClient, IN mAllocator *pAllocator, const mIPAddress_v4 &ipv4, const uint16_t port);
mFUNCTION(mTcpClient_Create, OUT mPtr<mTcpClient> *pClient, IN mAllocator *pAllocator, const mIPAddress_v6 &ipv6, const uint16_t port);

mFUNCTION(mTcpClient_Send, mPtr<mTcpClient> &tcpClient, IN const void *pData, const size_t length, OUT OPTIONAL size_t *pBytesSent = nullptr);
mFUNCTION(mTcpClient_Receive, mPtr<mTcpClient> &tcpClient, OUT void *pData, const size_t maxLength, OUT OPTIONAL size_t *pBytesReceived = nullptr);
mFUNCTION(mTcpCLient_GetReadableBytes, mPtr<mTcpClient> &tcpClient, OUT size_t *pReadableBytes, OPTIONAL size_t timeoutMs = 0);
mFUNCTION(mTcpCLient_GetWriteableBytes, mPtr<mTcpClient> &tcpClient, OUT size_t *pWriteableBytes, OPTIONAL size_t timeoutMs = 0);

#endif // mTCPSocket_h__
