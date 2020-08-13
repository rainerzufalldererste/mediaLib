#ifndef mNetwork_h__
#define mNetwork_h__

#include "mediaLib.h"
#include "mQueue.h"

#ifdef GIT_BUILD // Define __M_FILE__
  #ifdef __M_FILE__
    #undef __M_FILE__
  #endif
  #define __M_FILE__ "oDw8VlI5UrEL3BYL+N7+6XALnhEyfDhb0OUMVr90oHd1+HKGSFkxRBHFbqJb5IG/DbliSyzD22Ii/XTp"
#endif

struct mIPAddress_v4
{
#pragma warning(push)
#pragma warning(disable: 4201)
  union
  {
    struct
    {
      uint8_t _0, _1, _2, _3;
    };

    uint8_t bytes[4];
  };
#pragma warning(pop)

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
#pragma warning(push)
#pragma warning(disable: 4201)
  union
  {
    struct
    {
      uint16_t _0, _1, _2, _3, _4, _5, _6, _7;
    };

    uint8_t bytes[16];
  };
#pragma warning(pop)

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
    return mSprintf(string, maxLength, "%02" PRIx8 "%02" PRIx8 ":%02" PRIx8 "%02" PRIx8 ":%02" PRIx8 "%02" PRIx8 ":%02" PRIx8 "%02" PRIx8 ":%02" PRIx8 "%02" PRIx8 ":%02" PRIx8 "%02" PRIx8 ":%02" PRIx8 "%02" PRIx8 ":%02" PRIx8 "%02" PRIx8, bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6], bytes[7], bytes[8], bytes[9], bytes[10], bytes[11], bytes[12], bytes[13], bytes[14], bytes[15]);
  }
};

struct mIPAddress
{
  bool isIPv6;

#pragma warning(push)
#pragma warning(disable: 4201)
  union
  {
    uint8_t ipv4[4];
    uint8_t ipv6[16];
  };
#pragma warning(pop)

  inline mFUNCTION(ToString, OUT char *string, const size_t maxLength) const
  {
    if (isIPv6)
      return mSprintf(string, maxLength, "%02" PRIx8 "%02" PRIx8 ":%02" PRIx8 "%02" PRIx8 ":%02" PRIx8 "%02" PRIx8 ":%02" PRIx8 "%02" PRIx8 ":%02" PRIx8 "%02" PRIx8 ":%02" PRIx8 "%02" PRIx8 ":%02" PRIx8 "%02" PRIx8 ":%02" PRIx8 "%02" PRIx8, ipv6[0], ipv6[1], ipv6[2], ipv6[3], ipv6[4], ipv6[5], ipv6[6], ipv6[7], ipv6[8], ipv6[9], ipv6[10], ipv6[11], ipv6[12], ipv6[13], ipv6[14], ipv6[15]);
    else
      return mSprintf(string, maxLength, "%" PRIu8 ".%" PRIu8 ".%" PRIu8 ".%" PRIu8, ipv4[0], ipv4[1], ipv4[2], ipv4[3]);
  }
};

struct mNetworkAdapter
{
  mString name;

  mPtr<mQueue<mIPAddress>> addresses;
  mPtr<mQueue<mIPAddress>> gatewayAddresses;

  size_t sendBitsPerSecond, receiveBitsPerSecond;
};

mFUNCTION(mNetwork_Init);
mFUNCTION(mNetwork_GetLocalAddresses, OUT mPtr<mQueue<mIPAddress>> *pAddresses, IN mAllocator *pAllocator);
mFUNCTION(mNetwork_GetAdapters, OUT mPtr<mQueue<mNetworkAdapter>> *pAdapters, IN mAllocator *pAllocator);

#endif // mNetwork_h__
