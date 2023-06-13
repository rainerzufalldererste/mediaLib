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
    return mFormatTo(string, maxLength, _0, '.', _1, '.', _2, '.', _3);
  }

  inline bool operator == (const mIPAddress_v4 &other) const
  {
    return memcmp(bytes, other.bytes, sizeof(bytes)) == 0;
  }

  inline bool operator != (const mIPAddress_v4 &other) const
  {
    return !(*this == other);
  }
};

inline mFUNCTION(mIPAddress_v4_ToString, const mIPAddress_v4 &address, OUT char *string, const size_t maxLength)
{
  return address.ToString(string, maxLength);
}

struct mIPAddress_v6
{
#pragma warning(push)
#pragma warning(disable: 4201)
  union
  {
    struct
    {
      uint8_t _0, _1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11, _12, _13, _14, _15;
    };

    uint8_t bytes[16];
  };
#pragma warning(pop)

  inline mIPAddress_v6(const uint8_t _0, const uint8_t _1, const uint8_t _2, const uint8_t _3, const uint8_t _4, const uint8_t _5, const uint8_t _6, const uint8_t _7, const uint8_t _8, const uint8_t _9, const uint8_t _10, const uint8_t _11, const uint8_t _12, const uint8_t _13, const uint8_t _14, const uint8_t _15) :
    _0(_0),
    _1(_1),
    _2(_2),
    _3(_3),
    _4(_4),
    _5(_5),
    _6(_6),
    _7(_7),
    _8(_8),
    _9(_9),
    _10(_10),
    _11(_11),
    _12(_12),
    _13(_13),
    _14(_14),
    _15(_15)
  { }

  inline mFUNCTION(ToString, OUT char *string, const size_t maxLength) const
  {
    mFormatState &fs = mFormat_GetState();
    mFormatState copy(fs);
    mDEFER(fs.SetTo(copy));

    fs.minChars = 2;
    fs.integerBaseOption = mFBO_Hexadecimal;
    fs.hexadecimalUpperCase = false;
    fs.fillCharacterIsZero = true;
    fs.fillCharacter = '0';

    return mFormatTo(string, maxLength, _0, _1, ':', _2, _3, ':', _4, _5, ':', _6, _7, ':', _8, _9, ':', _10, _11, ':', _12, _13, ':', _14, _15);
  }

  inline bool operator == (const mIPAddress_v6 &other) const
  {
    return memcmp(bytes, other.bytes, sizeof(bytes)) == 0;
  }

  inline bool operator != (const mIPAddress_v6 &other) const
  {
    return !(*this == other);
  }
};

inline mFUNCTION(mIPAddress_v6_ToString, const mIPAddress_v6 &address, OUT char *string, const size_t maxLength)
{
  return address.ToString(string, maxLength);
}

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
    {
      mFormatState &fs = mFormat_GetState();
      mFormatState copy(fs);
      mDEFER(fs.SetTo(copy));

      fs.minChars = 2;
      fs.integerBaseOption = mFBO_Hexadecimal;
      fs.hexadecimalUpperCase = false;
      fs.fillCharacterIsZero = true;
      fs.fillCharacter = '0';

      return mFormatTo(string, maxLength, ipv6[0], ipv6[1], ':', ipv6[2], ipv6[3], ':', ipv6[4], ipv6[5], ':', ipv6[6], ipv6[7], ':', ipv6[8], ipv6[9], ':', ipv6[10], ipv6[11], ':', ipv6[12], ipv6[13], ':', ipv6[14], ipv6[15]);
    }
    else
    {
      return mFormatTo(string, maxLength, ipv4[0], '.', ipv4[1], '.', ipv4[2], '.', ipv4[3]);
    }
  }

  inline bool operator == (const mIPAddress &other) const
  {
    if (other.isIPv6 != isIPv6)
      return false;

    return memcmp(other.ipv6, ipv6, isIPv6 ? sizeof(ipv6) : sizeof(ipv4)) == 0;
  }

  inline bool operator != (const mIPAddress &other) const
  {
    return !(*this == other);
  }

  inline mIPAddress() { }

  inline mIPAddress(const mIPAddress_v4 &addr)
  {
    isIPv6 = false;
    mMemcpy(ipv4, addr.bytes, mARRAYSIZE(addr.bytes));
  }

  inline mIPAddress(const mIPAddress_v6 &addr)
  {
    isIPv6 = true;
    mMemcpy(ipv6, addr.bytes, mARRAYSIZE(addr.bytes));
  }
};

inline mFUNCTION(mIPAddress_ToString, const mIPAddress &address, OUT char *string, const size_t maxLength)
{
  return address.ToString(string, maxLength);
}

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
