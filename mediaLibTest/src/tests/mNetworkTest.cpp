#include "mTestLib.h"
#include "mNetwork.h"

mTEST(mNetwork, TestIPv4)
{
  mTEST_ALLOCATOR_SETUP();

  mIPAddress_v4 ipv4(192, 168, 0, 1);
  mIPAddress_v4 ipv4_1(255, 255, 255, 255);
  mIPAddress_v4 ipv4_2(0, 1, 2, 3);

  char buffer[3 * 4 + 3 + 1];
  mTEST_ASSERT_SUCCESS(mIPAddress_v4_ToString(ipv4, buffer, mARRAYSIZE(buffer)));
  mTEST_ASSERT_TRUE(0 == strncmp(buffer, "192.168.0.1", mARRAYSIZE(buffer)));
  mTEST_ASSERT_SUCCESS(mIPAddress_v4_ToString(ipv4_1, buffer, mARRAYSIZE(buffer)));
  mTEST_ASSERT_TRUE(0 == strncmp(buffer, "255.255.255.255", mARRAYSIZE(buffer)));
  mTEST_ASSERT_SUCCESS(mIPAddress_v4_ToString(ipv4_2, buffer, mARRAYSIZE(buffer)));
  mTEST_ASSERT_TRUE(0 == strncmp(buffer, "0.1.2.3", mARRAYSIZE(buffer)));

  mTEST_ASSERT_TRUE(ipv4 == ipv4);
  mTEST_ASSERT_FALSE(ipv4 != ipv4);
  mTEST_ASSERT_FALSE(ipv4 == ipv4_1);
  mTEST_ASSERT_TRUE(ipv4 != ipv4_1);

  mTEST_ALLOCATOR_ZERO_CHECK();
}

mTEST(mNetwork, TestIPv6)
{
  mTEST_ALLOCATOR_SETUP();

  mIPAddress_v6 ip(0x20, 0x01, 0x0d, 0xb8, 0x85, 0xa3, 0x00, 0x00, 0x00, 0x00, 0x8a, 0x2e, 0x03, 0x70, 0x73, 0x34);
  mIPAddress_v6 ip_1(0x20, 0x03, 0x00, 0xcc, 0x1f, 0x04, 0x53, 0x75, 0xb4, 0x37, 0x1d, 0x92, 0x8a, 0xa6, 0xbc, 0xe5);
  mIPAddress_v6 ip_2(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1);

  char buffer[4 * 8 + 7 + 1];
  mTEST_ASSERT_SUCCESS(mIPAddress_v6_ToString(ip, buffer, mARRAYSIZE(buffer)));
  mTEST_ASSERT_TRUE(0 == strncmp(buffer, "2001:0db8:85a3:0000:0000:8a2e:0370:7334", mARRAYSIZE(buffer)));
  mTEST_ASSERT_SUCCESS(mIPAddress_v6_ToString(ip_1, buffer, mARRAYSIZE(buffer)));
  mTEST_ASSERT_TRUE(0 == strncmp(buffer, "2003:00cc:1f04:5375:b437:1d92:8aa6:bce5", mARRAYSIZE(buffer)));
  mTEST_ASSERT_SUCCESS(mIPAddress_v6_ToString(ip_2, buffer, mARRAYSIZE(buffer)));
  mTEST_ASSERT_TRUE(0 == strncmp(buffer, "0000:0000:0000:0000:0000:0000:0000:0001", mARRAYSIZE(buffer)));

  mTEST_ASSERT_TRUE(ip == ip);
  mTEST_ASSERT_FALSE(ip != ip);
  mTEST_ASSERT_FALSE(ip == ip_1);
  mTEST_ASSERT_TRUE(ip != ip_1);

  mTEST_ALLOCATOR_ZERO_CHECK();
}

mTEST(mNetwork, TestIPAddress)
{
  mTEST_ALLOCATOR_SETUP();

  mPtr<mQueue<mIPAddress>> addresses;
  mTEST_ASSERT_SUCCESS(mNetwork_GetLocalAddresses(&addresses, pAllocator));

  mIPAddress loopbackV4;
  loopbackV4.isIPv6 = false;
  loopbackV4.ipv4[0] = 127;
  loopbackV4.ipv4[1] = 0;
  loopbackV4.ipv4[2] = 0;
  loopbackV4.ipv4[3] = 1;

  mIPAddress loopbackV6;
  loopbackV6.isIPv6 = true;
  loopbackV6.ipv6[0] = 0;
  loopbackV6.ipv6[1] = 0;
  loopbackV6.ipv6[2] = 0;
  loopbackV6.ipv6[3] = 0;
  loopbackV6.ipv6[4] = 0;
  loopbackV6.ipv6[5] = 0;
  loopbackV6.ipv6[6] = 0;
  loopbackV6.ipv6[7] = 0;
  loopbackV6.ipv6[8] = 0;
  loopbackV6.ipv6[9] = 0;
  loopbackV6.ipv6[10] = 0;
  loopbackV6.ipv6[11] = 0;
  loopbackV6.ipv6[12] = 0;
  loopbackV6.ipv6[13] = 0;
  loopbackV6.ipv6[14] = 0;
  loopbackV6.ipv6[15] = 1;

  mIPAddress garbageV6;
  garbageV6.isIPv6 = true;
  garbageV6.ipv6[0] = 255;
  garbageV6.ipv6[1] = 255;
  garbageV6.ipv6[2] = 255;
  garbageV6.ipv6[3] = 255;
  garbageV6.ipv6[4] = 255;
  garbageV6.ipv6[5] = 255;
  garbageV6.ipv6[6] = 255;
  garbageV6.ipv6[7] = 255;
  garbageV6.ipv6[8] = 255;
  garbageV6.ipv6[9] = 255;
  garbageV6.ipv6[10] = 255;
  garbageV6.ipv6[11] = 255;
  garbageV6.ipv6[12] = 255;
  garbageV6.ipv6[13] = 255;
  garbageV6.ipv6[14] = 255;
  garbageV6.ipv6[15] = 255;

  bool containsV4 = false;
  bool containsV6 = false;

  for (const auto &_ip : addresses->Iterate())
  {
    if (_ip == loopbackV4)
    {
      mTEST_ASSERT_FALSE(containsV4);
      containsV4 = true;

      char buffer[3 * 4 + 3 + 1];
      mTEST_ASSERT_SUCCESS(mIPAddress_ToString(_ip, buffer, mARRAYSIZE(buffer)));
      mTEST_ASSERT_TRUE(0 == strncmp(buffer, "127.0.0.1", mARRAYSIZE(buffer)));
    }
    
    if (_ip == loopbackV6)
    {
      mTEST_ASSERT_FALSE(containsV6);
      containsV6 = true;

      char buffer[4 * 8 + 7 + 1];
      mTEST_ASSERT_SUCCESS(mIPAddress_ToString(_ip, buffer, mARRAYSIZE(buffer)));
      mTEST_ASSERT_TRUE(0 == strncmp(buffer, "0000:0000:0000:0000:0000:0000:0000:0001", mARRAYSIZE(buffer)));
    }

    mTEST_ASSERT_FALSE(_ip == garbageV6);
  }

  mTEST_ASSERT_TRUE(containsV4);
  mTEST_ASSERT_TRUE(containsV6);

  mTEST_ALLOCATOR_ZERO_CHECK();
}
