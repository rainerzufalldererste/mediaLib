#pragma warning(push, 0)
#include <WinSock2.h>
#include <WS2tcpip.h>
#include <iphlpapi.h>
#pragma warning(pop)

#include "mNetwork.h"

#pragma comment(lib,"IPHLPAPI.lib")

#ifdef GIT_BUILD // Define __M_FILE__
  #ifdef __M_FILE__
    #undef __M_FILE__
  #endif
  #define __M_FILE__ "TW/D0ir9/B9WaGGm0AwCSRQd00GQJj2WJPwQcPTpr0zjd10M4Ek6WFMPGYdH4za/R+PFcelq9TcBKAGK"
#endif

static bool mNetwork_StaticallyInitiailized = false;

#ifdef mPLATFORM_WINDOWS
static WSADATA mNetwork_WinsockInfo;
#endif

int32_t WSACleanupWrapper()
{
  return WSACleanup();
}

mFUNCTION(mNetwork_Init)
{
  mFUNCTION_SETUP();

  mERROR_IF(mNetwork_StaticallyInitiailized, mR_Success);

  const int32_t result = WSAStartup(MAKEWORD(2, 2), &mNetwork_WinsockInfo);

  if (result != 0)
  {
    mPRINT_ERROR("Failed to initialize WinSock with error code 0x%" PRIx32 ".\n", result);
    mRETURN_RESULT(mR_OperationNotSupported);
  }

  onexit(WSACleanupWrapper);
  mNetwork_StaticallyInitiailized = true;

  mRETURN_SUCCESS();
}

mFUNCTION(mNetwork_GetLocalAddresses, OUT mPtr<mQueue<mIPAddress>> *pAddresses, IN mAllocator *pAllocator)
{
  mFUNCTION_SETUP();

  mERROR_IF(pAddresses == nullptr, mR_ArgumentNull);

  if (*pAddresses == nullptr)
    mERROR_CHECK(mQueue_Create(pAddresses, pAllocator));
  else
    mERROR_CHECK(mQueue_Clear(*pAddresses));

  mERROR_CHECK(mNetwork_Init());

  IP_ADAPTER_ADDRESSES *pAdapterAddresses = nullptr;
  mAllocator *pTempAllocator = &mDefaultTempAllocator;
  mDEFER_CALL_2(mAllocator_FreePtr, pTempAllocator, &pAdapterAddresses);

  ULONG bytes = 0;

  while (true)
  {  
    const ULONG result = GetAdaptersAddresses(AF_UNSPEC, GAA_FLAG_SKIP_ANYCAST | GAA_FLAG_SKIP_MULTICAST | GAA_FLAG_SKIP_DNS_SERVER | GAA_FLAG_SKIP_FRIENDLY_NAME, nullptr, pAdapterAddresses, &bytes);
    
    if (result != ERROR_SUCCESS)
      mERROR_IF(result != ERROR_BUFFER_OVERFLOW, mR_InternalError);
    else
      break;

    mERROR_CHECK(mAllocator_Reallocate(pTempAllocator, reinterpret_cast<uint8_t **>(&pAdapterAddresses), bytes));
  }

  for (IP_ADAPTER_ADDRESSES *pCurrentAdapter = pAdapterAddresses; pCurrentAdapter != nullptr; pCurrentAdapter = pCurrentAdapter->Next)
  {
    for (IP_ADAPTER_UNICAST_ADDRESS *pUnicastAddress = pCurrentAdapter->FirstUnicastAddress; pUnicastAddress != nullptr; pUnicastAddress = pUnicastAddress->Next)
    {
      if (pUnicastAddress->DadState == NldsInvalid || pUnicastAddress->DadState == NldsDeprecated || pUnicastAddress->DadState == NldsTentative || pUnicastAddress->DadState == NldsDuplicate)
        continue;

      mIPAddress ip;

      switch (pUnicastAddress->Address.lpSockaddr->sa_family)
      {
      case AF_INET:
      {
        ip.isIPv6 = false;

        const sockaddr_in *pIPv4Addr = reinterpret_cast<const sockaddr_in *>(pUnicastAddress->Address.lpSockaddr);
        memcpy(ip.ipv4, &pIPv4Addr->sin_addr.S_un, sizeof(ip.ipv4));

        break;
      }

      case AF_INET6:
      {
        ip.isIPv6 = true;

        const sockaddr_in6 *pIPv6Addr = reinterpret_cast<const sockaddr_in6 *>(pUnicastAddress->Address.lpSockaddr);
        memcpy(ip.ipv6, &pIPv6Addr->sin6_addr.u, sizeof(ip.ipv6));

        break;
      }

      default:
      {
        continue;
      }
      }

      mERROR_CHECK(mQueue_PushBack(*pAddresses, ip));
    }
  }

  mRETURN_SUCCESS();
}

mFUNCTION(mNetwork_GetAdapters, OUT mPtr<mQueue<mNetworkAdapter>> *pAdapters, IN mAllocator *pAllocator)
{
  mFUNCTION_SETUP();

  mERROR_IF(pAdapters == nullptr, mR_ArgumentNull);

  if (*pAdapters == nullptr)
    mERROR_CHECK(mQueue_Create(pAdapters, pAllocator));
  else
    mERROR_CHECK(mQueue_Clear(*pAdapters));

  mERROR_CHECK(mNetwork_Init());

  IP_ADAPTER_ADDRESSES *pAdapterAddresses = nullptr;
  mAllocator *pTempAllocator = &mDefaultTempAllocator;
  mDEFER_CALL_2(mAllocator_FreePtr, pTempAllocator, &pAdapterAddresses);

  ULONG bytes = 0;

  while (true)
  {
    const ULONG result = GetAdaptersAddresses(AF_UNSPEC, GAA_FLAG_SKIP_ANYCAST | GAA_FLAG_SKIP_MULTICAST | GAA_FLAG_SKIP_DNS_SERVER | GAA_FLAG_INCLUDE_GATEWAYS, nullptr, pAdapterAddresses, &bytes);

    if (result != ERROR_SUCCESS)
      mERROR_IF(result != ERROR_BUFFER_OVERFLOW, mR_InternalError);
    else
      break;

    mERROR_CHECK(mAllocator_Reallocate(pTempAllocator, reinterpret_cast<uint8_t **>(&pAdapterAddresses), bytes));
  }

  for (IP_ADAPTER_ADDRESSES *pCurrentAdapter = pAdapterAddresses; pCurrentAdapter != nullptr; pCurrentAdapter = pCurrentAdapter->Next)
  {
    if (pCurrentAdapter->OperStatus != IfOperStatusUp && pCurrentAdapter->OperStatus != IfOperStatusUnknown)
      continue;

    mNetworkAdapter adapter;

    adapter.sendBitsPerSecond = pCurrentAdapter->TransmitLinkSpeed;
    adapter.receiveBitsPerSecond = pCurrentAdapter->ReceiveLinkSpeed;

    mERROR_CHECK(mQueue_Create(&adapter.addresses, pAllocator));
    mERROR_CHECK(mQueue_Create(&adapter.gatewayAddresses, pAllocator));

    mERROR_CHECK(mString_Create(&adapter.name, pCurrentAdapter->FriendlyName, pAllocator));

    for (IP_ADAPTER_UNICAST_ADDRESS *pUnicastAddress = pCurrentAdapter->FirstUnicastAddress; pUnicastAddress != nullptr; pUnicastAddress = pUnicastAddress->Next)
    {
      if (pUnicastAddress->DadState == NldsInvalid || pUnicastAddress->DadState == NldsDeprecated || pUnicastAddress->DadState == NldsTentative || pUnicastAddress->DadState == NldsDuplicate)
        continue;

      mIPAddress ip;

      switch (pUnicastAddress->Address.lpSockaddr->sa_family)
      {
      case AF_INET:
      {
        ip.isIPv6 = false;

        const sockaddr_in *pIPv4Addr = reinterpret_cast<const sockaddr_in *>(pUnicastAddress->Address.lpSockaddr);
        memcpy(ip.ipv4, &pIPv4Addr->sin_addr.S_un, sizeof(ip.ipv4));

        break;
      }

      case AF_INET6:
      {
        ip.isIPv6 = true;

        const sockaddr_in6 *pIPv6Addr = reinterpret_cast<const sockaddr_in6 *>(pUnicastAddress->Address.lpSockaddr);
        memcpy(ip.ipv6, &pIPv6Addr->sin6_addr.u, sizeof(ip.ipv6));

        break;
      }

      default:
      {
        continue;
      }
      }

      mERROR_CHECK(mQueue_PushBack(adapter.addresses, ip));
    }

    for (IP_ADAPTER_GATEWAY_ADDRESS *pGatewayAddress = pCurrentAdapter->FirstGatewayAddress; pGatewayAddress != nullptr; pGatewayAddress = pGatewayAddress->Next)
    {
      mIPAddress ip;

      switch (pGatewayAddress->Address.lpSockaddr->sa_family)
      {
      case AF_INET:
      {
        ip.isIPv6 = false;

        const sockaddr_in *pIPv4Addr = reinterpret_cast<const sockaddr_in *>(pGatewayAddress->Address.lpSockaddr);
        memcpy(ip.ipv4, &pIPv4Addr->sin_addr.S_un, sizeof(ip.ipv4));

        break;
      }

      case AF_INET6:
      {
        ip.isIPv6 = true;

        const sockaddr_in6 *pIPv6Addr = reinterpret_cast<const sockaddr_in6 *>(pGatewayAddress->Address.lpSockaddr);
        memcpy(ip.ipv6, &pIPv6Addr->sin6_addr.u, sizeof(ip.ipv6));

        break;
      }

      default:
      {
        continue;
      }
      }

      mERROR_CHECK(mQueue_PushBack(adapter.gatewayAddresses, ip));
    }

    mERROR_CHECK(mQueue_PushBack(*pAdapters, std::move(adapter)));
  }

  mRETURN_SUCCESS();
}
