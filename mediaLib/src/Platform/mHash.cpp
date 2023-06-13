#include "mHash.h"

#ifdef GIT_BUILD // Define __M_FILE__
  #ifdef __M_FILE__
    #undef __M_FILE__
  #endif
  #define __M_FILE__ "KnlobfSd7qBcpERcl7UJFSK/nldU8u7hvNu5EanriIv8hZKXWdW0M85EY2vJNZ49iFY4pUkDwLtGWnnp"
#endif

uint64_t MurmurHash64A(const void *pKey, const size_t length, const uint64_t seed);

//////////////////////////////////////////////////////////////////////////

uint64_t mMurmurHash2(const void *pData, const size_t length, const uint64_t seed)
{
  return MurmurHash64A(pData, (int32_t)length, seed);
}

//-----------------------------------------------------------------------------
// MurmurHash2, 64-bit versions, by Austin Appleby

// The same caveats as 32-bit MurmurHash2 apply here - beware of alignment 
// and endian-ness issues if used across multiple platforms.

// 64-bit hash for 64-bit platforms

uint64_t MurmurHash64A(const void *pKey, const size_t length, const uint64_t seed)
{
  const uint64_t m = 0xC6A4A7935BD1E995;
  const int32_t r = 47;

  uint64_t h = seed ^ (length * m);

  const uint64_t *pData = reinterpret_cast<const uint64_t *>(pKey);
  const uint64_t *pEnd = pData + (length / sizeof(uint64_t));

  while (pData != pEnd)
  {
    uint64_t k = *pData++;

    k *= m;
    k ^= k >> r;
    k *= m;

    h ^= k;
    h *= m;
  }

  const uint8_t *pData2 = reinterpret_cast<const uint8_t *>(pData);

  switch (length & 7)
  {
  case 7: h ^= uint64_t(pData2[6]) << 48;
  case 6: h ^= uint64_t(pData2[5]) << 40;
  case 5: h ^= uint64_t(pData2[4]) << 32;
  case 4: h ^= uint64_t(pData2[3]) << 24;
  case 3: h ^= uint64_t(pData2[2]) << 16;
  case 2: h ^= uint64_t(pData2[1]) << 8;
  case 1: h ^= uint64_t(pData2[0]);
    h *= m;
  };

  h ^= h >> r;
  h *= m;
  h ^= h >> r;

  return h;
}
