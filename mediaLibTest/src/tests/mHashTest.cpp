#include "mTestLib.h"
#include "mHash.h"

mTEST(mHash, Test_mString)
{
  mString testString0 = "testString0";
  mString testString1 = "testString1";

  uint64_t hash0;
  mTEST_ASSERT_SUCCESS(mHash(testString0, &hash0));

  uint64_t hash1;
  mString testString0_ = "testString0";
  mTEST_ASSERT_SUCCESS(mHash(testString0_, &hash1));
  mTEST_ASSERT_EQUAL(hash0, hash1);

  uint64_t hash2;
  mTEST_ASSERT_SUCCESS(mHash(testString1, &hash2));
  mTEST_ASSERT_NOT_EQUAL(hash0, hash2);

  uint64_t hash3;
  mString testString1_ = "testString1";
  mTEST_ASSERT_SUCCESS(mHash(testString1_, &hash3));
  mTEST_ASSERT_EQUAL(hash2, hash3);
}

mTEST(mHash, Test_mInplaceString)
{
  mInplaceString<128> testString0;
  mInplaceString<128> testString1;

  mTEST_ASSERT_SUCCESS(mInplaceString_CreateRaw(&testString0, "testString0"));
  mTEST_ASSERT_SUCCESS(mInplaceString_CreateRaw(&testString1, "testString1"));

  uint64_t hash0;
  mTEST_ASSERT_SUCCESS(mHash(testString0, &hash0));

  uint64_t hash1;
  mTEST_ASSERT_SUCCESS(mHash(testString0, &hash1));
  mTEST_ASSERT_EQUAL(hash0, hash1);

  uint64_t hash2;
  mTEST_ASSERT_SUCCESS(mHash(testString1, &hash2));
  mTEST_ASSERT_NOT_EQUAL(hash0, hash2);

  uint64_t hash3;
  mTEST_ASSERT_SUCCESS(mHash(testString1, &hash3));
  mTEST_ASSERT_EQUAL(hash2, hash3);
}

mTEST(mHash, TestVeryShort)
{
  uint8_t v0 = 0xF1;
  uint8_t v1 = 0x1F;

  uint64_t hash0;
  mTEST_ASSERT_SUCCESS(mHash(&v0, &hash0));

  uint64_t hash1;
  mTEST_ASSERT_SUCCESS(mHash(&v0, &hash1));
  mTEST_ASSERT_EQUAL(hash0, hash1);

  uint64_t hash2;
  mTEST_ASSERT_SUCCESS(mHash(&v1, &hash2));
  mTEST_ASSERT_NOT_EQUAL(hash0, hash2);

  uint64_t hash3;
  mTEST_ASSERT_SUCCESS(mHash(&v1, &hash3));
  mTEST_ASSERT_EQUAL(hash2, hash3);
}
