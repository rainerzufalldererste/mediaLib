// Copyright 2018 Christoph Stiller
// 
// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files(the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions :
// 
// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
// 
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "mTestLib.h"
#include "mHash.h"

mTEST(mHash, Test_mString)
{
  mString testString0 = "testString0";
  mString testString1 = "testString1";

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

mTEST(mHash, TestTooShort)
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
