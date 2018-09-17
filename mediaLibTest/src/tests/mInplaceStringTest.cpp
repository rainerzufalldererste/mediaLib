// Copyright 2018 Christoph Stiller
// 
// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files(the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions :
// 
// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
// 
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "mTestLib.h"
#include "mString.h"

mTEST(mInplaceString, TestCreateNone)
{
  mInplaceString<128> string;
  mTEST_ASSERT_SUCCESS(mInplaceString_Create(&string));

  size_t count;
  mTEST_ASSERT_SUCCESS(mInplaceString_GetCount(string, &count));
  mTEST_ASSERT_EQUAL(count, 1);

  size_t size;
  mTEST_ASSERT_SUCCESS(mInplaceString_GetByteSize(string, &size));
  mTEST_ASSERT_EQUAL(size, 1);
}

mTEST(mInplaceString, TestCreateEmpty)
{
  mInplaceString<128> string;
  mTEST_ASSERT_SUCCESS(mInplaceString_CreateRaw(&string, ""));

  size_t count;
  mTEST_ASSERT_SUCCESS(mInplaceString_GetCount(string, &count));
  mTEST_ASSERT_EQUAL(count, 1);

  size_t size;
  mTEST_ASSERT_SUCCESS(mInplaceString_GetByteSize(string, &size));
  mTEST_ASSERT_EQUAL(size, 1);
}

mTEST(mInplaceString, TestCreateUTF8)
{
  mInplaceString<128> string;
  mTEST_ASSERT_SUCCESS(mInplaceString_Create(&string, "🌵🦎🎅test"));

  size_t count;
  mTEST_ASSERT_SUCCESS(mInplaceString_GetCount(string, &count));
  mTEST_ASSERT_EQUAL(count, 3 + 4 + 1);

  size_t size;
  mTEST_ASSERT_SUCCESS(mInplaceString_GetByteSize(string, &size));
  mTEST_ASSERT_EQUAL(size, 12 + 4 + 1);
}

mTEST(mInplaceString, TestCreateFromArray)
{
  mInplaceString<128> string;
  char text[32] = "🌵🦎🎅test";
  mTEST_ASSERT_SUCCESS(mInplaceString_Create(&string, (char [32])text));

  size_t count;
  mTEST_ASSERT_SUCCESS(mInplaceString_GetCount(string, &count));
  mTEST_ASSERT_EQUAL(count, 3 + 4 + 1);

  size_t size;
  mTEST_ASSERT_SUCCESS(mInplaceString_GetByteSize(string, &size));
  mTEST_ASSERT_EQUAL(size, 12 + 4 + 1);
}

mTEST(mInplaceString, TestCreateFromCharPtr)
{
  mInplaceString<128> string;
  char text[] = "🌵🦎🎅testמⴲ";
  mTEST_ASSERT_SUCCESS(mInplaceString_CreateRaw(&string, (char *)text));

  size_t count;
  mTEST_ASSERT_SUCCESS(mInplaceString_GetCount(string, &count));
  mTEST_ASSERT_EQUAL(count, 3 + 4 + 2 + 1);

  size_t size;
  mTEST_ASSERT_SUCCESS(mInplaceString_GetByteSize(string, &size));
  mTEST_ASSERT_EQUAL(size, 12 + 4 + 2 + 3 + 1);
}

mTEST(mInplaceString, TestCreateFromCharPtrSize)
{
  mInplaceString<128> string;
  char text[] = "🌵🦎🎅test";
  mTEST_ASSERT_SUCCESS(mInplaceString_Create(&string, (char *)text, mARRAYSIZE(text)));

  size_t count;
  mTEST_ASSERT_SUCCESS(mInplaceString_GetCount(string, &count));
  mTEST_ASSERT_EQUAL(count, 3 + 4 + 1);

  size_t size;
  mTEST_ASSERT_SUCCESS(mInplaceString_GetByteSize(string, &size));
  mTEST_ASSERT_EQUAL(size, 12 + 4 + 1);
}

mTEST(mInplaceString, TestCastTo_mString)
{
  mTEST_ALLOCATOR_SETUP();

  mInplaceString<128> string;
  char text[] = "🌵🦎🎅test";
  mTEST_ASSERT_SUCCESS(mInplaceString_Create(&string, (char *)text, mARRAYSIZE(text)));

  mString mstring;
  mDEFER_DESTRUCTION(&mstring, mString_Destroy);
  mTEST_ASSERT_SUCCESS(mString_Create(&mstring, string));

  size_t count;
  mTEST_ASSERT_SUCCESS(mInplaceString_GetCount(string, &count));
  size_t mStringCount;
  mTEST_ASSERT_SUCCESS(mString_GetCount(mstring, &mStringCount));
  mTEST_ASSERT_EQUAL(count, mStringCount);

  size_t size;
  mTEST_ASSERT_SUCCESS(mInplaceString_GetByteSize(string, &size));
  size_t mStringSize;
  mTEST_ASSERT_SUCCESS(mString_GetByteSize(mstring, &mStringSize));
  mTEST_ASSERT_EQUAL(size, mStringSize);

  mTEST_ALLOCATOR_ZERO_CHECK();
}
