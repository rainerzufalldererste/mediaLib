#include "mTestLib.h"
#include "mString.h"

mTEST(mString, TestCreateEmpty)
{
  mTEST_ALLOCATOR_SETUP();

  mString string;
  mDEFER_CALL(&string, mString_Destroy);
  mTEST_ASSERT_SUCCESS(mString_Create(&string, "", pAllocator));

  size_t count;
  mTEST_ASSERT_SUCCESS(mString_GetCount(string, &count));
  mTEST_ASSERT_EQUAL(count, 1);

  size_t size;
  mTEST_ASSERT_SUCCESS(mString_GetByteSize(string, &size));
  mTEST_ASSERT_EQUAL(size, 1);

  mTEST_ALLOCATOR_ZERO_CHECK();
}

mTEST(mString, TestCreateNullptr)
{
  mTEST_ALLOCATOR_SETUP();

  mString string;
  mDEFER_CALL(&string, mString_Destroy);
  mTEST_ASSERT_SUCCESS(mString_Create(&string, (char *)nullptr, pAllocator));

  mTEST_ASSERT_EQUAL(string.c_str(), nullptr);

  size_t count;
  mTEST_ASSERT_SUCCESS(mString_GetCount(string, &count));
  mTEST_ASSERT_EQUAL(count, 0);

  size_t size;
  mTEST_ASSERT_SUCCESS(mString_GetByteSize(string, &size));
  mTEST_ASSERT_EQUAL(size, 0);

  mTEST_ALLOCATOR_ZERO_CHECK();
}

mTEST(mString, TestSetToNullptr)
{
  mTEST_ALLOCATOR_SETUP();

  mString string;
  mDEFER_CALL(&string, mString_Destroy);
  mTEST_ASSERT_SUCCESS(mString_Create(&string, "not nullptr", pAllocator));

  mTEST_ASSERT_SUCCESS(mString_Create(&string, (char *)nullptr, pAllocator));

  mTEST_ASSERT_EQUAL(string, "");

  size_t count;
  mTEST_ASSERT_SUCCESS(mString_GetCount(string, &count));
  mTEST_ASSERT_EQUAL(count, 1);

  size_t size;
  mTEST_ASSERT_SUCCESS(mString_GetByteSize(string, &size));
  mTEST_ASSERT_EQUAL(size, 1);

  mTEST_ALLOCATOR_ZERO_CHECK();
}

mTEST(mString, TestCreateNullptrLength)
{
  mTEST_ALLOCATOR_SETUP();

  mString string;
  mDEFER_CALL(&string, mString_Destroy);
  mTEST_ASSERT_SUCCESS(mString_Create(&string, (char *)nullptr, 1337, pAllocator));

  mTEST_ASSERT_EQUAL(string.c_str(), nullptr);

  size_t count;
  mTEST_ASSERT_SUCCESS(mString_GetCount(string, &count));
  mTEST_ASSERT_EQUAL(count, 0);

  size_t size;
  mTEST_ASSERT_SUCCESS(mString_GetByteSize(string, &size));
  mTEST_ASSERT_EQUAL(size, 0);

  mTEST_ALLOCATOR_ZERO_CHECK();
}

mTEST(mString, TestSetToNullptrLength)
{
  mTEST_ALLOCATOR_SETUP();

  const char testText[] = "not nullptr";

  mString string;
  mDEFER_CALL(&string, mString_Destroy);
  mTEST_ASSERT_SUCCESS(mString_Create(&string, testText, mARRAYSIZE(testText), pAllocator));

  mTEST_ASSERT_SUCCESS(mString_Create(&string, (char *)nullptr, 1337, pAllocator));

  mTEST_ASSERT_EQUAL(string, "");

  size_t count;
  mTEST_ASSERT_SUCCESS(mString_GetCount(string, &count));
  mTEST_ASSERT_EQUAL(count, 1);

  size_t size;
  mTEST_ASSERT_SUCCESS(mString_GetByteSize(string, &size));
  mTEST_ASSERT_EQUAL(size, 1);

  mTEST_ALLOCATOR_ZERO_CHECK();
}

mTEST(mString, TestCreateNullptrWcharT)
{
  mTEST_ALLOCATOR_SETUP();

  mString string;
  mDEFER_CALL(&string, mString_Destroy);
  mTEST_ASSERT_SUCCESS(mString_Create(&string, (wchar_t *)nullptr, pAllocator));

  mTEST_ASSERT_EQUAL(string.c_str(), nullptr);

  size_t count;
  mTEST_ASSERT_SUCCESS(mString_GetCount(string, &count));
  mTEST_ASSERT_EQUAL(count, 0);

  size_t size;
  mTEST_ASSERT_SUCCESS(mString_GetByteSize(string, &size));
  mTEST_ASSERT_EQUAL(size, 0);

  mTEST_ALLOCATOR_ZERO_CHECK();
}

mTEST(mString, TestSetToNullptrWcharT)
{
  mTEST_ALLOCATOR_SETUP();

  mString string;
  mDEFER_CALL(&string, mString_Destroy);
  mTEST_ASSERT_SUCCESS(mString_Create(&string, "not nullptr", pAllocator));

  mTEST_ASSERT_SUCCESS(mString_Create(&string, (wchar_t *)nullptr, pAllocator));

  mTEST_ASSERT_EQUAL(string, "");

  size_t count;
  mTEST_ASSERT_SUCCESS(mString_GetCount(string, &count));
  mTEST_ASSERT_EQUAL(count, 1);

  size_t size;
  mTEST_ASSERT_SUCCESS(mString_GetByteSize(string, &size));
  mTEST_ASSERT_EQUAL(size, 1);

  mTEST_ALLOCATOR_ZERO_CHECK();
}

mTEST(mString, TestCreateNullptrLengthWcharT)
{
  mTEST_ALLOCATOR_SETUP();

  mString string;
  mDEFER_CALL(&string, mString_Destroy);
  mTEST_ASSERT_SUCCESS(mString_Create(&string, (wchar_t *)nullptr, 1337, pAllocator));

  mTEST_ASSERT_EQUAL(string.c_str(), nullptr);

  size_t count;
  mTEST_ASSERT_SUCCESS(mString_GetCount(string, &count));
  mTEST_ASSERT_EQUAL(count, 0);

  size_t size;
  mTEST_ASSERT_SUCCESS(mString_GetByteSize(string, &size));
  mTEST_ASSERT_EQUAL(size, 0);

  mTEST_ALLOCATOR_ZERO_CHECK();
}

mTEST(mString, TestSetToNullptrLengthWcharT)
{
  mTEST_ALLOCATOR_SETUP();

  const char testText[] = "not nullptr";

  mString string;
  mDEFER_CALL(&string, mString_Destroy);
  mTEST_ASSERT_SUCCESS(mString_Create(&string, testText, mARRAYSIZE(testText), pAllocator));

  mTEST_ASSERT_SUCCESS(mString_Create(&string, (wchar_t *)nullptr, 1337, pAllocator));

  mTEST_ASSERT_EQUAL(string, "");

  size_t count;
  mTEST_ASSERT_SUCCESS(mString_GetCount(string, &count));
  mTEST_ASSERT_EQUAL(count, 1);

  size_t size;
  mTEST_ASSERT_SUCCESS(mString_GetByteSize(string, &size));
  mTEST_ASSERT_EQUAL(size, 1);

  mTEST_ALLOCATOR_ZERO_CHECK();
}

mTEST(mString, TestCreateASCII)
{
  mTEST_ALLOCATOR_SETUP();

  mString string;
  mDEFER_CALL(&string, mString_Destroy);
  mTEST_ASSERT_SUCCESS(mString_Create(&string, "h", pAllocator));

  size_t count;
  mTEST_ASSERT_SUCCESS(mString_GetCount(string, &count));
  mTEST_ASSERT_EQUAL(count, 2);

  size_t size;
  mTEST_ASSERT_SUCCESS(mString_GetByteSize(string, &size));
  mTEST_ASSERT_EQUAL(size, 2);

  mTEST_ALLOCATOR_ZERO_CHECK();
}

mTEST(mString, TestCreateMultipleASCII)
{
  mTEST_ALLOCATOR_SETUP();

  mString string;
  mDEFER_CALL(&string, mString_Destroy);
  mTEST_ASSERT_SUCCESS(mString_Create(&string, "a b c", pAllocator));

  size_t count;
  mTEST_ASSERT_SUCCESS(mString_GetCount(string, &count));
  mTEST_ASSERT_EQUAL(count, 6);

  size_t size;
  mTEST_ASSERT_SUCCESS(mString_GetByteSize(string, &size));
  mTEST_ASSERT_EQUAL(size, 6);

  mTEST_ALLOCATOR_ZERO_CHECK();
}

mTEST(mString, TestCreateUTF8)
{
  mTEST_ALLOCATOR_SETUP();

  mString string;
  mDEFER_CALL(&string, mString_Destroy);
  mTEST_ASSERT_SUCCESS(mString_Create(&string, "🌵🦎🎅", pAllocator));

  size_t count;
  mTEST_ASSERT_SUCCESS(mString_GetCount(string, &count));
  mTEST_ASSERT_EQUAL(count, 4);

  size_t size;
  mTEST_ASSERT_SUCCESS(mString_GetByteSize(string, &size));
  mTEST_ASSERT_EQUAL(size, 12 + 1);

  mTEST_ALLOCATOR_ZERO_CHECK();
}

mTEST(mString, TestCreateMixed)
{
  mTEST_ALLOCATOR_SETUP();

  mString string;
  mDEFER_CALL(&string, mString_Destroy);
  mTEST_ASSERT_SUCCESS(mString_Create(&string, "🌵🦎🎅testמⴲ", pAllocator));

  size_t count;
  mTEST_ASSERT_SUCCESS(mString_GetCount(string, &count));
  mTEST_ASSERT_EQUAL(count, 3 + 4 + 2 + 1);

  size_t size;
  mTEST_ASSERT_SUCCESS(mString_GetByteSize(string, &size));
  mTEST_ASSERT_EQUAL(size, 12 + 4 + 2 + 3 + 1);

  mTEST_ALLOCATOR_ZERO_CHECK();
}

mTEST(mString, TestCreateLength)
{
  mTEST_ALLOCATOR_SETUP();

  const char text[] = "test123";

  mString string;
  mDEFER_CALL(&string, mString_Destroy);
  mTEST_ASSERT_SUCCESS(mString_Create(&string, text, 4, pAllocator));

  size_t count;
  mTEST_ASSERT_SUCCESS(mString_GetCount(string, &count));
  mTEST_ASSERT_EQUAL(count, 5);

  size_t size;
  mTEST_ASSERT_SUCCESS(mString_GetByteSize(string, &size));
  mTEST_ASSERT_EQUAL(size, 5);

  mTEST_ASSERT_EQUAL('\0', string.c_str()[string.Size() - 1]);

  mTEST_ALLOCATOR_ZERO_CHECK();
}

mTEST(mString, TestCreateWideLength)
{
  mTEST_ALLOCATOR_SETUP();

  const wchar_t text[] = L"test123";

  mString string;
  mDEFER_CALL(&string, mString_Destroy);
  mTEST_ASSERT_SUCCESS(mString_Create(&string, text, 4, pAllocator));

  size_t count;
  mTEST_ASSERT_SUCCESS(mString_GetCount(string, &count));
  mTEST_ASSERT_EQUAL(count, 5);

  size_t size;
  mTEST_ASSERT_SUCCESS(mString_GetByteSize(string, &size));
  mTEST_ASSERT_EQUAL(size, 5);

  mTEST_ASSERT_EQUAL('\0', string.c_str()[string.Size() - 1]);

  mTEST_ALLOCATOR_ZERO_CHECK();
}

mTEST(mString, TestCreateFromWchar)
{
  mTEST_ALLOCATOR_SETUP();

  mString string;
  mDEFER_CALL(&string, mString_Destroy);
  mTEST_ASSERT_SUCCESS(mString_Create(&string, L"Testמ", pAllocator));

  size_t count;
  mTEST_ASSERT_SUCCESS(mString_GetCount(string, &count));
  mTEST_ASSERT_EQUAL(count, 4 + 2 + 1);

  mTEST_ALLOCATOR_ZERO_CHECK();
}

mTEST(mString, TestCreateFromString)
{
  mTEST_ALLOCATOR_SETUP();

  mString stringA;
  mDEFER_CALL(&stringA, mString_Destroy);
  mTEST_ASSERT_SUCCESS(mString_Create(&stringA, "🌵🦎🎅testמⴲ", pAllocator));

  mString stringB;
  mDEFER_CALL(&stringB, mString_Destroy);
  mTEST_ASSERT_SUCCESS(mString_Create(&stringB, stringA));

  mTEST_ASSERT_EQUAL(stringA.bytes, stringB.bytes);
  mTEST_ASSERT_EQUAL(stringA.count, stringB.count);
  mTEST_ASSERT_NOT_EQUAL((void *)stringA.text, (void *)stringB.text);
  mTEST_ASSERT_TRUE(stringA == stringB);

  mTEST_ASSERT_SUCCESS(mString_Create(&stringA, stringB));

  mTEST_ASSERT_EQUAL(stringA.bytes, stringB.bytes);
  mTEST_ASSERT_EQUAL(stringA.count, stringB.count);
  mTEST_ASSERT_NOT_EQUAL((void *)stringA.text, (void *)stringB.text);
  mTEST_ASSERT_TRUE(stringA == stringB);

  mTEST_ALLOCATOR_ZERO_CHECK();
}

mTEST(mString, TestAt)
{
  mTEST_ALLOCATOR_SETUP();

  mString string;
  mDEFER_CALL(&string, mString_Destroy);
  mTEST_ASSERT_SUCCESS(mString_Create(&string, "🌵🦎🎅testמⴲ", pAllocator));

  mchar_t c = mToChar<4>("🌵");
  mTEST_ASSERT_EQUAL(c, string[0]);
  mTEST_ASSERT_NOT_EQUAL(c, string[1]);
  c = mToChar<4>("🦎");
  mTEST_ASSERT_EQUAL(c, string[1]);
  c = mToChar<4>("🎅");
  mTEST_ASSERT_EQUAL(c, string[2]);
  c = mToChar<1>("t");
  mTEST_ASSERT_EQUAL(c, string[3]);
  c = mToChar<1>("e");
  mTEST_ASSERT_EQUAL(c, string[4]);
  c = mToChar<1>("s");
  mTEST_ASSERT_EQUAL(c, string[5]);
  c = mToChar<1>("t");
  mTEST_ASSERT_EQUAL(c, string[6]);
  c = mToChar<2>("מ");
  mTEST_ASSERT_EQUAL(c, string[7]);
  c = mToChar<3>("ⴲ");
  mTEST_ASSERT_EQUAL(c, string[8]);

  mTEST_ALLOCATOR_ZERO_CHECK();
}

mTEST(mString, TestAppend)
{
  mTEST_ALLOCATOR_SETUP();

  mString string;
  mDEFER_CALL(&string, mString_Destroy);
  mTEST_ASSERT_SUCCESS(mString_Create(&string, "🌵🦎🎅test", pAllocator));

  mTEST_ASSERT_SUCCESS(mString_Append(string, mString("🦎T", pAllocator)));

  size_t count;
  mTEST_ASSERT_SUCCESS(mString_GetCount(string, &count));
  mTEST_ASSERT_EQUAL(count, 4 + 5 + 1);

  size_t size;
  mTEST_ASSERT_SUCCESS(mString_GetByteSize(string, &size));
  mTEST_ASSERT_EQUAL(size, 16 + 5 + 1);

  mTEST_ALLOCATOR_ZERO_CHECK();
}

mTEST(mString, TestAppendOperator)
{
  mTEST_ALLOCATOR_SETUP();

  mString string;
  mDEFER_CALL(&string, mString_Destroy);
  mTEST_ASSERT_SUCCESS(mString_Create(&string, "🌵🦎🎅test", pAllocator));

  string += mString("🦎T", pAllocator);
  mTEST_ASSERT_FALSE(string.hasFailed);

  size_t count;
  mTEST_ASSERT_SUCCESS(mString_GetCount(string, &count));
  mTEST_ASSERT_EQUAL(count, 4 + 5 + 1);

  size_t size;
  mTEST_ASSERT_SUCCESS(mString_GetByteSize(string, &size));
  mTEST_ASSERT_EQUAL(size, 16 + 5 + 1);

  mTEST_ALLOCATOR_ZERO_CHECK();
}

mTEST(mString, TestConcatOperator)
{
  mTEST_ALLOCATOR_SETUP();

  mString string;
  mDEFER_CALL(&string, mString_Destroy);
  mTEST_ASSERT_SUCCESS(mString_Create(&string, "🌵🦎🎅test", pAllocator));

  mString newString = string + mString("🦎T", pAllocator);
  mTEST_ASSERT_FALSE(newString.hasFailed);

  size_t count;
  mTEST_ASSERT_SUCCESS(mString_GetCount(newString, &count));
  mTEST_ASSERT_EQUAL(count, 4 + 5 + 1);

  size_t size;
  mTEST_ASSERT_SUCCESS(mString_GetByteSize(newString, &size));
  mTEST_ASSERT_EQUAL(size, 16 + 5 + 1);

  mTEST_ALLOCATOR_ZERO_CHECK();
}

mTEST(mString, TestAppendEmpty)
{
  mTEST_ALLOCATOR_SETUP();

  mString string;
  mDEFER_CALL(&string, mString_Destroy);
  mTEST_ASSERT_SUCCESS(mString_Create(&string, "🌵🦎🎅test", pAllocator));

  mTEST_ASSERT_SUCCESS(mString_Append(string, mString()));

  size_t count;
  mTEST_ASSERT_SUCCESS(mString_GetCount(string, &count));
  mTEST_ASSERT_EQUAL(count, 3 + 4 + 1);

  size_t size;
  mTEST_ASSERT_SUCCESS(mString_GetByteSize(string, &size));
  mTEST_ASSERT_EQUAL(size, 12 + 4 + 1);

  mTEST_ALLOCATOR_ZERO_CHECK();
}

mTEST(mString, TestAppendOperatorEmpty)
{
  mTEST_ALLOCATOR_SETUP();

  mString string;
  mDEFER_CALL(&string, mString_Destroy);
  mTEST_ASSERT_SUCCESS(mString_Create(&string, "🌵🦎🎅test", pAllocator));

  string += mString();
  mTEST_ASSERT_FALSE(string.hasFailed);

  size_t count;
  mTEST_ASSERT_SUCCESS(mString_GetCount(string, &count));
  mTEST_ASSERT_EQUAL(count, 3 + 4 + 1);

  size_t size;
  mTEST_ASSERT_SUCCESS(mString_GetByteSize(string, &size));
  mTEST_ASSERT_EQUAL(size, 12 + 4 + 1);

  mTEST_ALLOCATOR_ZERO_CHECK();
}

mTEST(mString, TestConcatOperatorEmpty)
{
  mTEST_ALLOCATOR_SETUP();

  mString string;
  mDEFER_CALL(&string, mString_Destroy);
  mTEST_ASSERT_SUCCESS(mString_Create(&string, "🌵🦎🎅test", pAllocator));

  mString newString = string + mString();
  mTEST_ASSERT_FALSE(newString.hasFailed);

  size_t count;
  mTEST_ASSERT_SUCCESS(mString_GetCount(newString, &count));
  mTEST_ASSERT_EQUAL(count, 3 + 4 + 1);

  size_t size;
  mTEST_ASSERT_SUCCESS(mString_GetByteSize(newString, &size));
  mTEST_ASSERT_EQUAL(size, 12 + 4 + 1);

  mTEST_ALLOCATOR_ZERO_CHECK();
}

mTEST(mString, TestAppendEmptyBase)
{
  mTEST_ALLOCATOR_SETUP();

  mString string;
  mDEFER_CALL(&string, mString_Destroy);
  mTEST_ASSERT_SUCCESS(mString_Create(&string, "🌵🦎🎅test", pAllocator));

  mString emptyString;
  mTEST_ASSERT_SUCCESS(mString_Append(emptyString, string));

  size_t count;
  mTEST_ASSERT_SUCCESS(mString_GetCount(emptyString, &count));
  mTEST_ASSERT_EQUAL(count, 3 + 4 + 1);

  size_t size;
  mTEST_ASSERT_SUCCESS(mString_GetByteSize(emptyString, &size));
  mTEST_ASSERT_EQUAL(size, 12 + 4 + 1);

  mTEST_ALLOCATOR_ZERO_CHECK();
}

mTEST(mString, TestAppendOperatorEmptyBase)
{
  mTEST_ALLOCATOR_SETUP();

  mString string;
  mDEFER_CALL(&string, mString_Destroy);
  mTEST_ASSERT_SUCCESS(mString_Create(&string, "🌵🦎🎅test", pAllocator));

  mString emptyString;
  emptyString += string;
  mTEST_ASSERT_FALSE(string.hasFailed);

  size_t count;
  mTEST_ASSERT_SUCCESS(mString_GetCount(emptyString, &count));
  mTEST_ASSERT_EQUAL(count, 3 + 4 + 1);

  size_t size;
  mTEST_ASSERT_SUCCESS(mString_GetByteSize(emptyString, &size));
  mTEST_ASSERT_EQUAL(size, 12 + 4 + 1);

  mTEST_ALLOCATOR_ZERO_CHECK();
}

mTEST(mString, TestConcatOperatorEmptyBase)
{
  mTEST_ALLOCATOR_SETUP();

  mString string;
  mDEFER_CALL(&string, mString_Destroy);
  mTEST_ASSERT_SUCCESS(mString_Create(&string, "🌵🦎🎅test", pAllocator));

  mString newString = mString() + string;
  mTEST_ASSERT_FALSE(newString.hasFailed);

  size_t count;
  mTEST_ASSERT_SUCCESS(mString_GetCount(newString, &count));
  mTEST_ASSERT_EQUAL(count, 3 + 4 + 1);

  size_t size;
  mTEST_ASSERT_SUCCESS(mString_GetByteSize(newString, &size));
  mTEST_ASSERT_EQUAL(size, 12 + 4 + 1);

  mTEST_ALLOCATOR_ZERO_CHECK();
}

mTEST(mString, TestWstringCompare)
{
  mTEST_ALLOCATOR_SETUP();

  mString string;
  mDEFER_CALL(&string, mString_Destroy);
  mTEST_ASSERT_SUCCESS(mString_Create(&string, "🌵🦎🎅testמⴲx", pAllocator));

  wchar_t wstring[1024];
  mTEST_ASSERT_SUCCESS(mString_ToWideString(string, wstring, mARRAYSIZE(wstring)));

  mString string2;
  mDEFER_CALL(&string2, mString_Destroy);
  mTEST_ASSERT_SUCCESS(mString_Create(&string2, wstring, pAllocator));

  mTEST_ASSERT_EQUAL(string, string2);

  mTEST_ALLOCATOR_ZERO_CHECK();
}

mTEST(mString, TestSubstring)
{
  mTEST_ALLOCATOR_SETUP();

  mString string;
  mDEFER_CALL(&string, mString_Destroy);
  mTEST_ASSERT_SUCCESS(mString_Create(&string, "🌵🦎🎅testמⴲx", pAllocator));

  mString subString;
  mDEFER_CALL(&subString, mString_Destroy);
  mTEST_ASSERT_SUCCESS(mString_Substring(string, &subString, 2, 7));

  mString comparisonString;
  mDEFER_CALL(&comparisonString, mString_Destroy);
  mTEST_ASSERT_SUCCESS(mString_Create(&comparisonString, "🎅testמⴲ", pAllocator));

  mTEST_ASSERT_EQUAL(comparisonString.count, subString.count);
  mTEST_ASSERT_EQUAL(comparisonString.bytes, subString.bytes);

  for (size_t i = 0; i < comparisonString.count; i++)
    mTEST_ASSERT_EQUAL(comparisonString[i], subString[i]);

  mTEST_ASSERT_SUCCESS(mString_Substring(string, &subString, 1));
  comparisonString = mString("🦎", pAllocator) + comparisonString + "x";

  mTEST_ASSERT_EQUAL(comparisonString.count, subString.count);
  mTEST_ASSERT_EQUAL(comparisonString.bytes, subString.bytes);

  for (size_t i = 0; i < comparisonString.count; i++)
    mTEST_ASSERT_EQUAL(comparisonString[i], subString[i]);

  mTEST_ALLOCATOR_ZERO_CHECK();
}

mTEST(mString, TestEquals)
{
  mTEST_ALLOCATOR_SETUP();

  mString stringA;
  mDEFER_CALL(&stringA, mString_Destroy);
  mTEST_ASSERT_SUCCESS(mString_Create(&stringA, "", pAllocator));

  mString stringB;
  mDEFER_CALL(&stringB, mString_Destroy);
  mTEST_ASSERT_SUCCESS(mString_Create(&stringB, "", pAllocator));

  mTEST_ASSERT_EQUAL(stringA, stringB);
  mTEST_ASSERT_FALSE(stringA != stringB);
  mTEST_ASSERT_TRUE(stringA == stringB);

  bool equal;
  mTEST_ASSERT_SUCCESS(mString_Equals(stringA, stringB, &equal));
  mTEST_ASSERT_TRUE(equal);

  mTEST_ASSERT_SUCCESS(mString_Create(&stringA, "🌵🦎🎅testמⴲx", pAllocator));
  mTEST_ASSERT_SUCCESS(mString_Create(&stringB, "🌵🦎🎅testמⴲx", pAllocator));

  mTEST_ASSERT_EQUAL(stringA, stringB);
  mTEST_ASSERT_FALSE(stringA != stringB);
  mTEST_ASSERT_TRUE(stringA == stringB);

  mTEST_ASSERT_SUCCESS(mString_Equals(stringA, stringB, &equal));
  mTEST_ASSERT_TRUE(equal);

  mTEST_ASSERT_SUCCESS(mString_Create(&stringA, "🌵🦎🎅 estמⴲx", pAllocator));
  mTEST_ASSERT_SUCCESS(mString_Create(&stringB, "🌵🦎🎅testמⴲx", pAllocator));

  mTEST_ASSERT_NOT_EQUAL(stringA, stringB);
  mTEST_ASSERT_TRUE(stringA != stringB);
  mTEST_ASSERT_FALSE(stringA == stringB);

  mTEST_ASSERT_SUCCESS(mString_Equals(stringA, stringB, &equal));
  mTEST_ASSERT_FALSE(equal);

  mTEST_ASSERT_SUCCESS(mString_Create(&stringA, "🌵🦎🎅testמⴲ", pAllocator));
  mTEST_ASSERT_SUCCESS(mString_Create(&stringB, "🌵🦎🎅testמⴲx", pAllocator));

  mTEST_ASSERT_NOT_EQUAL(stringA, stringB);
  mTEST_ASSERT_TRUE(stringA != stringB);
  mTEST_ASSERT_FALSE(stringA == stringB);

  mTEST_ASSERT_SUCCESS(mString_Equals(stringA, stringB, &equal));
  mTEST_ASSERT_FALSE(equal);

  mTEST_ASSERT_SUCCESS(mString_Create(&stringA, "🌵🦎🎅testמⴲy", pAllocator));
  mTEST_ASSERT_SUCCESS(mString_Create(&stringB, "🌵🦎🎅testמⴲx", pAllocator));

  mTEST_ASSERT_NOT_EQUAL(stringA, stringB);
  mTEST_ASSERT_TRUE(stringA != stringB);
  mTEST_ASSERT_FALSE(stringA == stringB);

  mTEST_ASSERT_SUCCESS(mString_Equals(stringA, stringB, &equal));
  mTEST_ASSERT_FALSE(equal);

  mTEST_ASSERT_SUCCESS(mString_Create(&stringA, "🌵🦎🎅testמⴲ🌵מk", pAllocator));
  mTEST_ASSERT_SUCCESS(mString_Create(&stringB, "🌵🦎🎅testמⴲמ🌵k", pAllocator));

  mTEST_ASSERT_NOT_EQUAL(stringA, stringB);
  mTEST_ASSERT_TRUE(stringA != stringB);
  mTEST_ASSERT_FALSE(stringA == stringB);

  mTEST_ASSERT_SUCCESS(mString_Equals(stringA, stringB, &equal));
  mTEST_ASSERT_FALSE(equal);

  mTEST_ASSERT_SUCCESS(mString_Create(&stringA, "🌵🦎🎅testמⴲמ", pAllocator));
  mTEST_ASSERT_SUCCESS(mString_Create(&stringB, "🦎🌵🎅testמⴲמ", pAllocator));

  mTEST_ASSERT_NOT_EQUAL(stringA, stringB);
  mTEST_ASSERT_TRUE(stringA != stringB);
  mTEST_ASSERT_FALSE(stringA == stringB);

  mTEST_ASSERT_SUCCESS(mString_Equals(stringA, stringB, &equal));
  mTEST_ASSERT_FALSE(equal);

  mTEST_ALLOCATOR_ZERO_CHECK();
}

mTEST(mString, TestToDirectoryPath)
{
  mTEST_ALLOCATOR_SETUP();

  mString string;
  mDEFER_CALL(&string, mString_Destroy);
  mTEST_ASSERT_SUCCESS(mString_Create(&string, "C:/Some Folder🌵//Path/\\p", pAllocator));

  mString path;
  mDEFER_CALL(&path, mString_Destroy);
  mTEST_ASSERT_SUCCESS(mString_ToDirectoryPath(&path, string));

  mTEST_ASSERT_EQUAL(path, "C:\\Some Folder🌵\\Path\\p\\");

  mTEST_ASSERT_SUCCESS(mString_Create(&string, "../Some Folder//Path\\\\p/", pAllocator));
  mTEST_ASSERT_SUCCESS(mString_ToDirectoryPath(&path, string));

  mTEST_ASSERT_EQUAL(path, "..\\Some Folder\\Path\\p\\");

  mTEST_ASSERT_SUCCESS(mString_Create(&string, "/Some Folder//Path/\\/p\\", pAllocator));
  mTEST_ASSERT_SUCCESS(mString_ToDirectoryPath(&path, string));

  mTEST_ASSERT_EQUAL(path, "\\Some Folder\\Path\\p\\");

  mTEST_ASSERT_SUCCESS(mString_Create(&string, "test", pAllocator));
  mTEST_ASSERT_SUCCESS(mString_ToDirectoryPath(&path, string));

  mTEST_ASSERT_EQUAL(path, "test\\");

  mTEST_ASSERT_SUCCESS(mString_Create(&string, "\\\\NetworkPath//Path/\\/p\\", pAllocator));
  mTEST_ASSERT_SUCCESS(mString_ToDirectoryPath(&path, string));

  mTEST_ASSERT_EQUAL(path, "\\\\NetworkPath\\Path\\p\\");

  mTEST_ALLOCATOR_ZERO_CHECK();
}

mTEST(mString, TestToFilePath)
{
  mTEST_ALLOCATOR_SETUP();

  mString string;
  mDEFER_CALL(&string, mString_Destroy);
  mTEST_ASSERT_SUCCESS(mString_Create(&string, "C:/Some Folder//Path/\\p", pAllocator));

  mString path;
  mDEFER_CALL(&path, mString_Destroy);
  mTEST_ASSERT_SUCCESS(mString_ToFilePath(&path, string));

  mTEST_ASSERT_EQUAL(path, "C:\\Some Folder\\Path\\p");

  mTEST_ASSERT_SUCCESS(mString_Create(&string, "../Some Folder//Path\\\\p", pAllocator));
  mTEST_ASSERT_SUCCESS(mString_ToFilePath(&path, string));

  mTEST_ASSERT_EQUAL(path, "..\\Some Folder\\Path\\p");

  mTEST_ASSERT_SUCCESS(mString_Create(&string, "/Some Folder//Path/\\/p", pAllocator));
  mTEST_ASSERT_SUCCESS(mString_ToFilePath(&path, string));

  mTEST_ASSERT_EQUAL(path, "\\Some Folder\\Path\\p");

  mTEST_ASSERT_SUCCESS(mString_Create(&string, "test", pAllocator));
  mTEST_ASSERT_SUCCESS(mString_ToFilePath(&path, string));

  mTEST_ASSERT_EQUAL(path, "test");

  mTEST_ASSERT_SUCCESS(mString_Create(&string, "\\\\NetworkPath//Path/\\/p", pAllocator));
  mTEST_ASSERT_SUCCESS(mString_ToFilePath(&path, string));

  mTEST_ASSERT_EQUAL(path, "\\\\NetworkPath\\Path\\p");

  mTEST_ALLOCATOR_ZERO_CHECK();
}

mTEST(mString, TestToDirectoryPathEmpty)
{
  mTEST_ALLOCATOR_SETUP();

  mString string;
  mDEFER_CALL(&string, mString_Destroy);

  mString path;
  mDEFER_CALL(&path, mString_Destroy);
  mTEST_ASSERT_SUCCESS(mString_ToDirectoryPath(&path, string));

  mTEST_ASSERT_EQUAL(path, "");

  mTEST_ALLOCATOR_ZERO_CHECK();
}

mTEST(mString, TestToFilePathEmpty)
{
  mTEST_ALLOCATOR_SETUP();

  mString string;
  mDEFER_CALL(&string, mString_Destroy);

  mString path;
  mDEFER_CALL(&path, mString_Destroy);
  mTEST_ASSERT_SUCCESS(mString_ToFilePath(&path, string));

  mTEST_ASSERT_EQUAL(path, "");

  mTEST_ALLOCATOR_ZERO_CHECK();
}

mTEST(mString, TestAppendEqualsStringFunction)
{
  mTEST_ALLOCATOR_SETUP();

  mString string;
  mDEFER_CALL(&string, mString_Destroy);
  mTEST_ASSERT_SUCCESS(mString_Create(&string, "", pAllocator));

  mString appendedString;
  mDEFER_CALL(&appendedString, mString_Destroy);
  mTEST_ASSERT_SUCCESS(mString_Create(&appendedString, "test string 1 2 3", pAllocator));

  for (size_t i = 0; i < 1024; i++)
    mTEST_ASSERT_SUCCESS(mString_Append(string, appendedString));

  mTEST_ALLOCATOR_ZERO_CHECK();
}

mTEST(mString, TestAppendEqualsStringOperator)
{
  mTEST_ALLOCATOR_SETUP();

  mString string;
  mDEFER_CALL(&string, mString_Destroy);
  mTEST_ASSERT_SUCCESS(mString_Create(&string, "", pAllocator));

  for (size_t i = 0; i < 1024; i++)
    string += "test string 1 2 3";

  mTEST_ALLOCATOR_ZERO_CHECK();
}

mTEST(mString, TestAppendStringOperator)
{
  mTEST_ALLOCATOR_SETUP();

  mString string;
  mDEFER_CALL(&string, mString_Destroy);
  mTEST_ASSERT_SUCCESS(mString_Create(&string, "", pAllocator));

  for (size_t i = 0; i < 8; i++)
    string += string + "test string 1 2 3" + string + "test";

  mTEST_ALLOCATOR_ZERO_CHECK();
}

mTEST(mString, TestIterate)
{
  mTEST_ALLOCATOR_SETUP();

  mString string;
  mDEFER_CALL(&string, mString_Destroy);
  mTEST_ASSERT_SUCCESS(mString_Create(&string, "🌵🦎🎅testמⴲx", pAllocator));

  size_t charSize[] = { 4, 4, 4, 1, 1, 1, 1, 2, 3, 1 };

  size_t count = 0;

  for (auto &&_char : string.begin())
  {
    if (count == 0)
      mTEST_ASSERT_EQUAL(*(uint32_t *)string.text, *(uint32_t *)_char.character);

    mTEST_ASSERT_EQUAL(_char.index, count);
    mTEST_ASSERT_EQUAL(_char.characterSize, charSize[count]);
    count++;
  }

  mTEST_ASSERT_EQUAL(string.Count() - 1, count); // we don't care about the '\0' character.

  mTEST_ALLOCATOR_ZERO_CHECK();
}

mTEST(mString, TestCreateFromEmpty)
{
  mTEST_ALLOCATOR_SETUP();

  mString emptyString;
  mDEFER_CALL(&emptyString, mString_Destroy);

  mString string;
  mDEFER_CALL(&string, mString_Destroy);
  mTEST_ASSERT_SUCCESS(mString_Create(&string, emptyString, pAllocator));

  mTEST_ASSERT_EQUAL(string.bytes, 0);
  mTEST_ASSERT_EQUAL(string.count, 0);
  mTEST_ASSERT_EQUAL(string.text, nullptr);

  mTEST_ALLOCATOR_ZERO_CHECK();
}

mTEST(mString, TestCreateFromNowEmpty)
{
  mTEST_ALLOCATOR_SETUP();

  mString emptyString;
  mDEFER_CALL(&emptyString, mString_Destroy);
  mTEST_ASSERT_SUCCESS(mString_Create(&emptyString, "THIS IS NOT EMPTY", pAllocator));
  mTEST_ASSERT_SUCCESS(mString_Create(&emptyString, (char *)nullptr, pAllocator));

  mString string;
  mDEFER_CALL(&string, mString_Destroy);
  mTEST_ASSERT_SUCCESS(mString_Create(&string, emptyString, pAllocator));

  mTEST_ASSERT_EQUAL(string.bytes, 0);
  mTEST_ASSERT_EQUAL(string.count, 0);
  mTEST_ASSERT_EQUAL(string.text, nullptr);

  mTEST_ALLOCATOR_ZERO_CHECK();
}

mTEST(mString, TestCreateFromTooBigInvalidUTF8)
{
  mTEST_ALLOCATOR_SETUP();

  char text[] = { '\xf0', '\x9f', '\x8c', '\xb5', 'X', '\0', 'A', '\x9D', '\x8C', '\x86', '\x20' };

  mString string;
  mDEFER_CALL(&string, mString_Destroy);
  mTEST_ASSERT_SUCCESS(mString_Create(&string, text, pAllocator));

  mTEST_ASSERT_EQUAL(string.bytes, 6);
  mTEST_ASSERT_EQUAL(string.count, 3);

  string = "";
  mTEST_ASSERT_EQUAL(string.bytes, 1);
  mTEST_ASSERT_EQUAL(string.count, 1);

  mTEST_ASSERT_SUCCESS(mString_Create(&string, text, mARRAYSIZE(text), pAllocator));

  mTEST_ASSERT_EQUAL(string.bytes, 6);
  mTEST_ASSERT_EQUAL(string.count, 3);

  mTEST_ALLOCATOR_ZERO_CHECK();
}

mTEST(mString, TestAppendWCharT)
{
  mTEST_ALLOCATOR_SETUP();

  mString string;
  mTEST_ASSERT_SUCCESS(mString_Create(&string, "testString, blah, blah", pAllocator));

  string = string + L"horrible WCharT String";

  const char compString[] = "testString, blah, blah" "horrible WCharT String";

  mTEST_ASSERT_EQUAL(string, mString(compString));
  mTEST_ASSERT_EQUAL(strlen(string.c_str()), strlen(compString));

  mTEST_ALLOCATOR_ZERO_CHECK();
}

mTEST(mString, TestSetToWCharT)
{
  mTEST_ALLOCATOR_SETUP();

  mString string;
  mTEST_ASSERT_SUCCESS(mString_Create(&string, "this is a very long testString to allocate a sufficient amount of memory for the wchar_t string", pAllocator));

  mTEST_ASSERT_SUCCESS(mString_Create(&string, L"horrible WCharT String"));

  const char compString[] = "horrible WCharT String";

  mTEST_ASSERT_EQUAL(string, mString(compString));
  mTEST_ASSERT_EQUAL(strlen(string.c_str()), strlen(compString));

  mTEST_ALLOCATOR_ZERO_CHECK();
}

mTEST(mString, TestWstringLargerBufferSizeCreate)
{
  mTEST_ALLOCATOR_SETUP();

  wchar_t testString[10] = { 'b', 'i', 'n', '\0', '\0', '\0', '\0', '\0', '\0', '\0' };
  mString string;

  mTEST_ASSERT_SUCCESS(mString_Create(&string, testString, mARRAYSIZE(testString), pAllocator));

  mTEST_ALLOCATOR_ZERO_CHECK();
}

mTEST(mString, TestStartsWith)
{
  mTEST_ALLOCATOR_SETUP();

  mString testString;
  mTEST_ASSERT_SUCCESS(mString_Create(&testString, "\xF0\x93\x83\xA0\xF0\x93\x83\xA1\xF0\x93\x83\xA2\xF0\x93\x83\xA3\xF0\x93\x83\xA4\xF0\x93\x83\xA5\xF0\x93\x83\xA6\xF0\x93\x83\xA7\xF0\x93\x83\xA8\xF0\x93\x83\xA9\xF0\x93\x83\xAA\xF0\x93\x83\xAB\xF0\x93\x83\xAC\xF0\x93\x83\xAD", pAllocator));

  mString empty;

  mString emptyInit;
  mTEST_ASSERT_SUCCESS(mString_Create(&emptyInit, "", pAllocator));

  mString invalid;
  invalid.hasFailed = true;

  mString notTheStart;
  mTEST_ASSERT_SUCCESS(mString_Create(&notTheStart, "\xe1\x86\xb0\xe1\x86\xb1\xe1\x86\xb2\xe1\x86\xb3\xe1\x86\xb4", pAllocator));

  mString theStart;
  mTEST_ASSERT_SUCCESS(mString_Create(&theStart, "\xf0\x93\x83\xa0\xf0\x93\x83\xa1\xf0\x93\x83\xa2\xf0\x93\x83\xa3\xf0\x93\x83\xa4\xf0\x93\x83\xa5", pAllocator));

  mString invalidStart = "\xF0\x93\x83\xA0\xF0\x93\x83";

  bool startsWith;
  mTEST_ASSERT_SUCCESS(mString_StartsWith(testString, empty, &startsWith));
  mTEST_ASSERT_TRUE(startsWith);
  mTEST_ASSERT_TRUE(testString.StartsWith(empty));

  mTEST_ASSERT_SUCCESS(mString_StartsWith(testString, emptyInit, &startsWith));
  mTEST_ASSERT_TRUE(startsWith);

  mTEST_ASSERT_SUCCESS(mString_StartsWith(empty, testString, &startsWith));
  mTEST_ASSERT_FALSE(startsWith);

  mTEST_ASSERT_EQUAL(mR_InvalidParameter, mString_StartsWith(testString, invalid, &startsWith));
  mTEST_ASSERT_FALSE(startsWith);
  mTEST_ASSERT_FALSE(testString.StartsWith(invalid));

  mTEST_ASSERT_EQUAL(mR_InvalidParameter, mString_StartsWith(invalid, testString, &startsWith));
  mTEST_ASSERT_FALSE(startsWith);

  mTEST_ASSERT_SUCCESS(mString_StartsWith(testString, testString, &startsWith));
  mTEST_ASSERT_TRUE(startsWith);

  mTEST_ASSERT_EQUAL(mR_InvalidParameter, mString_StartsWith(invalid, invalidStart, &startsWith));
  mTEST_ASSERT_FALSE(startsWith);

  mTEST_ASSERT_SUCCESS(mString_StartsWith(testString, notTheStart, &startsWith));
  mTEST_ASSERT_FALSE(startsWith);

  mTEST_ASSERT_SUCCESS(mString_StartsWith(testString, theStart, &startsWith));
  mTEST_ASSERT_TRUE(startsWith);

  mTEST_ALLOCATOR_ZERO_CHECK();
}

mTEST(mString, TestEndsWith)
{
  mTEST_ALLOCATOR_SETUP();

  mString testString;
  mTEST_ASSERT_SUCCESS(mString_Create(&testString, "\xe0\xa6\x93\xe0\xa6\x94\xe0\xa6\x95\xe0\xa6\x96\xe0\xa6\x98\xe0\xa6\x99\xe0\xa6\x9a\xe0\xa6\x9b\xe0\xa6\x9c\xe0\xa6\x9d\xe0\xa6\x9e\xe0\xa6\x9f", pAllocator));

  mString empty;

  mString emptyInit;
  mTEST_ASSERT_SUCCESS(mString_Create(&emptyInit, "", pAllocator));

  mString invalid;
  invalid.hasFailed = true;

  mString notTheEnd;
  mTEST_ASSERT_SUCCESS(mString_Create(&notTheEnd, "\xe1\x86\xb0\xe1\x86\xb1\xe1\x86\xb2\xe1\x86\xb3\xe1\x86\xb4", pAllocator));

  mString theEnd;
  mTEST_ASSERT_SUCCESS(mString_Create(&theEnd, "\xe0\xa6\x9c\xe0\xa6\x9d\xe0\xa6\x9e\xe0\xa6\x9f", pAllocator));

  mString invalidEnd = "\xe0\xa6\x9c\xe0\xa6\x9d\xe0\xa6\x9e\xe0\xa6";

  bool endsWith;
  mTEST_ASSERT_SUCCESS(mString_EndsWith(testString, empty, &endsWith));
  mTEST_ASSERT_TRUE(endsWith);
  mTEST_ASSERT_TRUE(testString.EndsWith(empty));

  mTEST_ASSERT_SUCCESS(mString_EndsWith(testString, emptyInit, &endsWith));
  mTEST_ASSERT_TRUE(endsWith);

  mTEST_ASSERT_SUCCESS(mString_EndsWith(empty, testString, &endsWith));
  mTEST_ASSERT_FALSE(endsWith);

  mTEST_ASSERT_EQUAL(mR_InvalidParameter, mString_EndsWith(testString, invalid, &endsWith));
  mTEST_ASSERT_FALSE(endsWith);
  mTEST_ASSERT_FALSE(testString.EndsWith(invalid));

  mTEST_ASSERT_EQUAL(mR_InvalidParameter, mString_EndsWith(invalid, testString, &endsWith));
  mTEST_ASSERT_FALSE(endsWith);

  mTEST_ASSERT_SUCCESS(mString_EndsWith(testString, testString, &endsWith));
  mTEST_ASSERT_TRUE(endsWith);

  mTEST_ASSERT_EQUAL(mR_InvalidParameter, mString_EndsWith(invalid, invalidEnd, &endsWith));
  mTEST_ASSERT_FALSE(endsWith);

  mTEST_ASSERT_SUCCESS(mString_EndsWith(testString, notTheEnd, &endsWith));
  mTEST_ASSERT_FALSE(endsWith);

  mTEST_ASSERT_SUCCESS(mString_EndsWith(testString, theEnd, &endsWith));
  mTEST_ASSERT_TRUE(endsWith);

  mTEST_ALLOCATOR_ZERO_CHECK();
}

mTEST(mString, TestKmpAscii)
{
  mTEST_ALLOCATOR_SETUP();

  void mString_GetKmp(const mString &string, IN_OUT mchar_t *pString, IN_OUT size_t *pKmp);

  mString testString;
  mTEST_ASSERT_SUCCESS(mString_Create(&testString, "abcabcaabbccabcabcabcabcbacababac", pAllocator));

  size_t *pKmp = nullptr;
  mchar_t *pChars = nullptr;

  mDEFER_CALL_2(mAllocator_FreePtr, &mDefaultTempAllocator, &pKmp);
  mDEFER_CALL_2(mAllocator_FreePtr, &mDefaultTempAllocator, &pChars);
  mTEST_ASSERT_SUCCESS(mAllocator_AllocateZero(&mDefaultTempAllocator, &pKmp, testString.count - 1));
  mTEST_ASSERT_SUCCESS(mAllocator_Allocate(&mDefaultTempAllocator, &pChars, testString.count - 1));

  mString_GetKmp(testString, pChars, pKmp);

  size_t kmp[] = { 0, 0, 0, 1, 2, 3, 4, 1, 2, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 5, 6, 7, 5, 6, 0, 1, 0, 1, 2, 1, 2, 1, 0 };

  for (size_t i = 0; i < mARRAYSIZE(kmp); i++)
    mTEST_ASSERT_EQUAL(kmp[i], pKmp[i]);

  mTEST_ALLOCATOR_ZERO_CHECK();
}

mTEST(mString, TestKmpUnicode)
{
  mTEST_ALLOCATOR_SETUP();

  void mString_GetKmp(const mString &string, IN_OUT mchar_t *pString, IN_OUT size_t *pKmp);

  mString testString;
  mTEST_ASSERT_SUCCESS(mString_Create(&testString, "Я中CЯ中CЯЯ中中CCЯ中CЯ中CЯ中CЯ中C中ЯCЯ中Я中ЯC", pAllocator));

  size_t *pKmp = nullptr;
  mchar_t *pChars = nullptr;

  mDEFER_CALL_2(mAllocator_FreePtr, &mDefaultTempAllocator, &pKmp);
  mDEFER_CALL_2(mAllocator_FreePtr, &mDefaultTempAllocator, &pChars);
  mTEST_ASSERT_SUCCESS(mAllocator_AllocateZero(&mDefaultTempAllocator, &pKmp, testString.count - 1));
  mTEST_ASSERT_SUCCESS(mAllocator_Allocate(&mDefaultTempAllocator, &pChars, testString.count - 1));

  mString_GetKmp(testString, pChars, pKmp);

  size_t kmp[] = { 0, 0, 0, 1, 2, 3, 4, 1, 2, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 5, 6, 7, 5, 6, 0, 1, 0, 1, 2, 1, 2, 1, 0 };

  for (size_t i = 0; i < mARRAYSIZE(kmp); i++)
    mTEST_ASSERT_EQUAL(kmp[i], pKmp[i]);

  mTEST_ALLOCATOR_ZERO_CHECK();
}

mTEST(mString, TestContainsAscii)
{
  mTEST_ALLOCATOR_SETUP();

  mString string;
  mTEST_ASSERT_SUCCESS(mString_Create(&string, "ABCABCABCAAABBBCCCABCABCABCAACCBBBAABCABCBABCBBACBAAAABBABABBCCX", pAllocator));

  mString empty;

  mString emptyInit;
  mTEST_ASSERT_SUCCESS(mString_Create(&emptyInit, "", pAllocator));

  mString invalid;
  invalid.hasFailed = true;

  mString notContained;
  mTEST_ASSERT_SUCCESS(mString_Create(&notContained, "CAACCBBBAABCABAAAA", pAllocator));

  mString notContainedChar;
  mTEST_ASSERT_SUCCESS(mString_Create(&notContainedChar, "V", pAllocator));

  mString contained;
  mTEST_ASSERT_SUCCESS(mString_Create(&contained, "CAACCBBBAABCAB", pAllocator));

  mString start;
  mTEST_ASSERT_SUCCESS(mString_Create(&start, "ABCABCABCAAABB", pAllocator));

  mString end;
  mTEST_ASSERT_SUCCESS(mString_Create(&end, "ABBCC", pAllocator));

  mString startChar;
  mTEST_ASSERT_SUCCESS(mString_Create(&startChar, "A", pAllocator));

  mString endChar;
  mTEST_ASSERT_SUCCESS(mString_Create(&endChar, "X", pAllocator));

  mString midChar;
  mTEST_ASSERT_SUCCESS(mString_Create(&midChar, "C", pAllocator));

  bool isContained = false;
  mTEST_ASSERT_SUCCESS(mString_Contains(string, empty, &isContained));
  mTEST_ASSERT_TRUE(isContained);

  isContained = false;
  mTEST_ASSERT_SUCCESS(mString_Contains(string, emptyInit, &isContained));
  mTEST_ASSERT_TRUE(isContained);

  isContained = true;
  mTEST_ASSERT_EQUAL(mR_InvalidParameter, mString_Contains(string, invalid, &isContained));
  mTEST_ASSERT_FALSE(isContained);

  isContained = true;
  mTEST_ASSERT_EQUAL(mR_InvalidParameter, mString_Contains(invalid, string, &isContained));
  mTEST_ASSERT_FALSE(isContained);

  isContained = true;
  mTEST_ASSERT_SUCCESS(mString_Contains(string, notContained, &isContained));
  mTEST_ASSERT_FALSE(isContained);

  isContained = true;
  mTEST_ASSERT_SUCCESS(mString_Contains(string, notContainedChar, &isContained));
  mTEST_ASSERT_FALSE(isContained);

  isContained = false;
  mTEST_ASSERT_SUCCESS(mString_Contains(string, contained, &isContained));
  mTEST_ASSERT_TRUE(isContained);

  isContained = false;
  mTEST_ASSERT_SUCCESS(mString_Contains(string, start, &isContained));
  mTEST_ASSERT_TRUE(isContained);

  isContained = false;
  mTEST_ASSERT_SUCCESS(mString_Contains(string, end, &isContained));
  mTEST_ASSERT_TRUE(isContained);

  isContained = false;
  mTEST_ASSERT_SUCCESS(mString_Contains(string, startChar, &isContained));
  mTEST_ASSERT_TRUE(isContained);

  isContained = false;
  mTEST_ASSERT_SUCCESS(mString_Contains(string, endChar, &isContained));
  mTEST_ASSERT_TRUE(isContained);

  isContained = false;
  mTEST_ASSERT_SUCCESS(mString_Contains(string, midChar, &isContained));
  mTEST_ASSERT_TRUE(isContained);

  mTEST_ALLOCATOR_ZERO_CHECK();
}

mTEST(mString, TestContainsUnicode)
{
  mTEST_ALLOCATOR_SETUP();

  mString string;
  mTEST_ASSERT_SUCCESS(mString_Create(&string, "\xe0\xbc\x80\xe0\xbc\x81\xe0\xbc\x82\xe0\xbc\x83\xe0\xbc\x84\xe0\xbc\x85\xe0\xbc\x86\xe0\xbc\x87\xe0\xbc\x88\xe0\xbc\x89\xe0\xbc\x8a\xe0\xbc\x95\xe0\xbc\x96\xe0\xbc\x97\xe0\xbc\x98\xe0\xbc\x99\xe0\xbf\x90\xe1\x8f\xa0\xe1\x8f\xa1\xe1\x8f\xa2\xe1\x8f\xa3\xe1\x8f\xa4\xe1\x8f\xa5\xe1\x8f\xa6\xe1\x8f\xa7\xe1\x8f\xa8\xe1\x8f\xa9\xe1\x8f\xaa\xe1\x8f\xab\xe1\x8f\xac\xe1\x8f\xad\xe1\x8f\xae\xe1\x8f\xaf\xe0\xb8\xaa\xe0\xb8\xa7\xe0\xb8\xb1\xe0\xb8\xaa\xe0\xb8\x94\xe0\xb8\xb5\xe0\xb8\x8a\xe0\xb8\xb2\xe0\xb8\xa7\xe0\xb9\x82\xe0\xb8\xa5\xe0\xb8\x81", pAllocator));

  mString empty;

  mString emptyInit;
  mTEST_ASSERT_SUCCESS(mString_Create(&emptyInit, "", pAllocator));

  mString invalid;
  invalid.hasFailed = true;

  mString notContained;
  mTEST_ASSERT_SUCCESS(mString_Create(&notContained, "\xe0\xbc\x82\xe0\xbc\x83\xe0\xbc\x84\xe0\xbc\x85\xe1\x8f\xa8", pAllocator));

  mString notContainedChar;
  mTEST_ASSERT_SUCCESS(mString_Create(&notContainedChar, "V", pAllocator));

  mString contained;
  mTEST_ASSERT_SUCCESS(mString_Create(&contained, "\xe0\xbc\x96\xe0\xbc\x97\xe0\xbc\x98\xe0\xbc\x99\xe0\xbf\x90\xe1\x8f\xa0", pAllocator));

  mString start;
  mTEST_ASSERT_SUCCESS(mString_Create(&start, "\xe0\xbc\x80\xe0\xbc\x81\xe0\xbc\x82\xe0\xbc\x83\xe0\xbc\x84\xe0\xbc\x85", pAllocator));

  mString end;
  mTEST_ASSERT_SUCCESS(mString_Create(&end, "\xe0\xb8\xa7\xe0\xb9\x82\xe0\xb8\xa5\xe0\xb8\x81", pAllocator));

  mString startChar;
  mTEST_ASSERT_SUCCESS(mString_Create(&startChar, "\xe0\xbc\x80", pAllocator));

  mString endChar;
  mTEST_ASSERT_SUCCESS(mString_Create(&endChar, "\xe0\xb8\x81", pAllocator));

  mString midChar;
  mTEST_ASSERT_SUCCESS(mString_Create(&midChar, "\xe1\x8f\xa4", pAllocator));

  bool isContained = false;
  mTEST_ASSERT_SUCCESS(mString_Contains(string, empty, &isContained));
  mTEST_ASSERT_TRUE(isContained);

  isContained = false;
  mTEST_ASSERT_SUCCESS(mString_Contains(string, emptyInit, &isContained));
  mTEST_ASSERT_TRUE(isContained);

  isContained = true;
  mTEST_ASSERT_EQUAL(mR_InvalidParameter, mString_Contains(string, invalid, &isContained));
  mTEST_ASSERT_FALSE(isContained);

  isContained = true;
  mTEST_ASSERT_EQUAL(mR_InvalidParameter, mString_Contains(invalid, string, &isContained));
  mTEST_ASSERT_FALSE(isContained);

  isContained = true;
  mTEST_ASSERT_SUCCESS(mString_Contains(string, notContained, &isContained));
  mTEST_ASSERT_FALSE(isContained);

  isContained = true;
  mTEST_ASSERT_SUCCESS(mString_Contains(string, notContainedChar, &isContained));
  mTEST_ASSERT_FALSE(isContained);

  isContained = false;
  mTEST_ASSERT_SUCCESS(mString_Contains(string, contained, &isContained));
  mTEST_ASSERT_TRUE(isContained);

  isContained = false;
  mTEST_ASSERT_SUCCESS(mString_Contains(string, start, &isContained));
  mTEST_ASSERT_TRUE(isContained);

  isContained = false;
  mTEST_ASSERT_SUCCESS(mString_Contains(string, end, &isContained));
  mTEST_ASSERT_TRUE(isContained);

  isContained = false;
  mTEST_ASSERT_SUCCESS(mString_Contains(string, startChar, &isContained));
  mTEST_ASSERT_TRUE(isContained);

  isContained = false;
  mTEST_ASSERT_SUCCESS(mString_Contains(string, endChar, &isContained));
  mTEST_ASSERT_TRUE(isContained);

  isContained = false;
  mTEST_ASSERT_SUCCESS(mString_Contains(string, midChar, &isContained));
  mTEST_ASSERT_TRUE(isContained);

  mTEST_ALLOCATOR_ZERO_CHECK();
}

mTEST(mString, TestFindFirst)
{
  mTEST_ALLOCATOR_SETUP();

  mString string;
  mTEST_ASSERT_SUCCESS(mString_Create(&string, "\xe0\xbc\x80\xe0\xbc\x81\xe0\xbc\x82\xe0\xbc\x83\xe0\xbc\x84\xe0\xbc\x85\xe0\xbc\x86\xe0\xbc\x87\xe0\xbc\x88\xe0\xbc\x89\xe0\xbc\x8a\xe0\xbc\x95\xe0\xbc\x96\xe0\xbc\x97\xe0\xbc\x98\xe0\xbc\x99\xe0\xbf\x90\xe1\x8f\xa0\xe1\x8f\xa1\xe1\x8f\xa2\xe1\x8f\xa3\xe1\x8f\xa4\xe1\x8f\xa5\xe1\x8f\xa6\xe1\x8f\xa7\xe1\x8f\xa8\xe1\x8f\xa9\xe1\x8f\xaa\xe1\x8f\xab\xe1\x8f\xac\xe1\x8f\xad\xe1\x8f\xae\xe1\x8f\xaf\xe0\xb8\xaa\xe0\xb8\xa7\xe0\xb8\xb1\xe0\xb8\xaa\xe0\xb8\x94\xe0\xb8\xb5\xe0\xb8\x8a\xe0\xb8\xb2\xe0\xb8\xa7\xe0\xb9\x82\xe0\xb8\xa5\xe0\xb8\x81", pAllocator));

  mString empty;

  mString emptyInit;
  mTEST_ASSERT_SUCCESS(mString_Create(&emptyInit, "", pAllocator));

  mString invalid;
  invalid.hasFailed = true;

  mString notContained;
  mTEST_ASSERT_SUCCESS(mString_Create(&notContained, "\xe0\xbc\x82\xe0\xbc\x83\xe0\xbc\x84\xe0\xbc\x85\xe1\x8f\xa8", pAllocator));

  mString notContainedChar;
  mTEST_ASSERT_SUCCESS(mString_Create(&notContainedChar, "V", pAllocator));

  mString contained;
  mTEST_ASSERT_SUCCESS(mString_Create(&contained, "\xe0\xbc\x96\xe0\xbc\x97\xe0\xbc\x98\xe0\xbc\x99\xe0\xbf\x90\xe1\x8f\xa0", pAllocator));

  mString start;
  mTEST_ASSERT_SUCCESS(mString_Create(&start, "\xe0\xbc\x80\xe0\xbc\x81\xe0\xbc\x82\xe0\xbc\x83\xe0\xbc\x84\xe0\xbc\x85", pAllocator));

  mString end;
  mTEST_ASSERT_SUCCESS(mString_Create(&end, "\xe0\xb8\xa7\xe0\xb9\x82\xe0\xb8\xa5\xe0\xb8\x81", pAllocator));

  mString startChar;
  mTEST_ASSERT_SUCCESS(mString_Create(&startChar, "\xe0\xbc\x80", pAllocator));

  mString endChar;
  mTEST_ASSERT_SUCCESS(mString_Create(&endChar, "\xe0\xb8\x81", pAllocator));

  mString midChar;
  mTEST_ASSERT_SUCCESS(mString_Create(&midChar, "\xe1\x8f\xa4", pAllocator));

  bool isContained = false;
  size_t index = 0;
  mTEST_ASSERT_SUCCESS(mString_FindFirst(string, empty, &index, &isContained));
  mTEST_ASSERT_TRUE(isContained);
  mTEST_ASSERT_EQUAL(0, index);

  isContained = false;
  mTEST_ASSERT_SUCCESS(mString_FindFirst(string, emptyInit, &index, &isContained));
  mTEST_ASSERT_TRUE(isContained);
  mTEST_ASSERT_EQUAL(0, index);

  isContained = true;
  mTEST_ASSERT_EQUAL(mR_InvalidParameter, mString_FindFirst(string, invalid, &index, &isContained));
  mTEST_ASSERT_FALSE(isContained);

  isContained = true;
  mTEST_ASSERT_EQUAL(mR_InvalidParameter, mString_FindFirst(invalid, string, &index, &isContained));
  mTEST_ASSERT_FALSE(isContained);

  isContained = true;
  mTEST_ASSERT_SUCCESS(mString_FindFirst(string, notContained, &index, &isContained));
  mTEST_ASSERT_FALSE(isContained);

  isContained = true;
  mTEST_ASSERT_SUCCESS(mString_FindFirst(string, notContainedChar, &index, &isContained));
  mTEST_ASSERT_FALSE(isContained);

  isContained = false;
  mTEST_ASSERT_SUCCESS(mString_FindFirst(string, contained, &index, &isContained));
  mTEST_ASSERT_TRUE(isContained);
  
  for (size_t i = 0; i < contained.count - 1; i++)
    mTEST_ASSERT_EQUAL(string[index + i], contained[i]);

  isContained = false;
  mTEST_ASSERT_SUCCESS(mString_FindFirst(string, start, &index, &isContained));
  mTEST_ASSERT_TRUE(isContained);

  for (size_t i = 0; i < start.count - 1; i++)
    mTEST_ASSERT_EQUAL(string[index + i], start[i]);


  isContained = false;
  mTEST_ASSERT_SUCCESS(mString_FindFirst(string, end, &index, &isContained));
  mTEST_ASSERT_TRUE(isContained);

  for (size_t i = 0; i < end.count - 1; i++)
    mTEST_ASSERT_EQUAL(string[index + i], end[i]);


  isContained = false;
  mTEST_ASSERT_SUCCESS(mString_FindFirst(string, startChar, &index, &isContained));
  mTEST_ASSERT_TRUE(isContained);

  for (size_t i = 0; i < startChar.count - 1; i++)
    mTEST_ASSERT_EQUAL(string[index + i], startChar[i]);


  isContained = false;
  mTEST_ASSERT_SUCCESS(mString_FindFirst(string, endChar, &index, &isContained));
  mTEST_ASSERT_TRUE(isContained);

  for (size_t i = 0; i < endChar.count - 1; i++)
    mTEST_ASSERT_EQUAL(string[index + i], endChar[i]);


  isContained = false;
  mTEST_ASSERT_SUCCESS(mString_FindFirst(string, midChar, &index, &isContained));
  mTEST_ASSERT_TRUE(isContained);

  for (size_t i = 0; i < midChar.count - 1; i++)
    mTEST_ASSERT_EQUAL(string[index + i], midChar[i]);


  mTEST_ALLOCATOR_ZERO_CHECK();
}

mTEST(mString, TestTrimStart)
{
  mTEST_ALLOCATOR_SETUP();

  mchar_t trimmedChar = mToChar<2>(" ");
  mchar_t nonTrimmableChar = mToChar<2>("\xe0\xbc\x82");
  char result[] = "\xe1\x8f\xa4 b c    d e \xe0\xb8\x81  ";

  mString string;
  mTEST_ASSERT_SUCCESS(mString_Create(&string, "     \xe1\x8f\xa4 b c    d e \xe0\xb8\x81  ", pAllocator));

  mString trimmedString;
  mTEST_ASSERT_SUCCESS(mString_TrimStart(string, trimmedChar, &trimmedString));
  mTEST_ASSERT_EQUAL(trimmedString, result);

  mTEST_ASSERT_SUCCESS(mString_TrimStart(string, nonTrimmableChar, &trimmedString));
  mTEST_ASSERT_EQUAL(string, trimmedString);

  mTEST_ALLOCATOR_ZERO_CHECK();
}

mTEST(mString, TestTrimEnd)
{
  mTEST_ALLOCATOR_SETUP();

  mchar_t trimmedChar = mToChar<2>(" ");
  mchar_t nonTrimmableChar = mToChar<4>("\xe0\xbc\x82");
  char result[] = "     \xe1\x8f\xa4 b c    d e \xe0\xb8\x81";

  mString string;
  mTEST_ASSERT_SUCCESS(mString_Create(&string, "     \xe1\x8f\xa4 b c    d e \xe0\xb8\x81  ", pAllocator));

  mString trimmedString;
  mTEST_ASSERT_SUCCESS(mString_TrimEnd(string, trimmedChar, &trimmedString));
  mTEST_ASSERT_EQUAL(trimmedString, result);

  mTEST_ASSERT_SUCCESS(mString_TrimEnd(string, nonTrimmableChar, &trimmedString));
  mTEST_ASSERT_EQUAL(string, trimmedString);

  mTEST_ALLOCATOR_ZERO_CHECK();
}

mTEST(mString, TestRemoveChar)
{
  mTEST_ALLOCATOR_SETUP();

  mchar_t replacedChar = mToChar<4>("\xe1\x8f\xa4");
  mchar_t replacedChar2 = mToChar<4>("\xe0\xb8\x81");
  mchar_t nonReplaceableChar = mToChar<2>(" ");
  char result[] = "\x62\x63\x64\xe0\xb8\x81\xe0\xb8\x81\x65\xe0\xb8\x81";
  char result2[] = "\xe1\x8f\xa4\xe1\x8f\xa4\x62\xe1\x8f\xa4\x63\x64\x65\xe1\x8f\xa4";

  mString string;
  mTEST_ASSERT_SUCCESS(mString_Create(&string, "\xe1\x8f\xa4\xe1\x8f\xa4\x62\xe1\x8f\xa4\x63\x64\xe0\xb8\x81\xe0\xb8\x81\x65\xe1\x8f\xa4\xe0\xb8\x81", pAllocator));

  mString resultString;
  mTEST_ASSERT_SUCCESS(mString_RemoveChar(string, replacedChar, &resultString));
  mTEST_ASSERT_EQUAL(resultString, result);

  mTEST_ASSERT_SUCCESS(mString_RemoveChar(string, replacedChar2, &resultString));
  mTEST_ASSERT_EQUAL(resultString, result2);

  mTEST_ASSERT_SUCCESS(mString_RemoveChar(string, nonReplaceableChar, &resultString));
  mTEST_ASSERT_EQUAL(string, resultString);

  mTEST_ALLOCATOR_ZERO_CHECK();
}

mTEST(mString, TestRemoveString)
{
  mTEST_ALLOCATOR_SETUP();

  mString replace;

  mString string;
  mTEST_ASSERT_SUCCESS(mString_Create(&string, "\xE1\xA0\xAC\xE1\xA0\xAD\x68\xE1\x80\xA9\xE1\x8C\xAC\xE1\x80\xA9\xE1\x8C\xAC\xF0\x90\x8C\x86\xE1\xA0\xAC\xE1\xA0\xAD\x65\xE1\x8C\xAC\xE1\xA0\xAC\xE1\xA0\xAD\xE1\x8C\xAC\xE1\xA0\xAC\xE1\xA0\xAD\x72\xE1\x80\xA9\xF0\x94\x93\x98\x67\xE1\xA0\xAC\xE1\xA0\xAD\x68\xF0\x90\x8C\x86\xE1\xA0\xAC\xE1\xA0\xAD\xE1\x80\xA9\xE1\x8C\xAC\xE1\x80\xA9\xF0\x94\x93\x98\xE1\xA0\xAC\xE1\xA0\xAD\x68\xE1\x80\xA9\xE1\x8C\xAC\xF0\x90\x8C\x86\x70\x70\x6c\xE1\x80\xA9\x63\xF0\x90\x8C\x86\xE1\xA0\xAC\xE1\xA0\xAD\xE1\x80\xA9\x6f\xF0\x94\x93\x98\xE1\x80\xA9\xE1\x80\xA9\xE1\x8C\xAC\xE1\x8C\xAC"));

  mString result;
  mTEST_ASSERT_SUCCESS(mString_RemoveString(string, string, &result));
  mTEST_ASSERT_EQUAL(result.count, 1);
  mTEST_ASSERT_EQUAL(result.bytes, 1);
  mTEST_ASSERT_EQUAL(result.text[0], '\0');

  mTEST_ASSERT_SUCCESS(mString_Create(&replace, "\xE1\x80\xA9\xE1\x8C\xAC", pAllocator));
  mTEST_ASSERT_SUCCESS(mString_RemoveString(string, replace, &result));
  mTEST_ASSERT_EQUAL(result, "\xE1\xA0\xAC\xE1\xA0\xAD\x68\xF0\x90\x8C\x86\xE1\xA0\xAC\xE1\xA0\xAD\x65\xE1\x8C\xAC\xE1\xA0\xAC\xE1\xA0\xAD\xE1\x8C\xAC\xE1\xA0\xAC\xE1\xA0\xAD\x72\xE1\x80\xA9\xF0\x94\x93\x98\x67\xE1\xA0\xAC\xE1\xA0\xAD\x68\xF0\x90\x8C\x86\xE1\xA0\xAC\xE1\xA0\xAD\xE1\x80\xA9\xF0\x94\x93\x98\xE1\xA0\xAC\xE1\xA0\xAD\x68\xF0\x90\x8C\x86\x70\x70\x6c\xE1\x80\xA9\x63\xF0\x90\x8C\x86\xE1\xA0\xAC\xE1\xA0\xAD\xE1\x80\xA9\x6f\xF0\x94\x93\x98\xE1\x80\xA9\xE1\x8C\xAC");

  mTEST_ASSERT_SUCCESS(mString_Create(&replace, "\x6f\xF0\x94\x93\x98\xE1\x80\xA9\xE1\x80\xA9\xE1\x8C\xAC\xE1\x8C\xAC", pAllocator));
  mTEST_ASSERT_SUCCESS(mString_RemoveString(string, replace, &result));
  mTEST_ASSERT_EQUAL(result, "\xE1\xA0\xAC\xE1\xA0\xAD\x68\xE1\x80\xA9\xE1\x8C\xAC\xE1\x80\xA9\xE1\x8C\xAC\xF0\x90\x8C\x86\xE1\xA0\xAC\xE1\xA0\xAD\x65\xE1\x8C\xAC\xE1\xA0\xAC\xE1\xA0\xAD\xE1\x8C\xAC\xE1\xA0\xAC\xE1\xA0\xAD\x72\xE1\x80\xA9\xF0\x94\x93\x98\x67\xE1\xA0\xAC\xE1\xA0\xAD\x68\xF0\x90\x8C\x86\xE1\xA0\xAC\xE1\xA0\xAD\xE1\x80\xA9\xE1\x8C\xAC\xE1\x80\xA9\xF0\x94\x93\x98\xE1\xA0\xAC\xE1\xA0\xAD\x68\xE1\x80\xA9\xE1\x8C\xAC\xF0\x90\x8C\x86\x70\x70\x6c\xE1\x80\xA9\x63\xF0\x90\x8C\x86\xE1\xA0\xAC\xE1\xA0\xAD\xE1\x80\xA9");

  mTEST_ASSERT_SUCCESS(mString_Create(&replace, "\xE1\xA0\xAC\xE1\xA0\xAD\x68\xE1\x80\xA9\xE1\x8C\xAC\xE1\x80\xA9\xE1\x8C\xAC\xF0\x90\x8C\x86\xE1\xA0\xAC\xE1\xA0\xAD\x65\xE1\x8C\xAC\xE1\xA0\xAC\xE1\xA0\xAD\xE1\x8C\xAC\xE1\xA0\xAC\xE1\xA0\xAD\x72\xE1\x80\xA9\xF0\x94\x93\x98\x67", pAllocator));
  mTEST_ASSERT_SUCCESS(mString_RemoveString(string, replace, &result));
  mTEST_ASSERT_EQUAL(result, "\xE1\xA0\xAC\xE1\xA0\xAD\x68\xF0\x90\x8C\x86\xE1\xA0\xAC\xE1\xA0\xAD\xE1\x80\xA9\xE1\x8C\xAC\xE1\x80\xA9\xF0\x94\x93\x98\xE1\xA0\xAC\xE1\xA0\xAD\x68\xE1\x80\xA9\xE1\x8C\xAC\xF0\x90\x8C\x86\x70\x70\x6c\xE1\x80\xA9\x63\xF0\x90\x8C\x86\xE1\xA0\xAC\xE1\xA0\xAD\xE1\x80\xA9\x6f\xF0\x94\x93\x98\xE1\x80\xA9\xE1\x80\xA9\xE1\x8C\xAC\xE1\x8C\xAC");
  
  mTEST_ASSERT_SUCCESS(mString_Create(&replace, "\xE1\x80\xA9\xF0\x94\x93\x98\x76\xF0\x90\x8C\x86\x6c\xE1\x80\xA9\x64", pAllocator));
  mTEST_ASSERT_SUCCESS(mString_RemoveString(string, replace, &result));
  mTEST_ASSERT_EQUAL(string, result);

  mTEST_ALLOCATOR_ZERO_CHECK();
}

mTEST(mString, TestReplaceString)
{
  mTEST_ALLOCATOR_SETUP();

  mString replace;
  mString with;

  mString string;
  mTEST_ASSERT_SUCCESS(mString_Create(&string, "\xE1\xA0\xAC\xE1\xA0\xAD\x68\xE1\x80\xA9\xE1\x8C\xAC\xE1\x80\xA9\xE1\x8C\xAC\xF0\x90\x8C\x86\xE1\xA0\xAC\xE1\xA0\xAD\x65\xE1\x8C\xAC\xE1\xA0\xAC\xE1\xA0\xAD\xE1\x8C\xAC\xE1\xA0\xAC\xE1\xA0\xAD\x72\xE1\x80\xA9\xF0\x94\x93\x98\x67\xE1\xA0\xAC\xE1\xA0\xAD\x68\xF0\x90\x8C\x86\xE1\xA0\xAC\xE1\xA0\xAD\xE1\x80\xA9\xE1\x8C\xAC\xE1\x80\xA9\xF0\x94\x93\x98\xE1\xA0\xAC\xE1\xA0\xAD\x68\xE1\x80\xA9\xE1\x8C\xAC\xF0\x90\x8C\x86\x70\x70\x6c\xE1\x80\xA9\x63\xF0\x90\x8C\x86\xE1\xA0\xAC\xE1\xA0\xAD\xE1\x80\xA9\x6f\xF0\x94\x93\x98\xE1\x80\xA9\xE1\x80\xA9\xE1\x8C\xAC\xE1\x8C\xAC"));

  mString result;
  mTEST_ASSERT_SUCCESS(mString_RemoveString(string, string, &result));
  mTEST_ASSERT_EQUAL(result.count, 1);
  mTEST_ASSERT_EQUAL(result.bytes, 1);
  mTEST_ASSERT_EQUAL(result.text[0], '\0');

  mTEST_ASSERT_SUCCESS(mString_Create(&replace, "\xE1\x80\xA9\xE1\x8C\xAC", pAllocator));
  mTEST_ASSERT_SUCCESS(mString_Create(&with, "\xE1\x8F\xAA\xE1\x8F\xAB\xF0\x90\x8C\x86\xF0\x94\x93\x98\xF0\x90\x8C\x86\xF0\x94\x93\x98\xF0\x90\x8C\x86\xE1\x80\xA9\xE1\x8C\xAC\xE1\x8F\xAA\xE1\x8F\xAB\xF0\x90\x8C\x86\xF0\x94\x93\x98\xF0\x90\x8C\x86\xF0\x94\x93\x98\xF0\x90\x8C\x86", pAllocator));
  mTEST_ASSERT_SUCCESS(mString_Replace(string, replace, with, &result));
  mTEST_ASSERT_EQUAL(result, "\xE1\xA0\xAC\xE1\xA0\xAD\x68\xE1\x8F\xAA\xE1\x8F\xAB\xF0\x90\x8C\x86\xF0\x94\x93\x98\xF0\x90\x8C\x86\xF0\x94\x93\x98\xF0\x90\x8C\x86\xE1\x80\xA9\xE1\x8C\xAC\xE1\x8F\xAA\xE1\x8F\xAB\xF0\x90\x8C\x86\xF0\x94\x93\x98\xF0\x90\x8C\x86\xF0\x94\x93\x98\xF0\x90\x8C\x86\xE1\x8F\xAA\xE1\x8F\xAB\xF0\x90\x8C\x86\xF0\x94\x93\x98\xF0\x90\x8C\x86\xF0\x94\x93\x98\xF0\x90\x8C\x86\xE1\x80\xA9\xE1\x8C\xAC\xE1\x8F\xAA\xE1\x8F\xAB\xF0\x90\x8C\x86\xF0\x94\x93\x98\xF0\x90\x8C\x86\xF0\x94\x93\x98\xF0\x90\x8C\x86\xF0\x90\x8C\x86\xE1\xA0\xAC\xE1\xA0\xAD\x65\xE1\x8C\xAC\xE1\xA0\xAC\xE1\xA0\xAD\xE1\x8C\xAC\xE1\xA0\xAC\xE1\xA0\xAD\x72\xE1\x80\xA9\xF0\x94\x93\x98\x67\xE1\xA0\xAC\xE1\xA0\xAD\x68\xF0\x90\x8C\x86\xE1\xA0\xAC\xE1\xA0\xAD\xE1\x8F\xAA\xE1\x8F\xAB\xF0\x90\x8C\x86\xF0\x94\x93\x98\xF0\x90\x8C\x86\xF0\x94\x93\x98\xF0\x90\x8C\x86\xE1\x80\xA9\xE1\x8C\xAC\xE1\x8F\xAA\xE1\x8F\xAB\xF0\x90\x8C\x86\xF0\x94\x93\x98\xF0\x90\x8C\x86\xF0\x94\x93\x98\xF0\x90\x8C\x86\xE1\x80\xA9\xF0\x94\x93\x98\xE1\xA0\xAC\xE1\xA0\xAD\x68\xE1\x8F\xAA\xE1\x8F\xAB\xF0\x90\x8C\x86\xF0\x94\x93\x98\xF0\x90\x8C\x86\xF0\x94\x93\x98\xF0\x90\x8C\x86\xE1\x80\xA9\xE1\x8C\xAC\xE1\x8F\xAA\xE1\x8F\xAB\xF0\x90\x8C\x86\xF0\x94\x93\x98\xF0\x90\x8C\x86\xF0\x94\x93\x98\xF0\x90\x8C\x86\xF0\x90\x8C\x86\x70\x70\x6c\xE1\x80\xA9\x63\xF0\x90\x8C\x86\xE1\xA0\xAC\xE1\xA0\xAD\xE1\x80\xA9\x6f\xF0\x94\x93\x98\xE1\x80\xA9\xE1\x8F\xAA\xE1\x8F\xAB\xF0\x90\x8C\x86\xF0\x94\x93\x98\xF0\x90\x8C\x86\xF0\x94\x93\x98\xF0\x90\x8C\x86\xE1\x80\xA9\xE1\x8C\xAC\xE1\x8F\xAA\xE1\x8F\xAB\xF0\x90\x8C\x86\xF0\x94\x93\x98\xF0\x90\x8C\x86\xF0\x94\x93\x98\xF0\x90\x8C\x86\xE1\x8C\xAC");

  mTEST_ASSERT_SUCCESS(mString_Create(&replace, "\x6f\xF0\x94\x93\x98\xE1\x80\xA9\xE1\x80\xA9\xE1\x8C\xAC\xE1\x8C\xAC", pAllocator));
  mTEST_ASSERT_SUCCESS(mString_Create(&with, "", pAllocator));
  mTEST_ASSERT_SUCCESS(mString_Replace(string, replace, with, &result));
  mTEST_ASSERT_EQUAL(result, "\xE1\xA0\xAC\xE1\xA0\xAD\x68\xE1\x80\xA9\xE1\x8C\xAC\xE1\x80\xA9\xE1\x8C\xAC\xF0\x90\x8C\x86\xE1\xA0\xAC\xE1\xA0\xAD\x65\xE1\x8C\xAC\xE1\xA0\xAC\xE1\xA0\xAD\xE1\x8C\xAC\xE1\xA0\xAC\xE1\xA0\xAD\x72\xE1\x80\xA9\xF0\x94\x93\x98\x67\xE1\xA0\xAC\xE1\xA0\xAD\x68\xF0\x90\x8C\x86\xE1\xA0\xAC\xE1\xA0\xAD\xE1\x80\xA9\xE1\x8C\xAC\xE1\x80\xA9\xF0\x94\x93\x98\xE1\xA0\xAC\xE1\xA0\xAD\x68\xE1\x80\xA9\xE1\x8C\xAC\xF0\x90\x8C\x86\x70\x70\x6c\xE1\x80\xA9\x63\xF0\x90\x8C\x86\xE1\xA0\xAC\xE1\xA0\xAD\xE1\x80\xA9");

  mTEST_ASSERT_SUCCESS(mString_Create(&replace, "\xE1\xA0\xAC\xE1\xA0\xAD\x68\xE1\x80\xA9\xE1\x8C\xAC\xE1\x80\xA9\xE1\x8C\xAC\xF0\x90\x8C\x86\xE1\xA0\xAC\xE1\xA0\xAD\x65\xE1\x8C\xAC\xE1\xA0\xAC\xE1\xA0\xAD\xE1\x8C\xAC\xE1\xA0\xAC\xE1\xA0\xAD\x72\xE1\x80\xA9\xF0\x94\x93\x98\x67"));
  mTEST_ASSERT_SUCCESS(mString_Replace(string, replace, with, &result));
  mTEST_ASSERT_EQUAL(result, "\xE1\xA0\xAC\xE1\xA0\xAD\x68\xF0\x90\x8C\x86\xE1\xA0\xAC\xE1\xA0\xAD\xE1\x80\xA9\xE1\x8C\xAC\xE1\x80\xA9\xF0\x94\x93\x98\xE1\xA0\xAC\xE1\xA0\xAD\x68\xE1\x80\xA9\xE1\x8C\xAC\xF0\x90\x8C\x86\x70\x70\x6c\xE1\x80\xA9\x63\xF0\x90\x8C\x86\xE1\xA0\xAC\xE1\xA0\xAD\xE1\x80\xA9\x6f\xF0\x94\x93\x98\xE1\x80\xA9\xE1\x80\xA9\xE1\x8C\xAC\xE1\x8C\xAC");
  
  mTEST_ASSERT_SUCCESS(mString_Create(&replace, "\xE1\x80\xA9\xF0\x94\x93\x98\x76\xF0\x90\x8C\x86\x6c\xE1\x80\xA9\x64"));
  mTEST_ASSERT_SUCCESS(mString_Replace(string, replace, with, &result));
  mTEST_ASSERT_EQUAL(string, result);

  mTEST_ALLOCATOR_ZERO_CHECK();
}
