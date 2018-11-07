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

mTEST(mString, TestCreateFromWchar)
{
  mTEST_ALLOCATOR_SETUP();

  mString string;
  mDEFER_CALL(&string, mString_Destroy);
  mTEST_ASSERT_SUCCESS(mString_Create(&string, L"Testמ", pAllocator));

  size_t count;
  mTEST_ASSERT_SUCCESS(mString_GetCount(string, &count));
  mTEST_ASSERT_EQUAL(count, 4 + 1 + 1);

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

  std::wstring wstring0;
  std::wstring wstring1;

  mTEST_ASSERT_SUCCESS(mString_ToWideString(string, &wstring0));
  wstring1 = (std::wstring)string;

  mTEST_ASSERT_EQUAL(wstring0.size(), wstring1.size());

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
