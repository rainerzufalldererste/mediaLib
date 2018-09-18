// Copyright 2018 Christoph Stiller
// 
// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files(the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions :
// 
// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
// 
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "mTestLib.h"
#include "mString.h"

mTEST(mString, TestCreateEmpty)
{
  mTEST_ALLOCATOR_SETUP();

  mString string;
  mDEFER_DESTRUCTION(&string, mString_Destroy);
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
  mDEFER_DESTRUCTION(&string, mString_Destroy);
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
  mDEFER_DESTRUCTION(&string, mString_Destroy);
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
  mDEFER_DESTRUCTION(&string, mString_Destroy);
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
  mDEFER_DESTRUCTION(&string, mString_Destroy);
  mTEST_ASSERT_SUCCESS(mString_Create(&string, "🌵🦎🎅testÿⴲ", pAllocator));

  size_t count;
  mTEST_ASSERT_SUCCESS(mString_GetCount(string, &count));
  mTEST_ASSERT_EQUAL(count, 3 + 4 + 2 + 1);

  size_t size;
  mTEST_ASSERT_SUCCESS(mString_GetByteSize(string, &size));
  mTEST_ASSERT_EQUAL(size, 12 + 4 + 2 + 3 + 1);

  mTEST_ALLOCATOR_ZERO_CHECK();
}

mTEST(mString, TestAt)
{
  mTEST_ALLOCATOR_SETUP();

  mString string;
  mDEFER_DESTRUCTION(&string, mString_Destroy);
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
  mDEFER_DESTRUCTION(&string, mString_Destroy);
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
  mDEFER_DESTRUCTION(&string, mString_Destroy);
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
  mDEFER_DESTRUCTION(&string, mString_Destroy);
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
  mDEFER_DESTRUCTION(&string, mString_Destroy);
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
  mDEFER_DESTRUCTION(&string, mString_Destroy);
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
  mDEFER_DESTRUCTION(&string, mString_Destroy);
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
  mDEFER_DESTRUCTION(&string, mString_Destroy);
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
  mDEFER_DESTRUCTION(&string, mString_Destroy);
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
  mDEFER_DESTRUCTION(&string, mString_Destroy);
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
  mDEFER_DESTRUCTION(&string, mString_Destroy);
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
  mDEFER_DESTRUCTION(&string, mString_Destroy);
  mTEST_ASSERT_SUCCESS(mString_Create(&string, "🌵🦎🎅testמⴲx", pAllocator));

  mString subString;
  mDEFER_DESTRUCTION(&subString, mString_Destroy);
  mTEST_ASSERT_SUCCESS(mString_Substring(string, &subString, 2, 7));

  mString comparisonString;
  mDEFER_DESTRUCTION(&comparisonString, mString_Destroy);
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
  mDEFER_DESTRUCTION(&stringA, mString_Destroy);
  mTEST_ASSERT_SUCCESS(mString_Create(&stringA, "", pAllocator));

  mString stringB;
  mDEFER_DESTRUCTION(&stringB, mString_Destroy);
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
  mDEFER_DESTRUCTION(&string, mString_Destroy);
  mTEST_ASSERT_SUCCESS(mString_Create(&string, "C:/Some Folder🌵//Path/\\p", pAllocator));

  mString path;
  mDEFER_DESTRUCTION(&path, mString_Destroy);
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
  mDEFER_DESTRUCTION(&string, mString_Destroy);
  mTEST_ASSERT_SUCCESS(mString_Create(&string, "C:/Some Folder//Path/\\p", pAllocator));

  mString path;
  mDEFER_DESTRUCTION(&path, mString_Destroy);
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
