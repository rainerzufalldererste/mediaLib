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

  mTEST_RETURN_SUCCESS();
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

  mTEST_RETURN_SUCCESS();
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

  mTEST_RETURN_SUCCESS();
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

  mTEST_RETURN_SUCCESS();
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

  mTEST_RETURN_SUCCESS();
}

mTEST(mInplaceString, TestCreateFromCharPtrSize)
{
  mInplaceString<128> string;
  char text[32] = "🌵🦎🎅test";
  mTEST_ASSERT_SUCCESS(mInplaceString_Create(&string, (char *)text, mARRAYSIZE(text)));

  size_t count;
  mTEST_ASSERT_SUCCESS(mInplaceString_GetCount(string, &count));
  mTEST_ASSERT_EQUAL(count, 3 + 4 + 1);

  size_t size;
  mTEST_ASSERT_SUCCESS(mInplaceString_GetByteSize(string, &size));
  mTEST_ASSERT_EQUAL(size, 12 + 4 + 1);

  mTEST_RETURN_SUCCESS();
}

mTEST(mInplaceString, TestCastTo_mString)
{
  mTEST_ALLOCATOR_SETUP();

  mInplaceString<128> string;
  char text[] = "🌵🦎🎅test";
  mTEST_ASSERT_SUCCESS(mInplaceString_Create(&string, (char *)text, mARRAYSIZE(text)));

  mString mstring;
  mDEFER_CALL(&mstring, mString_Destroy);
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

mTEST(mInplaceString, TestCreateTooLong)
{
  mInplaceString<5> string;
  char text[] = "🌵🦎🎅testמⴲ";
  mTEST_ASSERT_EQUAL(mR_ArgumentOutOfBounds, mInplaceString_CreateRaw(&string, (char *)text));
  mTEST_ASSERT_EQUAL(mR_ArgumentOutOfBounds, mInplaceString_Create(&string, text));
  mTEST_ASSERT_EQUAL(mR_ArgumentOutOfBounds, mInplaceString_Create(&string, (mString)text));
  mTEST_ASSERT_EQUAL(mR_ArgumentOutOfBounds, mInplaceString_Create(&string, text, mARRAYSIZE(text)));

  mTEST_RETURN_SUCCESS();
}

mTEST(mInplaceString, TestSet)
{
  mInplaceString<128> stringA;
  mInplaceString<128> stringB;
  mInplaceString<128> stringC;
  
  char text[] = "🌵🦎🎅testמⴲ";
  mTEST_ASSERT_SUCCESS(mInplaceString_Create(&stringA, (char *)text, mARRAYSIZE(text)));
  mTEST_ASSERT_SUCCESS(mInplaceString_Create(&stringC, (char *)text, mARRAYSIZE(text)));

  stringB = stringA;
  mTEST_ASSERT_EQUAL(stringA, stringB);
  stringB = std::move(stringA);
  mTEST_ASSERT_EQUAL(stringA, stringB);

  mTEST_ASSERT_SUCCESS(mInplaceString_Create(&stringA, (char *)text, mARRAYSIZE(text)));

  new (&stringB) mInplaceString<128>(stringA);
  mTEST_ASSERT_EQUAL(stringC, stringB);

  new (&stringB) mInplaceString<128>(std::move(stringA));
  mTEST_ASSERT_EQUAL(stringC, stringB);

  mTEST_RETURN_SUCCESS();
}

mTEST(mInplaceString, TestEqual)
{
  mInplaceString<255> stringA;
  mInplaceString<255> stringB;

  mTEST_ASSERT_SUCCESS(mInplaceString_Create(&stringA, "C:/Windows/Fonts/seguiemj.ttf"));
  mTEST_ASSERT_SUCCESS(mInplaceString_Create(&stringB, stringA));
  mTEST_ASSERT_EQUAL(stringA, stringB);

  mTEST_RETURN_SUCCESS();
}

mTEST(mInplaceString, TestIterate)
{
  mTEST_ALLOCATOR_SETUP();

  mInplaceString<255> string;
  mTEST_ASSERT_SUCCESS(mInplaceString_Create(&string, "🌵🦎🎅testמⴲx"));

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

  mTEST_ASSERT_EQUAL(string.count - 1, count); // we don't care about the '\0' character.

  mTEST_ALLOCATOR_ZERO_CHECK();
}

mTEST(mString, TestAppendInplaceString)
{
  mTEST_ALLOCATOR_SETUP();

  mString string;
  mDEFER_CALL(&string, mString_Destroy);
  mTEST_ASSERT_SUCCESS(mString_Create(&string, "🌵🦎🎅test", pAllocator));

  mInplaceString<0xF> appended;
  mTEST_ASSERT_SUCCESS(mInplaceString_Create(&appended, "🦎T"));

  mTEST_ASSERT_SUCCESS(mString_Append(string, appended));

  size_t count;
  mTEST_ASSERT_SUCCESS(mString_GetCount(string, &count));
  mTEST_ASSERT_EQUAL(count, 4 + 5 + 1);

  size_t size;
  mTEST_ASSERT_SUCCESS(mString_GetByteSize(string, &size));
  mTEST_ASSERT_EQUAL(size, 16 + 5 + 1);

  mTEST_ALLOCATOR_ZERO_CHECK();
}
