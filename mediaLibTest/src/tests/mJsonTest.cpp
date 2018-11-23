#include "mTestLib.h"
#include "mJson.h"

mTEST(mJsonWriter, TestCreate)
{
  mTEST_ALLOCATOR_SETUP();

  mPtr<mJsonWriter> jsonWriter;
  mDEFER_CALL(&jsonWriter, mJsonWriter_Destroy);

  mTEST_ASSERT_SUCCESS(mJsonWriter_Create(&jsonWriter, pAllocator));
  mTEST_ASSERT_NOT_EQUAL(jsonWriter, nullptr);

  mTEST_ALLOCATOR_ZERO_CHECK();
}

mTEST(mJsonWriter, TestCleanupDouble)
{
  mTEST_ALLOCATOR_SETUP();

  mPtr<mJsonWriter> jsonWriter;
  mDEFER_CALL(&jsonWriter, mJsonWriter_Destroy);

  mTEST_ASSERT_SUCCESS(mJsonWriter_Create(&jsonWriter, pAllocator));

  const char *name = "a";
  mTEST_ASSERT_SUCCESS(mJsonWriter_BeginArray(jsonWriter, name));

  for (size_t i = 0; i < 1024 * 8; i++)
  {
    mTEST_ASSERT_SUCCESS(mJsonWriter_AddArrayValue(jsonWriter, (double_t)i));
  }

  mTEST_ASSERT_SUCCESS(mJsonWriter_EndArray(jsonWriter));

  mTEST_ALLOCATOR_ZERO_CHECK();
}

mTEST(mJsonWriter, TestCleanupMString)
{
  mTEST_ALLOCATOR_SETUP();

  mPtr<mJsonWriter> jsonWriter;
  mDEFER_CALL(&jsonWriter, mJsonWriter_Destroy);

  mTEST_ASSERT_SUCCESS(mJsonWriter_Create(&jsonWriter, pAllocator));

  const char *name = "a";
  mTEST_ASSERT_SUCCESS(mJsonWriter_BeginArray(jsonWriter, name));

  for (size_t i = 0; i < 1024 * 8; i++)
  {
    mString string;
    mDEFER_CALL(&string, mString_Destroy);
    mTEST_ASSERT_SUCCESS(mString_CreateFormat(&string, pAllocator, "%" PRIu64, i));
    mTEST_ASSERT_SUCCESS(mJsonWriter_AddArrayValue(jsonWriter, string));
  }

  mTEST_ASSERT_SUCCESS(mJsonWriter_EndArray(jsonWriter));

  mTEST_ALLOCATOR_ZERO_CHECK();
}

mTEST(mJsonWriter, TestCleanupCString)
{
  mTEST_ALLOCATOR_SETUP();

  mPtr<mJsonWriter> jsonWriter;
  mDEFER_CALL(&jsonWriter, mJsonWriter_Destroy);
  mTEST_ASSERT_SUCCESS(mJsonWriter_Create(&jsonWriter, pAllocator));

  const char *name = "a";
  char testString[255];
  mTEST_ASSERT_SUCCESS(mJsonWriter_BeginArray(jsonWriter, name));

  for (size_t i = 0; i < 1024 * 8; i++)
  {
    sprintf_s(testString, "%" PRIu64, i);
    mTEST_ASSERT_SUCCESS(mJsonWriter_AddArrayValue(jsonWriter, testString));
  }

  mTEST_ASSERT_SUCCESS(mJsonWriter_EndArray(jsonWriter));

  mTEST_ALLOCATOR_ZERO_CHECK();
}

mTEST(mJsonWriter, TestCleanupBool)
{
  mTEST_ALLOCATOR_SETUP();

  mPtr<mJsonWriter> jsonWriter;
  mDEFER_CALL(&jsonWriter, mJsonWriter_Destroy);

  mTEST_ASSERT_SUCCESS(mJsonWriter_Create(&jsonWriter, pAllocator));

  const char *name = "a";
  mTEST_ASSERT_SUCCESS(mJsonWriter_BeginArray(jsonWriter, name));

  for (size_t i = 0; i < 1024 * 8; i++)
  {
    mTEST_ASSERT_SUCCESS(mJsonWriter_AddArrayValue(jsonWriter, i % 2 == 0));
  }

  mTEST_ASSERT_SUCCESS(mJsonWriter_EndArray(jsonWriter));

  mTEST_ALLOCATOR_ZERO_CHECK();
}

mTEST(mJsonWriter, TestAddNamed)
{
  mTEST_ALLOCATOR_SETUP();

  mPtr<mJsonWriter> jsonWriter;
  mDEFER_CALL(&jsonWriter, mJsonWriter_Destroy);

  mTEST_ASSERT_SUCCESS(mJsonWriter_Create(&jsonWriter, pAllocator));

  const char *name = "testString";

  mString string;
  mDEFER_CALL(&string, mString_Destroy);
  mTEST_ASSERT_SUCCESS(mString_Create(&string, name, pAllocator));

  mTEST_ASSERT_SUCCESS(mJsonWriter_BeginNamed(jsonWriter, name));

  mTEST_ASSERT_SUCCESS(mJsonWriter_AddValue(jsonWriter, "double 1", 1.0));
  mTEST_ASSERT_SUCCESS(mJsonWriter_AddValue(jsonWriter, "double 2", 0.0));
  mTEST_ASSERT_SUCCESS(mJsonWriter_AddValue(jsonWriter, "double 3", -1.0));
  mTEST_ASSERT_SUCCESS(mJsonWriter_AddValue(jsonWriter, "double 4", 0.1));
  mTEST_ASSERT_SUCCESS(mJsonWriter_AddValue(jsonWriter, "double 5", -0.1));
  mTEST_ASSERT_SUCCESS(mJsonWriter_AddValue(jsonWriter, "double Infinity", DBL_MIN));
  mTEST_ASSERT_SUCCESS(mJsonWriter_AddValue(jsonWriter, "double Infinity", DBL_MAX));
  mTEST_ASSERT_SUCCESS(mJsonWriter_AddValue(jsonWriter, "double epislon", DBL_EPSILON));
  mTEST_ASSERT_SUCCESS(mJsonWriter_AddValue(jsonWriter, "mString", string));
  mTEST_ASSERT_SUCCESS(mJsonWriter_AddValue(jsonWriter, "string", name));
  mTEST_ASSERT_SUCCESS(mJsonWriter_AddValue(jsonWriter, "boolean true", true));
  mTEST_ASSERT_SUCCESS(mJsonWriter_AddValue(jsonWriter, "boolean false", false));
  mTEST_ASSERT_SUCCESS(mJsonWriter_AddValueX(jsonWriter, "Vec2f", mVec2f(1, 0)));
  mTEST_ASSERT_SUCCESS(mJsonWriter_AddValueX(jsonWriter, "Vec3f", mVec3f(1, 0, 1)));
  mTEST_ASSERT_SUCCESS(mJsonWriter_AddValueX(jsonWriter, "vec4f", mVec4f(0, 1, 0, 1)));
  mTEST_ASSERT_SUCCESS(mJsonWriter_AddValue(jsonWriter, "nullptr", nullptr));

  mTEST_ASSERT_SUCCESS(mJsonWriter_EndNamed(jsonWriter));

  mTEST_ALLOCATOR_ZERO_CHECK();
}

mTEST(mJsonWriter, TestAddArray)
{
  mTEST_ALLOCATOR_SETUP();

  mPtr<mJsonWriter> jsonWriter;
  mDEFER_CALL(&jsonWriter, mJsonWriter_Destroy);

  mTEST_ASSERT_SUCCESS(mJsonWriter_Create(&jsonWriter, pAllocator));

  const char *name = "testString";

  mString string;
  mDEFER_CALL(&string, mString_Destroy);
  mTEST_ASSERT_SUCCESS(mString_Create(&string, name, pAllocator));

  mTEST_ASSERT_SUCCESS(mJsonWriter_BeginArray(jsonWriter, name));

  mTEST_ASSERT_SUCCESS(mJsonWriter_AddArrayValue(jsonWriter, 1.0));
  mTEST_ASSERT_SUCCESS(mJsonWriter_AddArrayValue(jsonWriter, 0.0));
  mTEST_ASSERT_SUCCESS(mJsonWriter_AddArrayValue(jsonWriter, -1.0));
  mTEST_ASSERT_SUCCESS(mJsonWriter_AddArrayValue(jsonWriter, 0.1));
  mTEST_ASSERT_SUCCESS(mJsonWriter_AddArrayValue(jsonWriter, -0.1));
  mTEST_ASSERT_SUCCESS(mJsonWriter_AddArrayValue(jsonWriter, DBL_MIN));
  mTEST_ASSERT_SUCCESS(mJsonWriter_AddArrayValue(jsonWriter, DBL_MAX));
  mTEST_ASSERT_SUCCESS(mJsonWriter_AddArrayValue(jsonWriter, DBL_EPSILON));
  mTEST_ASSERT_SUCCESS(mJsonWriter_AddArrayValue(jsonWriter, string));
  mTEST_ASSERT_SUCCESS(mJsonWriter_AddArrayValue(jsonWriter, name));
  mTEST_ASSERT_SUCCESS(mJsonWriter_AddArrayValue(jsonWriter, true));
  mTEST_ASSERT_SUCCESS(mJsonWriter_AddArrayValue(jsonWriter, false));
  mTEST_ASSERT_SUCCESS(mJsonWriter_AddArrayValueX(jsonWriter, mVec2f(1, 0)));
  mTEST_ASSERT_SUCCESS(mJsonWriter_AddArrayValueX(jsonWriter, mVec3f(1, 0, 1)));
  mTEST_ASSERT_SUCCESS(mJsonWriter_AddArrayValueX(jsonWriter, mVec4f(0, 1, 0, 1)));
  mTEST_ASSERT_SUCCESS(mJsonWriter_AddArrayValue(jsonWriter, nullptr));

  mTEST_ASSERT_SUCCESS(mJsonWriter_EndArray(jsonWriter));

  mTEST_ALLOCATOR_ZERO_CHECK();
}

mTEST(mJsonWriter, TestQuotationLevels)
{
  mTEST_ALLOCATOR_SETUP();

  mPtr<mJsonWriter> jsonWriter;
  mDEFER_CALL(&jsonWriter, mJsonWriter_Destroy);

  mTEST_ASSERT_SUCCESS(mJsonWriter_Create(&jsonWriter, pAllocator));

  const char *name = "test";

  mString string;
  mDEFER_CALL(&string, mString_Destroy);
  mTEST_ASSERT_SUCCESS(mString_Create(&string, name, pAllocator));

  char tempString[255];
  mTEST_ASSERT_SUCCESS(mJsonWriter_BeginNamed(jsonWriter, name));

  for (size_t i = 0; i < 4; i++)
  {
    sprintf_s(tempString, "%" PRIu64, i);
    mTEST_ASSERT_SUCCESS(mJsonWriter_AddValue(jsonWriter, tempString, 1.0));
    sprintf_s(tempString, "%g", (double_t)i);
    mTEST_ASSERT_SUCCESS(mJsonWriter_BeginNamed(jsonWriter, tempString));
  }

  mTEST_ASSERT_SUCCESS(mJsonWriter_BeginArray(jsonWriter, name));

  mTEST_ASSERT_SUCCESS(mJsonWriter_AddArrayValue(jsonWriter, 1.0));
  mTEST_ASSERT_SUCCESS(mJsonWriter_AddArrayValue(jsonWriter, 0.0));
  mTEST_ASSERT_SUCCESS(mJsonWriter_AddArrayValue(jsonWriter, -1.0));
  mTEST_ASSERT_SUCCESS(mJsonWriter_AddArrayValue(jsonWriter, 0.1));
  mTEST_ASSERT_SUCCESS(mJsonWriter_AddArrayValue(jsonWriter, -0.1));
  mTEST_ASSERT_SUCCESS(mJsonWriter_AddArrayValue(jsonWriter, DBL_MIN));
  mTEST_ASSERT_SUCCESS(mJsonWriter_AddArrayValue(jsonWriter, DBL_MAX));
  mTEST_ASSERT_SUCCESS(mJsonWriter_AddArrayValue(jsonWriter, DBL_EPSILON));
  mTEST_ASSERT_SUCCESS(mJsonWriter_AddArrayValue(jsonWriter, string));
  mTEST_ASSERT_SUCCESS(mJsonWriter_AddArrayValue(jsonWriter, name));
  mTEST_ASSERT_SUCCESS(mJsonWriter_AddArrayValue(jsonWriter, true));
  mTEST_ASSERT_SUCCESS(mJsonWriter_AddArrayValue(jsonWriter, false));
  mTEST_ASSERT_SUCCESS(mJsonWriter_AddArrayValueX(jsonWriter, mVec2f(1, 0)));
  mTEST_ASSERT_SUCCESS(mJsonWriter_AddArrayValueX(jsonWriter, mVec3f(1, 0, 1)));
  mTEST_ASSERT_SUCCESS(mJsonWriter_AddArrayValueX(jsonWriter, mVec4f(0, 1, 0, 1)));
  mTEST_ASSERT_SUCCESS(mJsonWriter_AddArrayValue(jsonWriter, nullptr));

  for (size_t i = 0; i < 4; i++)
  {
    mTEST_ASSERT_SUCCESS(mJsonWriter_EndArray(jsonWriter));
  }

  mTEST_ASSERT_SUCCESS(mJsonWriter_EndArray(jsonWriter));

  mTEST_ALLOCATOR_ZERO_CHECK();
}

mTEST(mJsonReader, CreateFromString)
{
  mTEST_ALLOCATOR_SETUP();

  mPtr<mJsonReader> jsonReader;
  mDEFER_CALL(&jsonReader, mJsonReader_Destroy);

  const char *text = "\"a\" : { x : 1 }";
  mTEST_ASSERT_SUCCESS(mJsonReader_CreateFromString(&jsonReader, pAllocator, text));

  mTEST_ALLOCATOR_ZERO_CHECK();
}

mTEST(mJsonReader, TestReadVecs)
{
  mTEST_ALLOCATOR_SETUP();

  mPtr<mJsonReader> jsonReader;
  mDEFER_CALL(&jsonReader, mJsonReader_Destroy);

  const char *text = "{\"a\" : [11,12], \"b\" : [11,12,13], \"c\" : [11,12,13,14], \"d\" : [[21,22],[21,22,23],[21,22,23,24]], \"e\" : { \"ae\" : 1, \"be\" : true, \"ce\" : \"abcd\", \"de\" : \"dcba\", \"ee\" : [[31,32],[31,32,33],[31,32,33,34]] } }";
  mTEST_ASSERT_SUCCESS(mJsonReader_CreateFromString(&jsonReader, pAllocator, text));

  mVec2f v2;
  mVec3f v3;
  mVec4f v4;

  mTEST_ASSERT_SUCCESS(mJsonReader_ReadNamedValue(jsonReader, "b", &v3));
  mTEST_ASSERT_SUCCESS(mJsonReader_ReadNamedValue(jsonReader, "a", &v2));
  mTEST_ASSERT_SUCCESS(mJsonReader_ReadNamedValue(jsonReader, "c", &v4));

  mTEST_ASSERT_SUCCESS(mJsonReader_StepIntoArray(jsonReader, "d"));

  mTEST_ASSERT_SUCCESS(mJsonReader_ReadArrayValue(jsonReader, 2, &v4));
  mTEST_ASSERT_SUCCESS(mJsonReader_ReadArrayValue(jsonReader, 0, &v2));
  mTEST_ASSERT_SUCCESS(mJsonReader_ReadArrayValue(jsonReader, 1, &v3));

  mTEST_ASSERT_SUCCESS(mJsonReader_ExitArray(jsonReader));

  mTEST_ALLOCATOR_ZERO_CHECK();
}

mTEST(mJsonReader, TestRead)
{
  mTEST_ALLOCATOR_SETUP();

  mPtr<mJsonReader> jsonReader;
  mDEFER_CALL(&jsonReader, mJsonReader_Destroy);

  const char *text = "{\"a\" : [11,12], \"b\" : [11,12,13], \"c\" : [11,12,13,14], \"d\" : [[21,22],[21,22,23],[21,22,23,24]], \"e\" : { \"ae\" : 1, \"be\" : true, \"ce\" : \"abcd\", \"de\" : \"dcba\", \"ee\" : [[31,32],[31,32,33],[31,32,33,34]] } }";

  mVec2f v2;
  mVec3f v3;
  mVec4f v4;

  double_t d;
  bool b;

  mString string;
  mDEFER_CALL(&string, mString_Destroy);

  mTEST_ASSERT_SUCCESS(mJsonReader_CreateFromString(&jsonReader, pAllocator, text));

  mTEST_ASSERT_SUCCESS(mJsonReader_ReadNamedValue(jsonReader, "a", &v2));
  mTEST_ASSERT_SUCCESS(mJsonReader_ReadNamedValue(jsonReader, "b", &v3));
  mTEST_ASSERT_SUCCESS(mJsonReader_ReadNamedValue(jsonReader, "c", &v4));

  mTEST_ASSERT_SUCCESS(mJsonReader_StepIntoArray(jsonReader, "d"));

  mTEST_ASSERT_SUCCESS(mJsonReader_ReadArrayValue(jsonReader, 0, &v2));
  mTEST_ASSERT_SUCCESS(mJsonReader_ReadArrayValue(jsonReader, 1, &v3));
  mTEST_ASSERT_SUCCESS(mJsonReader_ReadArrayValue(jsonReader, 2, &v4));

  mTEST_ASSERT_SUCCESS(mJsonReader_ExitArray(jsonReader));

  mTEST_ASSERT_SUCCESS(mJsonReader_StepIntoNamed(jsonReader, "e"));
  mTEST_ASSERT_SUCCESS(mJsonReader_ReadNamedValue(jsonReader, "ae", &d));
  mTEST_ASSERT_SUCCESS(mJsonReader_ReadNamedValue(jsonReader, "be", &b));
  mTEST_ASSERT_SUCCESS(mJsonReader_ReadNamedValue(jsonReader, "ce", &string));

  mTEST_ASSERT_SUCCESS(mJsonReader_ReadNamedValue(jsonReader, "de", &string));

  mTEST_ASSERT_SUCCESS(mJsonReader_StepIntoArray(jsonReader, "ee"));
  mTEST_ASSERT_SUCCESS(mJsonReader_ReadArrayValue(jsonReader, 0, &v2));
  mTEST_ASSERT_SUCCESS(mJsonReader_ReadArrayValue(jsonReader, 1, &v3));
  mTEST_ASSERT_SUCCESS(mJsonReader_ReadArrayValue(jsonReader, 2, &v4));
  mTEST_ASSERT_SUCCESS(mJsonReader_ExitArray(jsonReader));

  mTEST_ALLOCATOR_ZERO_CHECK();
}

mTEST(mJsonReader, TestWriteAndRead)
{
  mTEST_ALLOCATOR_SETUP();

  mVec2f v2;
  mVec3f v3;
  mVec4f v4;

  double_t d;

  mString string;
  mDEFER_CALL(&string, mString_Destroy);

  mPtr<mJsonWriter> jsonWriter;
  mDEFER_CALL(&jsonWriter, mJsonWriter_Destroy);
  mTEST_ASSERT_SUCCESS(mJsonWriter_Create(&jsonWriter, pAllocator));

  mTEST_ASSERT_SUCCESS(mJsonWriter_BeginNamed(jsonWriter, "obj1"));
  mTEST_ASSERT_SUCCESS(mJsonWriter_AddValue(jsonWriter, "a", 1.1));
  mTEST_ASSERT_SUCCESS(mJsonWriter_AddValue(jsonWriter, "b", 2.2));
  mTEST_ASSERT_SUCCESS(mJsonWriter_AddValue(jsonWriter, "c", 3.3));
  mTEST_ASSERT_SUCCESS(mJsonWriter_AddValue(jsonWriter, "s", "abcd"));
  mTEST_ASSERT_SUCCESS(mJsonWriter_EndNamed(jsonWriter));

  mTEST_ASSERT_SUCCESS(mJsonWriter_BeginNamed(jsonWriter, "obj2"));
  mTEST_ASSERT_SUCCESS(mJsonWriter_AddValue(jsonWriter, "aaaa", 1.1));
  mTEST_ASSERT_SUCCESS(mJsonWriter_AddValue(jsonWriter, "bbbb", 2.2));
  mTEST_ASSERT_SUCCESS(mJsonWriter_AddValue(jsonWriter, "cccc", 3.3));
  mTEST_ASSERT_SUCCESS(mJsonWriter_AddValue(jsonWriter, "ssss", "abcd"));
  mTEST_ASSERT_SUCCESS(mJsonWriter_EndNamed(jsonWriter));

  mTEST_ASSERT_SUCCESS(mJsonWriter_BeginArray(jsonWriter, "vectors"));
  mTEST_ASSERT_SUCCESS(mJsonWriter_AddArrayValueX(jsonWriter, mVec2f(1, 1)));
  mTEST_ASSERT_SUCCESS(mJsonWriter_AddArrayValueX(jsonWriter, mVec3f(-1, 0, 1)));
  mTEST_ASSERT_SUCCESS(mJsonWriter_AddArrayValueX(jsonWriter, mVec4f(10, -1, 1, 10)));
  mTEST_ASSERT_SUCCESS(mJsonWriter_EndArray(jsonWriter));

  mTEST_ASSERT_SUCCESS(mJsonWriter_BeginNamed(jsonWriter, "obj3"));

  mTEST_ASSERT_SUCCESS(mJsonWriter_BeginArray(jsonWriter, "o3v2"));
  mTEST_ASSERT_SUCCESS(mJsonWriter_AddArrayValueX(jsonWriter, mVec2f(0, 1)));
  mTEST_ASSERT_SUCCESS(mJsonWriter_AddArrayValueX(jsonWriter, mVec2f(1, 1)));
  mTEST_ASSERT_SUCCESS(mJsonWriter_EndArray(jsonWriter));

  mTEST_ASSERT_SUCCESS(mJsonWriter_BeginArray(jsonWriter, "o3v3"));
  mTEST_ASSERT_SUCCESS(mJsonWriter_AddArrayValueX(jsonWriter, mVec3f(0, 1, 2)));
  mTEST_ASSERT_SUCCESS(mJsonWriter_AddArrayValueX(jsonWriter, mVec3f(1, 1, 2)));
  mTEST_ASSERT_SUCCESS(mJsonWriter_AddArrayValueX(jsonWriter, mVec3f(2, 1, 2)));
  mTEST_ASSERT_SUCCESS(mJsonWriter_EndArray(jsonWriter));

  mTEST_ASSERT_SUCCESS(mJsonWriter_BeginArray(jsonWriter, "o3v4"));
  mTEST_ASSERT_SUCCESS(mJsonWriter_AddArrayValueX(jsonWriter, mVec4f(0, 1, 2, 3)));
  mTEST_ASSERT_SUCCESS(mJsonWriter_AddArrayValueX(jsonWriter, mVec4f(1, 1, 2, 3)));
  mTEST_ASSERT_SUCCESS(mJsonWriter_AddArrayValueX(jsonWriter, mVec4f(2, 1, 2, 3)));
  mTEST_ASSERT_SUCCESS(mJsonWriter_AddArrayValueX(jsonWriter, mVec4f(3, 1, 2, 3)));
  mTEST_ASSERT_SUCCESS(mJsonWriter_EndArray(jsonWriter));

  mTEST_ASSERT_SUCCESS(mJsonWriter_AddValue(jsonWriter, "numVec2", 2.0));
  mTEST_ASSERT_SUCCESS(mJsonWriter_AddValue(jsonWriter, "numVec3", 3.0));
  mTEST_ASSERT_SUCCESS(mJsonWriter_AddValue(jsonWriter, "numVec4", 4.0));
  mTEST_ASSERT_SUCCESS(mJsonWriter_AddValueX(jsonWriter, "colour", mVec4f(1, 1, 0, 1)));

  mTEST_ASSERT_SUCCESS(mJsonWriter_EndNamed(jsonWriter));

  mTEST_ASSERT_SUCCESS(mJsonWriter_ToString(jsonWriter, &string));

  mPtr<mJsonReader> jsonReader;
  mDEFER_CALL(&jsonReader, mJsonReader_Destroy);
  mTEST_ASSERT_SUCCESS(mJsonReader_CreateFromString(&jsonReader, pAllocator, string));

  mTEST_ASSERT_SUCCESS(mJsonReader_StepIntoNamed(jsonReader, "obj1"));
  mTEST_ASSERT_SUCCESS(mJsonReader_ReadNamedValue(jsonReader, "a", &d));
  mTEST_ASSERT_SUCCESS(mJsonReader_ReadNamedValue(jsonReader, "b", &d));
  mTEST_ASSERT_SUCCESS(mJsonReader_ReadNamedValue(jsonReader, "c", &d));
  mTEST_ASSERT_SUCCESS(mJsonReader_ReadNamedValue(jsonReader, "s", &string));
  mTEST_ASSERT_SUCCESS(mJsonReader_ExitNamed(jsonReader));

  mTEST_ASSERT_SUCCESS(mJsonReader_StepIntoNamed(jsonReader, "obj2"));
  mTEST_ASSERT_SUCCESS(mJsonReader_ReadNamedValue(jsonReader, "aaaa", &d));
  mTEST_ASSERT_SUCCESS(mJsonReader_ReadNamedValue(jsonReader, "bbbb", &d));
  mTEST_ASSERT_SUCCESS(mJsonReader_ReadNamedValue(jsonReader, "cccc", &d));
  mTEST_ASSERT_SUCCESS(mJsonReader_ReadNamedValue(jsonReader, "ssss", &string));
  mTEST_ASSERT_SUCCESS(mJsonReader_ExitNamed(jsonReader));

  mTEST_ASSERT_SUCCESS(mJsonReader_StepIntoArray(jsonReader, "vectors"));

  mTEST_ASSERT_SUCCESS(mJsonReader_ReadArrayValue(jsonReader, 0, &v2));
  mTEST_ASSERT_SUCCESS(mJsonReader_ReadArrayValue(jsonReader, 1, &v3));
  mTEST_ASSERT_SUCCESS(mJsonReader_ReadArrayValue(jsonReader, 2, &v4));
  mTEST_ASSERT_SUCCESS(mJsonReader_ExitArray(jsonReader));

  mTEST_ASSERT_SUCCESS(mJsonReader_StepIntoNamed(jsonReader, "obj3"));
  mTEST_ASSERT_SUCCESS(mJsonReader_StepIntoArray(jsonReader, "o3v2"));
  mTEST_ASSERT_SUCCESS(mJsonReader_ReadArrayValue(jsonReader, 0, &v2));
  mTEST_ASSERT_SUCCESS(mJsonReader_ReadArrayValue(jsonReader, 1, &v2));
  mTEST_ASSERT_SUCCESS(mJsonReader_ExitArray(jsonReader));

  mTEST_ASSERT_SUCCESS(mJsonReader_StepIntoArray(jsonReader, "o3v3"));
  mTEST_ASSERT_SUCCESS(mJsonReader_ReadArrayValue(jsonReader, 0, &v3));
  mTEST_ASSERT_SUCCESS(mJsonReader_ReadArrayValue(jsonReader, 1, &v3));
  mTEST_ASSERT_SUCCESS(mJsonReader_ReadArrayValue(jsonReader, 2, &v3));
  mTEST_ASSERT_SUCCESS(mJsonReader_ExitArray(jsonReader));

  mTEST_ASSERT_SUCCESS(mJsonReader_StepIntoArray(jsonReader, "o3v4"));
  mTEST_ASSERT_SUCCESS(mJsonReader_ReadArrayValue(jsonReader, 0, &v4));
  mTEST_ASSERT_SUCCESS(mJsonReader_ReadArrayValue(jsonReader, 1, &v4));
  mTEST_ASSERT_SUCCESS(mJsonReader_ReadArrayValue(jsonReader, 2, &v4));
  mTEST_ASSERT_SUCCESS(mJsonReader_ReadArrayValue(jsonReader, 3, &v4));
  mTEST_ASSERT_SUCCESS(mJsonReader_ExitArray(jsonReader));

  mTEST_ASSERT_SUCCESS(mJsonReader_ReadNamedValue(jsonReader, "numVec2", &d));
  mTEST_ASSERT_SUCCESS(mJsonReader_ReadNamedValue(jsonReader, "numVec3", &d));
  mTEST_ASSERT_SUCCESS(mJsonReader_ReadNamedValue(jsonReader, "numVec4", &d));
  mTEST_ASSERT_SUCCESS(mJsonReader_ReadNamedValue(jsonReader, "colour", &v4));

  mTEST_ASSERT_SUCCESS(mJsonReader_ExitNamed(jsonReader));

  mTEST_ALLOCATOR_ZERO_CHECK();
}

mTEST(mJsonReader, TestWriteNullptrString)
{
  mTEST_ALLOCATOR_SETUP();

  mPtr<mJsonWriter> jsonWriter;
  mDEFER_CALL(&jsonWriter, mJsonWriter_Destroy);
  mTEST_ASSERT_SUCCESS(mJsonWriter_Create(&jsonWriter, pAllocator));

  const char uninitializedStringName[] = "uninitializedString";
  const char charNullPtrName[] = "charNullPtr";

  // Write empty strings.
  {
    mString uninitializedString;
    mTEST_ASSERT_SUCCESS(mJsonWriter_AddValue(jsonWriter, uninitializedStringName, uninitializedString));

    const char *charNullPtr = nullptr;
    mTEST_ASSERT_SUCCESS(mJsonWriter_AddValue(jsonWriter, charNullPtrName, charNullPtr));
  }

  mString jsonString;
  mTEST_ASSERT_SUCCESS(mJsonWriter_ToString(jsonWriter, &jsonString));

  mPtr<mJsonReader> jsonReader;
  mTEST_ASSERT_SUCCESS(mJsonReader_CreateFromString(&jsonReader, pAllocator, jsonString));

  // Read empty strings.
  {
    mString readString0;
    mTEST_ASSERT_SUCCESS(mJsonReader_ReadNamedValue(jsonReader, uninitializedStringName, &readString0));
    mTEST_ASSERT_EQUAL(readString0.bytes, 1);

    mString readString1;
    mTEST_ASSERT_SUCCESS(mJsonReader_ReadNamedValue(jsonReader, charNullPtrName, &readString1));
    mTEST_ASSERT_EQUAL(readString1.bytes, 1);
  }

  mTEST_ALLOCATOR_ZERO_CHECK();
}
