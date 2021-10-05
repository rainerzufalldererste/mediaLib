#include "mTestLib.h"
#include "mBlobReader.h"
#include "mBlobWriter.h"

mTEST(mBlob, TestReadWrite)
{
  mTEST_ALLOCATOR_SETUP();

  const uint8_t testData[] = { 0xFF, 0xAA, 0x12, 0x36, 0x79, 0x10, 0x20, 0x02, 0x01 };

  mPtr<mBlobWriter> writer;
  mTEST_ASSERT_SUCCESS(mBlobWriter_Create(&writer, pAllocator));

  mTEST_ASSERT_SUCCESS(mBlobWriter_BeginContainer(writer));
  mTEST_ASSERT_SUCCESS(mBlobWriter_AddValue(writer, (uint8_t)124));
  mTEST_ASSERT_SUCCESS(mBlobWriter_AddValue(writer, (uint32_t)0xDEADF00D));
  mTEST_ASSERT_SUCCESS(mBlobWriter_AddValue(writer, (int32_t)-12345));
  mTEST_ASSERT_SUCCESS(mBlobWriter_AddValue(writer, (uint64_t)0x123456789ABCDEF0));
  mTEST_ASSERT_SUCCESS(mBlobWriter_AddValue(writer, (int64_t)-9876543210));
  mTEST_ASSERT_SUCCESS(mBlobWriter_AddValue(writer, (float_t)mPIf));
  mTEST_ASSERT_SUCCESS(mBlobWriter_AddValue(writer, (double_t)mSQRT2));
  mTEST_ASSERT_SUCCESS(mBlobWriter_AddValue(writer, true));
  mTEST_ASSERT_SUCCESS(mBlobWriter_AddValue(writer, mVec2s(124, 0xAAAABCDEF)));
  mTEST_ASSERT_SUCCESS(mBlobWriter_AddValue(writer, mVec2i(-124, -987698769876)));
  mTEST_ASSERT_SUCCESS(mBlobWriter_AddValue(writer, mVec2f(0.1f, -0.5f)));
  mTEST_ASSERT_SUCCESS(mBlobWriter_AddValue(writer, mVec3f(0.2f, -0.12f, 0)));
  mTEST_ASSERT_SUCCESS(mBlobWriter_AddValue(writer, mVec4f(0.6f, -0.25f, 1e10f, -1e10f)));
  mTEST_ASSERT_SUCCESS(mBlobWriter_AddValue(writer, mVec2d(0.12, -0.55)));
  mTEST_ASSERT_SUCCESS(mBlobWriter_AddValue(writer, mVec3d(0.22, -0.125, -1234)));
  mTEST_ASSERT_SUCCESS(mBlobWriter_AddValue(writer, mVec4d(0.62, -0.255, -1e-10, 1e-10)));
  mTEST_ASSERT_SUCCESS(mBlobWriter_AddValue(writer, mString("This is a test string")));
  mTEST_ASSERT_SUCCESS(mBlobWriter_AddValue(writer, testData, mARRAYSIZE(testData)));

  mTEST_ASSERT_SUCCESS(mBlobWriter_BeginBaseTypeArray(writer));

  for (size_t i = 0; i < 256; i++)
    mTEST_ASSERT_SUCCESS(mBlobWriter_AddValue(writer, mVec3f(0.001f * i, -12.0f / (i + 1), (float_t)i)));

  mTEST_ASSERT_SUCCESS(mBlobWriter_EndBaseTypeArray(writer));

  mTEST_ASSERT_SUCCESS(mBlobWriter_BeginContainer(writer));
  mTEST_ASSERT_SUCCESS(mBlobWriter_BeginContainer(writer));

  for (size_t i = 0; i < 3; i++)
    mTEST_ASSERT_SUCCESS(mBlobWriter_AddValue(writer, testData, mARRAYSIZE(testData)));

  mTEST_ASSERT_SUCCESS(mBlobWriter_EndContainer(writer));

  mTEST_ASSERT_SUCCESS(mBlobWriter_BeginArray(writer));

  for (size_t i = 0; i < 32; i++)
  {
    mTEST_ASSERT_SUCCESS(mBlobWriter_AddValue(writer, ((i % 11) & 0b101) != 0));
    mTEST_ASSERT_SUCCESS(mBlobWriter_AddValue(writer, (uint8_t)((i % 11) & 0b101)));
  }

  mTEST_ASSERT_SUCCESS(mBlobWriter_EndArray(writer));

  mTEST_ASSERT_SUCCESS(mBlobWriter_EndContainer(writer));

  mTEST_ASSERT_SUCCESS(mBlobWriter_EndContainer(writer));

  mTEST_ASSERT_SUCCESS(mBlobWriter_BeginBaseTypeArray(writer));

  for (size_t i = 0; i < 256; i++)
  {
    mTEST_ASSERT_SUCCESS(mBlobWriter_AddValue(writer, -0.001f * i));
    mTEST_ASSERT_SUCCESS(mBlobWriter_AddValue(writer, 0.001f * (i + 1)));
  }

  mTEST_ASSERT_SUCCESS(mBlobWriter_EndBaseTypeArray(writer));
  
  const uint8_t *pBlobData = nullptr;
  size_t blobDataSize = 0;
  mTEST_ASSERT_SUCCESS(mBlobWriter_GetData(writer, &pBlobData, &blobDataSize));

  mPtr<mBlobReader> reader;
  mTEST_ASSERT_SUCCESS(mBlobReader_Create(&reader, pAllocator, pBlobData, blobDataSize));

  mTEST_ASSERT_SUCCESS(mBlobReader_StepIntoContainer(reader));

  uint8_t u8;
  uint32_t u32;
  int32_t i32;
  uint64_t u64;
  int64_t i64;
  float_t f32;
  double_t f64;
  bool b;
  mVec2s u64_2;
  mVec2i i64_2;
  mVec2f f32_2;
  mVec3f f32_3;
  mVec4f f32_4;
  mVec2d f64_2;
  mVec3d f64_3;
  mVec4d f64_4;
  mString s;
  const uint8_t *pD;
  size_t dSize;

  s.pAllocator = pAllocator;

  mTEST_ASSERT_SUCCESS(mBlobReader_ReadValue(reader, &u8));
  mTEST_ASSERT_SUCCESS(mBlobReader_ReadValue(reader, &u32));
  mTEST_ASSERT_SUCCESS(mBlobReader_ReadValue(reader, &i32));
  mTEST_ASSERT_SUCCESS(mBlobReader_ReadValue(reader, &u64));
  mTEST_ASSERT_SUCCESS(mBlobReader_ReadValue(reader, &i64));
  mTEST_ASSERT_SUCCESS(mBlobReader_ReadValue(reader, &f32));
  mTEST_ASSERT_SUCCESS(mBlobReader_ReadValue(reader, &f64));
  mTEST_ASSERT_SUCCESS(mBlobReader_ReadValue(reader, &b));
  mTEST_ASSERT_SUCCESS(mBlobReader_ReadValue(reader, &u64_2));
  mTEST_ASSERT_SUCCESS(mBlobReader_ReadValue(reader, &i64_2));
  mTEST_ASSERT_SUCCESS(mBlobReader_ReadValue(reader, &f32_2));
  mTEST_ASSERT_SUCCESS(mBlobReader_ReadValue(reader, &f32_3));
  mTEST_ASSERT_SUCCESS(mBlobReader_ReadValue(reader, &f32_4));
  mTEST_ASSERT_SUCCESS(mBlobReader_ReadValue(reader, &f64_2));
  mTEST_ASSERT_SUCCESS(mBlobReader_ReadValue(reader, &f64_3));
  mTEST_ASSERT_SUCCESS(mBlobReader_ReadValue(reader, &f64_4));
  mTEST_ASSERT_SUCCESS(mBlobReader_ReadValue(reader, &s));
  mTEST_ASSERT_SUCCESS(mBlobReader_ReadValue(reader, &pD, &dSize));

  mTEST_ASSERT_TRUE(u8 == (uint8_t)124);
  mTEST_ASSERT_TRUE(u32 == (uint32_t)0xDEADF00D);
  mTEST_ASSERT_TRUE(i32 == (int32_t)-12345);
  mTEST_ASSERT_TRUE(u64 == (uint64_t)0x123456789ABCDEF0);
  mTEST_ASSERT_TRUE(i64 == (int64_t)-9876543210);
  mTEST_ASSERT_TRUE(f32 == (float_t)mPIf);
  mTEST_ASSERT_TRUE(f64 == (double_t)mSQRT2);
  mTEST_ASSERT_TRUE(b == true);
  mTEST_ASSERT_TRUE(u64_2 == mVec2s(124, 0xAAAABCDEF));
  mTEST_ASSERT_TRUE(i64_2 == mVec2i(-124, -987698769876));
  mTEST_ASSERT_TRUE(f32_2 == mVec2f(0.1f, -0.5f));
  mTEST_ASSERT_TRUE(f32_3 == mVec3f(0.2f, -0.12f, 0));
  mTEST_ASSERT_TRUE(f32_4 == mVec4f(0.6f, -0.25f, 1e10f, -1e10f));
  mTEST_ASSERT_TRUE(f64_2 == mVec2d(0.12, -0.55));
  mTEST_ASSERT_TRUE(f64_3 == mVec3d(0.22, -0.125, -1234));
  mTEST_ASSERT_TRUE(f64_4 == mVec4d(0.62, -0.255, -1e-10, 1e-10));
  mTEST_ASSERT_TRUE(s == mString("This is a test string"));
  mTEST_ASSERT_TRUE(dSize == mARRAYSIZE(testData));
  mTEST_ASSERT_TRUE(0 == memcmp(pD, testData, mARRAYSIZE(testData)));

  mTEST_ASSERT_EQUAL(mR_ResourceIncompatible, mBlobReader_ReadValue(reader, &u8));
  mTEST_ASSERT_EQUAL(mR_ResourceIncompatible, mBlobReader_ReadValue(reader, &u32));
  mTEST_ASSERT_EQUAL(mR_ResourceIncompatible, mBlobReader_ReadValue(reader, &i32));
  mTEST_ASSERT_EQUAL(mR_ResourceIncompatible, mBlobReader_ReadValue(reader, &u64));
  mTEST_ASSERT_EQUAL(mR_ResourceIncompatible, mBlobReader_ReadValue(reader, &i64));
  mTEST_ASSERT_EQUAL(mR_ResourceIncompatible, mBlobReader_ReadValue(reader, &f32));
  mTEST_ASSERT_EQUAL(mR_ResourceIncompatible, mBlobReader_ReadValue(reader, &f64));
  mTEST_ASSERT_EQUAL(mR_ResourceIncompatible, mBlobReader_ReadValue(reader, &b));
  mTEST_ASSERT_EQUAL(mR_ResourceIncompatible, mBlobReader_ReadValue(reader, &u64_2));
  mTEST_ASSERT_EQUAL(mR_ResourceIncompatible, mBlobReader_ReadValue(reader, &i64_2));
  mTEST_ASSERT_EQUAL(mR_ResourceIncompatible, mBlobReader_ReadValue(reader, &f32_2));
  mTEST_ASSERT_EQUAL(mR_ResourceIncompatible, mBlobReader_ReadValue(reader, &f32_3));
  mTEST_ASSERT_EQUAL(mR_ResourceIncompatible, mBlobReader_ReadValue(reader, &f32_4));
  mTEST_ASSERT_EQUAL(mR_ResourceIncompatible, mBlobReader_ReadValue(reader, &f64_2));
  mTEST_ASSERT_EQUAL(mR_ResourceIncompatible, mBlobReader_ReadValue(reader, &f64_3));
  mTEST_ASSERT_EQUAL(mR_ResourceIncompatible, mBlobReader_ReadValue(reader, &f64_4));
  mTEST_ASSERT_EQUAL(mR_ResourceIncompatible, mBlobReader_ReadValue(reader, &s));
  mTEST_ASSERT_EQUAL(mR_ResourceIncompatible, mBlobReader_ReadValue(reader, &pD, &dSize));
  
  mTEST_ASSERT_SUCCESS(mBlobReader_StepIntoContainer(reader));

  mTEST_ASSERT_EQUAL(mR_ResourceIncompatible, mBlobReader_ReadValue(reader, &u8));
  mTEST_ASSERT_EQUAL(mR_ResourceIncompatible, mBlobReader_ReadValue(reader, &u32));
  mTEST_ASSERT_EQUAL(mR_ResourceIncompatible, mBlobReader_ReadValue(reader, &i32));
  mTEST_ASSERT_EQUAL(mR_ResourceIncompatible, mBlobReader_ReadValue(reader, &u64));
  mTEST_ASSERT_EQUAL(mR_ResourceIncompatible, mBlobReader_ReadValue(reader, &i64));
  mTEST_ASSERT_EQUAL(mR_ResourceIncompatible, mBlobReader_ReadValue(reader, &f32));
  mTEST_ASSERT_EQUAL(mR_ResourceIncompatible, mBlobReader_ReadValue(reader, &f64));
  mTEST_ASSERT_EQUAL(mR_ResourceIncompatible, mBlobReader_ReadValue(reader, &b));
  mTEST_ASSERT_EQUAL(mR_ResourceIncompatible, mBlobReader_ReadValue(reader, &u64_2));
  mTEST_ASSERT_EQUAL(mR_ResourceIncompatible, mBlobReader_ReadValue(reader, &i64_2));
  mTEST_ASSERT_EQUAL(mR_ResourceIncompatible, mBlobReader_ReadValue(reader, &f32_2));
  mTEST_ASSERT_EQUAL(mR_ResourceIncompatible, mBlobReader_ReadValue(reader, &f32_4));
  mTEST_ASSERT_EQUAL(mR_ResourceIncompatible, mBlobReader_ReadValue(reader, &f64_2));
  mTEST_ASSERT_EQUAL(mR_ResourceIncompatible, mBlobReader_ReadValue(reader, &f64_3));
  mTEST_ASSERT_EQUAL(mR_ResourceIncompatible, mBlobReader_ReadValue(reader, &f64_4));
  mTEST_ASSERT_EQUAL(mR_ResourceInvalid, mBlobReader_ReadValue(reader, &s));
  mTEST_ASSERT_EQUAL(mR_ResourceInvalid, mBlobReader_ReadValue(reader, &pD, &dSize));
  mTEST_ASSERT_EQUAL(mR_ResourceInvalid, mBlobReader_StepIntoContainer(reader));

  for (size_t i = 0; i < 256; i++)
  {
    mTEST_ASSERT_SUCCESS(mBlobReader_ReadValue(reader, &f32_3));

    mTEST_ASSERT_TRUE(mVec3f(0.001f * i, -12.0f / (i + 1), (float_t)i) == f32_3);
  }

  mTEST_ASSERT_EQUAL(mR_IndexOutOfBounds, mBlobReader_ReadValue(reader, &f32_3));
  mTEST_ASSERT_SUCCESS(mBlobReader_ExitContainer(reader));

  mTEST_ASSERT_SUCCESS(mBlobReader_StepIntoContainer(reader));
  mTEST_ASSERT_SUCCESS(mBlobReader_StepIntoContainer(reader));
  mTEST_ASSERT_EQUAL(mR_ResourceIncompatible, mBlobReader_StepIntoContainer(reader));

  for (size_t i = 0; i < 3; i++)
  {
    mTEST_ASSERT_SUCCESS(mBlobReader_ReadValue(reader, &pD, &dSize));

    mTEST_ASSERT_TRUE(dSize == mARRAYSIZE(testData));
    mTEST_ASSERT_TRUE(0 == memcmp(pD, testData, mARRAYSIZE(testData)));
  }

  mTEST_ASSERT_SUCCESS(mBlobReader_ExitContainer(reader));
  mTEST_ASSERT_SUCCESS(mBlobReader_StepIntoContainer(reader));
  mTEST_ASSERT_SUCCESS(mBlobReader_GetArrayCount(reader, &u64));
  mTEST_ASSERT_TRUE(u64 == 64);

  for (size_t i = 0; i < 32; i++)
  {
    mTEST_ASSERT_SUCCESS(mBlobReader_ReadValue(reader, &b));
    mTEST_ASSERT_SUCCESS(mBlobReader_ReadValue(reader, &u8));

    mTEST_ASSERT_TRUE(b == (((i % 11) & 0b101) != 0));
    mTEST_ASSERT_TRUE(u8 == (uint8_t)((i % 11) & 0b101));
  }

  mTEST_ASSERT_SUCCESS(mBlobReader_ExitContainer(reader));
  mTEST_ASSERT_SUCCESS(mBlobReader_ExitContainer(reader));
  mTEST_ASSERT_SUCCESS(mBlobReader_ExitContainer(reader));
  mTEST_ASSERT_EQUAL(mR_ResourceNotFound, mBlobReader_ExitContainer(reader));

  mTEST_ASSERT_SUCCESS(mBlobReader_ResetToContainerFront(reader));

  mTEST_ASSERT_SUCCESS(mBlobReader_StepIntoContainer(reader));

  mTEST_ASSERT_SUCCESS(mBlobReader_SkipValue(reader));
  mTEST_ASSERT_SUCCESS(mBlobReader_SkipValue(reader));
  mTEST_ASSERT_SUCCESS(mBlobReader_SkipValue(reader));
  mTEST_ASSERT_SUCCESS(mBlobReader_SkipValue(reader));
  mTEST_ASSERT_SUCCESS(mBlobReader_SkipValue(reader));
  mTEST_ASSERT_SUCCESS(mBlobReader_SkipValue(reader));
  mTEST_ASSERT_SUCCESS(mBlobReader_SkipValue(reader));
  mTEST_ASSERT_SUCCESS(mBlobReader_SkipValue(reader));
  mTEST_ASSERT_SUCCESS(mBlobReader_SkipValue(reader));
  mTEST_ASSERT_SUCCESS(mBlobReader_SkipValue(reader));
  mTEST_ASSERT_SUCCESS(mBlobReader_SkipValue(reader));
  mTEST_ASSERT_SUCCESS(mBlobReader_SkipValue(reader));
  mTEST_ASSERT_SUCCESS(mBlobReader_SkipValue(reader));
  mTEST_ASSERT_SUCCESS(mBlobReader_SkipValue(reader));
  mTEST_ASSERT_SUCCESS(mBlobReader_SkipValue(reader));
  mTEST_ASSERT_SUCCESS(mBlobReader_SkipValue(reader));
  mTEST_ASSERT_SUCCESS(mBlobReader_SkipValue(reader));
  mTEST_ASSERT_SUCCESS(mBlobReader_SkipValue(reader));

  mTEST_ASSERT_SUCCESS(mBlobReader_StepIntoContainer(reader));

  mTEST_ASSERT_SUCCESS(mBlobReader_SkipValue(reader));

  mTEST_ASSERT_SUCCESS(mBlobReader_ExitContainer(reader));

  mTEST_ASSERT_SUCCESS(mBlobReader_SkipValue(reader));

  mTEST_ASSERT_SUCCESS(mBlobReader_ResetToContainerFront(reader));

  mTEST_ASSERT_SUCCESS(mBlobReader_ReadValue(reader, &u8));
  mTEST_ASSERT_TRUE(u8 == (uint8_t)124);

  mTEST_ASSERT_SUCCESS(mBlobReader_ExitContainer(reader));

  mTEST_ASSERT_SUCCESS(mBlobReader_StepIntoContainer(reader));

  for (size_t i = 0; i < 256; i++)
  {
    mTEST_ASSERT_SUCCESS(mBlobReader_ReadValue(reader, &f32));
    mTEST_ASSERT_TRUE(f32 == -0.001f * i);

    mTEST_ASSERT_SUCCESS(mBlobReader_ReadValue(reader, &f32));
    mTEST_ASSERT_TRUE(f32 == 0.001f * (i + 1));
  }

  mTEST_ASSERT_SUCCESS(mBlobReader_ExitContainer(reader));

  mTEST_ALLOCATOR_ZERO_CHECK();
}
