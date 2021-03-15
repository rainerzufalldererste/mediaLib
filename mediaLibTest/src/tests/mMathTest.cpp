#include "mTestLib.h"

mTEST(mMath, TestIsPrime)
{
  // https://oeis.org/A010051
  const bool isPrime[] = { 0, 0, 1, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, };

  for (size_t i = 0; i < mARRAYSIZE(isPrime); i++)
    if (isPrime[i] != mIsPrime(i))
      __debugbreak();

  mTEST_ASSERT_TRUE(mIsPrime(0x10001));
  mTEST_ASSERT_FALSE(mIsPrime(0x10000));
  mTEST_ASSERT_FALSE(mIsPrime(0xFFFF));

  mTEST_RETURN_SUCCESS();
}

mTEST(mMath, TestVec2Constuct)
{
  const mVec2f v = mVec2f(0, FLT_MAX);

  mTEST_ASSERT_EQUAL(v.x, 0);
  mTEST_ASSERT_EQUAL(v.y, FLT_MAX);
  mTEST_ASSERT_EQUAL(v.asArray[0], 0);
  mTEST_ASSERT_EQUAL(v.asArray[1], FLT_MAX);

  const mVec2i v2 = mVec2i(20, -20);

  mTEST_ASSERT_EQUAL(v2.x, 20);
  mTEST_ASSERT_EQUAL(v2.y, -20);
  mTEST_ASSERT_EQUAL(v2.asArray[0], 20);
  mTEST_ASSERT_EQUAL(v2.asArray[1], -20);

  mTEST_RETURN_SUCCESS();
}

mTEST(mMath, TestVec3Constuct)
{
  const mVec3f v = mVec3f(FLT_MIN, 0, FLT_MAX);

  mTEST_ASSERT_EQUAL(v.x, FLT_MIN);
  mTEST_ASSERT_EQUAL(v.y, 0);
  mTEST_ASSERT_EQUAL(v.z, FLT_MAX);
  mTEST_ASSERT_EQUAL(v.asArray[0], FLT_MIN);
  mTEST_ASSERT_EQUAL(v.asArray[1], 0);
  mTEST_ASSERT_EQUAL(v.asArray[2], FLT_MAX);

  const mVec3i v2 = mVec3i(-10, 10, 200);

  mTEST_ASSERT_EQUAL(v2.x, -10);
  mTEST_ASSERT_EQUAL(v2.y, 10);
  mTEST_ASSERT_EQUAL(v2.z, 200);
  mTEST_ASSERT_EQUAL(v2.asArray[0], -10);
  mTEST_ASSERT_EQUAL(v2.asArray[1], 10);
  mTEST_ASSERT_EQUAL(v2.asArray[2], 200);

  mTEST_RETURN_SUCCESS();
}

mTEST(mMath, TestVec4Constuct)
{
  const mVec4f v = mVec4f(-FLT_MAX, FLT_MIN, 0, FLT_MAX);

  mTEST_ASSERT_EQUAL(v.x, -FLT_MAX);
  mTEST_ASSERT_EQUAL(v.y, FLT_MIN);
  mTEST_ASSERT_EQUAL(v.z, 0);
  mTEST_ASSERT_EQUAL(v.w, FLT_MAX);
  mTEST_ASSERT_EQUAL(v.asArray[0], -FLT_MAX);
  mTEST_ASSERT_EQUAL(v.asArray[1], FLT_MIN);
  mTEST_ASSERT_EQUAL(v.asArray[2], 0);
  mTEST_ASSERT_EQUAL(v.asArray[3], FLT_MAX);

  const mVec4i v2 = mVec4i(1, -10, 10, 200);

  mTEST_ASSERT_EQUAL(v2.x, 1);
  mTEST_ASSERT_EQUAL(v2.y, -10);
  mTEST_ASSERT_EQUAL(v2.z, 10);
  mTEST_ASSERT_EQUAL(v2.w, 200);
  mTEST_ASSERT_EQUAL(v2.asArray[0], 1);
  mTEST_ASSERT_EQUAL(v2.asArray[1], -10);
  mTEST_ASSERT_EQUAL(v2.asArray[2], 10);
  mTEST_ASSERT_EQUAL(v2.asArray[3], 200);

  mTEST_RETURN_SUCCESS();
}

mTEST(mMath, TestVec2ConstuctSingle)
{
  const mVec2f v = mVec2f(FLT_MAX);

  mTEST_ASSERT_EQUAL(v.x, FLT_MAX);
  mTEST_ASSERT_EQUAL(v.y, FLT_MAX);
  mTEST_ASSERT_EQUAL(v.asArray[0], FLT_MAX);
  mTEST_ASSERT_EQUAL(v.asArray[1], FLT_MAX);

  const mVec2i v2 = mVec2i(20);

  mTEST_ASSERT_EQUAL(v2.x, 20);
  mTEST_ASSERT_EQUAL(v2.y, 20);
  mTEST_ASSERT_EQUAL(v2.asArray[0], 20);
  mTEST_ASSERT_EQUAL(v2.asArray[1], 20);

  mTEST_RETURN_SUCCESS();
}

mTEST(mMath, TestVec3ConstuctSingle)
{
  const mVec3f v = mVec3f(FLT_MIN);

  mTEST_ASSERT_EQUAL(v.x, FLT_MIN);
  mTEST_ASSERT_EQUAL(v.y, FLT_MIN);
  mTEST_ASSERT_EQUAL(v.z, FLT_MIN);
  mTEST_ASSERT_EQUAL(v.asArray[0], FLT_MIN);
  mTEST_ASSERT_EQUAL(v.asArray[1], FLT_MIN);
  mTEST_ASSERT_EQUAL(v.asArray[2], FLT_MIN);

  const mVec3i v2 = mVec3i(-10);

  mTEST_ASSERT_EQUAL(v2.x, -10);
  mTEST_ASSERT_EQUAL(v2.y, -10);
  mTEST_ASSERT_EQUAL(v2.z, -10);
  mTEST_ASSERT_EQUAL(v2.asArray[0], -10);
  mTEST_ASSERT_EQUAL(v2.asArray[1], -10);
  mTEST_ASSERT_EQUAL(v2.asArray[2], -10);

  mTEST_RETURN_SUCCESS();
}

mTEST(mMath, TestVec3ConstuctVec2)
{
  const mVec3f v = mVec3f(mVec2f(0, FLT_MAX), -FLT_MAX);

  mTEST_ASSERT_EQUAL(v.x, 0);
  mTEST_ASSERT_EQUAL(v.y, FLT_MAX);
  mTEST_ASSERT_EQUAL(v.z, -FLT_MAX);
  mTEST_ASSERT_EQUAL(v.asArray[0], 0);
  mTEST_ASSERT_EQUAL(v.asArray[1], FLT_MAX);
  mTEST_ASSERT_EQUAL(v.asArray[2], -FLT_MAX);

  const mVec3i v2 = mVec3i(mVec2i(0, -10), -100);

  mTEST_ASSERT_EQUAL(v2.x, 0);
  mTEST_ASSERT_EQUAL(v2.y, -10);
  mTEST_ASSERT_EQUAL(v2.z, -100);
  mTEST_ASSERT_EQUAL(v2.asArray[0], 0);
  mTEST_ASSERT_EQUAL(v2.asArray[1], -10);
  mTEST_ASSERT_EQUAL(v2.asArray[2], -100);

  mTEST_RETURN_SUCCESS();
}

mTEST(mMath, TestVec4ConstuctSingle)
{
  const mVec4f v = mVec4f(-FLT_MAX);

  mTEST_ASSERT_EQUAL(v.x, -FLT_MAX);
  mTEST_ASSERT_EQUAL(v.y, -FLT_MAX);
  mTEST_ASSERT_EQUAL(v.z, -FLT_MAX);
  mTEST_ASSERT_EQUAL(v.w, -FLT_MAX);
  mTEST_ASSERT_EQUAL(v.asArray[0], -FLT_MAX);
  mTEST_ASSERT_EQUAL(v.asArray[1], -FLT_MAX);
  mTEST_ASSERT_EQUAL(v.asArray[2], -FLT_MAX);
  mTEST_ASSERT_EQUAL(v.asArray[3], -FLT_MAX);

  const mVec4i v2 = mVec4i(1);

  mTEST_ASSERT_EQUAL(v2.x, 1);
  mTEST_ASSERT_EQUAL(v2.y, 1);
  mTEST_ASSERT_EQUAL(v2.z, 1);
  mTEST_ASSERT_EQUAL(v2.w, 1);
  mTEST_ASSERT_EQUAL(v2.asArray[0], 1);
  mTEST_ASSERT_EQUAL(v2.asArray[1], 1);
  mTEST_ASSERT_EQUAL(v2.asArray[2], 1);
  mTEST_ASSERT_EQUAL(v2.asArray[3], 1);

  mTEST_RETURN_SUCCESS();
}

mTEST(mMath, TestVec4ConstuctVec3)
{
  const mVec4f v = mVec4f(mVec3f(0, -FLT_MAX, FLT_MAX), FLT_MIN);

  mTEST_ASSERT_EQUAL(v.x, 0);
  mTEST_ASSERT_EQUAL(v.y, -FLT_MAX);
  mTEST_ASSERT_EQUAL(v.z, FLT_MAX);
  mTEST_ASSERT_EQUAL(v.w, FLT_MIN);
  mTEST_ASSERT_EQUAL(v.asArray[0], 0);
  mTEST_ASSERT_EQUAL(v.asArray[1], -FLT_MAX);
  mTEST_ASSERT_EQUAL(v.asArray[2], FLT_MAX);
  mTEST_ASSERT_EQUAL(v.asArray[3], FLT_MIN);

  const mVec4i v2 = mVec4i(mVec3i(0, 1, 2), 3);

  mTEST_ASSERT_EQUAL(v2.x, 0);
  mTEST_ASSERT_EQUAL(v2.y, 1);
  mTEST_ASSERT_EQUAL(v2.z, 2);
  mTEST_ASSERT_EQUAL(v2.w, 3);
  mTEST_ASSERT_EQUAL(v2.asArray[0], 0);
  mTEST_ASSERT_EQUAL(v2.asArray[1], 1);
  mTEST_ASSERT_EQUAL(v2.asArray[2], 2);
  mTEST_ASSERT_EQUAL(v2.asArray[3], 3);

  mTEST_RETURN_SUCCESS();
}

mTEST(mMath, TestVec3ToVec2)
{
  const mVec2f v = mVec3f(FLT_MIN, 0, FLT_MAX).ToVector2();

  mTEST_ASSERT_EQUAL(v.x, FLT_MIN);
  mTEST_ASSERT_EQUAL(v.y, 0);
  mTEST_ASSERT_EQUAL(v.asArray[0], FLT_MIN);
  mTEST_ASSERT_EQUAL(v.asArray[1], 0);

  const mVec2i v2 = mVec3i(-10, 10, 200).ToVector2();

  mTEST_ASSERT_EQUAL(v2.x, -10);
  mTEST_ASSERT_EQUAL(v2.y, 10);
  mTEST_ASSERT_EQUAL(v2.asArray[0], -10);
  mTEST_ASSERT_EQUAL(v2.asArray[1], 10);

  mTEST_RETURN_SUCCESS();
}

mTEST(mMath, TestVec4ToVec2)
{
  const mVec2f v = mVec4f(FLT_MIN, 0, FLT_MAX, -100).ToVector2();

  mTEST_ASSERT_EQUAL(v.x, FLT_MIN);
  mTEST_ASSERT_EQUAL(v.y, 0);
  mTEST_ASSERT_EQUAL(v.asArray[0], FLT_MIN);
  mTEST_ASSERT_EQUAL(v.asArray[1], 0);

  const mVec2i v2 = mVec4i(-10, 10, 200, -100).ToVector2();

  mTEST_ASSERT_EQUAL(v2.x, -10);
  mTEST_ASSERT_EQUAL(v2.y, 10);
  mTEST_ASSERT_EQUAL(v2.asArray[0], -10);
  mTEST_ASSERT_EQUAL(v2.asArray[1], 10);

  mTEST_RETURN_SUCCESS();
}

mTEST(mMath, TestVec4ToVec3)
{
  const mVec3f v = mVec4f(FLT_MIN, 0, FLT_MAX, -100).ToVector3();

  mTEST_ASSERT_EQUAL(v.x, FLT_MIN);
  mTEST_ASSERT_EQUAL(v.y, 0);
  mTEST_ASSERT_EQUAL(v.z, FLT_MAX);
  mTEST_ASSERT_EQUAL(v.asArray[0], FLT_MIN);
  mTEST_ASSERT_EQUAL(v.asArray[1], 0);
  mTEST_ASSERT_EQUAL(v.asArray[2], FLT_MAX);

  const mVec3i v2 = mVec4i(-10, 10, 200, -100).ToVector3();

  mTEST_ASSERT_EQUAL(v2.x, -10);
  mTEST_ASSERT_EQUAL(v2.y, 10);
  mTEST_ASSERT_EQUAL(v2.z, 200);
  mTEST_ASSERT_EQUAL(v2.asArray[0], -10);
  mTEST_ASSERT_EQUAL(v2.asArray[1], 10);
  mTEST_ASSERT_EQUAL(v2.asArray[2], 200);

  mTEST_RETURN_SUCCESS();
}

mTEST(mMath, TestMin)
{
  mTEST_ASSERT_EQUAL(1, mMin(1, 1));
  mTEST_ASSERT_EQUAL(1, mMin(100, 1));
  mTEST_ASSERT_EQUAL(1, mMin(1, 100));
  mTEST_ASSERT_EQUAL(-1, mMin(-1, -1));
  mTEST_ASSERT_EQUAL(-1, mMin(100, -1));
  mTEST_ASSERT_EQUAL(-1, mMin(-1, 100));
  mTEST_ASSERT_EQUAL(-100, mMin(-100, -1));
  mTEST_ASSERT_EQUAL(-100, mMin(-1, -100));
  mTEST_ASSERT_EQUAL(0, mMin(0, 0));

  mTEST_RETURN_SUCCESS();
}

mTEST(mMath, TestMax)
{
  mTEST_ASSERT_EQUAL(1, mMax(1, 1));
  mTEST_ASSERT_EQUAL(100, mMax(100, 1));
  mTEST_ASSERT_EQUAL(100, mMax(1, 100));
  mTEST_ASSERT_EQUAL(-1, mMax(-1, -1));
  mTEST_ASSERT_EQUAL(1, mMax(-100, 1));
  mTEST_ASSERT_EQUAL(1, mMax(1, -100));
  mTEST_ASSERT_EQUAL(-1, mMax(-100, -1));
  mTEST_ASSERT_EQUAL(-1, mMax(-1, -100));
  mTEST_ASSERT_EQUAL(0, mMin(0, 0));

  mTEST_RETURN_SUCCESS();
}

mTEST(mMath, TestClamp)
{
#define RUNTIME_AND_STATIC_ASSERT_EQUAL(a, b) mTEST_ASSERT_EQUAL(a, b); mTEST_STATIC_ASSERT(a == (b))

  RUNTIME_AND_STATIC_ASSERT_EQUAL(1, mClamp(1, 1, 1));
  RUNTIME_AND_STATIC_ASSERT_EQUAL(1, mClamp(0, 1, 1));
  RUNTIME_AND_STATIC_ASSERT_EQUAL(1, mClamp(1, 0, 1));
  RUNTIME_AND_STATIC_ASSERT_EQUAL(1, mClamp(1, 0, 2));
  RUNTIME_AND_STATIC_ASSERT_EQUAL(0, mClamp(0, 0, 1));
  RUNTIME_AND_STATIC_ASSERT_EQUAL(0, mClamp(-1, 0, 1));
  RUNTIME_AND_STATIC_ASSERT_EQUAL(1, mClamp(2, 0, 1));
  RUNTIME_AND_STATIC_ASSERT_EQUAL(0.5, mClamp(0.5, 0.0, 1.0));
  RUNTIME_AND_STATIC_ASSERT_EQUAL(0.0, mClamp(-0.5, 0.0, 1.0));
  RUNTIME_AND_STATIC_ASSERT_EQUAL(1.0, mClamp(1.5, 0.0, 1.0));

  mTEST_RETURN_SUCCESS();
}

#define mNUM_x 1
#define mNUM_y 2
#define mNUM_z 3
#define mNUM_w 4

#define mNUMS_2 mNUM_x, mNUM_y
#define mNUMS_3 mNUMS_2, mNUM_z
#define mNUMS_4 mNUMS_3, mNUM_w

#define mVECTOR_TEST_SUBSET2(n, a, b) mTEST(mMath, TestVector ## n ## SubSet_ ## a ## _ ## b) { mTEST_ASSERT_EQUAL(mVec ## n ## s(mNUMS_ ## n).a ## b(), mVec2s(mNUM_ ## a, mNUM_ ## b)); mTEST_RETURN_SUCCESS(); }

#define mVECTOR_TEST_SUBSET3(n, a, b, c) mTEST(mMath, TestVector ## n ## SubSet_ ## a ## _ ## b ## _ ## c) { mTEST_ASSERT_EQUAL(mVec ## n ## s(mNUMS_ ## n).a ## b ## c(), mVec3s(mNUM_ ## a, mNUM_ ## b, mNUM_ ## c)); mTEST_RETURN_SUCCESS(); }

#define mVECTOR_TEST_SUBSET4(n, a, b, c, d) mTEST(mMath, TestVector ## n ## SubSet_ ## a ## _ ## b ## _ ## c ## _ ## d) { mTEST_ASSERT_EQUAL(mVec ## n ## s(mNUMS_ ## n).a ## b ## c ## d(), mVec4s(mNUM_ ## a, mNUM_ ## b, mNUM_ ## c, mNUM_ ## d)); mTEST_RETURN_SUCCESS(); }

mVECTOR_TEST_SUBSET2(2, y, x);

mVECTOR_TEST_SUBSET2(3, x, y);
mVECTOR_TEST_SUBSET2(3, x, z);
mVECTOR_TEST_SUBSET2(3, y, x);
mVECTOR_TEST_SUBSET2(3, y, z);
mVECTOR_TEST_SUBSET2(3, z, x);
mVECTOR_TEST_SUBSET2(3, z, y);

mVECTOR_TEST_SUBSET2(4, x, y);
mVECTOR_TEST_SUBSET2(4, x, z);
mVECTOR_TEST_SUBSET2(4, x, w);
mVECTOR_TEST_SUBSET2(4, y, x);
mVECTOR_TEST_SUBSET2(4, y, z);
mVECTOR_TEST_SUBSET2(4, y, w);
mVECTOR_TEST_SUBSET2(4, z, x);
mVECTOR_TEST_SUBSET2(4, z, y);
mVECTOR_TEST_SUBSET2(4, z, w);
mVECTOR_TEST_SUBSET2(4, w, x);
mVECTOR_TEST_SUBSET2(4, w, y);
mVECTOR_TEST_SUBSET2(4, w, z);

mVECTOR_TEST_SUBSET3(3, x, z, y);
mVECTOR_TEST_SUBSET3(3, y, x, z);
mVECTOR_TEST_SUBSET3(3, y, z, x);
mVECTOR_TEST_SUBSET3(3, z, x, y);
mVECTOR_TEST_SUBSET3(3, z, y, x);

mVECTOR_TEST_SUBSET3(4, x, y, z);
mVECTOR_TEST_SUBSET3(4, x, y, w);
mVECTOR_TEST_SUBSET3(4, x, z, y);
mVECTOR_TEST_SUBSET3(4, x, z, w);
mVECTOR_TEST_SUBSET3(4, x, w, y);
mVECTOR_TEST_SUBSET3(4, x, w, z);

mVECTOR_TEST_SUBSET3(4, y, x, z);
mVECTOR_TEST_SUBSET3(4, y, x, w);
mVECTOR_TEST_SUBSET3(4, y, z, x);
mVECTOR_TEST_SUBSET3(4, y, z, w);
mVECTOR_TEST_SUBSET3(4, y, w, x);
mVECTOR_TEST_SUBSET3(4, y, w, z);

mVECTOR_TEST_SUBSET4(4, x, y, w, z);
mVECTOR_TEST_SUBSET4(4, x, y, z, w);
mVECTOR_TEST_SUBSET4(4, x, z, y, w);
mVECTOR_TEST_SUBSET4(4, x, z, w, y);
mVECTOR_TEST_SUBSET4(4, x, w, y, z);
mVECTOR_TEST_SUBSET4(4, x, w, z, y);

mVECTOR_TEST_SUBSET4(4, y, x, z, w);
mVECTOR_TEST_SUBSET4(4, y, x, w, z);
mVECTOR_TEST_SUBSET4(4, y, z, x, w);
mVECTOR_TEST_SUBSET4(4, y, z, w, x);
mVECTOR_TEST_SUBSET4(4, y, w, x, z);
mVECTOR_TEST_SUBSET4(4, y, w, z, x);

mVECTOR_TEST_SUBSET4(4, z, x, y, w);
mVECTOR_TEST_SUBSET4(4, z, x, w, y);
mVECTOR_TEST_SUBSET4(4, z, y, x, w);
mVECTOR_TEST_SUBSET4(4, z, y, w, x);
mVECTOR_TEST_SUBSET4(4, z, w, x, y);
mVECTOR_TEST_SUBSET4(4, z, w, y, x);

mVECTOR_TEST_SUBSET4(4, w, x, y, z);
mVECTOR_TEST_SUBSET4(4, w, x, z, y);
mVECTOR_TEST_SUBSET4(4, w, y, x, z);
mVECTOR_TEST_SUBSET4(4, w, y, z, x);
mVECTOR_TEST_SUBSET4(4, w, z, x, y);
mVECTOR_TEST_SUBSET4(4, w, z, y, x);

mTEST(mRectangle2D, TestConstruct)
{
  {
    mRectangle2D<int32_t> rect;

    mTEST_ASSERT_EQUAL(rect.x, 0);
    mTEST_ASSERT_EQUAL(rect.y, 0);
    mTEST_ASSERT_EQUAL(rect.w, 0);
    mTEST_ASSERT_EQUAL(rect.h, 0);
  }

  {
    mRectangle2D<int32_t> rect(1, 2, 3, 4);

    mTEST_ASSERT_EQUAL(rect.x, 1);
    mTEST_ASSERT_EQUAL(rect.y, 2);
    mTEST_ASSERT_EQUAL(rect.w, 3);
    mTEST_ASSERT_EQUAL(rect.h, 4);
    mTEST_ASSERT_EQUAL(rect.width, 3);
    mTEST_ASSERT_EQUAL(rect.height, 4);
    mTEST_ASSERT_EQUAL(rect.position, mVec2t<int32_t>(1, 2));
    mTEST_ASSERT_EQUAL(rect.size, mVec2t<int32_t>(3, 4));
    mTEST_ASSERT_EQUAL(rect.asVector4, mVec4t<int32_t>(1, 2, 3, 4));
    mTEST_ASSERT_EQUAL(rect.asArray[0], 1);
    mTEST_ASSERT_EQUAL(rect.asArray[1], 2);
    mTEST_ASSERT_EQUAL(rect.asArray[2], 3);
    mTEST_ASSERT_EQUAL(rect.asArray[3], 4);
  }

  {
    mRectangle2D<int32_t> rect(mVec2t<int32_t>(1, 2), mVec2t<int32_t>(3, 4));

    mTEST_ASSERT_EQUAL(rect.x, 1);
    mTEST_ASSERT_EQUAL(rect.y, 2);
    mTEST_ASSERT_EQUAL(rect.w, 3);
    mTEST_ASSERT_EQUAL(rect.h, 4);
  }

  {
    mRectangle2D<int32_t> rect = mRectangle2D<int32_t>(1, 2, 3, 4);

    mTEST_ASSERT_EQUAL(rect.x, 1);
    mTEST_ASSERT_EQUAL(rect.y, 2);
    mTEST_ASSERT_EQUAL(rect.w, 3);
    mTEST_ASSERT_EQUAL(rect.h, 4);
  }

  mTEST_RETURN_SUCCESS();
}

mTEST(mRectangle2D, TestEqual)
{
  mRectangle2D<int32_t> rect(1, 2, 3, 4);

  mTEST_ASSERT_TRUE(rect == mRectangle2D<int32_t>(1, 2, 3, 4));
  mTEST_ASSERT_TRUE(rect != mRectangle2D<int32_t>(0, 2, 3, 4));
  mTEST_ASSERT_TRUE(rect != mRectangle2D<int32_t>(1, 0, 3, 4));
  mTEST_ASSERT_TRUE(rect != mRectangle2D<int32_t>(1, 2, 0, 4));
  mTEST_ASSERT_TRUE(rect != mRectangle2D<int32_t>(1, 2, 3, 0));

  mTEST_RETURN_SUCCESS();
}

mTEST(mRectangle2D, TestContainsVector)
{
  {
    mRectangle2D<int32_t> rect(0, 0, 2, 2);

    mTEST_ASSERT_TRUE(rect.Contains(mVec2t<int32_t>(0, 0)));
    mTEST_ASSERT_TRUE(rect.Contains(mVec2t<int32_t>(0, 1)));
    mTEST_ASSERT_TRUE(rect.Contains(mVec2t<int32_t>(1, 0)));
    mTEST_ASSERT_TRUE(rect.Contains(mVec2t<int32_t>(1, 1)));

    mTEST_ASSERT_FALSE(rect.Contains(mVec2t<int32_t>(0, -1)));
    mTEST_ASSERT_FALSE(rect.Contains(mVec2t<int32_t>(-1, 0)));
    mTEST_ASSERT_FALSE(rect.Contains(mVec2t<int32_t>(-1, -1)));

    mTEST_ASSERT_FALSE(rect.Contains(mVec2t<int32_t>(0, 3)));
    mTEST_ASSERT_FALSE(rect.Contains(mVec2t<int32_t>(3, 0)));
    mTEST_ASSERT_FALSE(rect.Contains(mVec2t<int32_t>(3, 2)));
  }

  {
    constexpr int32_t startX = 20;
    constexpr int32_t startY = 2;
    constexpr int32_t sizeX = 11;
    constexpr int32_t sizeY = 8;

    mRectangle2D<int32_t> rect(startX, startY, sizeX, sizeY);

    mTEST_ASSERT_TRUE(rect.Contains(mVec2t<int32_t>(startX, startY)));
    mTEST_ASSERT_TRUE(rect.Contains(mVec2t<int32_t>(startX, startY + sizeY - 1)));
    mTEST_ASSERT_TRUE(rect.Contains(mVec2t<int32_t>(startX + sizeX - 1, startY)));
    mTEST_ASSERT_TRUE(rect.Contains(mVec2t<int32_t>(startX + sizeX - 1, startY + sizeY - 1)));

    mTEST_ASSERT_TRUE(rect.Contains(mVec2t<int32_t>(startX + sizeX / 2, startY + sizeY / 2)));

    mTEST_ASSERT_FALSE(rect.Contains(mVec2t<int32_t>(startX, startY - 1)));
    mTEST_ASSERT_FALSE(rect.Contains(mVec2t<int32_t>(startX - 1, startY)));
    mTEST_ASSERT_FALSE(rect.Contains(mVec2t<int32_t>(startX - 1, startY - 1)));

    mTEST_ASSERT_FALSE(rect.Contains(mVec2t<int32_t>(startX, startY + sizeY)));
    mTEST_ASSERT_FALSE(rect.Contains(mVec2t<int32_t>(startX + sizeX, startY)));
    mTEST_ASSERT_FALSE(rect.Contains(mVec2t<int32_t>(startX + sizeX, startY + sizeY)));
  }

  {
    constexpr int32_t startX = -10;
    constexpr int32_t startY = -4;
    constexpr int32_t sizeX = 1380;
    constexpr int32_t sizeY = 124;

    mRectangle2D<int32_t> rect(startX, startY, sizeX, sizeY);

    mTEST_ASSERT_TRUE(rect.Contains(mVec2t<int32_t>(startX, startY)));
    mTEST_ASSERT_TRUE(rect.Contains(mVec2t<int32_t>(startX, startY + sizeY - 1)));
    mTEST_ASSERT_TRUE(rect.Contains(mVec2t<int32_t>(startX + sizeX - 1, startY)));
    mTEST_ASSERT_TRUE(rect.Contains(mVec2t<int32_t>(startX + sizeX - 1, startY + sizeY - 1)));

    mTEST_ASSERT_TRUE(rect.Contains(mVec2t<int32_t>(startX + sizeX / 2, startY + sizeY / 2)));

    mTEST_ASSERT_FALSE(rect.Contains(mVec2t<int32_t>(startX, startY - 1)));
    mTEST_ASSERT_FALSE(rect.Contains(mVec2t<int32_t>(startX - 1, startY)));
    mTEST_ASSERT_FALSE(rect.Contains(mVec2t<int32_t>(startX - 1, startY - 1)));

    mTEST_ASSERT_FALSE(rect.Contains(mVec2t<int32_t>(startX, startY + sizeY)));
    mTEST_ASSERT_FALSE(rect.Contains(mVec2t<int32_t>(startX + sizeX, startY)));
    mTEST_ASSERT_FALSE(rect.Contains(mVec2t<int32_t>(startX + sizeX, startY + sizeY)));
  }

  {
    constexpr int32_t startX = 10;
    constexpr int32_t startY = -4;
    constexpr int32_t sizeX = 1;
    constexpr int32_t sizeY = 124;

    mRectangle2D<int32_t> rect(startX, startY, sizeX, sizeY);

    mTEST_ASSERT_TRUE(rect.Contains(mVec2t<int32_t>(startX, startY)));
    mTEST_ASSERT_TRUE(rect.Contains(mVec2t<int32_t>(startX, startY + sizeY - 1)));
    mTEST_ASSERT_TRUE(rect.Contains(mVec2t<int32_t>(startX + sizeX - 1, startY)));
    mTEST_ASSERT_TRUE(rect.Contains(mVec2t<int32_t>(startX + sizeX - 1, startY + sizeY - 1)));

    mTEST_ASSERT_TRUE(rect.Contains(mVec2t<int32_t>(startX + sizeX / 2, startY + sizeY / 2)));

    mTEST_ASSERT_FALSE(rect.Contains(mVec2t<int32_t>(startX, startY - 1)));
    mTEST_ASSERT_FALSE(rect.Contains(mVec2t<int32_t>(startX - 1, startY)));
    mTEST_ASSERT_FALSE(rect.Contains(mVec2t<int32_t>(startX - 1, startY - 1)));

    mTEST_ASSERT_FALSE(rect.Contains(mVec2t<int32_t>(startX, startY + sizeY)));
    mTEST_ASSERT_FALSE(rect.Contains(mVec2t<int32_t>(startX + sizeX, startY)));
    mTEST_ASSERT_FALSE(rect.Contains(mVec2t<int32_t>(startX + sizeX, startY + sizeY)));
  }

  {
    constexpr int32_t startX = -15;
    constexpr int32_t startY = 4;
    constexpr int32_t sizeX = 10;
    constexpr int32_t sizeY = 24;

    mRectangle2D<int32_t> rect(startX, startY, sizeX, sizeY);

    mTEST_ASSERT_TRUE(rect.Contains(mVec2t<int32_t>(startX, startY)));
    mTEST_ASSERT_TRUE(rect.Contains(mVec2t<int32_t>(startX, startY + sizeY - 1)));
    mTEST_ASSERT_TRUE(rect.Contains(mVec2t<int32_t>(startX + sizeX - 1, startY)));
    mTEST_ASSERT_TRUE(rect.Contains(mVec2t<int32_t>(startX + sizeX - 1, startY + sizeY - 1)));

    mTEST_ASSERT_TRUE(rect.Contains(mVec2t<int32_t>(startX + sizeX / 2, startY + sizeY / 2)));

    mTEST_ASSERT_FALSE(rect.Contains(mVec2t<int32_t>(startX, startY - 1)));
    mTEST_ASSERT_FALSE(rect.Contains(mVec2t<int32_t>(startX - 1, startY)));
    mTEST_ASSERT_FALSE(rect.Contains(mVec2t<int32_t>(startX - 1, startY - 1)));

    mTEST_ASSERT_FALSE(rect.Contains(mVec2t<int32_t>(startX, startY + sizeY)));
    mTEST_ASSERT_FALSE(rect.Contains(mVec2t<int32_t>(startX + sizeX, startY)));
    mTEST_ASSERT_FALSE(rect.Contains(mVec2t<int32_t>(startX + sizeX, startY + sizeY)));
  }

  mTEST_RETURN_SUCCESS();
}

mTEST(mRectangle2D, TestContainsRectangle)
{
  {
    constexpr int32_t startX = -1, startY = -10, sizeX = 10, sizeY = 20;

    mRectangle2D<int32_t> rect(startX, startY, sizeX, sizeY);

    mTEST_ASSERT_TRUE(rect.Contains(mRectangle2D<int32_t>(startX, startY, 1, 1)));
    mTEST_ASSERT_TRUE(rect.Contains(mRectangle2D<int32_t>(startX, startY, 0, 0)));
    mTEST_ASSERT_TRUE(rect.Contains(mRectangle2D<int32_t>(startX, startY, sizeX, sizeY)));
    mTEST_ASSERT_TRUE(rect.Contains(mRectangle2D<int32_t>(startX, startY, sizeX - 1, sizeY - 1)));
    mTEST_ASSERT_TRUE(rect.Contains(mRectangle2D<int32_t>(startX, startY + 1, sizeX, sizeY - 1)));
    mTEST_ASSERT_TRUE(rect.Contains(mRectangle2D<int32_t>(startX + 1, startY, sizeX - 1, sizeY)));
    mTEST_ASSERT_TRUE(rect.Contains(mRectangle2D<int32_t>(startX + 1, startY + 1, sizeX - 1, sizeY - 1)));

    mTEST_ASSERT_FALSE(rect.Contains(mRectangle2D<int32_t>(startX, startY + 1, sizeX, sizeY)));
    mTEST_ASSERT_FALSE(rect.Contains(mRectangle2D<int32_t>(startX + 1, startY, sizeX, sizeY)));
    mTEST_ASSERT_FALSE(rect.Contains(mRectangle2D<int32_t>(startX + 1, startY + 1, sizeX, sizeY)));
    mTEST_ASSERT_FALSE(rect.Contains(mRectangle2D<int32_t>(startX - 1, startY, 1, 1)));
    mTEST_ASSERT_FALSE(rect.Contains(mRectangle2D<int32_t>(startX - 1, startY, sizeX + 1, 1)));
    mTEST_ASSERT_FALSE(rect.Contains(mRectangle2D<int32_t>(startX - 1, startY, sizeX + 2, 1)));
    mTEST_ASSERT_FALSE(rect.Contains(mRectangle2D<int32_t>(startX, startY - 1, 1, 1)));
    mTEST_ASSERT_FALSE(rect.Contains(mRectangle2D<int32_t>(startX, startY - 1, sizeX, sizeY + 1)));
    mTEST_ASSERT_FALSE(rect.Contains(mRectangle2D<int32_t>(startX, startY - 1, sizeX, sizeY + 2)));
    mTEST_ASSERT_FALSE(rect.Contains(mRectangle2D<int32_t>(startX - 1, startY - 1, sizeX + 2, sizeY + 2)));
  }

  mTEST_RETURN_SUCCESS();
}

mTEST(mRectangle2D, TestScale)
{
  {
    mRectangle2D<int32_t> rect(10, 12, 13, 14);
    mRectangle2D<int32_t> rect0 = rect.ScaleCopy(mVec2t<int32_t>(10, 20));
    mRectangle2D<int32_t> rect1 = rect.ScaleSelf(mVec2t<int32_t>(10, 20));

    mTEST_ASSERT_EQUAL(rect0, rect1);
    
    mTEST_ASSERT_EQUAL(rect0, mRectangle2D<int32_t>(10 * 10, 12 * 20, 13 * 10, 14 * 20));
    mTEST_ASSERT_EQUAL(rect1, mRectangle2D<int32_t>(10 * 10, 12 * 20, 13 * 10, 14 * 20));
  }

  mTEST_RETURN_SUCCESS();
}

mTEST(mRectangle2D, TestOffset)
{
  {
    mRectangle2D<int32_t> rect(10, 12, 13, 14);
    mRectangle2D<int32_t> rect0 = rect.OffsetCopy(mVec2t<int32_t>(10, 20));
    mRectangle2D<int32_t> rect1 = rect.OffsetSelf(mVec2t<int32_t>(10, 20));

    mTEST_ASSERT_EQUAL(rect0, rect1);

    mTEST_ASSERT_EQUAL(rect0, mRectangle2D<int32_t>(10 + 10, 12 + 20, 13, 14));
    mTEST_ASSERT_EQUAL(rect1, mRectangle2D<int32_t>(10 + 10, 12 + 20, 13, 14));
  }

  mTEST_RETURN_SUCCESS();
}

mTEST(mRectangle2D, TestGrowToContainRectangle)
{
  {
    mRectangle2D<int64_t> rect(12, 2, 8, 18);

    mTEST_ASSERT_EQUAL(mRectangle2D<int64_t>(rect).GrowToContain(mRectangle2D<int64_t>(rect)), mRectangle2D<int64_t>(rect));
    mTEST_ASSERT_EQUAL(mRectangle2D<int64_t>(rect).GrowToContain(mRectangle2D<int64_t>(15, 10, 4, 6)), mRectangle2D<int64_t>(rect));
    mTEST_ASSERT_EQUAL(mRectangle2D<int64_t>(rect).GrowToContain(mRectangle2D<int64_t>(15, 10, 5, 6)), mRectangle2D<int64_t>(rect));
    mTEST_ASSERT_EQUAL(mRectangle2D<int64_t>(rect).GrowToContain(mRectangle2D<int64_t>(15, 10, 5, 10)), mRectangle2D<int64_t>(rect));
    mTEST_ASSERT_EQUAL(mRectangle2D<int64_t>(rect).GrowToContain(mRectangle2D<int64_t>(10, 10, 5, 5)), mRectangle2D<int64_t>(10, 2, 10, 18));
    mTEST_ASSERT_EQUAL(mRectangle2D<int64_t>(rect).GrowToContain(mRectangle2D<int64_t>(0, 10, 40, 5)), mRectangle2D<int64_t>(0, 2, 40, 18));
    mTEST_ASSERT_EQUAL(mRectangle2D<int64_t>(rect).GrowToContain(mRectangle2D<int64_t>(12, 0, 8, 1)), mRectangle2D<int64_t>(12, 0, 8, 20));
    mTEST_ASSERT_EQUAL(mRectangle2D<int64_t>(rect).GrowToContain(mRectangle2D<int64_t>(0, 0, 1, 1)), mRectangle2D<int64_t>(0, 0, 20, 20));
    mTEST_ASSERT_EQUAL(mRectangle2D<int64_t>(rect).GrowToContain(mRectangle2D<int64_t>(0, 0, 40, 40)), mRectangle2D<int64_t>(0, 0, 40, 40));
  }
  {
    mRectangle2D<int64_t> rect(-1, -1, 2, 2);

    mTEST_ASSERT_EQUAL(mRectangle2D<int64_t>(rect).GrowToContain(mRectangle2D<int64_t>(rect)), mRectangle2D<int64_t>(rect));
    mTEST_ASSERT_EQUAL(mRectangle2D<int64_t>(rect).GrowToContain(mRectangle2D<int64_t>(0, 0, 1, 1)), mRectangle2D<int64_t>(rect));
    mTEST_ASSERT_EQUAL(mRectangle2D<int64_t>(rect).GrowToContain(mRectangle2D<int64_t>(1, 1, 0, 0)), mRectangle2D<int64_t>(rect));
    mTEST_ASSERT_EQUAL(mRectangle2D<int64_t>(rect).GrowToContain(mRectangle2D<int64_t>(-1, -1, 1, 1)), mRectangle2D<int64_t>(rect));
    mTEST_ASSERT_EQUAL(mRectangle2D<int64_t>(rect).GrowToContain(mRectangle2D<int64_t>(-2, -2, 1, 1)), mRectangle2D<int64_t>(-2, -2, 3, 3));
    mTEST_ASSERT_EQUAL(mRectangle2D<int64_t>(rect).GrowToContain(mRectangle2D<int64_t>(1, 1, 1, 1)), mRectangle2D<int64_t>(-1, -1, 3, 3));
  }

  mTEST_RETURN_SUCCESS();
}

mTEST(mRectangle2D, TestGrowToContainVector)
{
  mRectangle2D<int64_t> rect(-1, -1, 2, 2);
  mRectangle2D<int64_t> originalRect(rect);

  mTEST_ASSERT_EQUAL(rect.GrowToContain(mVec2i(-1, -1)), originalRect);
  mTEST_ASSERT_EQUAL(rect.GrowToContain(mVec2i(0, 0)), originalRect);
  mTEST_ASSERT_EQUAL(rect.GrowToContain(mVec2i(1, 1)), mRectangle2D<int64_t>(-1, -1, 3, 3));
  mTEST_ASSERT_EQUAL(rect.GrowToContain(mVec2i(-1, -2)), mRectangle2D<int64_t>(-1, -2, 3, 4));
  mTEST_ASSERT_EQUAL(rect.GrowToContain(mVec2i(-2, 2)), mRectangle2D<int64_t>(-2, -2, 4, 5));

  mTEST_RETURN_SUCCESS();
}

mTEST(mAngleDiff, Test)
{
  mTEST_ASSERT_FLOAT_EQUALS(mAngleCompare(mPIf, mPIf), 1.f);
  mTEST_ASSERT_FLOAT_EQUALS(mAngleCompare(mPIf, 3 * mPIf), 1.f);
  mTEST_ASSERT_FLOAT_EQUALS(mAngleCompare(mPIf, 0.f), 0.f);
  mTEST_ASSERT_FLOAT_EQUALS(mAngleCompare(3 * mPIf, 0.f), 0.f);
  mTEST_ASSERT_FLOAT_EQUALS(mAngleCompare(mPIf, 2 * mPIf), 0.f);
  mTEST_ASSERT_FLOAT_EQUALS(mAngleCompare(mPIf, 4 * mPIf), 0.f);
  mTEST_ASSERT_FLOAT_EQUALS(mAngleCompare(mPIf, 5 * mPIf), 1.f);
  mTEST_ASSERT_FLOAT_EQUALS(mAngleCompare(mPIf, -5 * mPIf), 1.f);
  mTEST_ASSERT_FLOAT_EQUALS(mAngleCompare(-11 * mPIf, -5 * mPIf), 1.f);
  mTEST_ASSERT_FLOAT_EQUALS(mAngleCompare(4 * mPIf, 2 * mPIf), 1.f);
  mTEST_ASSERT_FLOAT_EQUALS(mAngleCompare(4 * mPIf, -2 * mPIf), 1.f);
  mTEST_ASSERT_FLOAT_EQUALS(mAngleCompare(-4 * mPIf, -2 * mPIf), 1.f);
  mTEST_ASSERT_FLOAT_EQUALS(mAngleCompare(mPIf, 1.5f * mPIf), 0.5f);
  mTEST_ASSERT_FLOAT_EQUALS(mAngleCompare(mPIf, 0.5f * mPIf), 0.5f);
  mTEST_ASSERT_FLOAT_EQUALS(mAngleCompare(mPIf, -0.5f * mPIf), 0.5f);
  mTEST_ASSERT_FLOAT_EQUALS(mAngleCompare(mPIf, -1.5f * mPIf), 0.5f);

  mTEST_RETURN_SUCCESS();
}
