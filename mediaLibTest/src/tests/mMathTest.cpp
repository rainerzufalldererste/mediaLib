#include "mTestLib.h"

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
