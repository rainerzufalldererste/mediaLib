#include "mediaLib.h"

static_assert(mAbs(1) == 1, "Test Failed.");
static_assert(mAbs(0) == 0, "Test Failed.");
static_assert(mAbs(-1) == 1, "Test Failed.");
static_assert(mClamp(1, 0, 1) == 1, "Test Failed.");
static_assert(mClamp(0, 0, 1) == 0, "Test Failed.");
static_assert(mClamp(1, 0, 2) == 1, "Test Failed.");
static_assert(mClamp(0, -1, 1) == 0, "Test Failed.");
static_assert(mClamp(-1, 1, 2) == 1, "Test Failed.");
static_assert(mClamp(10, -5, 1) == 1, "Test Failed.");
static_assert(mSign(10) == 1, "Test Failed.");
static_assert(mSign(-10) == -1, "Test Failed.");
static_assert(mSign(10) == 1, "Test Failed.");
static_assert(mSign<size_t>(10) == 1, "Test Failed.");
static_assert(mSign(0) == 0, "Test Failed.");
static_assert(mSign<size_t>(0) == 0, "Test Failed.");
static_assert(mMin(0, 1) == 0, "Test Failed.");
static_assert(mMin(1, 0) == 0, "Test Failed.");
static_assert(mMin(-1, 1) == -1, "Test Failed.");
static_assert(mMin(1, -1) == -1, "Test Failed.");
static_assert(mMin(1, 1) == 1, "Test Failed.");
static_assert(mMax(0, 1) == 1, "Test Failed.");
static_assert(mMax(1, 0) == 1, "Test Failed.");
static_assert(mMax(-1, 2) == 2, "Test Failed.");
static_assert(mMax(2, -1) == 2, "Test Failed.");
static_assert(mMax(1, 1) == 1, "Test Failed.");

bool mIsPrime(const size_t n)
{
  if (n < 2)
    return false;

  const size_t max = (size_t)mSqrt(n);

  for (size_t i = 2; i <= max; i++)
    if (n % i == 0)
      return false;
  
  return true;
}

__host__ __device__ mVec3f mColor_UnpackBgraToVec3f(const uint32_t bgraColor)
{
  const float_t v(1.0f / 0xFF);
  return mVec3f((float_t)((bgraColor & 0x00FF0000) >> 0x10), (float_t)((bgraColor & 0x0000FF00) >> 0x8), (float_t)(bgraColor & 0x000000FF)) * v;
}

__host__ __device__ mVec3d mColor_UnpackBgraToVec3d(const uint32_t bgraColor)
{
  const double_t v(1.0 / 0xFF);
  return mVec3d((double_t)((bgraColor & 0x00FF0000) >> 0x10), (double_t)((bgraColor & 0x0000FF00) >> 0x8), (double_t)(bgraColor & 0x000000FF)) * v;
}

__host__ __device__ mVec4f mColor_UnpackBgraToVec4f(const uint32_t bgraColor)
{
  static const float_t v(1.0f / 0xFF);
  return mVec4f((float_t)((bgraColor & 0x00FF0000) >> 0x10), (float_t)((bgraColor & 0x0000FF00) >> 0x8), (float_t)(bgraColor & 0x000000FF), (float_t)((bgraColor & 0xFF000000) >> 0x18)) * v;
}

__host__ __device__ mVec4d mColor_UnpackBgraToVec4d(const uint32_t bgraColor)
{
  const double_t v(1.0 / 0xFF);
  return mVec4d((double_t)((bgraColor & 0x00FF0000) >> 0x10), (double_t)((bgraColor & 0x0000FF00) >> 0x8), (double_t)(bgraColor & 0x000000FF), (double_t)((bgraColor & 0xFF000000) >> 0x18)) * v;
}

__host__ __device__ mVector mColor_UnpackBgraToVector(const uint32_t bgraColor)
{
  const float_t v = 1.0f / 0xFF;
  return mVector((float_t)((bgraColor & 0x00FF0000) >> 0x10), (float_t)((bgraColor & 0x0000FF00) >> 0x8), (float_t)(bgraColor & 0x000000FF), (float_t)((bgraColor & 0xFF000000) >> 0x18)) * v;
}

__host__ __device__ uint32_t mColor_PackVec3fToBgra(const mVec3f rgbVector)
{
  const float_t v = (float_t)0xFF;
  const mVec3f v0 = rgbVector * v;
  return (mClamp((uint32_t)v0.x, 0u, 0xFFu) << 0x10) | (mClamp((uint32_t)v0.y, 0u, 0xFFu) << 0x8) | mClamp((uint32_t)v0.z, 0u, 0xFFu) | 0xFF000000;
}

__host__ __device__ uint32_t mColor_PackVec3dToBgra(const mVec3d rgbVector)
{
  const double_t v = (double_t)0xFF;
  const mVec3d v0 = rgbVector * v;
  return (mClamp((uint32_t)v0.x, 0u, 0xFFu) << 0x10) | (mClamp((uint32_t)v0.y, 0u, 0xFFu) << 0x8) | mClamp((uint32_t)v0.z, 0u, 0xFFu) | 0xFF000000;
}

__host__ __device__ uint32_t mColor_PackVec4fToBgra(const mVec4f rgbaVector)
{
  const float_t v = (float_t)0xFF;
  const mVec4f v0 = rgbaVector * v;
  return (mClamp((uint32_t)v0.x, 0u, 0xFFu) << 0x10) | (mClamp((uint32_t)v0.y, 0u, 0xFFu) << 0x8) | mClamp((uint32_t)v0.z, 0u, 0xFFu) | (mClamp((uint32_t)v0.w, 0u, 0xFFu) << 0x18);
}

__host__ __device__ uint32_t mColor_PackVec4dToBgra(const mVec4d rgbaVector)
{
  const double_t v = (double_t)0xFF;
  const mVec4d v0 = rgbaVector * v;
  return (mClamp((uint32_t)v0.x, 0u, 0xFFu) << 0x10) | (mClamp((uint32_t)v0.y, 0u, 0xFFu) << 0x8) | mClamp((uint32_t)v0.z, 0u, 0xFFu) | (mClamp((uint32_t)v0.w, 0u, 0xFFu) << 0x18);
}

__host__ __device__ uint32_t mColor_PackVectorToBgra(const mVector rgbaVector)
{
  const float_t v = (float_t)0xFF;
  const mVector v0 = rgbaVector * v;
  return (mClamp((uint32_t)v0.x, 0u, 0xFFu) << 0x10) | (mClamp((uint32_t)v0.y, 0u, 0xFFu) << 0x8) | mClamp((uint32_t)v0.z, 0u, 0xFFu) | (mClamp((uint32_t)v0.w, 0u, 0xFFu) << 0x18);
}

//////////////////////////////////////////////////////////////////////////

mFraction mToFraction(const double_t value, const int64_t precision /* = 1000000000 */)
{
  double_t integralPart;
  const double_t fractionalPart = modf(value, &integralPart);
  const int64_t fracPrec = (int64_t)round(fractionalPart * precision);
  const int64_t gcd = mGreatestCommonDivisor(fracPrec, precision);

  return { (int64_t)round(integralPart), fracPrec / gcd, precision / gcd };
}

int64_t mGreatestCommonDivisor(const int64_t a, const int64_t b)
{
  if (a == 0)
    return b;
  if (b == 0)
    return a;

  if (a < b)
    return mGreatestCommonDivisor(a, b % a);

  return mGreatestCommonDivisor(b, a % b);
}
