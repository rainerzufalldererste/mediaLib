#include "mediaLib.h"

#ifdef GIT_BUILD // Define __M_FILE__
  #ifdef __M_FILE__
    #undef __M_FILE__
  #endif
  #define __M_FILE__ "hUtGZWjbZNNzNag8i41uTW8LybEHgriSjm+4GeOEV3sWL4IRkCtdjsUmdK7Ff4ZKJZevCXNjcTMDkkxc"
#endif

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

//////////////////////////////////////////////////////////////////////////

uint32_t mColor_PackVectorToBgra(const mVector rgbaVector)
{
  constexpr float_t v = (float_t)0xFF;
  const mVector v0 = rgbaVector * v;
  return (mClamp((uint32_t)v0.x, 0u, 0xFFu) << 0x10) | (mClamp((uint32_t)v0.y, 0u, 0xFFu) << 0x8) | mClamp((uint32_t)v0.z, 0u, 0xFFu) | (mClamp((uint32_t)v0.w, 0u, 0xFFu) << 0x18);
}

mVector mColor_UnpackBgraToVector(const uint32_t bgraColor)
{
  constexpr float_t v = 1.0f / 0xFF;
  return mVector((float_t)((bgraColor & 0x00FF0000) >> 0x10), (float_t)((bgraColor & 0x0000FF00) >> 0x8), (float_t)(bgraColor & 0x000000FF), (float_t)((bgraColor & 0xFF000000) >> 0x18)) * v;
}

uint32_t mColor_PackVectorToRgba(const mVector rgbaVector)
{
  constexpr float_t v = (float_t)0xFF;
  const mVector v0 = rgbaVector * v;
  return (mClamp((uint32_t)v0.z, 0u, 0xFFu) << 0x10) | (mClamp((uint32_t)v0.y, 0u, 0xFFu) << 0x8) | mClamp((uint32_t)v0.x, 0u, 0xFFu) | (mClamp((uint32_t)v0.w, 0u, 0xFFu) << 0x18);
}

mVector mColor_UnpackRgbaToVector(const uint32_t rgbaColor)
{
  constexpr float_t v = 1.0f / 0xFF;
  return mVector((float_t)(rgbaColor & 0x000000FF), (float_t)((rgbaColor & 0x0000FF00) >> 0x8), (float_t)((rgbaColor & 0x00FF0000) >> 0x10), (float_t)((rgbaColor & 0xFF000000) >> 0x18)) * v;
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
