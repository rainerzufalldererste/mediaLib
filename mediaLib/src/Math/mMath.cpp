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

template<>
int8_t mMinValue<int8_t>()
{
  return INT8_MIN;
}

template<>
int16_t mMinValue<int16_t>()
{
  return INT16_MIN;
}

template<>
int32_t mMinValue<int32_t>()
{
  return INT32_MIN;
}

template<>
int64_t mMinValue<int64_t>()
{
  return INT64_MIN;
}

template<>
uint8_t mMinValue<uint8_t>()
{
  return 0;
}

template<>
uint16_t mMinValue<uint16_t>()
{
  return 0;
}

template<>
uint32_t mMinValue<uint32_t>()
{
  return 0;
}

template<>
uint64_t mMinValue<uint64_t>()
{
  return 0;
}

template<>
float_t mMinValue<float_t>()
{
  return -FLT_MAX;
}

template<>
double_t mMinValue<double_t>()
{
  return -DBL_MAX;
}

template<>
int8_t mMaxValue<int8_t>()
{
  return INT8_MAX;
}

template<>
int16_t mMaxValue<int16_t>()
{
  return INT16_MAX;
}

template<>
int32_t mMaxValue<int32_t>()
{
  return INT32_MAX;
}

template<>
int64_t mMaxValue<int64_t>()
{
  return INT64_MAX;
}

template<>
uint8_t mMaxValue<uint8_t>()
{
  return UINT8_MAX;
}

template<>
uint16_t mMaxValue<uint16_t>()
{
  return UINT16_MAX;
}

template<>
uint32_t mMaxValue<uint32_t>()
{
  return UINT32_MAX;
}

template<>
uint64_t mMaxValue<uint64_t>()
{
  return UINT64_MAX;
}

template<>
float_t mMaxValue<float_t>()
{
  return FLT_MAX;
}

template<>
double_t mMaxValue<double_t>()
{
  return DBL_MAX;
}

template<>
float_t mSmallest<float_t>()
{
  return FLT_EPSILON;
};

template<>
double_t mSmallest<double_t>()
{
  return DBL_EPSILON;
};

template <>
uint64_t constexpr mMod<uint64_t>(uint64_t value, uint64_t modulus)
{
  return value % modulus;
}

template <>
int64_t constexpr mMod<int64_t>(int64_t value, int64_t modulus)
{
  return value % modulus;
}

template <>
uint32_t constexpr mMod<uint32_t>(uint32_t value, uint32_t modulus)
{
  return value % modulus;
}

template <>
int32_t constexpr mMod<int32_t>(int32_t value, int32_t modulus)
{
  return value % modulus;
}

template <>
uint16_t constexpr mMod<uint16_t>(uint16_t value, uint16_t modulus)
{
  return value % modulus;
}

template <>
int16_t constexpr mMod<int16_t>(int16_t value, int16_t modulus)
{
  return value % modulus;
}

template <>
uint8_t constexpr mMod<uint8_t>(uint8_t value, uint8_t modulus)
{
  return value % modulus;
}

template <>
int8_t constexpr mMod<int8_t>(int8_t value, int8_t modulus)
{
  return value % modulus;
}

template <>
float_t mMod<float_t>(float_t value, float_t modulus)
{
  return fmodf(value, modulus);
}

template <>
double_t mMod<double_t>(double_t value, double_t modulus)
{
  return fmod(value, modulus);
}
