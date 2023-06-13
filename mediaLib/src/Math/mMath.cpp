#include "mediaLib.h"

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
