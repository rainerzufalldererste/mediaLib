#ifndef mMath_h__
#define mMath_h__

#include "default.h"

#ifndef __CUDA_ARCH__
#define __host__
#define __device__
#endif // !__CUDA_ARCH__

template <typename T> constexpr T mAbs(const T value) { return value >= 0 ? value : -value; }
template <typename T> auto mSqrt(const T value)->decltype(sqrt(value)) { return sqrt(value); }
template <typename T> auto mSin(const T value)->decltype(sin(value)) { return sin(value); }
template <typename T> auto mCos(const T value)->decltype(cos(value)) { return cos(value); }
template <typename T> auto mTan(const T value)->decltype(tan(value)) { return tan(value); }
template <typename T> auto mASin(const T value)->decltype(asin(value)) { return asin(value); }
template <typename T> auto mACos(const T value)->decltype(acos(value)) { return acos(value); }
template <typename T> auto mATan(const T value)->decltype(atan(value)) { return atan(value); }
template <typename T, typename U> auto mATan2(const T value, const U value2)->decltype(atan2(value, value2)) { return atan2(value, value2); }
template <typename T, typename U> auto mPow(const T value, const U value2)->decltype(pow(value, value2)) { return pow(value, value2); }

template <typename T> constexpr T mMax(const T a, const T b) { return (a >= b) ? a : b; }
template <typename T> constexpr T mMin(const T a, const T b) { return (a <= b) ? a : b; }

template <typename T, typename U>
constexpr auto mLerp(const T a, const T b, const U ratio) -> decltype(a + (b - a) * ratio) { return a + (b - a) * ratio; }

template <typename T, typename U>
auto mBiLerp(const T a, const T b, const T c, const T d, const U ratio1, const U ratio2) -> decltype(mLerp(mLerp(a, b, ratio1), mLerp(c, d, ratio1), ratio2)) { return mLerp(mLerp(a, b, ratio1), mLerp(c, d, ratio1), ratio2); }

template <typename T, typename U>
T mTriLerp(const T a, const T b, const T c, const T d, const T e, const T f, const T g, const T h, const U factor1, const U factor2, const U factor3) 
{
  const U inverseFactor1 = (U)1 - factor1;
  const U inverseFactor2 = (U)1 - factor2;

  const T c00 = a * inverseFactor1 + e * factor1;
  const T c01 = b * inverseFactor1 + f * factor1;
  const T c10 = c * inverseFactor1 + g * factor1;
  const T c11 = d * inverseFactor1 + h * factor1;

  const T c0 = c00 * inverseFactor2 + c10 * factor2;
  const T c1 = c01 * inverseFactor2 + c11 * factor2;

  return c0 * (1.0 - factor3) + c1 * factor3;
}

template <typename T, typename U>
T mInterpolateQuad(const T pos1, const T pos2, const T pos3, const T pos4, const U factor1, const U factor2)
{
  if (factor1 + factor2 <= (U)1)
    return pos1 + (pos2 - pos1) * factor1 + (pos3 - pos1) * factor2;
  else
    return pos4 + (pos3 - pos4) * ((U)1 - factor1) + (pos2 - pos4) * ((U)1 - factor2);
}

//////////////////////////////////////////////////////////////////////////

template <typename T> mINLINE T mClamp(T value, const T &min, const T &max)
{
  if (value > max)
    value = max;

  if (value < min)
    value = min;

  return value;
};

template <typename T>
struct mVec2t
{
  T x, y;

  __host__ __device__ inline mVec2t() : x(0), y(0) {}
  __host__ __device__ inline mVec2t(T _v) : x(_v), y(_v) {}
  __host__ __device__ inline mVec2t(T _x, T _y) : x(_x), y(_y) {}

  template <typename T2> __host__ __device__ inline explicit mVec2t(const mVec2t<T2> &cast) : x((T)cast.x), y((T)cast.y) { }

  __host__ __device__ inline mVec2t<T>  operator +  (const mVec2t<T> &a) const { return mVec2t<T>(x + a.x, y + a.y); };
  __host__ __device__ inline mVec2t<T>  operator -  (const mVec2t<T> &a) const { return mVec2t<T>(x - a.x, y - a.y); };
  __host__ __device__ inline mVec2t<T>  operator *  (const mVec2t<T> &a) const { return mVec2t<T>(x * a.x, y * a.y); };
  __host__ __device__ inline mVec2t<T>  operator /  (const mVec2t<T> &a) const { return mVec2t<T>(x / a.x, y / a.y); };
  __host__ __device__ inline mVec2t<T>& operator += (const mVec2t<T> &a) { return *this = mVec2t<T>(x + a.x, y + a.y); };
  __host__ __device__ inline mVec2t<T>& operator -= (const mVec2t<T> &a) { return *this = mVec2t<T>(x - a.x, y - a.y); };
  __host__ __device__ inline mVec2t<T>& operator *= (const mVec2t<T> &a) { return *this = mVec2t<T>(x * a.x, y * a.y); };
  __host__ __device__ inline mVec2t<T>& operator /= (const mVec2t<T> &a) { return *this = mVec2t<T>(x / a.x, y / a.y); };
  __host__ __device__ inline mVec2t<T>  operator *  (const T a) const { return mVec2t<T>(x * a, y * a); };
  __host__ __device__ inline mVec2t<T>  operator /  (const T a) const { return mVec2t<T>(x / a, y / a); };
  __host__ __device__ inline mVec2t<T>& operator *= (const T a) { return *this = mVec2t<T>(x * a, y * a); };
  __host__ __device__ inline mVec2t<T>& operator /= (const T a) { return *this = mVec2t<T>(x / a, y / a); };
  __host__ __device__ inline mVec2t<T>  operator << (const T a) const { return mVec2t<T>(x << a, y << a); };
  __host__ __device__ inline mVec2t<T>  operator >> (const T a) const { return mVec2t<T>(x >> a, y >> a); };
  __host__ __device__ inline mVec2t<T>& operator <<= (const T a) { return *this = mVec2t<T>(x << a, y << a); };
  __host__ __device__ inline mVec2t<T>& operator >>= (const T a) { return *this = mVec2t<T>(x >> a, y >> a); };

  __host__ __device__ inline mVec2t<T>  operator -  () const { return mVec2t<T>(-x, -y); };

  __host__ __device__ inline bool       operator == (const mVec2t<T> &a) const { return x == a.x && y == a.y; };
  __host__ __device__ inline bool       operator != (const mVec2t<T> &a) const { return x != a.x || y != a.y; };

  __host__ __device__ inline double Length() const { return mSqrt(x * x + y * y); };
  __host__ __device__ inline T LengthSquared() const { return x * x + y * y; };
};

typedef mVec2t<size_t> mVec2s;
typedef mVec2t<int64_t> mVec2i;
typedef mVec2t<uint64_t> mVec2u;
typedef mVec2t<float_t> mVec2f;
typedef mVec2t<double_t> mVec2d;

template <typename T>
struct mVec3t
{
  T x, y, z;

  __host__ __device__ inline mVec3t() : x(0), y(0), z(0) {}
  __host__ __device__ inline mVec3t(T _v) : x(_v), y(_v), z(_v) {}
  __host__ __device__ inline mVec3t(T _x, T _y, T _z) : x(_x), y(_y), z(_z) {}
  __host__ __device__ inline mVec3t(mVec2t<T> vector2, T _z) : x(vector2.x), y(vector2.y), z(_z) {}

  template <typename T2> __host__ __device__ inline explicit mVec3t(const mVec3t<T2> &cast) : x((T)cast.x), y((T)cast.y), z((T)cast.z) {}

  __host__ __device__ inline mVec3t<T>  operator +  (const mVec3t<T> &a) const { return mVec3t<T>(x + a.x, y + a.y, z + a.z); };
  __host__ __device__ inline mVec3t<T>  operator -  (const mVec3t<T> &a) const { return mVec3t<T>(x - a.x, y - a.y, z - a.z); };
  __host__ __device__ inline mVec3t<T>  operator *  (const mVec3t<T> &a) const { return mVec3t<T>(x * a.x, y * a.y, z * a.z); };
  __host__ __device__ inline mVec3t<T>  operator /  (const mVec3t<T> &a) const { return mVec3t<T>(x / a.x, y / a.y, z / a.z); };
  __host__ __device__ inline mVec3t<T>& operator += (const mVec3t<T> &a) { return *this = mVec3t<T>(x + a.x, y + a.y, z + a.z); };
  __host__ __device__ inline mVec3t<T>& operator -= (const mVec3t<T> &a) { return *this = mVec3t<T>(x - a.x, y - a.y, z - a.z); };
  __host__ __device__ inline mVec3t<T>& operator *= (const mVec3t<T> &a) { return *this = mVec3t<T>(x * a.x, y * a.y, z * a.z); };
  __host__ __device__ inline mVec3t<T>& operator /= (const mVec3t<T> &a) { return *this = mVec3t<T>(x / a.x, y / a.y, z / a.z); };
  __host__ __device__ inline mVec3t<T>  operator *  (const T a) const { return mVec3t<T>(x * a, y * a, z * a); };
  __host__ __device__ inline mVec3t<T>  operator /  (const T a) const { return mVec3t<T>(x / a, y / a, z / a); };
  __host__ __device__ inline mVec3t<T>& operator *= (const T a) { return *this = mVec3t<T>(x * a, y * a, z * a); };
  __host__ __device__ inline mVec3t<T>& operator /= (const T a) { return *this = mVec3t<T>(x / a, y / a, z / a); };

  __host__ __device__ inline mVec3t<T>  operator -  () const { return mVec3t<T>(-x, -y, -z); };

  __host__ __device__ inline bool       operator == (const mVec3t<T> &a) const { return x == a.x && y == a.y && z == a.z; };
  __host__ __device__ inline bool       operator != (const mVec3t<T> &a) const { return x != a.x || y != a.y || z != a.z; };

  __host__ __device__ inline double Length() const { return mSqrt(x * x + y * y + z * z); };
  __host__ __device__ inline T LengthSquared() const { return x * x + y * y + z * z; };

  __host__ __device__ inline mVec2t<T> ToVector2() const { return mVec2t<T>(x, y) };
};

typedef mVec3t<size_t> mVec3s;
typedef mVec3t<int64_t> mVec3i;
typedef mVec3t<uint64_t> mVec3u;
typedef mVec3t<float_t> mVec3f;
typedef mVec3t<double_t> mVec3d;

template <typename T>
struct mVec4t
{
  T x, y, z, w;

  __host__ __device__ inline mVec4t() : x(0), y(0), z(0), w(0) {}
  __host__ __device__ inline mVec4t(T _v) : x(_v), y(_v), z(_v), w(_v) {}
  __host__ __device__ inline mVec4t(T _x, T _y, T _z, T _w) : x(_x), y(_y), z(_z), w(_w) {}
  __host__ __device__ inline mVec4t(mVec3t<T> vector3, T _w) : x(vector3.x), y(vector3.y), z(vector3.z), w(_w) {}

  template <typename T2> __host__ __device__ inline explicit mVec4t(const mVec4t<T2> &cast) : x((T)cast.x), y((T)cast.y), z((T)cast.z), w((T)cast.w) {}

  __host__ __device__ inline mVec4t<T>  operator +  (const mVec4t<T> &a) const { return mVec4t<T>(x + a.x, y + a.y, z + a.z, w + a.w); };
  __host__ __device__ inline mVec4t<T>  operator -  (const mVec4t<T> &a) const { return mVec4t<T>(x - a.x, y - a.y, z - a.z, w - a.w); };
  __host__ __device__ inline mVec4t<T>  operator *  (const mVec4t<T> &a) const { return mVec4t<T>(x * a.x, y * a.y, z * a.z, w * a.w); };
  __host__ __device__ inline mVec4t<T>  operator /  (const mVec4t<T> &a) const { return mVec4t<T>(x / a.x, y / a.y, z / a.z, w / a.w); };
  __host__ __device__ inline mVec4t<T>& operator += (const mVec4t<T> &a) { return *this = mVec4t<T>(x + a.x, y + a.y, z + a.z, a.w + a.w); };
  __host__ __device__ inline mVec4t<T>& operator -= (const mVec4t<T> &a) { return *this = mVec4t<T>(x - a.x, y - a.y, z - a.z, a.w - a.w); };
  __host__ __device__ inline mVec4t<T>& operator *= (const mVec4t<T> &a) { return *this = mVec4t<T>(x * a.x, y * a.y, z * a.z, a.w * a.w); };
  __host__ __device__ inline mVec4t<T>& operator /= (const mVec4t<T> &a) { return *this = mVec4t<T>(x / a.x, y / a.y, z / a.z, a.w / a.w); };
  __host__ __device__ inline mVec4t<T>  operator *  (const T a) const { return mVec4t<T>(x * a, y * a, z * a, w * a); };
  __host__ __device__ inline mVec4t<T>  operator /  (const T a) const { return mVec4t<T>(x / a, y / a, z / a, w / a); };
  __host__ __device__ inline mVec4t<T>& operator *= (const T a) { return *this = mVec4t<T>(x * a, y * a, z * a, w * a); };
  __host__ __device__ inline mVec4t<T>& operator /= (const T a) { return *this = mVec4t<T>(x / a, y / a, z / a, w / a); };

  __host__ __device__ inline mVec4t<T>  operator -  () const { return mVec4t<T>(-x, -y, -z, -w); };

  __host__ __device__ inline bool       operator == (const mVec4t<T> &a) const { return x == a.x && y == a.y && z == a.z && w == a.w; };
  __host__ __device__ inline bool       operator != (const mVec4t<T> &a) const { return x != a.x || y != a.y || z != a.z || w != a.w; };

  __host__ __device__ inline double Length() const { return mSqrt(x * x + y * y + z * z + w * w); };
  __host__ __device__ inline T LengthSquared() const { return x * x + y * y + z * z + w * w; };

  __host__ __device__ inline mVec2t<T> ToVector2() const { return mVec2t<T>(x, y) };
  __host__ __device__ inline mVec3t<T> ToVector3() const { return mVec3t<T>(x, y, z) };
};

typedef mVec4t<size_t> mVec4s;
typedef mVec4t<int64_t> mVec4i;
typedef mVec4t<uint64_t> mVec4u;
typedef mVec4t<float_t> mVec4f;
typedef mVec4t<double_t> mVec4d;

template <typename T>
struct mRectangle2D
{
  T x, y, w, h;

  __host__ __device__ inline mRectangle2D(const T x, const T y, const T w, const T h) : x(x), y(y), w(w), h(h) { }
};

struct mVector;

mVec3f mColor_UnpackBgraToVec3f(const uint32_t bgraColor);
mVec3d mColor_UnpackBgraToVec3d(const uint32_t bgraColor);
mVec4f mColor_UnpackBgraToVec4f(const uint32_t bgraColor);
mVec4d mColor_UnpackBgraToVec4d(const uint32_t bgraColor);
mVector mColor_UnpackBgraToVector(const uint32_t bgraColor);

uint32_t mColor_PackVec3fToBgra(const mVec3f rgbVector);
uint32_t mColor_PackVec3dToBgra(const mVec3d rgbVector);
uint32_t mColor_PackVec4fToBgra(const mVec4f rgbaVector);
uint32_t mColor_PackVec4dToBgra(const mVec4d rgbaVector);
uint32_t mColor_PackVectorToBgra(const mVector rgbaVector);

__host__ __device__ inline mVec3f mColor_HueToVec3f(const float_t hue)
{
  const float_t h = hue * 6;
  const float_t r = mAbs(h - 3) - 1;
  const float_t g = 2 - mAbs(h - 2);
  const float_t b = 2 - mAbs(h - 4);

  return mVec3f(mClamp(r, 0.0f, 1.0f), mClamp(g, 0.0f, 1.0f), mClamp(b, 0.0f, 1.0f));
}

__host__ __device__ inline uint32_t mColor_HueToBgra(const float_t hue)
{
  return mColor_PackVec3fToBgra(mColor_HueToVec3f(hue));
}

#endif // mMath_h__
