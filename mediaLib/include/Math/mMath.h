#ifndef mMath_h__
#define mMath_h__

#include "mediaLib.h"

#ifndef __CUDA_ARCH__
#define __host__
#define __device__
#endif // !__CUDA_ARCH__

#define mPI M_PI
#define mTWOPI 6.283185307179586476925286766559
#define mHALFPI M_PI_2
#define mQUARTERPI M_PI_4
#define mSQRT2 M_SQRT2
#define mINVSQRT2 M_SQRT1_2
#define mSQRT3 1.414213562373095048801688724209698
#define mINV_SQRT3 0.5773502691896257645091487805019574556
#define mPIf 3.141592653589793f
#define mTWOPIf 6.283185307179586f
#define mHALFPIf ((float)M_PI_2)
#define mQUARTERPIf ((float)M_PI_4)
#define mSQRT2f 1.414213562373095f
#define mINVSQRT2f 0.7071067811865475f
#define mSQRT3f 1.414213562373095f
#define mINVSQRT3f 0.57735026918962576f

#define mDEG2RAD (mPI / 180.0)
#define mDEG2RADf (mPIf / 180.0f)
#define mRAD2DEG (180.0 / mPI)
#define mRAD2DEGf (180.0f / mPIf)

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

//////////////////////////////////////////////////////////////////////////

template <typename T> mINLINE T constexpr mClamp(const T value, const T min, const T max)
{
  return value <= max ? (value >= min ? value : min) : max;
};

template <typename T>
T mMinValue();

template <typename T>
T mMaxValue();

template <typename T>
T mSmallest();

template <typename T>
T mSmallest(const T scale);

template <typename T>
struct mVec2t
{
#pragma warning(push)
#pragma warning(disable: 4201)
  union
  {
    T asArray[2];

    struct
    {
      T x, y;
    };
  };
#pragma warning(pop)

  __host__ __device__ inline mVec2t() : x(0), y(0) {}
  __host__ __device__ inline explicit mVec2t(T _v) : x(_v), y(_v) {}
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
  __host__ __device__ inline mVec2t<T> Normalize() { return *this / (T)Length(); };
};

template <typename T>
__host__ __device__ inline mVec2t<T>  operator *  (const T a, const mVec2t<T> b) { return mVec2t<T>(a * b.x, a * b.y); };

template <typename T>
__host__ __device__ inline mVec2t<T>  operator /  (const T a, const mVec2t<T> b) { return mVec2t<T>(a / b.x, a / b.y); };

typedef mVec2t<size_t> mVec2s;
typedef mVec2t<int64_t> mVec2i;
typedef mVec2t<uint64_t> mVec2u;
typedef mVec2t<float_t> mVec2f;
typedef mVec2t<double_t> mVec2d;

template <typename T>
struct mVec3t
{
#pragma warning(push)
#pragma warning(disable: 4201)
  union
  {
    T asArray[3];

    struct
    {
      T x, y, z;
    };
  };
#pragma warning(pop)

  __host__ __device__ inline mVec3t() : x(0), y(0), z(0) {}
  __host__ __device__ inline explicit mVec3t(T _v) : x(_v), y(_v), z(_v) {}
  __host__ __device__ inline mVec3t(T _x, T _y, T _z) : x(_x), y(_y), z(_z) {}
  __host__ __device__ inline explicit mVec3t(mVec2t<T> vector2, T _z) : x(vector2.x), y(vector2.y), z(_z) {}

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
  __host__ __device__ inline mVec3t<T> Normalize() { return *this / (T)Length(); };

  __host__ __device__ inline mVec2t<T> ToVector2() const { return mVec2t<T>(x, y); };
};

template <typename T>
__host__ __device__ inline mVec3t<T>  operator *  (const T a, const mVec3t<T> b) { return mVec3t<T>(a * b.x, a * b.y, a * b.z); };

template <typename T>
__host__ __device__ inline mVec3t<T>  operator /  (const T a, const mVec3t<T> b) { return mVec3t<T>(a / b.x, a / b.y, a / b.z); };

typedef mVec3t<size_t> mVec3s;
typedef mVec3t<int64_t> mVec3i;
typedef mVec3t<uint64_t> mVec3u;
typedef mVec3t<float_t> mVec3f;
typedef mVec3t<double_t> mVec3d;

template <typename T>
struct mVec4t
{
#pragma warning(push)
#pragma warning(disable: 4201)
  union
  {
    T asArray[4];

    struct
    {
      T x, y, z, w;
    };
  };
#pragma warning(pop)

  __host__ __device__ inline mVec4t() : x(0), y(0), z(0), w(0) {}
  __host__ __device__ inline explicit mVec4t(T _v) : x(_v), y(_v), z(_v), w(_v) {}
  __host__ __device__ inline mVec4t(T _x, T _y, T _z, T _w) : x(_x), y(_y), z(_z), w(_w) {}
  __host__ __device__ inline explicit mVec4t(mVec3t<T> vector3, T _w) : x(vector3.x), y(vector3.y), z(vector3.z), w(_w) {}

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
  __host__ __device__ inline mVec4t<T> Normalize() { return *this / (T)Length(); };

  __host__ __device__ inline mVec2t<T> ToVector2() const { return mVec2t<T>(x, y); };
  __host__ __device__ inline mVec3t<T> ToVector3() const { return mVec3t<T>(x, y, z); };
};

template <typename T>
__host__ __device__ inline mVec4t<T>  operator *  (const T a, const mVec4t<T> b) { return mVec4t<T>(a * b.x, a * b.y, a * b.z, a * b.w); };

template <typename T>
__host__ __device__ inline mVec4t<T>  operator /  (const T a, const mVec4t<T> b) { return mVec4t<T>(a / b.x, a / b.y, a / b.z, a / b.w); };

typedef mVec4t<size_t> mVec4s;
typedef mVec4t<int64_t> mVec4i;
typedef mVec4t<uint64_t> mVec4u;
typedef mVec4t<float_t> mVec4f;
typedef mVec4t<double_t> mVec4d;

template <typename T>
struct mRectangle2D
{
#pragma warning(push)
#pragma warning(disable: 4201)
  union
  {
    T asArray[4];

    struct
    {
      T x, y;
      
      union
      {
        struct
        {
          T width, height;
        };

        struct
        {
          T w, h;
        };
      };
    };
  };
#pragma warning(pop)

  __host__ __device__ inline mRectangle2D() : x(0), y(0), w(0), h(0) { }
  __host__ __device__ inline mRectangle2D(const T x, const T y, const T w, const T h) : x(x), y(y), w(w), h(h) { }
  __host__ __device__ inline mRectangle2D(const mVec2t<T> &position, const mVec2t<T> size) : x(position.x), y(position.y), w(size.x), h(size.y) { }

  __host__ __device__ inline bool Contains(const mVec2t<T> &position)
  {
    return position.x >= x && position.y >= y && position.x - x < w && position.y - y < h;
  }
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

template<typename T, typename U>
inline mVec2t<U> mBarycentricInterpolationFactors(const T &p, const T &q, const U &x)
{
  const U wp = (x - q) / (q - p);
  return mVec2t<U>(wp, 1 - wp);
}

template<typename T, typename U>
inline mVec3t<U> mBarycentricInterpolationFactors(const mVec2t<T> &p, const mVec2t<T> &q, const mVec2t<T> &r, const mVec2t<U> &x)
{
  const U divisor = (p.x*(r.y - q.y) + q.x*(p.y - r.y) + r.x*(q.y - p.y));
  const U wp = (x.x*(r.y - q.y) + q.x*(x.y - r.y) + r.x*(q.y - x.y)) / divisor;
  const U wq = -(x.x*(r.y - p.y) + p.x*(x.y - r.y) + r.x*(p.y - x.y)) / divisor;

  return mVec3t<U>(wp, wq, 1 - wp - wq);
}

template<typename T, typename U>
inline mVec4t<U> mBarycentricInterpolationFactors(const mVec3t<T> &p, const mVec3t<T> &q, const mVec3t<T> &r, const mVec3t<T> &s, const mVec3t<U> &x)
{
  const U val0 = (p.y*(s.z - r.z) + r.y*(p.z - s.z) + (r.z - p.z)*s.y);
  const U val1 = (s.y*(x.z - r.z) + r.y*(s.z - x.z) + (r.z - s.z)*x.y);
  const U val2 = (p.y*(x.z - s.z) + s.y*(p.z - x.z) + (s.z - p.z)*x.y);

  const U divisor = (q.x*val0 + p.x*(r.y*(s.z - q.z) + q.y*(r.z - s.z) + (q.z - r.z)*s.y) + r.x*(q.y*(s.z - p.z) + p.y*(q.z - s.z) + (p.z - q.z)*s.y) + (p.y*(r.z - q.z) + q.y*(p.z - r.z) + (q.z - p.z)*r.y)*s.x);
  const U wp = -(r.x*(q.y*(x.z - s.z) + s.y*(q.z - x.z) + (s.z - q.z)*x.y) + q.x*val1 + s.x*(r.y*(x.z - q.z) + q.y*(r.z - x.z) + (q.z - r.z)*x.y) + (q.y*(s.z - r.z) + r.y*(q.z - s.z) + (r.z - q.z)*s.y)*x.x) / divisor;
  const U wq = (r.x*val2 + p.x*val1 + s.x*(r.y*(x.z - p.z) + p.y*(r.z - x.z) + (p.z - r.z)*x.y) + val0*x.x) / divisor;
  const U wr = -(q.x*val2 + p.x*(s.y*(x.z - q.z) + q.y*(s.z - x.z) + (q.z - s.z)*x.y) + s.x*(q.y*(x.z - p.z) + p.y*(q.z - x.z) + (p.z - q.z)*x.y) + (p.y*(s.z - q.z) + q.y*(p.z - s.z) + (q.z - p.z)*s.y)*x.x) / divisor;

  return mVec4t<U>(wp, wq, wr, 1 - wp - wq - wr)
}

//////////////////////////////////////////////////////////////////////////

template<>
inline int8_t mMinValue<int8_t>()
{
  return INT8_MIN;
}

template<>
inline int16_t mMinValue<int16_t>()
{
  return INT16_MIN;
}

template<>
inline int32_t mMinValue<int32_t>()
{
  return INT32_MIN;
}

template<>
inline int64_t mMinValue<int64_t>()
{
  return INT64_MIN;
}

template<>
inline uint8_t mMinValue<uint8_t>()
{
  return 0;
}

template<>
inline uint16_t mMinValue<uint16_t>()
{
  return 0;
}

template<>
inline uint32_t mMinValue<uint32_t>()
{
  return 0;
}

template<>
inline uint64_t mMinValue<uint64_t>()
{
  return 0;
}

template<>
inline float_t mMinValue<float_t>()
{
  return -FLT_MAX;
}

template<>
inline double_t mMinValue<double_t>()
{
  return -DBL_MAX;
}

template<>
inline int8_t mMaxValue<int8_t>()
{
  return INT8_MAX;
}

template<>
inline int16_t mMaxValue<int16_t>()
{
  return INT16_MAX;
}

template<>
inline int32_t mMaxValue<int32_t>()
{
  return INT32_MAX;
}

template<>
inline int64_t mMaxValue<int64_t>()
{
  return INT64_MAX;
}

template<>
inline uint8_t mMaxValue<uint8_t>()
{
  return UINT8_MAX;
}

template<>
inline uint16_t mMaxValue<uint16_t>()
{
  return UINT16_MAX;
}

template<>
inline uint32_t mMaxValue<uint32_t>()
{
  return UINT32_MAX;
}

template<>
inline uint64_t mMaxValue<uint64_t>()
{
  return UINT64_MAX;
}

template<>
inline float_t mMaxValue<float_t>()
{
  return FLT_MAX;
}

template<>
inline double_t mMaxValue<double_t>()
{
  return DBL_MAX;
}

template<>
inline int8_t mSmallest<int8_t>()
{
  return 1;
}

template<>
inline uint8_t mSmallest<uint8_t>()
{
  return 1;
};

template<>
inline int16_t mSmallest<int16_t>()
{
  return 1;
};

template<>
inline uint16_t mSmallest<uint16_t>()
{
  return 1;
};

template<>
inline int32_t mSmallest<int32_t>()
{
  return 1;
};

template<>
inline uint32_t mSmallest<uint32_t>()
{
  return 1;
};

template<>
inline int64_t mSmallest<int64_t>()
{
  return 1;
};

template<>
inline uint64_t mSmallest<uint64_t>()
{
  return 1;
};

template<>
inline float_t mSmallest<float_t>()
{
  return FLT_EPSILON;
};

template<>
inline double_t mSmallest<double_t>()
{
  return DBL_EPSILON;
};

template<>
inline float_t mSmallest<float>(const float_t scale)
{
  return mSmallest<float>() * scale;
};

template<>
inline double_t mSmallest<double>(const double_t scale)
{
  return mSmallest<double>() * scale;
};

template<typename T>
inline T mSmallest(const T scale)
{
  udUnused(scale);
  return mSmallest<T>();
};

#endif // mMath_h__
