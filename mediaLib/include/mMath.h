#ifndef mMath_h__
#define mMath_h__

#include "default.h"

#ifndef __CUDA_ARCH__
#define __host__
#define __device__
#endif // !__CUDA_ARCH__


template <typename T> T mClamp(T value, const T &min, const T &max)
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

  __host__ __device__ inline mVec2t(mVec2t<T> &&move) : x(move.x), y(move.y) {}
  __host__ __device__ inline mVec2t(const mVec2t<T> &copy) : x(copy.x), y(copy.y) {}
  template <typename T2> __host__ __device__ inline explicit mVec2t(const mVec2t<T2> &cast) : x((T)cast.x), y((T)cast.y) { }
  __host__ __device__ inline explicit mVec2t(const vec2 &cast) : x((T)cast.x), y((T)cast.y) { }

  __host__ __device__ inline mVec2t<T>& operator =  (mVec2t<T> &&move) { x = move.x; y = move.y; return *this; }
  __host__ __device__ inline mVec2t<T>& operator =  (const mVec2t<T> &copy) { x = copy.x; y = copy.y; return *this; }

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

  __host__ __device__ inline bool      operator == (const mVec2t<T> &a) const { return x == a.x && y == a.y; };

  __host__ __device__ inline double Length() { return sqrt(x * x + y * y); };
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

  __host__ __device__ inline mVec3t(mVec3t<T> &&move) : x(move.x), y(move.y), z(move.z) {}
  __host__ __device__ inline mVec3t(const mVec3t<T> &copy) : x(copy.x), y(copy.y), z(copy.z) {}
  template <typename T2> __host__ __device__ inline explicit mVec3t(const mVec3t<T2> &cast) : x((T)cast.x), y((T)cast.y), z((T)cast.z) {}
  __host__ __device__ inline explicit mVec3t(const vec3 &cast) : x((T)cast.x), y((T)cast.y), z((T)cast.z) {}

  __host__ __device__ inline mVec3t<T>& operator =  (mVec3t<T> &&move) { x = move.x; y = move.y; z = move.z; return *this; }
  __host__ __device__ inline mVec3t<T>& operator =  (const mVec3t<T> &copy) { x = copy.x; y = copy.y; z = copy.z; return *this; }

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

  __host__ __device__ inline bool      operator == (const mVec3t<T> &a) const { return x == a.x && y == a.y && z == a.z; };

  __host__ __device__ inline double Length() { return sqrt(x * x + y * y + z * z); };
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

  __host__ __device__ inline mVec4t(mVec4t<T> &&move) : x(move.x), y(move.y), z(move.z), w(move.w) {}
  __host__ __device__ inline mVec4t(const mVec4t<T> &copy) : x(copy.x), y(copy.y), z(copy.z), w(copy.w) {}
  template <typename T2> __host__ __device__ inline explicit mVec4t(const mVec4t<T2> &cast) : x((T)cast.x), y((T)cast.y), z((T)cast.z), w((T)cast.w) {}
  __host__ __device__ inline explicit mVec4t(const vec4 &cast) : x((T)cast.x), y((T)cast.y), z((T)cast.z), w((T)cast.w) {}

  __host__ __device__ inline mVec4t<T>& operator =  (mVec4t<T> &&move) { x = move.x; y = move.y; z = move.z; w = move.w; return *this; }
  __host__ __device__ inline mVec4t<T>& operator =  (const mVec4t<T> &copy) { x = copy.x; y = copy.y; z = copy.z; w = copy.w; return *this; }

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

  __host__ __device__ inline bool      operator == (const mVec4t<T> &a) const { return x == a.x && y == a.y && z == a.z && w == a.w; };

  __host__ __device__ inline double Length() { return sqrt(x * x + y * y + z * z + w * w); };
};

typedef mVec4t<size_t> mVec4s;
typedef mVec4t<int64_t> mVec4i;
typedef mVec4t<uint64_t> mVec4u;
typedef mVec4t<float_t> mVec4f;
typedef mVec4t<double_t> mVec4d;


#endif // mMath_h__
