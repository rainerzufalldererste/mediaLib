// Copyright 2018 Christoph Stiller
// 
// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files(the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions :
// 
// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
// 
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "default.h"
#include "mFastMath.h"

mVector::mVector(const float_t s)
{
  const DirectX::XMFLOAT4 f(s, s, s, s);
  v = DirectX::XMLoadFloat4(&f);
}

mVector mVector::Scalar(const float_t s) { return mVector(DirectX::XMLoadFloat(&s)); }

mVector::mVector(const float_t x, const float_t y)
{
  const DirectX::XMFLOAT2 f(x, y);
  v = DirectX::XMLoadFloat2(&f);
}

mVector::mVector(const float_t x, const float_t y, const float_t z)
{
  const DirectX::XMFLOAT3 f(x, y, z);
  v = DirectX::XMLoadFloat3(&f);
}

mVector::mVector(const float_t x, const float_t y, const float_t z, const float_t w)
{
  const DirectX::XMFLOAT4 f(x, y, z, w);
  v = DirectX::XMLoadFloat4(&f);
}

mVector::mVector(const mVec2f & v) : mVector(v.x, v.y) { }
mVector::mVector(const mVec3f & v) : mVector(v.x, v.y, v.z) { }
mVector::mVector(const mVec4f & v) : mVector(v.x, v.y, v.z, v.w) { }

mVector::mVector(DirectX::XMVECTOR _v) : v(_v) { }

mVector::mVector(DirectX::XMFLOAT2 _v) { v = DirectX::XMLoadFloat2(&_v); }
mVector::mVector(DirectX::XMFLOAT3 _v) { v = DirectX::XMLoadFloat3(&_v); }
mVector::mVector(DirectX::XMFLOAT4 _v) { v = DirectX::XMLoadFloat4(&_v); }

inline mVector mVector::operator+(const mVector &a) const { return mVector(DirectX::XMVectorAdd(v, a.x)); }
inline mVector mVector::operator-(const mVector &a) const { return mVector(DirectX::XMVectorSubtract(v, a.v)); }
inline mVector mVector::operator*(const mVector &a) const { return mVector(DirectX::XMVectorMultiply(v, a.v)); }
inline mVector mVector::operator/(const mVector &a) const { return mVector(DirectX::XMVectorDivide(v, a.v)); }

inline mVector & mVector::operator+=(const mVector & a) { return *this = (*this + a); }
inline mVector & mVector::operator-=(const mVector & a) { return *this = (*this - a); }
inline mVector & mVector::operator*=(const mVector & a) { return *this = (*this * a); }
inline mVector & mVector::operator/=(const mVector & a) { return *this = (*this / a); }

inline mVector mVector::operator*(const float_t a) const { return *this * mVector(a); }
inline mVector mVector::operator/(const float_t a) const { return *this / mVector(a); }

inline mVector & mVector::operator*=(const float_t a) { return *this = (*this * mVector(a)); }
inline mVector & mVector::operator/=(const float_t a) { return *this = (*this / mVector(a)); }

inline mVector mVector::operator-() const { return mVector(DirectX::XMVectorNegate(v)); }

inline float_t mVector::x(const float_t _x)
{
  DirectX::XMFLOAT4 f = (DirectX::XMFLOAT4)(*this);
  f.x = _x;
  v = DirectX::XMLoadFloat4(&f);
  return _x;
}

inline float_t mVector::y(const float_t _y)
{
  DirectX::XMFLOAT4 f = (DirectX::XMFLOAT4)(*this);
  f.y = _y;
  v = DirectX::XMLoadFloat4(&f);
  return _y;
}

inline float_t mVector::z(const float_t _z)
{
  DirectX::XMFLOAT4 f = (DirectX::XMFLOAT4)(*this);
  f.z = _z;
  v = DirectX::XMLoadFloat4(&f);
  return _z;
}

inline float_t mVector::w(const float_t _w)
{
  DirectX::XMFLOAT4 f = (DirectX::XMFLOAT4)(*this);
  f.w = _w;
  v = DirectX::XMLoadFloat4(&f);
  return _w;
}

inline float_t mVector::x() const { return ((DirectX::XMFLOAT4)*this).x; }
inline float_t mVector::y() const { return ((DirectX::XMFLOAT4)*this).y; }
inline float_t mVector::z() const { return ((DirectX::XMFLOAT4)*this).z; }
inline float_t mVector::w() const { return ((DirectX::XMFLOAT4)*this).w; }

inline mVector::operator mVec2f() const
{
  DirectX::XMFLOAT2 f;
  DirectX::XMStoreFloat2(&f, v);
  return mVec2f(f.x, f.y);
}

inline mVector::operator mVec3f() const
{
  DirectX::XMFLOAT3 f;
  DirectX::XMStoreFloat3(&f, v);
  return mVec3f(f.x, f.y, f.z);
}

inline mVector::operator mVec4f() const
{
  DirectX::XMFLOAT4 f;
  DirectX::XMStoreFloat4(&f, v);
  return mVec4f(f.x, f.y, f.z, f.w);
}

inline mVector::operator DirectX::XMFLOAT2() const
{
  DirectX::XMFLOAT2 f;
  DirectX::XMStoreFloat2(&f, v);
  return f;
}

inline mVector::operator DirectX::XMFLOAT3() const
{
  DirectX::XMFLOAT3 f;
  DirectX::XMStoreFloat3(&f, v);
  return f;
}

inline mVector::operator DirectX::XMFLOAT4() const
{
  DirectX::XMFLOAT4 f;
  DirectX::XMStoreFloat4(&f, v);
  return f;
}

mVec4t<bool> mVector::ComponentEquals(const mVector & a)
{
  DirectX::XMVECTOR _v = DirectX::XMVectorEqual(v, a.v);
  DirectX::XMFLOAT4 f;
  DirectX::XMStoreFloat4(&f, _v);
  return mVec4t<bool>(f.x, f.y, f.z, f.w);
}
