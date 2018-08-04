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

inline mVec4t<bool> mVector::ComponentEquals(const mVector & a) const
{
  DirectX::XMVECTOR _v = DirectX::XMVectorEqual(v, a.v);
  DirectX::XMFLOAT4 f;
  DirectX::XMStoreFloat4(&f, _v);
  return mVec4t<bool>(f.x, f.y, f.z, f.w);
}

inline mVector mVector::AngleBetweenNormals2(const mVector & a) const { return mVector(DirectX::XMVector2AngleBetweenNormals(v, a.v)); }
inline mVector mVector::AngleBetweenNormalsEst2(const mVector & a) const { return mVector(DirectX::XMVector2AngleBetweenNormalsEst(v, a.v)); }
inline mVector mVector::AngleBetweenVectors2(const mVector & a) const { return mVector(DirectX::XMVector2AngleBetweenVectors(v, a.v)); }
inline mVector mVector::ClampLength2(const float_t min, const float_t max) const { return mVector(DirectX::XMVector2ClampLength(v, min, max)); }
inline mVector mVector::ClampLengthVectors2(const mVector & min, const mVector & max) const { return mVector(DirectX::XMVector2ClampLengthV(v, min.v, max.v)); }
inline mVector mVector::Dot2(const mVector & a) const { return mVector(DirectX::XMVector2Dot(v, a.v)); }
inline mVector mVector::Cross2(const mVector & a) const { return mVector(DirectX::XMVector2Cross(v, a.v)); }
inline bool mVector::Equals2(const mVector & a) const { return DirectX::XMVector2Equal(v, a.v); }
inline bool mVector::NotEqualTo2(const mVector & a) const { return DirectX::XMVector2NotEqual(v, a.v); }
inline bool mVector::EqualsApproximately2(const mVector & a, const mVector & epsilon) const { return DirectX::XMVector2NearEqual(v, a.v, epsilon.v); }
inline bool mVector::Greater2(const mVector & a) const { return DirectX::XMVector2Greater(v, a.v); }
inline bool mVector::GreaterOrEqual2(const mVector & a) const { return DirectX::XMVector2GreaterOrEqual(v, a.v); }
inline bool mVector::Less2(const mVector & a) const { return DirectX::XMVector2Less(v, a.v); }
inline bool mVector::LessOrEqual2(const mVector & a) const { return DirectX::XMVector2LessOrEqual(v, a.v); }
inline bool mVector::InBounds(const mVector & a) const { return DirectX::XMVector2InBounds(v, a.v); }
inline mVector mVector::IntersectLine2(const mVector & line1Point1, const mVector & line1Point2, const mVector & line2Point1, const mVector & line2Point2) { return mVector(DirectX::XMVector2IntersectLine(line1Point1.v, line1Point2.v, line2Point1.v, line2Point2.v)); }
inline float_t mVector::LinePointDistance2(const mVector & line1Point1, const mVector & line1Point2, const mVector & point) { return mVector(DirectX::XMVector2LinePointDistance(line1Point1.v, line1Point2.v, point.v))._x; }
inline float_t mVector::Length2() const { return mVector(DirectX::XMVector2Length(v))._x; }
inline float_t mVector::LengthEst2() const { return mVector(DirectX::XMVector2LengthEst(v))._x; }
inline float_t mVector::LengthSquared2() const { return mVector(DirectX::XMVector2LengthSq(v))._x; }
inline mVector mVector::Normalize2() const { return mVector(DirectX::XMVector2Normalize(v)); }
inline mVector mVector::NormalizeEst2() const { return mVector(DirectX::XMVector2NormalizeEst(v)); }
inline mVector mVector::Orthogonal2() const { return mVector(DirectX::XMVector2Orthogonal(v)); }
inline mVector mVector::ReciprocalLength2() const { return mVector(DirectX::XMVector2ReciprocalLength(v)); }
inline mVector mVector::ReciprocalLengthEst2() const { return mVector(DirectX::XMVector2ReciprocalLengthEst(v)); }
inline mVector mVector::Reflect2(const mVector & incident, const mVector & normal) { return mVector(DirectX::XMVector2Reflect(incident.v, normal.v)); }
inline mVector mVector::Refract2(const mVector & incident, const mVector & normal, const float_t refractionIndex) { return mVector(DirectX::XMVector2Refract(incident.v, normal.v, refractionIndex)); }
inline mVector mVector::RefractVector2(const mVector & incident, const mVector & normal, const mVector & refractionIndex) { return mVector(DirectX::XMVector2RefractV(incident.v, normal.v, refractionIndex.v)); }
inline mVector mVector::Transform2(const mMatrix &matrix) const { return mVector(DirectX::XMVector2Transform(v, (DirectX::FXMMATRIX)matrix.m)); }
inline mVector mVector::TransformCoord2(const mMatrix & matrix) const { return mVector(DirectX::XMVector2TransformCoord(v, (DirectX::FXMMATRIX)matrix.m)); }
inline mVector mVector::TransformNormal2(const mMatrix & matrix) const { return mVector(DirectX::XMVector2TransformNormal(v, (DirectX::FXMMATRIX)matrix.m)); }

inline mFUNCTION(mVector::TransformCoordStream2, OUT DirectX::XMFLOAT2 * pOutputData, const size_t outputStride, IN DirectX::XMFLOAT2 * pInputData, const size_t inputStride, const size_t inputLength, const mMatrix & matrix)
{
  mFUNCTION_SETUP();

  mERROR_IF(pOutputData == nullptr || pInputData == nullptr, mR_ArgumentNull);
  DirectX::XMVector2TransformCoordStream(pOutputData, outputStride, pInputData, inputStride, inputLength, matrix.m);

  mRETURN_SUCCESS();
}

inline mFUNCTION(mVector::TransformNormalStream2, OUT DirectX::XMFLOAT2 * pOutputData, const size_t outputStride, IN DirectX::XMFLOAT2 * pInputData, const size_t inputStride, const size_t inputLength, const mMatrix & matrix)
{
  mFUNCTION_SETUP();

  mERROR_IF(pOutputData == nullptr || pInputData == nullptr, mR_ArgumentNull);
  DirectX::XMVector2TransformNormalStream(pOutputData, outputStride, pInputData, inputStride, inputLength, matrix.m);

  mRETURN_SUCCESS();
}
