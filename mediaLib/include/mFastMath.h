// Copyright 2018 Christoph Stiller
// 
// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files(the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions :
// 
// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
// 
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef mFastMath_h__
#define mFastMath_h__

#include "DirectXMath.h"
#include "mMath.h"

struct mVector
{
  DirectX::XMVECTOR v;

  explicit mVector(const float_t s);
  static mVector Scalar(const float_t s);

  mVector(const float_t x, const float_t y);
  mVector(const float_t x, const float_t y, const float_t z);
  mVector(const float_t x, const float_t y, const float_t z, const float_t w);

  explicit mVector(const mVec2f &v);
  explicit mVector(const mVec3f &v);
  explicit mVector(const mVec4f &v);

  explicit mVector(DirectX::XMVECTOR _v);
  explicit mVector(DirectX::XMFLOAT2 _v);
  explicit mVector(DirectX::XMFLOAT3 _v);
  explicit mVector(DirectX::XMFLOAT4 _v);

  inline mVector  operator +    (const mVector &a) const;
  inline mVector  operator -    (const mVector &a) const;
  inline mVector  operator *    (const mVector &a) const;
  inline mVector  operator /    (const mVector &a) const;
  inline mVector& operator +=   (const mVector &a);
  inline mVector& operator -=   (const mVector &a);
  inline mVector& operator *=   (const mVector &a);
  inline mVector& operator /=   (const mVector &a);
  inline mVector  operator *    (const float_t a) const;
  inline mVector  operator /    (const float_t a) const;
  inline mVector& operator *=   (const float_t a);
  inline mVector& operator /=   (const float_t a);

  inline mVector  operator -    () const;

  inline float_t x(const float_t _x);
  inline float_t x() const;
  inline float_t y(const float_t _y);
  inline float_t y() const;
  inline float_t z(const float_t _z);
  inline float_t z() const;
  inline float_t w(const float_t _w);
  inline float_t w() const;

  inline explicit operator mVec2f() const;
  inline explicit operator mVec3f() const;
  inline explicit operator mVec4f() const;

  inline explicit operator DirectX::XMFLOAT2() const;
  inline explicit operator DirectX::XMFLOAT3() const;
  inline explicit operator DirectX::XMFLOAT4() const;

  mVec4t<bool> ComponentEquals(const mVector &a);
};

#endif // mFastMath_h__
