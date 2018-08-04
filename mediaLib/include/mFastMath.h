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

struct mMatrix
{
  DirectX::FXMMATRIX m;
};

struct mVector
{
  union
  {
    DirectX::XMVECTOR v;
    struct { float_t _x, _y, _z, _w; };
    float_t _v[4];
  };

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
  inline float_t y(const float_t _y);
  inline float_t z(const float_t _z);
  inline float_t w(const float_t _w);

  inline explicit operator mVec2f() const;
  inline explicit operator mVec3f() const;
  inline explicit operator mVec4f() const;

  inline explicit operator DirectX::XMFLOAT2() const;
  inline explicit operator DirectX::XMFLOAT3() const;
  inline explicit operator DirectX::XMFLOAT4() const;

  inline mVec4t<bool> ComponentEquals(const mVector &a) const;

  inline mVector AngleBetweenNormals2(const mVector &a) const;
  inline mVector AngleBetweenNormalsEst2(const mVector &a) const;
  inline mVector AngleBetweenVectors2(const mVector &a) const;
  inline mVector ClampLength2(const float_t min, const float_t max) const;
  inline mVector ClampLengthVectors2(const mVector &min, const mVector &max) const;
  inline mVector Dot2(const mVector &a) const;
  inline mVector Cross2(const mVector &a) const;
  inline bool Equals2(const mVector &a) const;
  inline bool NotEqualTo2(const mVector &a) const;
  inline bool EqualsApproximately2(const mVector &a, const mVector &epsilon) const;
  inline bool Greater2(const mVector &a) const;
  inline bool GreaterOrEqual2(const mVector &a) const;
  inline bool Less2(const mVector &a) const;
  inline bool LessOrEqual2(const mVector &a) const;
  inline bool InBounds(const mVector &a) const;
  inline static mVector IntersectLine2(const mVector &line1Point1, const mVector &line1Point2, const mVector &line2Point1, const mVector &line2Point2);
  inline static float_t LinePointDistance2(const mVector &line1Point1, const mVector &line1Point2, const mVector &point);
  inline float_t Length2() const;
  inline float_t LengthEst2() const;
  inline float_t LengthSquared2() const;
  inline mVector Normalize2() const;
  inline mVector NormalizeEst2() const;
  inline mVector Orthogonal2() const;
  inline mVector ReciprocalLength2() const;
  inline mVector ReciprocalLengthEst2() const;
  inline static mVector Reflect2(const mVector &incident, const mVector &normal);
  inline static mVector Refract2(const mVector &incident, const mVector &normal, const float_t refractionIndex);
  inline static mVector RefractVector2(const mVector &incident, const mVector &normal, const mVector &refractionIndex);
  inline mVector Transform2(const mMatrix &matrix) const;
  inline mVector TransformCoord2(const mMatrix &matrix) const;
  inline mVector TransformNormal2(const mMatrix &matrix) const;
  inline static mFUNCTION(TransformCoordStream2, OUT DirectX::XMFLOAT2 *pOutputData, const size_t outputStride, IN DirectX::XMFLOAT2 *pInputData, const size_t inputStride, const size_t inputLength, const mMatrix &matrix);
  inline static mFUNCTION(TransformNormalStream2, OUT DirectX::XMFLOAT2 *pOutputData, const size_t outputStride, IN DirectX::XMFLOAT2 *pInputData, const size_t inputStride, const size_t inputLength, const mMatrix &matrix);
};

#endif // mFastMath_h__
