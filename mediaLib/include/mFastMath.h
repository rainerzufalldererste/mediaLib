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

struct mMatrix;
struct mQuaternion;

struct mVector
{
#pragma warning(push) 
#pragma warning(disable: 4201)
  union
  {
    DirectX::XMVECTOR v;
    struct { float_t x, y, z, w; };
    float_t _v[4];
  };
#pragma warning(pop)

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

  inline explicit operator mVec2f() const;
  inline explicit operator mVec3f() const;
  inline explicit operator mVec4f() const;

  inline explicit operator DirectX::XMFLOAT2() const;
  inline explicit operator DirectX::XMFLOAT3() const;
  inline explicit operator DirectX::XMFLOAT4() const;

  inline mVec4t<bool> ComponentEquals(const mVector &a) const;
  inline mVector Abs() const;

  inline static mVector Min(const mVector &a, const mVector &b);
  inline static mVector Max(const mVector &a, const mVector &b);
  inline static mVector Lerp(const mVector &a, const mVector &b, const float_t t);
  inline static mVector LerpVector(const mVector &a, const mVector &b, const mVector &t);
  inline static mVector Barycentric(const mVector &a, const mVector &b, const mVector &c, const float_t f, const float_t g);
  inline static mVector BarycentricVector(const mVector &a, const mVector &b, const mVector &c, const mVector &f, const mVector &g);
  inline static mVector CatmullRom(const mVector &a, const mVector &b, const mVector &c, const mVector &d, const float_t f);
  inline static mVector CatmullRomVector(const mVector &a, const mVector &b, const mVector &c, const mVector &d, const mVector &f);
  inline static mVector Hermite(const mVector &v1, const mVector &t1, const mVector &v2, const mVector &t2, const float_t f);
  inline static mVector HermiteVector(const mVector &v1, const mVector &t1, const mVector &v2, const mVector &t2, const mVector &f);

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
  inline bool InBounds2(const mVector &a) const;
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

  inline mVector AngleBetweenNormals3(const mVector &a) const;
  inline mVector AngleBetweenNormalsEst3(const mVector &a) const;
  inline mVector AngleBetweenVectors3(const mVector &a) const;
  inline mVector ClampLength3(const float_t min, const float_t max) const;
  inline mVector ClampLengthVectors3(const mVector &min, const mVector &max) const;
  inline mVector Dot3(const mVector &a) const;
  inline mVector Cross3(const mVector &a) const;
  inline bool Equals3(const mVector &a) const;
  inline bool NotEqualTo3(const mVector &a) const;
  inline bool EqualsApproximately3(const mVector &a, const mVector &epsilon) const;
  inline bool Greater3(const mVector &a) const;
  inline bool GreaterOrEqual3(const mVector &a) const;
  inline bool Less3(const mVector &a) const;
  inline bool LessOrEqual3(const mVector &a) const;
  inline bool InBounds3(const mVector &a) const;
  inline static float_t LinePointDistance3(const mVector &line1Point1, const mVector &line1Point2, const mVector &point);
  inline float_t Length3() const;
  inline float_t LengthEst3() const;
  inline float_t LengthSquared3() const;
  inline mVector Normalize3() const;
  inline mVector NormalizeEst3() const;
  inline mVector Orthogonal3() const;
  inline mVector ReciprocalLength3() const;
  inline mVector ReciprocalLengthEst3() const;
  inline static mVector Reflect3(const mVector &incident, const mVector &normal);
  inline static mVector Refract3(const mVector &incident, const mVector &normal, const float_t refractionIndex);
  inline static mVector RefractVector3(const mVector &incident, const mVector &normal, const mVector &refractionIndex);
  inline mVector Transform3(const mMatrix &matrix) const;
  inline mVector TransformCoord3(const mMatrix &matrix) const;
  inline mVector TransformNormal3(const mMatrix &matrix) const;
  inline mVector Rotate3(const mQuaternion &quaternion) const;
  inline mVector RotateInverse3(const mQuaternion &quaternion) const;

  inline static mFUNCTION(TransformCoordStream3, OUT DirectX::XMFLOAT3 *pOutputData, const size_t outputStride, IN DirectX::XMFLOAT3 *pInputData, const size_t inputStride, const size_t inputLength, const mMatrix &matrix);
  inline static mFUNCTION(TransformNormalStream3, OUT DirectX::XMFLOAT3 *pOutputData, const size_t outputStride, IN DirectX::XMFLOAT3 *pInputData, const size_t inputStride, const size_t inputLength, const mMatrix &matrix);
  inline static mFUNCTION(ComponentsFromNormal3, OUT mVector *pParallel, OUT mVector *pPerpendicular, const mVector &v, const mVector &normal);

  inline mVector AngleBetweenNormals4(const mVector &a) const;
  inline mVector AngleBetweenNormalsEst4(const mVector &a) const;
  inline mVector AngleBetweenVectors4(const mVector &a) const;
  inline mVector ClampLength4(const float_t min, const float_t max) const;
  inline mVector ClampLengthVectors4(const mVector &min, const mVector &max) const;
  inline mVector Dot4(const mVector &a) const;
  inline mVector Cross4(const mVector &a, const mVector & b) const;
  inline bool Equals4(const mVector &a) const;
  inline bool NotEqualTo4(const mVector &a) const;
  inline bool EqualsApproximately4(const mVector &a, const mVector &epsilon) const;
  inline bool Greater4(const mVector &a) const;
  inline bool GreaterOrEqual4(const mVector &a) const;
  inline bool Less4(const mVector &a) const;
  inline bool LessOrEqual4(const mVector &a) const;
  inline bool InBounds4(const mVector &a) const;
  inline float_t Length4() const;
  inline float_t LengthEst4() const;
  inline float_t LengthSquared4() const;
  inline mVector Normalize4() const;
  inline mVector NormalizeEst4() const;
  inline mVector Orthogonal4() const;
  inline mVector ReciprocalLength4() const;
  inline mVector ReciprocalLengthEst4() const;
  inline static mVector Reflect4(const mVector &incident, const mVector &normal);
  inline static mVector Refract4(const mVector &incident, const mVector &normal, const float_t refractionIndex);
  inline static mVector RefractVector4(const mVector &incident, const mVector &normal, const mVector &refractionIndex);
  inline mVector Transform4(const mMatrix &matrix) const;

  inline static mFUNCTION(TransformStream4, OUT DirectX::XMFLOAT4 *pOutputData, const size_t outputStride, IN DirectX::XMFLOAT4 *pInputData, const size_t inputStride, const size_t inputLength, const mMatrix &matrix);
};

struct mQuaternion
{
#pragma warning(push) 
#pragma warning(disable: 4201)
  union
  {
    DirectX::XMVECTOR q;
    struct { float_t x, y, z, w; };
    float_t _q[4];
  };
#pragma warning(pop)
};

struct mMatrix
{
#pragma warning(push) 
#pragma warning(disable: 4201)
  union
  {
    DirectX::XMMATRIX m;
    struct
    {
      float_t _11, _12, _13, _14;
      float_t _21, _22, _23, _24;
      float_t _31, _32, _33, _34;
      float_t _41, _42, _43, _44;
    };
    float_t _m[4][4];
  };
#pragma warning(pop)
};

#include "mFastMath.inl"

#endif // mFastMath_h__
