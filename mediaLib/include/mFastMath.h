// Copyright 2018 Christoph Stiller
// 
// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files(the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions :
// 
// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
// 
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef mFastMath_h__
#define mFastMath_h__

#include "default.h"

#pragma warning(push) 
#pragma warning(disable: 4577)
#include "DirectXMath.h"
#pragma warning(pop)

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

  inline static mVector mVECTORCALL Min(const mVector &a, const mVector &b);
  inline static mVector mVECTORCALL Max(const mVector &a, const mVector &b);
  inline static mVector mVECTORCALL Lerp(const mVector &a, const mVector &b, const float_t t);
  inline static mVector mVECTORCALL LerpVector(const mVector &a, const mVector &b, const mVector &t);
  inline static mVector mVECTORCALL Barycentric(const mVector &a, const mVector &b, const mVector &c, const float_t f, const float_t g);
  inline static mVector mVECTORCALL BarycentricVector(const mVector &a, const mVector &b, const mVector &c, const mVector &f, const mVector &g);
  inline static mVector mVECTORCALL CatmullRom(const mVector &a, const mVector &b, const mVector &c, const mVector &d, const float_t f);
  inline static mVector mVECTORCALL CatmullRomVector(const mVector &a, const mVector &b, const mVector &c, const mVector &d, const mVector &f);
  inline static mVector mVECTORCALL Hermite(const mVector &v1, const mVector &t1, const mVector &v2, const mVector &t2, const float_t f);
  inline static mVector mVECTORCALL HermiteVector(const mVector &v1, const mVector &t1, const mVector &v2, const mVector &t2, const mVector &f);

  inline mVector AngleBetweenNormals2(const mVector &a) const;
  inline mVector AngleBetweenNormalsEst2(const mVector &a) const;
  inline mVector AngleBetweenVectors2(const mVector &a) const;
  inline mVector ClampLength2(const float_t min, const float_t max) const;
  inline mVector ClampLengthVectors2(const mVector &min, const mVector &max) const;
  inline float_t Dot2(const mVector &a) const;
  inline mVector Cross2(const mVector &a) const;
  inline bool Equals2(const mVector &a) const;
  inline bool NotEqualTo2(const mVector &a) const;
  inline bool EqualsApproximately2(const mVector &a, const mVector &epsilon) const;
  inline bool Greater2(const mVector &a) const;
  inline bool GreaterOrEqual2(const mVector &a) const;
  inline bool Less2(const mVector &a) const;
  inline bool LessOrEqual2(const mVector &a) const;
  inline bool InBounds2(const mVector &a) const;
  inline static mVector mVECTORCALL IntersectLine2(const mVector &line1Point1, const mVector &line1Point2, const mVector &line2Point1, const mVector &line2Point2);
  inline static float_t mVECTORCALL LinePointDistance2(const mVector &line1Point1, const mVector &line1Point2, const mVector &point);
  inline float_t Length2() const;
  inline float_t LengthEst2() const;
  inline float_t LengthSquared2() const;
  inline mVector Normalize2() const;
  inline mVector NormalizeEst2() const;
  inline mVector Orthogonal2() const;
  inline mVector ReciprocalLength2() const;
  inline mVector ReciprocalLengthEst2() const;
  inline static mVector mVECTORCALL Reflect2(const mVector &incident, const mVector &normal);
  inline static mVector mVECTORCALL Refract2(const mVector &incident, const mVector &normal, const float_t refractionIndex);
  inline static mVector mVECTORCALL RefractVector2(const mVector &incident, const mVector &normal, const mVector &refractionIndex);
  inline mVector Transform2(const mMatrix &matrix) const;
  inline mVector TransformCoord2(const mMatrix &matrix) const;
  inline mVector TransformNormal2(const mMatrix &matrix) const;

  inline static mResult mVECTORCALL TransformCoordStream2(OUT DirectX::XMFLOAT2 *pOutputData, const size_t outputStride, IN DirectX::XMFLOAT2 *pInputData, const size_t inputStride, const size_t inputLength, const mMatrix &matrix);
  inline static mResult mVECTORCALL TransformNormalStream2(OUT DirectX::XMFLOAT2 *pOutputData, const size_t outputStride, IN DirectX::XMFLOAT2 *pInputData, const size_t inputStride, const size_t inputLength, const mMatrix &matrix);

  inline mVector AngleBetweenNormals3(const mVector &a) const;
  inline mVector AngleBetweenNormalsEst3(const mVector &a) const;
  inline mVector AngleBetweenVectors3(const mVector &a) const;
  inline mVector ClampLength3(const float_t min, const float_t max) const;
  inline mVector ClampLengthVectors3(const mVector &min, const mVector &max) const;
  inline float_t Dot3(const mVector &a) const;
  inline mVector Cross3(const mVector &a) const;
  inline bool Equals3(const mVector &a) const;
  inline bool NotEqualTo3(const mVector &a) const;
  inline bool EqualsApproximately3(const mVector &a, const mVector &epsilon) const;
  inline bool Greater3(const mVector &a) const;
  inline bool GreaterOrEqual3(const mVector &a) const;
  inline bool Less3(const mVector &a) const;
  inline bool LessOrEqual3(const mVector &a) const;
  inline bool InBounds3(const mVector &a) const;
  inline static float_t mVECTORCALL LinePointDistance3(const mVector &line1Point1, const mVector &line1Point2, const mVector &point);
  inline float_t Length3() const;
  inline float_t LengthEst3() const;
  inline float_t LengthSquared3() const;
  inline mVector Normalize3() const;
  inline mVector NormalizeEst3() const;
  inline mVector Orthogonal3() const;
  inline mVector ReciprocalLength3() const;
  inline mVector ReciprocalLengthEst3() const;
  inline static mVector mVECTORCALL Reflect3(const mVector &incident, const mVector &normal);
  inline static mVector mVECTORCALL Refract3(const mVector &incident, const mVector &normal, const float_t refractionIndex);
  inline static mVector mVECTORCALL RefractVector3(const mVector &incident, const mVector &normal, const mVector &refractionIndex);
  inline mVector Transform3(const mMatrix &matrix) const;
  inline mVector TransformCoord3(const mMatrix &matrix) const;
  inline mVector TransformNormal3(const mMatrix &matrix) const;
  inline mVector Rotate3(const mQuaternion &quaternion) const;
  inline mVector RotateInverse3(const mQuaternion &quaternion) const;

  inline static mResult mVECTORCALL TransformCoordStream3(OUT DirectX::XMFLOAT3 *pOutputData, const size_t outputStride, IN DirectX::XMFLOAT3 *pInputData, const size_t inputStride, const size_t inputLength, const mMatrix &matrix);
  inline static mResult mVECTORCALL TransformNormalStream3(OUT DirectX::XMFLOAT3 *pOutputData, const size_t outputStride, IN DirectX::XMFLOAT3 *pInputData, const size_t inputStride, const size_t inputLength, const mMatrix &matrix);
  inline static mResult mVECTORCALL ComponentsFromNormal3(OUT mVector *pParallel, OUT mVector *pPerpendicular, const mVector &v, const mVector &normal);

  inline mVector AngleBetweenNormals4(const mVector &a) const;
  inline mVector AngleBetweenNormalsEst4(const mVector &a) const;
  inline mVector AngleBetweenVectors4(const mVector &a) const;
  inline mVector ClampLength4(const float_t min, const float_t max) const;
  inline mVector ClampLengthVectors4(const mVector &min, const mVector &max) const;
  inline float_t Dot4(const mVector &a) const;
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
  inline static mVector mVECTORCALL Reflect4(const mVector &incident, const mVector &normal);
  inline static mVector mVECTORCALL Refract4(const mVector &incident, const mVector &normal, const float_t refractionIndex);
  inline static mVector mVECTORCALL RefractVector4(const mVector &incident, const mVector &normal, const mVector &refractionIndex);
  inline mVector Transform4(const mMatrix &matrix) const;

  inline static mResult mVECTORCALL TransformStream4(OUT DirectX::XMFLOAT4 *pOutputData, const size_t outputStride, IN DirectX::XMFLOAT4 *pInputData, const size_t inputStride, const size_t inputLength, const mMatrix &matrix);
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

  explicit mQuaternion(DirectX::XMVECTOR v);

  inline mQuaternion operator *(const mQuaternion &q1) const;
  inline mQuaternion& operator *=(const mQuaternion &q1);

  inline static mQuaternion mVECTORCALL BaryCentric(const mQuaternion &q0, const mQuaternion &q1, const mQuaternion &q2, const float_t f, const float_t g);
  inline static mQuaternion mVECTORCALL BaryCentricV(const mQuaternion &q0, const mQuaternion &q1, const mQuaternion &q2, const mVector &f, const mVector &g);
  inline mQuaternion Conjugate() const;
  inline float_t Dot(const mQuaternion &q2) const;
  inline bool Equals(const mQuaternion &q2) const;
  inline mQuaternion Exp() const;
  inline static mQuaternion mVECTORCALL Identity();
  inline mQuaternion Inverse() const;
  inline bool IsIdentity() const;
  inline float_t Length() const;
  inline float_t LengthSq() const;
  inline mQuaternion Ln() const;
  inline mQuaternion Multiply(const mQuaternion &q2) const;
  inline mQuaternion Normalize() const;
  inline mQuaternion NormalizeEst() const;
  inline bool NotEqualTo(const mQuaternion &q2) const;
  inline mVector ReciprocalLength() const;
  inline static mQuaternion mVECTORCALL RotationAxis(const mVector &axis, const float_t angle);
  inline static mQuaternion mVECTORCALL RotationMatrix(const mMatrix &m);
  inline static mQuaternion mVECTORCALL RotationNormal(const mVector &normalAxis, const float_t angle);
  inline static mQuaternion mVECTORCALL RotationRollPitchYaw(const float_t pitch, const float_t yaw, const float_t roll);
  inline static mQuaternion mVECTORCALL RotationRollPitchYawFromVector(const mVector &angles);
  inline static mQuaternion mVECTORCALL Slerp(const mQuaternion &q0, const mQuaternion &q1, const float_t t);
  inline static mQuaternion mVECTORCALL SlerpV(const mQuaternion &q0, const mQuaternion &q1, const mVector &t);
  inline static mQuaternion mVECTORCALL Squad(const mQuaternion &q0, const mQuaternion &q1, const mQuaternion &q2, const mQuaternion &q3, const float_t t);
  inline static mQuaternion mVECTORCALL SquadV(const mQuaternion &q0, const mQuaternion &q1, const mQuaternion &q2, const mQuaternion &q3, const mVector &t);
  inline mResult SquadSetup(OUT mVector *pA, OUT mVector *pB, OUT mVector *pC, const mQuaternion &q0, const mQuaternion &q1, const mQuaternion &q2, const mQuaternion &q3);
  inline mResult ToAxisAngle(OUT mVector *pAxis, OUT float_t *pAngle);
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

  explicit mMatrix(DirectX::XMMATRIX _m);

  inline mMatrix operator *(const mMatrix &q1) const;
  inline mMatrix& operator *=(const mMatrix &q1);
  inline mMatrix& operator =(const mMatrix &copy);

  inline static mMatrix mVECTORCALL AffineTransformation(const mVector &scaling, const mVector &rotationOrigin, const mQuaternion &rotationQuaternion, const mVector &translation);
  inline static mMatrix mVECTORCALL AffineTransformation2D(const mVector &scaling, const mVector &rotationOrigin, const float_t rotation, const mVector &translation);
  inline mResult Decompose(OUT mVector *pOutScale, OUT mQuaternion *pOutRotQuat, OUT mVector *pOutTrans) const;
  inline mVector Determinant() const;
  inline static mMatrix mVECTORCALL Identity();
  inline mMatrix Inverse(OUT OPTIONAL mVector *pDeterminant = nullptr) const;
  inline static mMatrix mVECTORCALL LookAtLH(const mVector &eyePosition, const mVector &focusPosition, const mVector &upDirection);
  inline static mMatrix mVECTORCALL LookAtRH(const mVector &eyePosition, const mVector &focusPosition, const mVector &upDirection);
  inline static mMatrix mVECTORCALL LookToLH(const mVector &eyePosition, const mVector &eyeDirection, const mVector &upDirection);
  inline static mMatrix mVECTORCALL LookToRH(const mVector &eyePosition, const mVector &eyeDirection, const mVector &upDirection);
  inline mMatrix Multiply(const mMatrix &m2) const;
  inline mMatrix MultiplyTranspose(const mMatrix &m2) const;
  inline static mMatrix mVECTORCALL OrthographicLH(const float_t viewWidth, const float_t viewHeight, const float_t nearZ, const float_t farZ);
  inline static mMatrix mVECTORCALL OrthographicRH(const float_t viewWidth, const float_t viewHeight, const float_t nearZ, const float_t farZ);
  inline static mMatrix mVECTORCALL OrthographicOffCenterLH(const float_t viewLeft, const float_t viewRight, const float_t viewBottom, const float_t viewTop, const float_t nearZ, const float_t farZ);
  inline static mMatrix mVECTORCALL OrthographicOffCenterRH(const float_t viewLeft, const float_t viewRight, const float_t viewBottom, const float_t viewTop, const float_t nearZ, const float_t farZ);
  inline static mMatrix mVECTORCALL PerspectiveLH(const float_t viewWidth, const float_t viewHeight, const float_t nearZ, const float_t farZ);
  inline static mMatrix mVECTORCALL PerspectiveRH(const float_t viewWidth, const float_t viewHeight, const float_t nearZ, const float_t farZ);
  inline static mMatrix mVECTORCALL PerspectiveFovLH(const float_t fovAngleY, const float_t aspectRatio, const float_t nearZ, const float_t farZ);
  inline static mMatrix mVECTORCALL PerspectiveFovRH(const float_t fovAngleY, const float_t aspectRatio, const float_t nearZ, const float_t farZ);
  inline static mMatrix mVECTORCALL PerspectiveOffCenterLH(const float_t viewLeft, const float_t viewRight, const float_t viewBottom, const float_t viewTop, const float_t nearZ, const float_t farZ);
  inline static mMatrix mVECTORCALL PerspectiveOffCenterRH(const float_t viewLeft, const float_t viewRight, const float_t viewBottom, const float_t viewTop, const float_t nearZ, const float_t farZ);
  inline static mMatrix mVECTORCALL Reflect(const mVector &reflectionPlane);
  inline static mMatrix mVECTORCALL RotationAxis(const mVector &axis, const float_t angle);
  inline static mMatrix mVECTORCALL RotationQuaternion(const mQuaternion &quaternion);
  inline static mMatrix mVECTORCALL RotationNormal(const mVector &normalAxis, const float_t angle);
  inline static mMatrix mVECTORCALL RotationRollPitchYaw(const float_t pitch, const float_t yaw, const float_t roll);
  inline static mMatrix mVECTORCALL RotationRollPitchYawFromVector(const mVector &angles);
  inline static mMatrix mVECTORCALL RotationX(const float_t angle);
  inline static mMatrix mVECTORCALL RotationY(const float_t angle);
  inline static mMatrix mVECTORCALL RotationZ(const float_t angle);
  inline static mMatrix mVECTORCALL Scale(const float_t scaleX, const float_t scaleY, const float_t scaleZ);
  inline static mMatrix mVECTORCALL ScalingFromVector(const mVector &scale);
  inline static mMatrix mVECTORCALL Shadow(const mVector &shadowPlane, const mVector &lightPosition);
  inline static mMatrix mVECTORCALL Transformation(const mVector &scalingOrigin, const mQuaternion &scalingOrientationQuaternion, const mVector &scaling, const mVector &rotationOrigin, const mQuaternion &rotationQuaternion, const mVector &translation);
  inline static mMatrix mVECTORCALL Transformation2D(const mVector &scalingOrigin, const float_t scalingOrientation, const mVector &scaling, const mVector &rotationOrigin, const float_t rotation, const mVector &translation);
  inline static mMatrix mVECTORCALL Translation(const float_t offsetX, const float_t offsetY, const float_t offsetZ);
  inline static mMatrix mVECTORCALL TranslationFromVector(const mVector &offset);
  inline mMatrix Transpose() const;
};

#include "mFastMath.inl"

#endif // mFastMath_h__
