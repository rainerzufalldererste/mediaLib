#ifndef mFastMath_h__
#define mFastMath_h__

#include "mediaLib.h"

#pragma warning(push) 
#pragma warning(disable: 4577)
#include "DirectXMath.h"
#include "DirectXCollision.h"
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

  mINLINE mVector() :
    x(0),
    y(0),
    z(0),
    w(1)
  { }

  mINLINE mVector(const mVector &a) :
    v(a.v)
  { }

  mINLINE explicit mVector(const float_t s)
  {
    const DirectX::XMFLOAT4 f(s, s, s, s);
    v = DirectX::XMLoadFloat4(&f);
  }

  mINLINE static mVector Scalar(const float_t s) { return mVector(DirectX::XMLoadFloat(&s)); }

  mINLINE mVector(const float_t x, const float_t y)
  {
    const DirectX::XMFLOAT2 f(x, y);
    v = DirectX::XMLoadFloat2(&f);
  }

  mINLINE mVector(const float_t x, const float_t y, const float_t z)
  {
    const DirectX::XMFLOAT3 f(x, y, z);
    v = DirectX::XMLoadFloat3(&f);
  }

  mINLINE mVector(const float_t x, const float_t y, const float_t z, const float_t w)
  {
    const DirectX::XMFLOAT4 f(x, y, z, w);
    v = DirectX::XMLoadFloat4(&f);
  }

  mINLINE mVector(const mVec2f &v) : mVector(v.x, v.y) { }
  mINLINE mVector(const mVec3f &v) : mVector(v.x, v.y, v.z) { }
  mINLINE mVector(const mVec4f &v) : mVector(v.x, v.y, v.z, v.w) { }

  mINLINE mVector(DirectX::XMVECTOR _v) : v(_v) { }

  mINLINE mVector(DirectX::XMFLOAT2 _v) { v = DirectX::XMLoadFloat2(&_v); }
  mINLINE mVector(DirectX::XMFLOAT3 _v) { v = DirectX::XMLoadFloat3(&_v); }
  mINLINE mVector(DirectX::XMFLOAT4 _v) { v = DirectX::XMLoadFloat4(&_v); }

  mINLINE mVector operator+(const mVector &a) const { return mVector(DirectX::XMVectorAdd(v, a.v)); }
  mINLINE mVector operator-(const mVector &a) const { return mVector(DirectX::XMVectorSubtract(v, a.v)); }
  mINLINE mVector operator*(const mVector &a) const { return mVector(DirectX::XMVectorMultiply(v, a.v)); }
  mINLINE mVector operator/(const mVector &a) const { return mVector(DirectX::XMVectorDivide(v, a.v)); }

  mINLINE mVector & operator+=(const mVector &a) { return *this = (*this + a); }
  mINLINE mVector & operator-=(const mVector &a) { return *this = (*this - a); }
  mINLINE mVector & operator*=(const mVector &a) { return *this = (*this * a); }
  mINLINE mVector & operator/=(const mVector &a) { return *this = (*this / a); }

  mINLINE mVector operator*(const float_t a) const { return mVector(DirectX::XMVectorScale(v, a)); }
  mINLINE mVector operator/(const float_t a) const { return mVector(DirectX::XMVectorScale(v, 1.0f / a)); }

  mINLINE mVector & operator*=(const float_t a) { return *this = (*this * a); }
  mINLINE mVector & operator/=(const float_t a) { return *this = (*this / a); }

  mINLINE mVector operator-() const { return mVector(DirectX::XMVectorNegate(v)); }

  mINLINE operator mVec2f() const
  {
    DirectX::XMFLOAT2 f;
    DirectX::XMStoreFloat2(&f, v);
    return mVec2f(f.x, f.y);
  }

  mINLINE operator mVec3f() const
  {
    DirectX::XMFLOAT3 f;
    DirectX::XMStoreFloat3(&f, v);
    return mVec3f(f.x, f.y, f.z);
  }

  mINLINE operator mVec4f() const
  {
    DirectX::XMFLOAT4 f;
    DirectX::XMStoreFloat4(&f, v);
    return mVec4f(f.x, f.y, f.z, f.w);
  }

  mINLINE operator DirectX::XMFLOAT2() const
  {
    DirectX::XMFLOAT2 f;
    DirectX::XMStoreFloat2(&f, v);
    return f;
  }

  mINLINE operator DirectX::XMFLOAT3() const
  {
    DirectX::XMFLOAT3 f;
    DirectX::XMStoreFloat3(&f, v);
    return f;
  }

  mINLINE operator DirectX::XMFLOAT4() const
  {
    DirectX::XMFLOAT4 f;
    DirectX::XMStoreFloat4(&f, v);
    return f;
  }

  mINLINE mVec4t<bool> ComponentEquals(const mVector &a) const
  {
    DirectX::XMVECTOR __v = DirectX::XMVectorEqual(v, a.v);
    DirectX::XMUINT4 f;
    DirectX::XMStoreUInt4(&f, __v);
    return mVec4t<bool>(f.x != 0, f.y != 0, f.z != 0, f.w != 0);
  }

  mINLINE mVector Abs() const { return mVector(DirectX::XMVectorAbs(v)); }

  mINLINE mVector AngleBetweenNormals2(const mVector &a) const { return mVector(DirectX::XMVector2AngleBetweenNormals(v, a.v)); }
  mINLINE mVector AngleBetweenNormalsEst2(const mVector &a) const { return mVector(DirectX::XMVector2AngleBetweenNormalsEst(v, a.v)); }
  mINLINE mVector AngleBetweenVectors2(const mVector &a) const { return mVector(DirectX::XMVector2AngleBetweenVectors(v, a.v)); }
  mINLINE mVector ClampLength2(const float_t min, const float_t max) const { return mVector(DirectX::XMVector2ClampLength(v, min, max)); }
  mINLINE mVector ClampLengthVectors2(const mVector &min, const mVector &max) const { return mVector(DirectX::XMVector2ClampLengthV(v, min.v, max.v)); }
  mINLINE float_t Dot2(const mVector &a) const { return mVector(DirectX::XMVector2Dot(v, a.v)).x; }
  mINLINE mVector Cross2(const mVector &a) const { return mVector(DirectX::XMVector2Cross(v, a.v)); }
  mINLINE bool Equals2(const mVector &a) const { return DirectX::XMVector2Equal(v, a.v); }
  mINLINE bool NotEqualTo2(const mVector &a) const { return DirectX::XMVector2NotEqual(v, a.v); }
  mINLINE bool EqualsApproximately2(const mVector &a, const mVector &epsilon) const { return DirectX::XMVector2NearEqual(v, a.v, epsilon.v); }
  mINLINE bool Greater2(const mVector &a) const { return DirectX::XMVector2Greater(v, a.v); }
  mINLINE bool GreaterOrEqual2(const mVector &a) const { return DirectX::XMVector2GreaterOrEqual(v, a.v); }
  mINLINE bool Less2(const mVector &a) const { return DirectX::XMVector2Less(v, a.v); }
  mINLINE bool LessOrEqual2(const mVector &a) const { return DirectX::XMVector2LessOrEqual(v, a.v); }
  mINLINE bool InBounds2(const mVector &a) const { return DirectX::XMVector2InBounds(v, a.v); }

  mINLINE static mVector mVECTORCALL Min(const mVector a, const mVector b) { return mVector(DirectX::XMVectorMin(a.v, b.v)); }
  mINLINE static mVector mVECTORCALL Max(const mVector a, const mVector b) { return mVector(DirectX::XMVectorMax(a.v, b.v)); }

  mINLINE static mVector mVECTORCALL Lerp(const mVector a, const mVector b, const float_t t) { return mVector(DirectX::XMVectorLerp(a.v, b.v, t)); }
  mINLINE static mVector mVECTORCALL LerpVector(const mVector a, const mVector b, const mVector t) { return mVector(DirectX::XMVectorLerpV(a.v, b.v, t.v)); }
  mINLINE static mVector mVECTORCALL Barycentric(const mVector a, const mVector b, const mVector c, const float_t f, const float_t g) { return mVector(DirectX::XMVectorBaryCentric(a.v, b.v, c.v, f, g)); }
  mINLINE static mVector mVECTORCALL BarycentricVector(const mVector a, const mVector b, const mVector c, const mVector f, const mVector g) { return mVector(DirectX::XMVectorBaryCentricV(a.v, b.v, c.v, f.v, g.v)); }
  mINLINE static mVector mVECTORCALL CatmullRom(const mVector a, const mVector b, const mVector c, const mVector d, const float_t f) { return mVector(DirectX::XMVectorCatmullRom(a.v, b.v, c.v, d.v, f)); }
  mINLINE static mVector mVECTORCALL CatmullRomVector(const mVector a, const mVector b, const mVector c, const mVector d, const mVector f) { return mVector(DirectX::XMVectorCatmullRomV(a.v, b.v, c.v, d.v, f.v)); }
  mINLINE static mVector mVECTORCALL Hermite(const mVector v1, const mVector t1, const mVector v2, const mVector t2, const float_t f) { return mVector(DirectX::XMVectorHermite(v1.v, t1.v, v2.v, t2.v, f)); }
  mINLINE static mVector mVECTORCALL HermiteVector(const mVector v1, const mVector t1, const mVector v2, const mVector t2, const  mVector &f) { return mVector(DirectX::XMVectorHermiteV(v1.v, t1.v, v2.v, t2.v, f.v)); }

  mINLINE static mVector mVECTORCALL IntersectLine2(const mVector line1Point1, const mVector line1Point2, const mVector line2Point1, const mVector line2Point2) { return mVector(DirectX::XMVector2IntersectLine(line1Point1.v, line1Point2.v, line2Point1.v, line2Point2.v)); }
  mINLINE static float_t mVECTORCALL LinePointDistance2(const mVector line1Point1, const mVector line1Point2, const mVector point) { return mVector(DirectX::XMVector2LinePointDistance(line1Point1.v, line1Point2.v, point.v)).x; }
  
  mINLINE float_t Length2() const { return mVector(DirectX::XMVector2Length(v)).x; }
  mINLINE float_t LengthEst2() const { return mVector(DirectX::XMVector2LengthEst(v)).x; }
  mINLINE float_t LengthSquared2() const { return mVector(DirectX::XMVector2LengthSq(v)).x; }
  mINLINE mVector Normalize2() const { return mVector(DirectX::XMVector2Normalize(v)); }
  mINLINE mVector NormalizeEst2() const { return mVector(DirectX::XMVector2NormalizeEst(v)); }
  mINLINE mVector Orthogonal2() const { return mVector(DirectX::XMVector2Orthogonal(v)); }
  mINLINE mVector ReciprocalLength2() const { return mVector(DirectX::XMVector2ReciprocalLength(v)); }
  mINLINE mVector ReciprocalLengthEst2() const { return mVector(DirectX::XMVector2ReciprocalLengthEst(v)); }

  mINLINE static mVector mVECTORCALL Reflect2(const mVector incident, const mVector normal) { return mVector(DirectX::XMVector2Reflect(incident.v, normal.v)); }
  mINLINE static mVector mVECTORCALL Refract2(const mVector incident, const mVector normal, const float_t refractionIndex) { return mVector(DirectX::XMVector2Refract(incident.v, normal.v, refractionIndex)); }
  mINLINE static mVector mVECTORCALL RefractVector2(const mVector incident, const mVector normal, const mVector refractionIndex) { return mVector(DirectX::XMVector2RefractV(incident.v, normal.v, refractionIndex.v)); }
  
  mVector Transform2(const mMatrix &matrix) const;
  mVector TransformCoord2(const mMatrix &matrix) const;
  mVector TransformNormal2(const mMatrix &matrix) const;

  mINLINE mVector AngleBetweenNormals3(const mVector &a) const { return mVector(DirectX::XMVector3AngleBetweenNormals(v, a.v)); }
  mINLINE mVector AngleBetweenNormalsEst3(const mVector &a) const { return mVector(DirectX::XMVector3AngleBetweenNormalsEst(v, a.v)); }
  mINLINE mVector AngleBetweenVectors3(const mVector &a) const { return mVector(DirectX::XMVector3AngleBetweenVectors(v, a.v)); }
  mINLINE mVector ClampLength3(const float_t min, const float_t max) const { return mVector(DirectX::XMVector3ClampLength(v, min, max)); }
  mINLINE mVector ClampLengthVectors3(const mVector &min, const mVector &max) const { return mVector(DirectX::XMVector3ClampLengthV(v, min.v, max.v)); }
  mINLINE float_t Dot3(const mVector &a) const { return mVector(DirectX::XMVector3Dot(v, a.v)).x; }
  mINLINE mVector Cross3(const mVector &a) const { return mVector(DirectX::XMVector3Cross(v, a.v)); }

  mINLINE bool Equals3(const mVector &a) const { return DirectX::XMVector3Equal(v, a.v); }
  mINLINE bool NotEqualTo3(const mVector &a) const { return DirectX::XMVector3NotEqual(v, a.v); }
  mINLINE bool EqualsApproximately3(const mVector &a, const mVector &epsilon) const { return DirectX::XMVector3NearEqual(v, a.v, epsilon.v); }
  mINLINE bool Greater3(const mVector &a) const { return DirectX::XMVector3Greater(v, a.v); }
  mINLINE bool GreaterOrEqual3(const mVector &a) const { return DirectX::XMVector3GreaterOrEqual(v, a.v); }
  mINLINE bool Less3(const mVector &a) const { return DirectX::XMVector3Less(v, a.v); }
  mINLINE bool LessOrEqual3(const mVector &a) const { return DirectX::XMVector3LessOrEqual(v, a.v); }
  mINLINE bool InBounds3(const mVector &a) const { return DirectX::XMVector3InBounds(v, a.v); }

  mINLINE static float_t LinePointDistance3(const mVector &line1Point1, const mVector &line1Point2, const mVector &point) { return mVector(DirectX::XMVector3LinePointDistance(line1Point1.v, line1Point2.v, point.v)).x; }
  
  mINLINE float_t Length3() const { return mVector(DirectX::XMVector3Length(v)).x; }
  mINLINE float_t LengthEst3() const { return mVector(DirectX::XMVector3LengthEst(v)).x; }
  mINLINE float_t LengthSquared3() const { return mVector(DirectX::XMVector3LengthSq(v)).x; }
  mINLINE mVector Normalize3() const { return mVector(DirectX::XMVector3Normalize(v)); }
  mINLINE mVector NormalizeEst3() const { return mVector(DirectX::XMVector3NormalizeEst(v)); }
  mINLINE mVector Orthogonal3() const { return mVector(DirectX::XMVector3Orthogonal(v)); }
  mINLINE mVector ReciprocalLength3() const { return mVector(DirectX::XMVector3ReciprocalLength(v)); }
  mINLINE mVector ReciprocalLengthEst3() const { return mVector(DirectX::XMVector3ReciprocalLengthEst(v)); }

  mINLINE static mVector mVECTORCALL Reflect3(const mVector incident, const mVector normal) { return mVector(DirectX::XMVector3Reflect(incident.v, normal.v)); }
  mINLINE static mVector mVECTORCALL Refract3(const mVector incident, const mVector normal, const float_t refractionIndex) { return mVector(DirectX::XMVector3Refract(incident.v, normal.v, refractionIndex)); }
  mINLINE static mVector mVECTORCALL RefractVector3(const mVector incident, const mVector normal, const mVector refractionIndex) { return mVector(DirectX::XMVector3RefractV(incident.v, normal.v, refractionIndex.v)); }

  mVector Transform3(const mMatrix &matrix) const;
  mVector TransformCoord3(const mMatrix &matrix) const;
  mVector TransformNormal3(const mMatrix &matrix) const;
  mVector Rotate3(const mQuaternion &quaternion) const;
  mVector RotateInverse3(const mQuaternion &quaternion) const;

  mINLINE mVector AngleBetweenNormals4(const mVector &a) const { return mVector(DirectX::XMVector4AngleBetweenNormals(v, a.v)); }
  mINLINE mVector AngleBetweenNormalsEst4(const mVector &a) const { return mVector(DirectX::XMVector4AngleBetweenNormalsEst(v, a.v)); }
  mINLINE mVector AngleBetweenVectors4(const mVector &a) const { return mVector(DirectX::XMVector4AngleBetweenVectors(v, a.v)); }
  mINLINE mVector ClampLength4(const float_t min, const float_t max) const { return mVector(DirectX::XMVector4ClampLength(v, min, max)); }
  mINLINE mVector ClampLengthVectors4(const mVector &min, const mVector &max) const { return mVector(DirectX::XMVector4ClampLengthV(v, min.v, max.v)); }
  mINLINE float_t Dot4(const mVector &a) const { return mVector(DirectX::XMVector4Dot(v, a.v)).x; }
  mINLINE mVector Cross4(const mVector &a, const mVector &b) const { return mVector(DirectX::XMVector4Cross(v, a.v, b.v)); }
  mINLINE bool Equals4(const mVector &a) const { return DirectX::XMVector4Equal(v, a.v); }
  mINLINE bool NotEqualTo4(const mVector &a) const { return DirectX::XMVector4NotEqual(v, a.v); }
  mINLINE bool EqualsApproximately4(const mVector &a, const mVector &epsilon) const { return DirectX::XMVector4NearEqual(v, a.v, epsilon.v); }
  mINLINE bool Greater4(const mVector &a) const { return DirectX::XMVector4Greater(v, a.v); }
  mINLINE bool GreaterOrEqual4(const mVector &a) const { return DirectX::XMVector4GreaterOrEqual(v, a.v); }
  mINLINE bool Less4(const mVector &a) const { return DirectX::XMVector4Less(v, a.v); }
  mINLINE bool LessOrEqual4(const mVector &a) const { return DirectX::XMVector4LessOrEqual(v, a.v); }
  mINLINE bool InBounds4(const mVector &a) const { return DirectX::XMVector4InBounds(v, a.v); }
  mINLINE float_t Length4() const { return mVector(DirectX::XMVector4Length(v)).x; }
  mINLINE float_t LengthEst4() const { return mVector(DirectX::XMVector4LengthEst(v)).x; }
  mINLINE float_t LengthSquared4() const { return mVector(DirectX::XMVector4LengthSq(v)).x; }
  mINLINE mVector Normalize4() const { return mVector(DirectX::XMVector4Normalize(v)); }
  mINLINE mVector NormalizeEst4() const { return mVector(DirectX::XMVector4NormalizeEst(v)); }
  mINLINE mVector Orthogonal4() const { return mVector(DirectX::XMVector4Orthogonal(v)); }
  mINLINE mVector ReciprocalLength4() const { return mVector(DirectX::XMVector4ReciprocalLength(v)); }
  mINLINE mVector ReciprocalLengthEst4() const { return mVector(DirectX::XMVector4ReciprocalLengthEst(v)); }

  mINLINE static mVector mVECTORCALL Reflect4(const mVector &incident, const mVector &normal) { return mVector(DirectX::XMVector4Reflect(incident.v, normal.v)); }
  mINLINE static mVector mVECTORCALL Refract4(const mVector &incident, const mVector &normal, const float_t refractionIndex) { return mVector(DirectX::XMVector4Refract(incident.v, normal.v, refractionIndex)); }
  mINLINE static mVector mVECTORCALL RefractVector4(const mVector &incident, const mVector &normal, const mVector &refractionIndex) { return mVector(DirectX::XMVector4RefractV(incident.v, normal.v, refractionIndex.v)); }
  
  mVector Transform4(const mMatrix &matrix) const;

  mINLINE static float_t Dot2(const mVector &a, const mVector &b) { return a.Dot2(b); }
  mINLINE static mVector Cross2(const mVector &a, const mVector &b) { return a.Cross2(b); }
  
  mINLINE static float_t Dot3(const mVector &a, const mVector &b) { return a.Dot3(b); }
  mINLINE static mVector Cross3(const mVector &a, const mVector &b) { return a.Cross3(b); }
  
  mINLINE static float_t Dot4(const mVector &a, const mVector &b) { return a.Dot4(b); }
  mINLINE static mVector Cross4(const mVector &a, const mVector &b, const mVector &c) { return a.Cross4(b, c); }

  static mResult mVECTORCALL TransformCoordStream2(OUT DirectX::XMFLOAT2 *pOutputData, const size_t outputStride, IN DirectX::XMFLOAT2 *pInputData, const size_t inputStride, const size_t inputLength, const mMatrix &matrix);
  static mResult mVECTORCALL TransformNormalStream2(OUT DirectX::XMFLOAT2 *pOutputData, const size_t outputStride, IN DirectX::XMFLOAT2 *pInputData, const size_t inputStride, const size_t inputLength, const mMatrix &matrix);

  static mResult mVECTORCALL TransformCoordStream3(OUT DirectX::XMFLOAT3 *pOutputData, const size_t outputStride, IN DirectX::XMFLOAT3 *pInputData, const size_t inputStride, const size_t inputLength, const mMatrix &matrix);
  static mResult mVECTORCALL TransformNormalStream3(OUT DirectX::XMFLOAT3 *pOutputData, const size_t outputStride, IN DirectX::XMFLOAT3 *pInputData, const size_t inputStride, const size_t inputLength, const mMatrix &matrix);
  static mResult mVECTORCALL ComponentsFromNormal3(OUT mVector *pParallel, OUT mVector *pPerpendicular, const mVector &v, const mVector &normal);

  static mResult mVECTORCALL TransformStream4(OUT DirectX::XMFLOAT4 *pOutputData, const size_t outputStride, IN DirectX::XMFLOAT4 *pInputData, const size_t inputStride, const size_t inputLength, const mMatrix &matrix);
};

//////////////////////////////////////////////////////////////////////////

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

  mINLINE mQuaternion() { *this = Identity(); }
  mINLINE explicit mQuaternion(DirectX::XMVECTOR v) : q(v) { }

  mINLINE mQuaternion Multiply(const mQuaternion &q2) const { return mQuaternion(DirectX::XMQuaternionMultiply(q, q2.q)); }

  mINLINE mQuaternion operator*(const mQuaternion &q1) const { return Multiply(q1); };
  mINLINE mQuaternion & operator*=(const mQuaternion &q1) { return *this = Multiply(q1); };

  mINLINE static mQuaternion BaryCentric(const mQuaternion &q0, const mQuaternion &q1, const mQuaternion &q2, const float_t f, const float_t g) { return mQuaternion(DirectX::XMQuaternionBaryCentric(q0.q, q1.q, q2.q, f, g)); }
  mINLINE static mQuaternion BaryCentricV(const mQuaternion &q0, const mQuaternion &q1, const mQuaternion &q2, const mVector &f, const mVector &g) { return mQuaternion(DirectX::XMQuaternionBaryCentricV(q0.q, q1.q, q2.q, f.v, g.v)); }
  
  mINLINE mQuaternion Conjugate() const { return mQuaternion(DirectX::XMQuaternionConjugate(q)); }
  mINLINE float_t Dot(const mQuaternion &q2) const { return mQuaternion(DirectX::XMQuaternionDot(q, q2.q)).x; }
  mINLINE bool Equals(const mQuaternion &q2) const { return DirectX::XMQuaternionEqual(q, q2.q); }
  mINLINE mQuaternion Exp() const { return mQuaternion(DirectX::XMQuaternionExp(q)); }
  mINLINE mQuaternion Inverse() const { return mQuaternion(DirectX::XMQuaternionInverse(q)); }
  mINLINE bool IsIdentity() const { return DirectX::XMQuaternionIsIdentity(q); }
  mINLINE float_t Length() const { return mQuaternion(DirectX::XMQuaternionLength(q)).x; }
  mINLINE float_t LengthSq() const { return mQuaternion(DirectX::XMQuaternionLengthSq(q)).x; }
  mINLINE mQuaternion Ln() const { return mQuaternion(DirectX::XMQuaternionLn(q)); }
  mINLINE mQuaternion Normalize() const { return mQuaternion(DirectX::XMQuaternionNormalize(q)); }
  mINLINE mQuaternion NormalizeEst() const { return mQuaternion(DirectX::XMQuaternionNormalizeEst(q)); }
  mINLINE bool NotEqualTo(const mQuaternion &q2) const { return DirectX::XMQuaternionNotEqual(q, q2.q); }
  mINLINE mVector ReciprocalLength() const { return mVector(DirectX::XMQuaternionReciprocalLength(q)); }

  mINLINE static mQuaternion Identity() { return mQuaternion(DirectX::XMQuaternionIdentity()); }
  
  static mQuaternion mVECTORCALL RotationMatrix(const mMatrix &m);

  mINLINE static mQuaternion mVECTORCALL RotationAxis(const mVector axis, const float_t angle) { return mQuaternion(DirectX::XMQuaternionRotationAxis(axis.v, angle)); }
  mINLINE static mQuaternion mVECTORCALL RotationNormal(const mVector normalAxis, const float_t angle) { return mQuaternion(DirectX::XMQuaternionRotationNormal(normalAxis.v, angle)); }
  mINLINE static mQuaternion mVECTORCALL RotationRollPitchYaw(const float_t pitch, const float_t yaw, const float_t roll) { return mQuaternion(DirectX::XMQuaternionRotationRollPitchYaw(pitch, yaw, roll)); }
  mINLINE static mQuaternion mVECTORCALL RotationRollPitchYawFromVector(const mVector angles) { return mQuaternion(DirectX::XMQuaternionRotationRollPitchYawFromVector(angles.v)); }
  mINLINE static mQuaternion mVECTORCALL Slerp(const mQuaternion q0, const mQuaternion q1, const float_t t) { return mQuaternion(DirectX::XMQuaternionSlerp(q0.q, q1.q, t)); }
  mINLINE static mQuaternion mVECTORCALL SlerpV(const mQuaternion q0, const mQuaternion q1, const mVector t) { return mQuaternion(DirectX::XMQuaternionSlerpV(q0.q, q1.q, t.v)); }
  mINLINE static mQuaternion mVECTORCALL Squad(const mQuaternion q0, const mQuaternion q1, const mQuaternion q2, const mQuaternion q3, const float_t t) { return mQuaternion(DirectX::XMQuaternionSquad(q0.q, q1.q, q2.q, q3.q, t)); }
  mINLINE static mQuaternion mVECTORCALL SquadV(const mQuaternion q0, const mQuaternion q1, const mQuaternion q2, const mQuaternion q3, const mVector t) { return mQuaternion(DirectX::XMQuaternionSquadV(q0.q, q1.q, q2.q, q3.q, t.v)); }

  mVec3f ToEulerAngles();
  mResult SquadSetup(OUT mVector *pA, OUT mVector *pB, OUT mVector *pC, const mQuaternion &q0, const mQuaternion &q1, const mQuaternion &q2, const mQuaternion &q3);
  mResult ToAxisAngle(OUT mVector *pAxis, OUT float_t *pAngle);
};

//////////////////////////////////////////////////////////////////////////

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


  mINLINE mMatrix() { }
  mINLINE mMatrix(DirectX::XMMATRIX _m) : m(_m) { }

  mINLINE mMatrix operator*(const mMatrix &q1) const { return Multiply(q1); }
  mINLINE mMatrix & operator*=(const mMatrix &q1) { return *this = Multiply(q1); }
  mINLINE mMatrix & operator=(const mMatrix &copy) { m = copy.m; return *this; }

  mINLINE static mMatrix mVECTORCALL Identity() { return mMatrix(DirectX::XMMatrixIdentity()); }
  mINLINE static mMatrix mVECTORCALL AffineTransformation(const mVector &scaling, const mVector &rotationOrigin, const mQuaternion &rotationQuaternion, const mVector &translation) { return mMatrix(DirectX::XMMatrixAffineTransformation(scaling.v, rotationOrigin.v, rotationQuaternion.q, translation.v)); }
  mINLINE static mMatrix mVECTORCALL AffineTransformation2D(const mVector &scaling, const mVector &rotationOrigin, const float_t rotation, const mVector &translation) { return mMatrix(DirectX::XMMatrixAffineTransformation2D(scaling.v, rotationOrigin.v, rotation, translation.v)); }

  mResult Decompose(OUT mVector *pOutScale, OUT mQuaternion *pOutRotQuat, OUT mVector *pOutTrans) const;
  
  mINLINE mVector Determinant() const { return mVector(DirectX::XMMatrixDeterminant(m)); }
  mINLINE mMatrix Inverse(OUT OPTIONAL mVector *pDeterminant /* = nullptr */) const { return mMatrix(DirectX::XMMatrixInverse(&pDeterminant->v, m)); }
  
  mINLINE static mMatrix mVECTORCALL LookAtLH(const mVector &eyePosition, const mVector &focusPosition, const mVector &upDirection) { return mMatrix(DirectX::XMMatrixLookAtLH(eyePosition.v, focusPosition.v, upDirection.v)); }
  mINLINE static mMatrix mVECTORCALL LookAtRH(const mVector &eyePosition, const mVector &focusPosition, const mVector &upDirection) { return mMatrix(DirectX::XMMatrixLookAtRH(eyePosition.v, focusPosition.v, upDirection.v)); }
  mINLINE static mMatrix mVECTORCALL LookToLH(const mVector &eyePosition, const mVector &eyeDirection, const mVector &upDirection) { return mMatrix(DirectX::XMMatrixLookToLH(eyePosition.v, eyeDirection.v, upDirection.v)); }
  mINLINE static mMatrix mVECTORCALL LookToRH(const mVector &eyePosition, const mVector &eyeDirection, const mVector &upDirection) { return mMatrix(DirectX::XMMatrixLookToRH(eyePosition.v, eyeDirection.v, upDirection.v)); }
  
  mINLINE mMatrix Multiply(const mMatrix &m2) const { return mMatrix(DirectX::XMMatrixMultiply(m, m2.m)); }
  mINLINE mMatrix MultiplyTranspose(const mMatrix &m2) const { return mMatrix(DirectX::XMMatrixMultiplyTranspose(m, m2.m)); }
  
  mINLINE static mMatrix mVECTORCALL OrthographicLH(const float_t viewWidth, const float_t viewHeight, const float_t nearZ, const float_t farZ) { return mMatrix(DirectX::XMMatrixOrthographicLH(viewWidth, viewHeight, nearZ, farZ)); }
  mINLINE static mMatrix mVECTORCALL OrthographicRH(const float_t viewWidth, const float_t viewHeight, const float_t nearZ, const float_t farZ) { return mMatrix(DirectX::XMMatrixOrthographicRH(viewWidth, viewHeight, nearZ, farZ)); }
  mINLINE static mMatrix mVECTORCALL OrthographicOffCenterLH(const float_t viewLeft, const float_t viewRight, const float_t viewBottom, const float_t viewTop, const float_t nearZ, const float_t farZ) { return mMatrix(DirectX::XMMatrixOrthographicOffCenterLH(viewLeft, viewRight, viewBottom, viewTop, nearZ, farZ)); }
  mINLINE static mMatrix mVECTORCALL OrthographicOffCenterRH(const float_t viewLeft, const float_t viewRight, const float_t viewBottom, const float_t viewTop, const float_t nearZ, const float_t farZ) { return mMatrix(DirectX::XMMatrixOrthographicOffCenterRH(viewLeft, viewRight, viewBottom, viewTop, nearZ, farZ)); }
  mINLINE static mMatrix mVECTORCALL PerspectiveLH(const float_t viewWidth, const float_t viewHeight, const float_t nearZ, const float_t farZ) { return mMatrix(DirectX::XMMatrixPerspectiveLH(viewWidth, viewHeight, nearZ, farZ)); }
  mINLINE static mMatrix mVECTORCALL PerspectiveRH(const float_t viewWidth, const float_t viewHeight, const float_t nearZ, const float_t farZ) { return mMatrix(DirectX::XMMatrixPerspectiveRH(viewWidth, viewHeight, nearZ, farZ)); }
  mINLINE static mMatrix mVECTORCALL PerspectiveFovLH(const float_t fovAngleY, const float_t aspectRatio, const float_t nearZ, const float_t farZ) { return mMatrix(DirectX::XMMatrixPerspectiveFovLH(fovAngleY, aspectRatio, nearZ, farZ)); }
  mINLINE static mMatrix mVECTORCALL PerspectiveFovRH(const float_t fovAngleY, const float_t aspectRatio, const float_t nearZ, const float_t farZ) { return mMatrix(DirectX::XMMatrixPerspectiveFovRH(fovAngleY, aspectRatio, nearZ, farZ)); }
  mINLINE static mMatrix mVECTORCALL PerspectiveOffCenterLH(const float_t viewLeft, const float_t viewRight, const float_t viewBottom, const float_t viewTop, const float_t nearZ, const float_t farZ) { return mMatrix(DirectX::XMMatrixPerspectiveOffCenterLH(viewLeft, viewRight, viewBottom, viewTop, nearZ, farZ)); }
  mINLINE static mMatrix mVECTORCALL PerspectiveOffCenterRH(const float_t viewLeft, const float_t viewRight, const float_t viewBottom, const float_t viewTop, const float_t nearZ, const float_t farZ) { return mMatrix(DirectX::XMMatrixPerspectiveOffCenterRH(viewLeft, viewRight, viewBottom, viewTop, nearZ, farZ)); }
  mINLINE static mMatrix mVECTORCALL Reflect(const mVector &reflectionPlane) { return mMatrix(DirectX::XMMatrixReflect(reflectionPlane.v)); }
  mINLINE static mMatrix mVECTORCALL RotationAxis(const mVector &axis, const float_t angle) { return mMatrix(DirectX::XMMatrixRotationAxis(axis.v, angle)); }
  mINLINE static mMatrix mVECTORCALL RotationQuaternion(const mQuaternion &quaternion) { return mMatrix(DirectX::XMMatrixRotationQuaternion(quaternion.q)); }
  mINLINE static mMatrix mVECTORCALL RotationNormal(const mVector &normalAxis, const float_t angle) { return mMatrix(DirectX::XMMatrixRotationNormal(normalAxis.v, angle)); }
  mINLINE static mMatrix mVECTORCALL RotationRollPitchYaw(const float_t pitch, const float_t yaw, const float_t roll) { return mMatrix(DirectX::XMMatrixRotationRollPitchYaw(pitch, yaw, roll)); }
  mINLINE static mMatrix mVECTORCALL RotationRollPitchYawFromVector(const mVector &angles) { return mMatrix(DirectX::XMMatrixRotationRollPitchYawFromVector(angles.v)); }
  mINLINE static mMatrix mVECTORCALL RotationX(const float_t angle) { return mMatrix(DirectX::XMMatrixRotationX(angle)); }
  mINLINE static mMatrix mVECTORCALL RotationY(const float_t angle) { return mMatrix(DirectX::XMMatrixRotationY(angle)); }
  mINLINE static mMatrix mVECTORCALL RotationZ(const float_t angle) { return mMatrix(DirectX::XMMatrixRotationZ(angle)); }
  mINLINE static mMatrix mVECTORCALL Scale(const float_t scaleX, const float_t scaleY, const float_t scaleZ) { return mMatrix(DirectX::XMMatrixScaling(scaleX, scaleY, scaleZ)); }
  mINLINE static mMatrix mVECTORCALL ScalingFromVector(const mVector &scale) { return mMatrix(DirectX::XMMatrixScalingFromVector(scale.v)); }
  mINLINE static mMatrix mVECTORCALL Shadow(const mVector &shadowPlane, const mVector &lightPosition) { return mMatrix(DirectX::XMMatrixShadow(shadowPlane.v, lightPosition.v)); }
  mINLINE static mMatrix mVECTORCALL Transformation(const mVector &scalingOrigin, const mQuaternion &scalingOrientationQuaternion, const mVector &scaling, const mVector &rotationOrigin, const mQuaternion &rotationQuaternion, const mVector &translation) { return mMatrix(DirectX::XMMatrixTransformation(scalingOrigin.v, scalingOrientationQuaternion.q, scaling.v, rotationOrigin.v, rotationQuaternion.q, translation.v)); }
  mINLINE static mMatrix mVECTORCALL Transformation2D(const mVector &scalingOrigin, const float_t scalingOrientation, const mVector &scaling, const mVector &rotationOrigin, const float_t rotation, const mVector &translation) { return mMatrix(DirectX::XMMatrixTransformation2D(scalingOrigin.v, scalingOrientation, scaling.v, rotationOrigin.v, rotation, translation.v)); }
  mINLINE static mMatrix mVECTORCALL Translation(const float_t offsetX, const float_t offsetY, const float_t offsetZ) { return mMatrix(DirectX::XMMatrixTranslation(offsetX, offsetY, offsetZ)); }
  mINLINE static mMatrix mVECTORCALL TranslationFromVector(const mVector &offset) { return mMatrix(DirectX::XMMatrixTranslationFromVector(offset.v)); }

  mINLINE static mMatrix mVECTORCALL Zero()
  {
    DirectX::XMVECTOR v[4];

    v[0] = _mm_setzero_ps();
    v[1] = _mm_setzero_ps();
    v[2] = _mm_setzero_ps();
    v[3] = _mm_setzero_ps();

    return mMatrix(*reinterpret_cast<DirectX::XMMATRIX *>(v));
  }

  mINLINE mMatrix Transpose() const { return mMatrix(DirectX::XMMatrixTranspose(m)); }
  mINLINE mVector mVECTORCALL TransformVector4(const mVector vector4) { return mVector(XMVector4Transform(vector4.v, m)); }
  mINLINE mVector mVECTORCALL TransformVector3(const mVector vector3) { return mVector(XMVector3Transform(vector3.v, m)); }
};

//////////////////////////////////////////////////////////////////////////

mINLINE mVector mVector::Transform2(const mMatrix &matrix) const { return mVector(DirectX::XMVector2Transform(v, (DirectX::FXMMATRIX)matrix.m)); }
mINLINE mVector mVector::TransformCoord2(const mMatrix &matrix) const { return mVector(DirectX::XMVector2TransformCoord(v, matrix.m)); }
mINLINE mVector mVector::TransformNormal2(const mMatrix &matrix) const { return mVector(DirectX::XMVector2TransformNormal(v, matrix.m)); }

mINLINE mVector mVector::Transform3(const mMatrix &matrix) const { return mVector(DirectX::XMVector3Transform(v, (DirectX::FXMMATRIX)matrix.m)); }
mINLINE mVector mVector::TransformCoord3(const mMatrix &matrix) const { return mVector(DirectX::XMVector3TransformCoord(v, matrix.m)); }
mINLINE mVector mVector::TransformNormal3(const mMatrix &matrix) const { return mVector(DirectX::XMVector3TransformNormal(v, matrix.m)); }
mINLINE mVector mVector::Rotate3(const mQuaternion &quaternion) const { return mVector(DirectX::XMVector3Rotate(v, quaternion.q)); }
mINLINE mVector mVector::RotateInverse3(const mQuaternion &quaternion) const { return mVector(DirectX::XMVector3InverseRotate(v, quaternion.q)); }

mINLINE mVector mVector::Transform4(const mMatrix &matrix) const { return mVector(DirectX::XMVector4Transform(v, (DirectX::FXMMATRIX)matrix.m)); };

//////////////////////////////////////////////////////////////////////////

mINLINE /* static */ mQuaternion mVECTORCALL mQuaternion::RotationMatrix(const mMatrix &m) { return mQuaternion(DirectX::XMQuaternionRotationMatrix(m.m)); };

template <>
mINLINE bool mIntersects<float_t>(const mTriangle3D<float_t> &triangle, const mLine3D<float_t> &line, OUT OPTIONAL float_t *pDistance /* = nullptr */)
{
  float_t distance;

  const bool ret = DirectX::TriangleTests::Intersects(mVector(line.position0).v, mVector(line.position1 - line.position0).v, mVector(triangle.position0).v, mVector(triangle.position1).v, mVector(triangle.position2).v, distance);

  if (pDistance != nullptr)
    *pDistance = distance;

  return ret;
}

#endif // mFastMath_h__
