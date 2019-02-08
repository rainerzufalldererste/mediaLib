#include "mediaLib.h"
#include "mFastMath.h"

mVector::mVector() :
  x(0),
  y(0),
  z(0),
  w(1)
{ }

mVector::mVector(const mVector &a) :
  v(a.v)
{ }

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

mVector::mVector(const mVec2f &v) : mVector(v.x, v.y) { }
mVector::mVector(const mVec3f &v) : mVector(v.x, v.y, v.z) { }
mVector::mVector(const mVec4f &v) : mVector(v.x, v.y, v.z, v.w) { }

mVector::mVector(DirectX::XMVECTOR _v) : v(_v) { }

mVector::mVector(DirectX::XMFLOAT2 _v) { v = DirectX::XMLoadFloat2(&_v); }
mVector::mVector(DirectX::XMFLOAT3 _v) { v = DirectX::XMLoadFloat3(&_v); }
mVector::mVector(DirectX::XMFLOAT4 _v) { v = DirectX::XMLoadFloat4(&_v); }

mQuaternion::mQuaternion() :
  mQuaternion(mQuaternion::Identity())
{ }

mQuaternion::mQuaternion(DirectX::XMVECTOR v) : q(v) { }

mMatrix::mMatrix(DirectX::XMMATRIX _m) : m(_m) { }

//////////////////////////////////////////////////////////////////////////

mVector mVector::operator+(const mVector &a) const { return mVector(DirectX::XMVectorAdd(v, a.v)); }
mVector mVector::operator-(const mVector &a) const { return mVector(DirectX::XMVectorSubtract(v, a.v)); }
mVector mVector::operator*(const mVector &a) const { return mVector(DirectX::XMVectorMultiply(v, a.v)); }
mVector mVector::operator/(const mVector &a) const { return mVector(DirectX::XMVectorDivide(v, a.v)); }

mVector & mVector::operator+=(const mVector &a) { return *this = (*this + a); }
mVector & mVector::operator-=(const mVector &a) { return *this = (*this - a); }
mVector & mVector::operator*=(const mVector &a) { return *this = (*this * a); }
mVector & mVector::operator/=(const mVector &a) { return *this = (*this / a); }

mVector mVector::operator*(const float_t a) const { return mVector(DirectX::XMVectorScale(v, a)); }
mVector mVector::operator/(const float_t a) const { return mVector(DirectX::XMVectorScale(v, 1.0f / a)); }

mVector & mVector::operator*=(const float_t a) { return *this = (*this * a); }
mVector & mVector::operator/=(const float_t a) { return *this = (*this / a); }

mVector mVector::operator-() const { return mVector(DirectX::XMVectorNegate(v)); }

mVector::operator mVec2f() const
{
  DirectX::XMFLOAT2 f;
  DirectX::XMStoreFloat2(&f, v);
  return mVec2f(f.x, f.y);
}

mVector::operator mVec3f() const
{
  DirectX::XMFLOAT3 f;
  DirectX::XMStoreFloat3(&f, v);
  return mVec3f(f.x, f.y, f.z);
}

mVector::operator mVec4f() const
{
  DirectX::XMFLOAT4 f;
  DirectX::XMStoreFloat4(&f, v);
  return mVec4f(f.x, f.y, f.z, f.w);
}

mVector::operator DirectX::XMFLOAT2() const
{
  DirectX::XMFLOAT2 f;
  DirectX::XMStoreFloat2(&f, v);
  return f;
}

mVector::operator DirectX::XMFLOAT3() const
{
  DirectX::XMFLOAT3 f;
  DirectX::XMStoreFloat3(&f, v);
  return f;
}

mVector::operator DirectX::XMFLOAT4() const
{
  DirectX::XMFLOAT4 f;
  DirectX::XMStoreFloat4(&f, v);
  return f;
}

mVec4t<bool> mVector::ComponentEquals(const mVector &a) const
{
  DirectX::XMVECTOR __v = DirectX::XMVectorEqual(v, a.v);
  DirectX::XMUINT4 f;
  DirectX::XMStoreUInt4(&f, __v);
  return mVec4t<bool>(f.x != 0, f.y != 0, f.z != 0, f.w != 0);
}

mVector mVector::Abs() const { return mVector(DirectX::XMVectorAbs(v)); }

mVector mVector::Min(const mVector &a, const mVector &b) { return mVector(DirectX::XMVectorMin(a.v, b.v)); }
mVector mVector::Max(const mVector &a, const mVector &b) { return mVector(DirectX::XMVectorMax(a.v, b.v)); }

mVector mVECTORCALL mVector::Lerp(const mVector &a, const mVector &b, const float_t t) { return mVector(DirectX::XMVectorLerp(a.v, b.v, t)); }
mVector mVECTORCALL mVector::LerpVector(const mVector &a, const mVector &b, const mVector &t) { return mVector(DirectX::XMVectorLerpV(a.v, b.v, t.v)); }
mVector mVECTORCALL mVector::Barycentric(const mVector &a, const mVector &b, const mVector &c, const float_t f, const float_t g) { return mVector(DirectX::XMVectorBaryCentric(a.v, b.v, c.v, f, g)); }
mVector mVECTORCALL mVector::BarycentricVector(const mVector &a, const mVector &b, const mVector &c, const mVector &f, const mVector &g) { return mVector(DirectX::XMVectorBaryCentricV(a.v, b.v, c.v, f.v, g.v)); }
mVector mVECTORCALL mVector::CatmullRom(const mVector &a, const mVector &b, const mVector &c, const mVector &d, const float_t f) { return mVector(DirectX::XMVectorCatmullRom(a.v, b.v, c.v, d.v, f)); }
mVector mVECTORCALL mVector::CatmullRomVector(const mVector &a, const mVector &b, const mVector &c, const mVector &d, const mVector &f) { return mVector(DirectX::XMVectorCatmullRomV(a.v, b.v, c.v, d.v, f.v)); }
mVector mVECTORCALL mVector::Hermite(const mVector &v1, const mVector &t1, const mVector &v2, const mVector &t2, const float_t f) { return mVector(DirectX::XMVectorHermite(v1.v, t1.v, v2.v, t2.v, f)); }
mVector mVECTORCALL mVector::HermiteVector(const mVector &v1, const mVector &t1, const mVector &v2, const mVector &t2, const  mVector &f) { return mVector(DirectX::XMVectorHermiteV(v1.v, t1.v, v2.v, t2.v, f.v)); }

mVector mVector::AngleBetweenNormals2(const mVector &a) const { return mVector(DirectX::XMVector2AngleBetweenNormals(v, a.v)); }
mVector mVector::AngleBetweenNormalsEst2(const mVector &a) const { return mVector(DirectX::XMVector2AngleBetweenNormalsEst(v, a.v)); }
mVector mVector::AngleBetweenVectors2(const mVector &a) const { return mVector(DirectX::XMVector2AngleBetweenVectors(v, a.v)); }
mVector mVector::ClampLength2(const float_t min, const float_t max) const { return mVector(DirectX::XMVector2ClampLength(v, min, max)); }
mVector mVector::ClampLengthVectors2(const mVector &min, const mVector &max) const { return mVector(DirectX::XMVector2ClampLengthV(v, min.v, max.v)); }
float_t mVector::Dot2(const mVector &a) const { return mVector(DirectX::XMVector2Dot(v, a.v)).x; }
mVector mVector::Cross2(const mVector &a) const { return mVector(DirectX::XMVector2Cross(v, a.v)); }
bool mVector::Equals2(const mVector &a) const { return DirectX::XMVector2Equal(v, a.v); }
bool mVector::NotEqualTo2(const mVector &a) const { return DirectX::XMVector2NotEqual(v, a.v); }
bool mVector::EqualsApproximately2(const mVector &a, const mVector &epsilon) const { return DirectX::XMVector2NearEqual(v, a.v, epsilon.v); }
bool mVector::Greater2(const mVector &a) const { return DirectX::XMVector2Greater(v, a.v); }
bool mVector::GreaterOrEqual2(const mVector &a) const { return DirectX::XMVector2GreaterOrEqual(v, a.v); }
bool mVector::Less2(const mVector &a) const { return DirectX::XMVector2Less(v, a.v); }
bool mVector::LessOrEqual2(const mVector &a) const { return DirectX::XMVector2LessOrEqual(v, a.v); }
bool mVector::InBounds2(const mVector &a) const { return DirectX::XMVector2InBounds(v, a.v); }
mVector mVECTORCALL mVector::IntersectLine2(const mVector &line1Point1, const mVector &line1Point2, const mVector &line2Point1, const mVector &line2Point2) { return mVector(DirectX::XMVector2IntersectLine(line1Point1.v, line1Point2.v, line2Point1.v, line2Point2.v)); }
float_t mVECTORCALL mVector::LinePointDistance2(const mVector &line1Point1, const mVector &line1Point2, const mVector &point) { return mVector(DirectX::XMVector2LinePointDistance(line1Point1.v, line1Point2.v, point.v)).x; }
float_t mVector::Length2() const { return mVector(DirectX::XMVector2Length(v)).x; }
float_t mVector::LengthEst2() const { return mVector(DirectX::XMVector2LengthEst(v)).x; }
float_t mVector::LengthSquared2() const { return mVector(DirectX::XMVector2LengthSq(v)).x; }
mVector mVector::Normalize2() const { return mVector(DirectX::XMVector2Normalize(v)); }
mVector mVector::NormalizeEst2() const { return mVector(DirectX::XMVector2NormalizeEst(v)); }
mVector mVector::Orthogonal2() const { return mVector(DirectX::XMVector2Orthogonal(v)); }
mVector mVector::ReciprocalLength2() const { return mVector(DirectX::XMVector2ReciprocalLength(v)); }
mVector mVector::ReciprocalLengthEst2() const { return mVector(DirectX::XMVector2ReciprocalLengthEst(v)); }
mVector mVECTORCALL mVector::Reflect2(const mVector &incident, const mVector &normal) { return mVector(DirectX::XMVector2Reflect(incident.v, normal.v)); }
mVector mVECTORCALL mVector::Refract2(const mVector &incident, const mVector &normal, const float_t refractionIndex) { return mVector(DirectX::XMVector2Refract(incident.v, normal.v, refractionIndex)); }
mVector mVECTORCALL mVector::RefractVector2(const mVector &incident, const mVector &normal, const mVector &refractionIndex) { return mVector(DirectX::XMVector2RefractV(incident.v, normal.v, refractionIndex.v)); }
mVector mVector::Transform2(const mMatrix &matrix) const { return mVector(DirectX::XMVector2Transform(v, (DirectX::FXMMATRIX)matrix.m)); }
mVector mVector::TransformCoord2(const mMatrix &matrix) const { return mVector(DirectX::XMVector2TransformCoord(v, matrix.m)); }
mVector mVector::TransformNormal2(const mMatrix &matrix) const { return mVector(DirectX::XMVector2TransformNormal(v, matrix.m)); }

mResult mVECTORCALL mVector::TransformCoordStream2(OUT DirectX::XMFLOAT2 *pOutputData, const size_t outputStride, IN DirectX::XMFLOAT2 *pInputData, const size_t inputStride, const size_t inputLength, const mMatrix &matrix)
{
  mFUNCTION_SETUP();

  mERROR_IF(pOutputData == nullptr || pInputData == nullptr, mR_ArgumentNull);
  DirectX::XMVector2TransformCoordStream(pOutputData, outputStride, pInputData, inputStride, inputLength, matrix.m);

  mRETURN_SUCCESS();
}

mResult mVECTORCALL mVector::TransformNormalStream2(OUT DirectX::XMFLOAT2 *pOutputData, const size_t outputStride, IN DirectX::XMFLOAT2 *pInputData, const size_t inputStride, const size_t inputLength, const mMatrix &matrix)
{
  mFUNCTION_SETUP();

  mERROR_IF(pOutputData == nullptr || pInputData == nullptr, mR_ArgumentNull);
  DirectX::XMVector2TransformNormalStream(pOutputData, outputStride, pInputData, inputStride, inputLength, matrix.m);

  mRETURN_SUCCESS();
}

mVector mVector::AngleBetweenNormals3(const mVector &a) const { return mVector(DirectX::XMVector3AngleBetweenNormals(v, a.v)); }
mVector mVector::AngleBetweenNormalsEst3(const mVector &a) const { return mVector(DirectX::XMVector3AngleBetweenNormalsEst(v, a.v)); }
mVector mVector::AngleBetweenVectors3(const mVector &a) const { return mVector(DirectX::XMVector3AngleBetweenVectors(v, a.v)); }
mVector mVector::ClampLength3(const float_t min, const float_t max) const { return mVector(DirectX::XMVector3ClampLength(v, min, max)); }
mVector mVector::ClampLengthVectors3(const mVector &min, const mVector &max) const { return mVector(DirectX::XMVector3ClampLengthV(v, min.v, max.v)); }
float_t mVector::Dot3(const mVector &a) const { return mVector(DirectX::XMVector3Dot(v, a.v)).x; }
mVector mVector::Cross3(const mVector &a) const { return mVector(DirectX::XMVector3Cross(v, a.v)); }
bool mVector::Equals3(const mVector &a) const { return DirectX::XMVector3Equal(v, a.v); }
bool mVector::NotEqualTo3(const mVector &a) const { return DirectX::XMVector3NotEqual(v, a.v); }
bool mVector::EqualsApproximately3(const mVector &a, const mVector &epsilon) const { return DirectX::XMVector3NearEqual(v, a.v, epsilon.v); }
bool mVector::Greater3(const mVector &a) const { return DirectX::XMVector3Greater(v, a.v); }
bool mVector::GreaterOrEqual3(const mVector &a) const { return DirectX::XMVector3GreaterOrEqual(v, a.v); }
bool mVector::Less3(const mVector &a) const { return DirectX::XMVector3Less(v, a.v); }
bool mVector::LessOrEqual3(const mVector &a) const { return DirectX::XMVector3LessOrEqual(v, a.v); }
bool mVector::InBounds3(const mVector &a) const { return DirectX::XMVector3InBounds(v, a.v); }
float_t mVector::LinePointDistance3(const mVector &line1Point1, const mVector &line1Point2, const mVector &point) { return mVector(DirectX::XMVector3LinePointDistance(line1Point1.v, line1Point2.v, point.v)).x; }
float_t mVector::Length3() const { return mVector(DirectX::XMVector3Length(v)).x; }
float_t mVector::LengthEst3() const { return mVector(DirectX::XMVector3LengthEst(v)).x; }
float_t mVector::LengthSquared3() const { return mVector(DirectX::XMVector3LengthSq(v)).x; }
mVector mVector::Normalize3() const { return mVector(DirectX::XMVector3Normalize(v)); }
mVector mVector::NormalizeEst3() const { return mVector(DirectX::XMVector3NormalizeEst(v)); }
mVector mVector::Orthogonal3() const { return mVector(DirectX::XMVector3Orthogonal(v)); }
mVector mVector::ReciprocalLength3() const { return mVector(DirectX::XMVector3ReciprocalLength(v)); }
mVector mVector::ReciprocalLengthEst3() const { return mVector(DirectX::XMVector3ReciprocalLengthEst(v)); }
mVector mVECTORCALL mVector::Reflect3(const mVector &incident, const mVector &normal) { return mVector(DirectX::XMVector3Reflect(incident.v, normal.v)); }
mVector mVECTORCALL mVector::Refract3(const mVector &incident, const mVector &normal, const float_t refractionIndex) { return mVector(DirectX::XMVector3Refract(incident.v, normal.v, refractionIndex)); }
mVector mVECTORCALL mVector::RefractVector3(const mVector &incident, const mVector &normal, const mVector &refractionIndex) { return mVector(DirectX::XMVector3RefractV(incident.v, normal.v, refractionIndex.v)); }
mVector mVector::Transform3(const mMatrix &matrix) const { return mVector(DirectX::XMVector3Transform(v, (DirectX::FXMMATRIX)matrix.m)); }
mVector mVector::TransformCoord3(const mMatrix &matrix) const { return mVector(DirectX::XMVector3TransformCoord(v, matrix.m)); }
mVector mVector::TransformNormal3(const mMatrix &matrix) const { return mVector(DirectX::XMVector3TransformNormal(v, matrix.m)); }
mVector mVector::Rotate3(const mQuaternion &quaternion) const { return mVector(DirectX::XMVector3Rotate(v, quaternion.q)); }
mVector mVector::RotateInverse3(const mQuaternion &quaternion) const { return mVector(DirectX::XMVector3InverseRotate(v, quaternion.q)); }

mResult mVECTORCALL mVector::TransformCoordStream3(OUT DirectX::XMFLOAT3 *pOutputData, const size_t outputStride, IN DirectX::XMFLOAT3 *pInputData, const size_t inputStride, const size_t inputLength, const mMatrix &matrix)
{
  mFUNCTION_SETUP();

  mERROR_IF(pOutputData == nullptr || pInputData == nullptr, mR_ArgumentNull);
  DirectX::XMVector3TransformCoordStream(pOutputData, outputStride, pInputData, inputStride, inputLength, matrix.m);

  mRETURN_SUCCESS();
}

mResult mVECTORCALL mVector::TransformNormalStream3(OUT DirectX::XMFLOAT3 *pOutputData, const size_t outputStride, IN DirectX::XMFLOAT3 *pInputData, const size_t inputStride, const size_t inputLength, const mMatrix &matrix)
{
  mFUNCTION_SETUP();

  mERROR_IF(pOutputData == nullptr || pInputData == nullptr, mR_ArgumentNull);
  DirectX::XMVector3TransformNormalStream(pOutputData, outputStride, pInputData, inputStride, inputLength, matrix.m);

  mRETURN_SUCCESS();
}

mResult mVECTORCALL mVector::ComponentsFromNormal3(OUT mVector *pParallel, OUT mVector *pPerpendicular, const mVector &v, const mVector &normal)
{
  mFUNCTION_SETUP();

  mERROR_IF(pParallel == nullptr || pParallel == nullptr, mR_ArgumentNull);
  DirectX::XMVector3ComponentsFromNormal(&pParallel->v, &pPerpendicular->v, v.v, normal.v);

  mRETURN_SUCCESS();
}

mVector mVector::AngleBetweenNormals4(const mVector &a) const { return mVector(DirectX::XMVector4AngleBetweenNormals(v, a.v)); }
mVector mVector::AngleBetweenNormalsEst4(const mVector &a) const { return mVector(DirectX::XMVector4AngleBetweenNormalsEst(v, a.v)); }
mVector mVector::AngleBetweenVectors4(const mVector &a) const { return mVector(DirectX::XMVector4AngleBetweenVectors(v, a.v)); }
mVector mVector::ClampLength4(const float_t min, const float_t max) const { return mVector(DirectX::XMVector4ClampLength(v, min, max)); }
mVector mVector::ClampLengthVectors4(const mVector &min, const mVector &max) const { return mVector(DirectX::XMVector4ClampLengthV(v, min.v, max.v)); }
float_t mVector::Dot4(const mVector &a) const { return mVector(DirectX::XMVector4Dot(v, a.v)).x; }
mVector mVector::Cross4(const mVector &a, const mVector &b) const { return mVector(DirectX::XMVector4Cross(v, a.v, b.v)); }
bool mVector::Equals4(const mVector &a) const { return DirectX::XMVector4Equal(v, a.v); }
bool mVector::NotEqualTo4(const mVector &a) const { return DirectX::XMVector4NotEqual(v, a.v); }
bool mVector::EqualsApproximately4(const mVector &a, const mVector &epsilon) const { return DirectX::XMVector4NearEqual(v, a.v, epsilon.v); }
bool mVector::Greater4(const mVector &a) const { return DirectX::XMVector4Greater(v, a.v); }
bool mVector::GreaterOrEqual4(const mVector &a) const { return DirectX::XMVector4GreaterOrEqual(v, a.v); }
bool mVector::Less4(const mVector &a) const { return DirectX::XMVector4Less(v, a.v); }
bool mVector::LessOrEqual4(const mVector &a) const { return DirectX::XMVector4LessOrEqual(v, a.v); }
bool mVector::InBounds4(const mVector &a) const { return DirectX::XMVector4InBounds(v, a.v); }
float_t mVector::Length4() const { return mVector(DirectX::XMVector4Length(v)).x; }
float_t mVector::LengthEst4() const { return mVector(DirectX::XMVector4LengthEst(v)).x; }
float_t mVector::LengthSquared4() const { return mVector(DirectX::XMVector4LengthSq(v)).x; }
mVector mVector::Normalize4() const { return mVector(DirectX::XMVector4Normalize(v)); }
mVector mVector::NormalizeEst4() const { return mVector(DirectX::XMVector4NormalizeEst(v)); }
mVector mVector::Orthogonal4() const { return mVector(DirectX::XMVector4Orthogonal(v)); }
mVector mVector::ReciprocalLength4() const { return mVector(DirectX::XMVector4ReciprocalLength(v)); }
mVector mVector::ReciprocalLengthEst4() const { return mVector(DirectX::XMVector4ReciprocalLengthEst(v)); }
mVector mVECTORCALL mVector::Reflect4(const mVector &incident, const mVector &normal) { return mVector(DirectX::XMVector4Reflect(incident.v, normal.v)); }
mVector mVECTORCALL mVector::Refract4(const mVector &incident, const mVector &normal, const float_t refractionIndex) { return mVector(DirectX::XMVector4Refract(incident.v, normal.v, refractionIndex)); }
mVector mVECTORCALL mVector::RefractVector4(const mVector &incident, const mVector &normal, const mVector &refractionIndex) { return mVector(DirectX::XMVector4RefractV(incident.v, normal.v, refractionIndex.v)); }
mVector mVector::Transform4(const mMatrix &matrix) const { return mVector(DirectX::XMVector4Transform(v, (DirectX::FXMMATRIX)matrix.m)); }

mResult mVECTORCALL mVector::TransformStream4(OUT DirectX::XMFLOAT4 *pOutputData, const size_t outputStride, IN DirectX::XMFLOAT4 *pInputData, const size_t inputStride, const size_t inputLength, const mMatrix &matrix)
{
  mFUNCTION_SETUP();

  mERROR_IF(pOutputData == nullptr || pInputData == nullptr, mR_ArgumentNull);
  DirectX::XMVector4TransformStream(pOutputData, outputStride, pInputData, inputStride, inputLength, matrix.m);

  mRETURN_SUCCESS();
}

mQuaternion mQuaternion::operator*(const mQuaternion &q1) const { return Multiply(q1); };
mQuaternion & mQuaternion::operator*=(const mQuaternion &q1) { return *this = Multiply(q1); };

mQuaternion mQuaternion::BaryCentric(const mQuaternion &q0, const mQuaternion &q1, const mQuaternion &q2, const float_t f, const float_t g) { return mQuaternion(DirectX::XMQuaternionBaryCentric(q0.q, q1.q, q2.q, f, g)); }
mQuaternion mQuaternion::BaryCentricV(const mQuaternion &q0, const mQuaternion &q1, const mQuaternion &q2, const mVector &f, const mVector &g) { return mQuaternion(DirectX::XMQuaternionBaryCentricV(q0.q, q1.q, q2.q, f.v, g.v)); }
mQuaternion mQuaternion::Conjugate() const { return mQuaternion(DirectX::XMQuaternionConjugate(q)); }
float_t mQuaternion::Dot(const mQuaternion &q2) const { return mQuaternion(DirectX::XMQuaternionDot(q, q2.q)).x; }
bool mQuaternion::Equals(const mQuaternion &q2) const { return DirectX::XMQuaternionEqual(q, q2.q); }
mQuaternion mQuaternion::Exp() const { return mQuaternion(DirectX::XMQuaternionExp(q)); }
mQuaternion mQuaternion::Identity() { return mQuaternion(DirectX::XMQuaternionIdentity()); }
mQuaternion mQuaternion::Inverse() const { return mQuaternion(DirectX::XMQuaternionInverse(q)); }
bool mQuaternion::IsIdentity() const { return DirectX::XMQuaternionIsIdentity(q); }
float_t mQuaternion::Length() const { return mQuaternion(DirectX::XMQuaternionLength(q)).x; }
float_t mQuaternion::LengthSq() const { return mQuaternion(DirectX::XMQuaternionLengthSq(q)).x; }
mQuaternion mQuaternion::Ln() const { return mQuaternion(DirectX::XMQuaternionLn(q)); }
mQuaternion mQuaternion::Multiply(const mQuaternion &q2) const { return mQuaternion(DirectX::XMQuaternionMultiply(q, q2.q)); }
mQuaternion mQuaternion::Normalize() const { return mQuaternion(DirectX::XMQuaternionNormalize(q)); }
mQuaternion mQuaternion::NormalizeEst() const { return mQuaternion(DirectX::XMQuaternionNormalizeEst(q)); }
bool mQuaternion::NotEqualTo(const mQuaternion &q2) const { return DirectX::XMQuaternionNotEqual(q, q2.q); }
mVector mQuaternion::ReciprocalLength() const { return mVector(DirectX::XMQuaternionReciprocalLength(q)); }
mQuaternion mVECTORCALL mQuaternion::RotationAxis(const mVector &axis, const float_t angle) { return mQuaternion(DirectX::XMQuaternionRotationAxis(axis.v, angle)); }
mQuaternion mVECTORCALL mQuaternion::RotationMatrix(const mMatrix &m) { return mQuaternion(DirectX::XMQuaternionRotationMatrix(m.m)); }
mQuaternion mVECTORCALL mQuaternion::RotationNormal(const mVector &normalAxis, const float_t angle) { return mQuaternion(DirectX::XMQuaternionRotationNormal(normalAxis.v, angle)); }
mQuaternion mVECTORCALL mQuaternion::RotationRollPitchYaw(const float_t pitch, const float_t yaw, const float_t roll) { return mQuaternion(DirectX::XMQuaternionRotationRollPitchYaw(pitch, yaw, roll)); }
mQuaternion mVECTORCALL mQuaternion::RotationRollPitchYawFromVector(const mVector &angles) { return mQuaternion(DirectX::XMQuaternionRotationRollPitchYawFromVector(angles.v)); }
mQuaternion mVECTORCALL mQuaternion::Slerp(const mQuaternion &q0, const mQuaternion &q1, const float_t t) { return mQuaternion(DirectX::XMQuaternionSlerp(q0.q, q1.q, t)); }
mQuaternion mVECTORCALL mQuaternion::SlerpV(const mQuaternion &q0, const mQuaternion &q1, const mVector &t) { return mQuaternion(DirectX::XMQuaternionSlerpV(q0.q, q1.q, t.v)); }
mQuaternion mVECTORCALL mQuaternion::Squad(const mQuaternion &q0, const mQuaternion &q1, const mQuaternion &q2, const mQuaternion &q3, const float_t t) { return mQuaternion(DirectX::XMQuaternionSquad(q0.q, q1.q, q2.q, q3.q, t)); }
mQuaternion mVECTORCALL mQuaternion::SquadV(const mQuaternion &q0, const mQuaternion &q1, const mQuaternion &q2, const mQuaternion &q3, const mVector &t) { return mQuaternion(DirectX::XMQuaternionSquadV(q0.q, q1.q, q2.q, q3.q, t.v)); }

mResult mQuaternion::SquadSetup(OUT mVector *pA, OUT mVector *pB, OUT mVector *pC, const mQuaternion &q0, const mQuaternion &q1, const mQuaternion &q2, const mQuaternion &q3)
{
  mFUNCTION_SETUP();

  mERROR_IF(pA == nullptr || pB == nullptr || pC == nullptr, mR_ArgumentNull);
  DirectX::XMQuaternionSquadSetup(&pA->v, &pB->v, &pC->v, q0.q, q1.q, q2.q, q3.q);

  mRETURN_SUCCESS();
}

mResult mQuaternion::ToAxisAngle(OUT mVector *pAxis, OUT float_t *pAngle)
{
  mFUNCTION_SETUP();

  mERROR_IF(pAxis == nullptr || pAngle == nullptr, mR_ArgumentNull);
  DirectX::XMQuaternionToAxisAngle(&pAxis->v, pAngle, q);

  mRETURN_SUCCESS();
}

mVec3f mQuaternion::ToEulerAngles()
{
  // x-axis rotation
  const float_t sinr_cosp = 2.0f * (w * x + y * z);
  const float_t cosr_cosp = 1.0f - 2.0f * (x * x + y * y);
  const float_t roll = atan2(sinr_cosp, cosr_cosp);

  // y-axis rotation
  const float_t sinp = 2.0f * (w * y - z * x);
  float_t pitch;

  if (fabsf(sinp) >= 1)
    pitch = copysign(mHALFPIf, sinp);
  else
    pitch = asin(sinp);

  // z-axis rotation
  const float_t siny_cosp = 2.0f * (w * z + x * y);
  const float_t cosy_cosp = 1.0f - 2.0f * (y * y + z * z);
  const float_t yaw = atan2(siny_cosp, cosy_cosp);

  return mVec3f(yaw, pitch, roll);
}

mMatrix mMatrix::operator*(const mMatrix &q1) const { return Multiply(q1); }
mMatrix & mMatrix::operator*=(const mMatrix &q1) { return *this = Multiply(q1); }
mMatrix & mMatrix::operator=(const mMatrix &copy) { m = copy.m; return *this; }

mResult mMatrix::Decompose(OUT mVector *pOutScale, OUT mQuaternion *pOutRotQuat, OUT mVector *pOutTrans) const
{
  mFUNCTION_SETUP();

  mERROR_IF(pOutScale == nullptr || pOutRotQuat == nullptr || pOutTrans == nullptr, mR_ArgumentNull);
  mERROR_IF(DirectX::XMMatrixDecompose(&pOutScale->v, &pOutRotQuat->q, &pOutTrans->v, m), mR_InternalError);

  mRETURN_SUCCESS();
}

mMatrix mVECTORCALL mMatrix::AffineTransformation(const mVector &scaling, const mVector &rotationOrigin, const mQuaternion &rotationQuaternion, const mVector &translation) { return mMatrix(DirectX::XMMatrixAffineTransformation(scaling.v, rotationOrigin.v, rotationQuaternion.q, translation.v)); }
mMatrix mVECTORCALL mMatrix::AffineTransformation2D(const mVector &scaling, const mVector &rotationOrigin, const float_t rotation, const mVector &translation) { return mMatrix(DirectX::XMMatrixAffineTransformation2D(scaling.v, rotationOrigin.v, rotation, translation.v)); }
mVector mMatrix::Determinant() const { return mVector(DirectX::XMMatrixDeterminant(m)); }
mMatrix mVECTORCALL mMatrix::Identity() { return mMatrix(DirectX::XMMatrixIdentity()); }
mMatrix mMatrix::Inverse(OUT OPTIONAL mVector *pDeterminant /* = nullptr */) const { return mMatrix(DirectX::XMMatrixInverse(&pDeterminant->v, m)); }
mMatrix mVECTORCALL mMatrix::LookAtLH(const mVector &eyePosition, const mVector &focusPosition, const mVector &upDirection) { return mMatrix(DirectX::XMMatrixLookAtLH(eyePosition.v, focusPosition.v, upDirection.v)); }
mMatrix mVECTORCALL mMatrix::LookAtRH(const mVector &eyePosition, const mVector &focusPosition, const mVector &upDirection) { return mMatrix(DirectX::XMMatrixLookAtRH(eyePosition.v, focusPosition.v, upDirection.v)); }
mMatrix mVECTORCALL mMatrix::LookToLH(const mVector &eyePosition, const mVector &eyeDirection, const mVector &upDirection) { return mMatrix(DirectX::XMMatrixLookToLH(eyePosition.v, eyeDirection.v, upDirection.v)); }
mMatrix mVECTORCALL mMatrix::LookToRH(const mVector &eyePosition, const mVector &eyeDirection, const mVector &upDirection) { return mMatrix(DirectX::XMMatrixLookToRH(eyePosition.v, eyeDirection.v, upDirection.v)); }
mMatrix mMatrix::Multiply(const mMatrix &m2) const { return mMatrix(DirectX::XMMatrixMultiply(m, m2.m)); }
mMatrix mMatrix::MultiplyTranspose(const mMatrix &m2) const { return mMatrix(DirectX::XMMatrixMultiplyTranspose(m, m2.m)); }
mMatrix mVECTORCALL mMatrix::OrthographicLH(const float_t viewWidth, const float_t viewHeight, const float_t nearZ, const float_t farZ) { return mMatrix(DirectX::XMMatrixOrthographicLH(viewWidth, viewHeight, nearZ, farZ)); }
mMatrix mVECTORCALL mMatrix::OrthographicRH(const float_t viewWidth, const float_t viewHeight, const float_t nearZ, const float_t farZ) { return mMatrix(DirectX::XMMatrixOrthographicRH(viewWidth, viewHeight, nearZ, farZ)); }
mMatrix mVECTORCALL mMatrix::OrthographicOffCenterLH(const float_t viewLeft, const float_t viewRight, const float_t viewBottom, const float_t viewTop, const float_t nearZ, const float_t farZ) { return mMatrix(DirectX::XMMatrixOrthographicOffCenterLH(viewLeft, viewRight, viewBottom, viewTop, nearZ, farZ)); }
mMatrix mVECTORCALL mMatrix::OrthographicOffCenterRH(const float_t viewLeft, const float_t viewRight, const float_t viewBottom, const float_t viewTop, const float_t nearZ, const float_t farZ) { return mMatrix(DirectX::XMMatrixOrthographicOffCenterRH(viewLeft, viewRight, viewBottom, viewTop, nearZ, farZ)); }
mMatrix mVECTORCALL mMatrix::PerspectiveLH(const float_t viewWidth, const float_t viewHeight, const float_t nearZ, const float_t farZ) { return mMatrix(DirectX::XMMatrixPerspectiveLH(viewWidth, viewHeight, nearZ, farZ)); }
mMatrix mVECTORCALL mMatrix::PerspectiveRH(const float_t viewWidth, const float_t viewHeight, const float_t nearZ, const float_t farZ) { return mMatrix(DirectX::XMMatrixPerspectiveRH(viewWidth, viewHeight, nearZ, farZ)); }
mMatrix mVECTORCALL mMatrix::PerspectiveFovLH(const float_t fovAngleY, const float_t aspectRatio, const float_t nearZ, const float_t farZ) { return mMatrix(DirectX::XMMatrixPerspectiveFovLH(fovAngleY, aspectRatio, nearZ, farZ)); }
mMatrix mVECTORCALL mMatrix::PerspectiveFovRH(const float_t fovAngleY, const float_t aspectRatio, const float_t nearZ, const float_t farZ) { return mMatrix(DirectX::XMMatrixPerspectiveFovRH(fovAngleY, aspectRatio, nearZ, farZ)); }
mMatrix mVECTORCALL mMatrix::PerspectiveOffCenterLH(const float_t viewLeft, const float_t viewRight, const float_t viewBottom, const float_t viewTop, const float_t nearZ, const float_t farZ) { return mMatrix(DirectX::XMMatrixPerspectiveOffCenterLH(viewLeft, viewRight, viewBottom, viewTop, nearZ, farZ)); }
mMatrix mVECTORCALL mMatrix::PerspectiveOffCenterRH(const float_t viewLeft, const float_t viewRight, const float_t viewBottom, const float_t viewTop, const float_t nearZ, const float_t farZ) { return mMatrix(DirectX::XMMatrixPerspectiveOffCenterRH(viewLeft, viewRight, viewBottom, viewTop, nearZ, farZ)); }
mMatrix mVECTORCALL mMatrix::Reflect(const mVector &reflectionPlane) { return mMatrix(DirectX::XMMatrixReflect(reflectionPlane.v)); }
mMatrix mVECTORCALL mMatrix::RotationAxis(const mVector &axis, const float_t angle) { return mMatrix(DirectX::XMMatrixRotationAxis(axis.v, angle)); }
mMatrix mVECTORCALL mMatrix::RotationQuaternion(const mQuaternion &quaternion) { return mMatrix(DirectX::XMMatrixRotationQuaternion(quaternion.q)); }
mMatrix mVECTORCALL mMatrix::RotationNormal(const mVector &normalAxis, const float_t angle) { return mMatrix(DirectX::XMMatrixRotationNormal(normalAxis.v, angle)); }
mMatrix mVECTORCALL mMatrix::RotationRollPitchYaw(const float_t pitch, const float_t yaw, const float_t roll) { return mMatrix(DirectX::XMMatrixRotationRollPitchYaw(pitch, yaw, roll)); }
mMatrix mVECTORCALL mMatrix::RotationRollPitchYawFromVector(const mVector &angles) { return mMatrix(DirectX::XMMatrixRotationRollPitchYawFromVector(angles.v)); }
mMatrix mVECTORCALL mMatrix::RotationX(const float_t angle) { return mMatrix(DirectX::XMMatrixRotationX(angle)); }
mMatrix mVECTORCALL mMatrix::RotationY(const float_t angle) { return mMatrix(DirectX::XMMatrixRotationY(angle)); }
mMatrix mVECTORCALL mMatrix::RotationZ(const float_t angle) { return mMatrix(DirectX::XMMatrixRotationZ(angle)); }
mMatrix mVECTORCALL mMatrix::Scale(const float_t scaleX, const float_t scaleY, const float_t scaleZ) { return mMatrix(DirectX::XMMatrixScaling(scaleX, scaleY, scaleZ)); }
mMatrix mVECTORCALL mMatrix::ScalingFromVector(const mVector &scale) { return mMatrix(DirectX::XMMatrixScalingFromVector(scale.v)); }
mMatrix mVECTORCALL mMatrix::Shadow(const mVector &shadowPlane, const mVector &lightPosition) { return mMatrix(DirectX::XMMatrixShadow(shadowPlane.v, lightPosition.v)); }
mMatrix mVECTORCALL mMatrix::Transformation(const mVector &scalingOrigin, const mQuaternion &scalingOrientationQuaternion, const mVector &scaling, const mVector &rotationOrigin, const mQuaternion &rotationQuaternion, const mVector &translation) { return mMatrix(DirectX::XMMatrixTransformation(scalingOrigin.v, scalingOrientationQuaternion.q, scaling.v, rotationOrigin.v, rotationQuaternion.q, translation.v)); }
mMatrix mVECTORCALL mMatrix::Transformation2D(const mVector &scalingOrigin, const float_t scalingOrientation, const mVector &scaling, const mVector &rotationOrigin, const float_t rotation, const mVector &translation) { return mMatrix(DirectX::XMMatrixTransformation2D(scalingOrigin.v, scalingOrientation, scaling.v, rotationOrigin.v, rotation, translation.v)); }
mMatrix mVECTORCALL mMatrix::Translation(const float_t offsetX, const float_t offsetY, const float_t offsetZ) { return mMatrix(DirectX::XMMatrixTranslation(offsetX, offsetY, offsetZ)); }
mMatrix mVECTORCALL mMatrix::TranslationFromVector(const mVector &offset) { return mMatrix(DirectX::XMMatrixTranslationFromVector(offset.v)); }
mMatrix mMatrix::Transpose() const { return mMatrix(DirectX::XMMatrixTranspose(m)); }
