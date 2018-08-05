// Copyright 2018 Christoph Stiller
// 
// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files(the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions :
// 
// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
// 
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

inline mVector mVector::operator+(const mVector &a) const { return mVector(DirectX::XMVectorAdd(v, a.v)); }
inline mVector mVector::operator-(const mVector &a) const { return mVector(DirectX::XMVectorSubtract(v, a.v)); }
inline mVector mVector::operator*(const mVector &a) const { return mVector(DirectX::XMVectorMultiply(v, a.v)); }
inline mVector mVector::operator/(const mVector &a) const { return mVector(DirectX::XMVectorDivide(v, a.v)); }

inline mVector & mVector::operator+=(const mVector & a) { return *this = (*this + a); }
inline mVector & mVector::operator-=(const mVector & a) { return *this = (*this - a); }
inline mVector & mVector::operator*=(const mVector & a) { return *this = (*this * a); }
inline mVector & mVector::operator/=(const mVector & a) { return *this = (*this / a); }

inline mVector mVector::operator*(const float_t a) const { return mVector(DirectX::XMVectorScale(v, a)); }
inline mVector mVector::operator/(const float_t a) const { return mVector(DirectX::XMVectorScale(v, 1.0f / a)); }

inline mVector & mVector::operator*=(const float_t a) { return *this = (*this * a); }
inline mVector & mVector::operator/=(const float_t a) { return *this = (*this / a); }

inline mVector mVector::operator-() const { return mVector(DirectX::XMVectorNegate(v)); }

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
  DirectX::XMVECTOR __v = DirectX::XMVectorEqual(v, a.v);
  DirectX::XMFLOAT4 f;
  DirectX::XMStoreFloat4(&f, __v);
  return mVec4t<bool>(f.x, f.y, f.z, f.w);
}

inline mVector mVector::Abs() const { return mVector(DirectX::XMVectorAbs(v)); }

inline mVector mVector::Min(const mVector & a, const mVector & b) { return mVector(DirectX::XMVectorMin(a.v, b.v)); }
inline mVector mVector::Max(const mVector & a, const mVector & b) { return mVector(DirectX::XMVectorMax(a.v, b.v)); }

inline mVector mVector::Lerp(const mVector & a, const mVector & b, const float_t t) { return mVector(DirectX::XMVectorLerp(a.v, b.v, t)); }
inline mVector mVector::LerpVector(const mVector & a, const mVector & b, const mVector & t) { return mVector(DirectX::XMVectorLerpV(a.v, b.v, t.v)); }
inline mVector mVector::Barycentric(const mVector & a, const mVector & b, const mVector & c, const float_t f, const float_t g) { return mVector(DirectX::XMVectorBaryCentric(a.v, b.v, c.v, f, g)); }
inline mVector mVector::BarycentricVector(const mVector & a, const mVector & b, const mVector & c, const mVector & f, const mVector & g) { return mVector(DirectX::XMVectorBaryCentricV(a.v, b.v, c.v, f.v, g.v)); }
inline mVector mVector::CatmullRom(const mVector & a, const mVector & b, const mVector & c, const mVector & d, const float_t f) { return mVector(DirectX::XMVectorCatmullRom(a.v, b.v, c.v, d.v, f)); }
inline mVector mVector::CatmullRomVector(const mVector & a, const mVector & b, const mVector & c, const mVector & d, const mVector & f) { return mVector(DirectX::XMVectorCatmullRomV(a.v, b.v, c.v, d.v, f.v)); }
inline mVector mVector::Hermite(const mVector & v1, const mVector & t1, const mVector & v2, const mVector & t2, const float_t f) { return mVector(DirectX::XMVectorHermite(v1.v, t1.v, v2.v, t2.v, f)); }
inline mVector mVector::HermiteVector(const mVector & v1, const mVector & t1, const mVector & v2, const mVector & t2, const  mVector & f) { return mVector(DirectX::XMVectorHermiteV(v1.v, t1.v, v2.v, t2.v, f.v)); }

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
inline bool mVector::InBounds2(const mVector & a) const { return DirectX::XMVector2InBounds(v, a.v); }
inline mVector mVector::IntersectLine2(const mVector & line1Point1, const mVector & line1Point2, const mVector & line2Point1, const mVector & line2Point2) { return mVector(DirectX::XMVector2IntersectLine(line1Point1.v, line1Point2.v, line2Point1.v, line2Point2.v)); }
inline float_t mVector::LinePointDistance2(const mVector & line1Point1, const mVector & line1Point2, const mVector & point) { return mVector(DirectX::XMVector2LinePointDistance(line1Point1.v, line1Point2.v, point.v)).x; }
inline float_t mVector::Length2() const { return mVector(DirectX::XMVector2Length(v)).x; }
inline float_t mVector::LengthEst2() const { return mVector(DirectX::XMVector2LengthEst(v)).x; }
inline float_t mVector::LengthSquared2() const { return mVector(DirectX::XMVector2LengthSq(v)).x; }
inline mVector mVector::Normalize2() const { return mVector(DirectX::XMVector2Normalize(v)); }
inline mVector mVector::NormalizeEst2() const { return mVector(DirectX::XMVector2NormalizeEst(v)); }
inline mVector mVector::Orthogonal2() const { return mVector(DirectX::XMVector2Orthogonal(v)); }
inline mVector mVector::ReciprocalLength2() const { return mVector(DirectX::XMVector2ReciprocalLength(v)); }
inline mVector mVector::ReciprocalLengthEst2() const { return mVector(DirectX::XMVector2ReciprocalLengthEst(v)); }
inline mVector mVector::Reflect2(const mVector & incident, const mVector & normal) { return mVector(DirectX::XMVector2Reflect(incident.v, normal.v)); }
inline mVector mVector::Refract2(const mVector & incident, const mVector & normal, const float_t refractionIndex) { return mVector(DirectX::XMVector2Refract(incident.v, normal.v, refractionIndex)); }
inline mVector mVector::RefractVector2(const mVector & incident, const mVector & normal, const mVector & refractionIndex) { return mVector(DirectX::XMVector2RefractV(incident.v, normal.v, refractionIndex.v)); }
inline mVector mVector::Transform2(const mMatrix &matrix) const { return mVector(DirectX::XMVector2Transform(v, (DirectX::FXMMATRIX)matrix.m)); }
inline mVector mVector::TransformCoord2(const mMatrix & matrix) const { return mVector(DirectX::XMVector2TransformCoord(v, matrix.m)); }
inline mVector mVector::TransformNormal2(const mMatrix & matrix) const { return mVector(DirectX::XMVector2TransformNormal(v, matrix.m)); }

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

inline mVector mVector::AngleBetweenNormals3(const mVector & a) const { return mVector(DirectX::XMVector3AngleBetweenNormals(v, a.v)); }
inline mVector mVector::AngleBetweenNormalsEst3(const mVector & a) const { return mVector(DirectX::XMVector3AngleBetweenNormalsEst(v, a.v)); }
inline mVector mVector::AngleBetweenVectors3(const mVector & a) const { return mVector(DirectX::XMVector3AngleBetweenVectors(v, a.v)); }
inline mVector mVector::ClampLength3(const float_t min, const float_t max) const { return mVector(DirectX::XMVector3ClampLength(v, min, max)); }
inline mVector mVector::ClampLengthVectors3(const mVector & min, const mVector & max) const { return mVector(DirectX::XMVector3ClampLengthV(v, min.v, max.v)); }
inline mVector mVector::Dot3(const mVector & a) const { return mVector(DirectX::XMVector3Dot(v, a.v)); }
inline mVector mVector::Cross3(const mVector & a) const { return mVector(DirectX::XMVector3Cross(v, a.v)); }
inline bool mVector::Equals3(const mVector & a) const { return DirectX::XMVector3Equal(v, a.v); }
inline bool mVector::NotEqualTo3(const mVector & a) const { return DirectX::XMVector3NotEqual(v, a.v); }
inline bool mVector::EqualsApproximately3(const mVector & a, const mVector & epsilon) const { return DirectX::XMVector3NearEqual(v, a.v, epsilon.v); }
inline bool mVector::Greater3(const mVector & a) const { return DirectX::XMVector3Greater(v, a.v); }
inline bool mVector::GreaterOrEqual3(const mVector & a) const { return DirectX::XMVector3GreaterOrEqual(v, a.v); }
inline bool mVector::Less3(const mVector & a) const { return DirectX::XMVector3Less(v, a.v); }
inline bool mVector::LessOrEqual3(const mVector & a) const { return DirectX::XMVector3LessOrEqual(v, a.v); }
inline bool mVector::InBounds3(const mVector & a) const { return DirectX::XMVector3InBounds(v, a.v); }
inline float_t mVector::LinePointDistance3(const mVector & line1Point1, const mVector & line1Point2, const mVector & point) { return mVector(DirectX::XMVector3LinePointDistance(line1Point1.v, line1Point2.v, point.v)).x; }
inline float_t mVector::Length3() const { return mVector(DirectX::XMVector3Length(v)).x; }
inline float_t mVector::LengthEst3() const { return mVector(DirectX::XMVector3LengthEst(v)).x; }
inline float_t mVector::LengthSquared3() const { return mVector(DirectX::XMVector3LengthSq(v)).x; }
inline mVector mVector::Normalize3() const { return mVector(DirectX::XMVector3Normalize(v)); }
inline mVector mVector::NormalizeEst3() const { return mVector(DirectX::XMVector3NormalizeEst(v)); }
inline mVector mVector::Orthogonal3() const { return mVector(DirectX::XMVector3Orthogonal(v)); }
inline mVector mVector::ReciprocalLength3() const { return mVector(DirectX::XMVector3ReciprocalLength(v)); }
inline mVector mVector::ReciprocalLengthEst3() const { return mVector(DirectX::XMVector3ReciprocalLengthEst(v)); }
inline mVector mVector::Reflect3(const mVector & incident, const mVector & normal) { return mVector(DirectX::XMVector3Reflect(incident.v, normal.v)); }
inline mVector mVector::Refract3(const mVector & incident, const mVector & normal, const float_t refractionIndex) { return mVector(DirectX::XMVector3Refract(incident.v, normal.v, refractionIndex)); }
inline mVector mVector::RefractVector3(const mVector & incident, const mVector & normal, const mVector & refractionIndex) { return mVector(DirectX::XMVector3RefractV(incident.v, normal.v, refractionIndex.v)); }
inline mVector mVector::Transform3(const mMatrix &matrix) const { return mVector(DirectX::XMVector3Transform(v, (DirectX::FXMMATRIX)matrix.m)); }
inline mVector mVector::TransformCoord3(const mMatrix & matrix) const { return mVector(DirectX::XMVector3TransformCoord(v, matrix.m)); }
inline mVector mVector::TransformNormal3(const mMatrix & matrix) const { return mVector(DirectX::XMVector3TransformNormal(v, matrix.m)); }
inline mVector mVector::Rotate3(const mQuaternion & quaternion) const { return mVector(DirectX::XMVector3Rotate(v, quaternion.q)); }
inline mVector mVector::RotateInverse3(const mQuaternion & quaternion) const { return mVector(DirectX::XMVector3InverseRotate(v, quaternion.q)); }

inline mFUNCTION(mVector::TransformCoordStream3, OUT DirectX::XMFLOAT3 * pOutputData, const size_t outputStride, IN DirectX::XMFLOAT3 * pInputData, const size_t inputStride, const size_t inputLength, const mMatrix & matrix)
{
  mFUNCTION_SETUP();

  mERROR_IF(pOutputData == nullptr || pInputData == nullptr, mR_ArgumentNull);
  DirectX::XMVector3TransformCoordStream(pOutputData, outputStride, pInputData, inputStride, inputLength, matrix.m);

  mRETURN_SUCCESS();
}

inline mFUNCTION(mVector::TransformNormalStream3, OUT DirectX::XMFLOAT3 * pOutputData, const size_t outputStride, IN DirectX::XMFLOAT3 * pInputData, const size_t inputStride, const size_t inputLength, const mMatrix & matrix)
{
  mFUNCTION_SETUP();

  mERROR_IF(pOutputData == nullptr || pInputData == nullptr, mR_ArgumentNull);
  DirectX::XMVector3TransformNormalStream(pOutputData, outputStride, pInputData, inputStride, inputLength, matrix.m);

  mRETURN_SUCCESS();
}

inline mFUNCTION(mVector::ComponentsFromNormal3, OUT mVector *pParallel, OUT mVector *pPerpendicular, const mVector &v, const mVector &normal)
{
  mFUNCTION_SETUP();

  mERROR_IF(pParallel == nullptr || pParallel == nullptr, mR_ArgumentNull);
  DirectX::XMVector3ComponentsFromNormal(&pParallel->v, &pPerpendicular->v, v.v, normal.v);

  mRETURN_SUCCESS();
}

inline mVector mVector::AngleBetweenNormals4(const mVector & a) const { return mVector(DirectX::XMVector4AngleBetweenNormals(v, a.v)); }
inline mVector mVector::AngleBetweenNormalsEst4(const mVector & a) const { return mVector(DirectX::XMVector4AngleBetweenNormalsEst(v, a.v)); }
inline mVector mVector::AngleBetweenVectors4(const mVector & a) const { return mVector(DirectX::XMVector4AngleBetweenVectors(v, a.v)); }
inline mVector mVector::ClampLength4(const float_t min, const float_t max) const { return mVector(DirectX::XMVector4ClampLength(v, min, max)); }
inline mVector mVector::ClampLengthVectors4(const mVector & min, const mVector & max) const { return mVector(DirectX::XMVector4ClampLengthV(v, min.v, max.v)); }
inline mVector mVector::Dot4(const mVector & a) const { return mVector(DirectX::XMVector4Dot(v, a.v)); }
inline mVector mVector::Cross4(const mVector & a, const mVector & b) const { return mVector(DirectX::XMVector4Cross(v, a.v, b.v)); }
inline bool mVector::Equals4(const mVector & a) const { return DirectX::XMVector4Equal(v, a.v); }
inline bool mVector::NotEqualTo4(const mVector & a) const { return DirectX::XMVector4NotEqual(v, a.v); }
inline bool mVector::EqualsApproximately4(const mVector & a, const mVector & epsilon) const { return DirectX::XMVector4NearEqual(v, a.v, epsilon.v); }
inline bool mVector::Greater4(const mVector & a) const { return DirectX::XMVector4Greater(v, a.v); }
inline bool mVector::GreaterOrEqual4(const mVector & a) const { return DirectX::XMVector4GreaterOrEqual(v, a.v); }
inline bool mVector::Less4(const mVector & a) const { return DirectX::XMVector4Less(v, a.v); }
inline bool mVector::LessOrEqual4(const mVector & a) const { return DirectX::XMVector4LessOrEqual(v, a.v); }
inline bool mVector::InBounds4(const mVector & a) const { return DirectX::XMVector4InBounds(v, a.v); }
inline float_t mVector::Length4() const { return mVector(DirectX::XMVector4Length(v)).x; }
inline float_t mVector::LengthEst4() const { return mVector(DirectX::XMVector4LengthEst(v)).x; }
inline float_t mVector::LengthSquared4() const { return mVector(DirectX::XMVector4LengthSq(v)).x; }
inline mVector mVector::Normalize4() const { return mVector(DirectX::XMVector4Normalize(v)); }
inline mVector mVector::NormalizeEst4() const { return mVector(DirectX::XMVector4NormalizeEst(v)); }
inline mVector mVector::Orthogonal4() const { return mVector(DirectX::XMVector4Orthogonal(v)); }
inline mVector mVector::ReciprocalLength4() const { return mVector(DirectX::XMVector4ReciprocalLength(v)); }
inline mVector mVector::ReciprocalLengthEst4() const { return mVector(DirectX::XMVector4ReciprocalLengthEst(v)); }
inline mVector mVector::Reflect4(const mVector & incident, const mVector & normal) { return mVector(DirectX::XMVector4Reflect(incident.v, normal.v)); }
inline mVector mVector::Refract4(const mVector & incident, const mVector & normal, const float_t refractionIndex) { return mVector(DirectX::XMVector4Refract(incident.v, normal.v, refractionIndex)); }
inline mVector mVector::RefractVector4(const mVector & incident, const mVector & normal, const mVector & refractionIndex) { return mVector(DirectX::XMVector4RefractV(incident.v, normal.v, refractionIndex.v)); }
inline mVector mVector::Transform4(const mMatrix &matrix) const { return mVector(DirectX::XMVector4Transform(v, (DirectX::FXMMATRIX)matrix.m)); }

inline mFUNCTION(mVector::TransformStream4, OUT DirectX::XMFLOAT4 * pOutputData, const size_t outputStride, IN DirectX::XMFLOAT4 * pInputData, const size_t inputStride, const size_t inputLength, const mMatrix & matrix)
{
  mFUNCTION_SETUP();

  mERROR_IF(pOutputData == nullptr || pInputData == nullptr, mR_ArgumentNull);
  DirectX::XMVector4TransformStream(pOutputData, outputStride, pInputData, inputStride, inputLength, matrix.m);

  mRETURN_SUCCESS();
}

inline mQuaternion mQuaternion::operator*(const mQuaternion & q1) { return Multiply(q1); };
inline mQuaternion & mQuaternion::operator*=(const mQuaternion & q1) { return *this = Multiply(q1); };

inline mQuaternion mQuaternion::BaryCentric(const mQuaternion & q0, const mQuaternion & q1, const mQuaternion & q2, const float_t f, const float_t g) { return mQuaternion(DirectX::XMQuaternionBaryCentric(q0.q, q1.q, q2.q, f, g)); }
inline mQuaternion mQuaternion::BaryCentricV(const mQuaternion & q0, const mQuaternion & q1, const mQuaternion & q2, const mVector & f, const mVector & g) { return mQuaternion(DirectX::XMQuaternionBaryCentricV(q0.q, q1.q, q2.q, f.v, g.v)); }
inline mQuaternion mQuaternion::Conjugate() const { return mQuaternion(DirectX::XMQuaternionConjugate(q)); }
inline mQuaternion mQuaternion::Dot(const mQuaternion & q2) const { return mQuaternion(DirectX::XMQuaternionDot(q, q2.q)); }
inline bool mQuaternion::Equals(const mQuaternion & q2) const { return DirectX::XMQuaternionEqual(q, q2.q); }
inline mQuaternion mQuaternion::Exp() const { return mQuaternion(DirectX::XMQuaternionExp(q)); }
inline mQuaternion mQuaternion::Identity() { return mQuaternion(DirectX::XMQuaternionIdentity()); }
inline mQuaternion mQuaternion::Inverse() const { return mQuaternion(DirectX::XMQuaternionInverse(q)); }
inline bool mQuaternion::IsIdentity() const { return DirectX::XMQuaternionIsIdentity(q); }
inline float_t mQuaternion::Length() const { return mQuaternion(DirectX::XMQuaternionLength(q)).x; }
inline float_t mQuaternion::LengthSq() const { return mQuaternion(DirectX::XMQuaternionLengthSq(q)).x; }
inline mQuaternion mQuaternion::Ln() const { return mQuaternion(DirectX::XMQuaternionLn(q)); }
inline mQuaternion mQuaternion::Multiply(const mQuaternion & q2) const { return mQuaternion(DirectX::XMQuaternionMultiply(q, q2.q)); }
inline mQuaternion mQuaternion::Normalize() const { return mQuaternion(DirectX::XMQuaternionNormalize(q)); }
inline mQuaternion mQuaternion::NormalizeEst() const { return mQuaternion(DirectX::XMQuaternionNormalizeEst(q)); }
inline bool mQuaternion::NotEqualTo(const mQuaternion & q2) const { return DirectX::XMQuaternionNotEqual(q, q2.q); }
inline mVector mQuaternion::ReciprocalLength() const { return mVector(DirectX::XMQuaternionReciprocalLength(q)); }
inline mQuaternion mQuaternion::RotationAxis(const mVector & axis, const float_t angle) { return mQuaternion(DirectX::XMQuaternionRotationAxis(axis.v, angle)); }
inline mQuaternion mQuaternion::RotationMatrix(const mMatrix & m) { return mQuaternion(DirectX::XMQuaternionRotationMatrix(m.m)); }
inline mQuaternion mQuaternion::RotationNormal(const mVector & normalAxis, const float_t angle) { return mQuaternion(DirectX::XMQuaternionRotationNormal(normalAxis.v, angle)); }
inline mQuaternion mQuaternion::RotationRollPitchYaw(const float_t pitch, const float_t yaw, const float_t roll) { return mQuaternion(DirectX::XMQuaternionRotationRollPitchYaw(pitch, yaw, roll)); }
inline mQuaternion mQuaternion::RotationRollPitchYawFromVector(const mVector & angles) { return mQuaternion(DirectX::XMQuaternionRotationRollPitchYawFromVector(angles.v)); }
inline mQuaternion mQuaternion::Slerp(const mQuaternion & q0, const mQuaternion & q1, const float_t t) { return mQuaternion(DirectX::XMQuaternionSlerp(q0.q, q1.q, t)); }
inline mQuaternion mQuaternion::SlerpV(const mQuaternion & q0, const mQuaternion & q1, const mVector & t) { return mQuaternion(DirectX::XMQuaternionSlerpV(q0.q, q1.q, t.v)); }
inline mQuaternion mQuaternion::Squad(const mQuaternion & q0, const mQuaternion & q1, const mQuaternion & q2, const mQuaternion & q3, const float_t t) { return mQuaternion(DirectX::XMQuaternionSquad(q0.q, q1.q, q2.q, q3.q, t)); }

inline mFUNCTION(mQuaternion::SquadSetup, OUT mVector * pA, OUT mVector * pB, OUT mVector * pC, const mQuaternion & q0, const mQuaternion & q1, const mQuaternion & q2, const mQuaternion & q3) 
{
  mFUNCTION_SETUP();

  mERROR_IF(pA == nullptr || pB == nullptr || pC == nullptr, mR_ArgumentNull);
  DirectX::XMQuaternionSquadSetup(&pA->v, &pB->v, &pC->v, q0.q, q1.q, q2.q, q3.q); 

  mRETURN_SUCCESS();
}

inline mQuaternion mQuaternion::SquadV(const mQuaternion & q0, const mQuaternion & q1, const mQuaternion & q2, const mQuaternion & q3, const mVector & t) { mQuaternion(DirectX::XMQuaternionSquadV(q0.q, q1.q, q2.q, q3.q, t.v)); }

inline mFUNCTION(mQuaternion::ToAxisAngle, OUT mVector *pAxis, OUT float_t *pAngle)
{
  mFUNCTION_SETUP();

  mERROR_IF(pAxis == nullptr || pAngle == nullptr, mR_ArgumentNull);
  DirectX::XMQuaternionToAxisAngle(&pAxis->v, pAngle, q);

  mRETURN_SUCCESS();
}
