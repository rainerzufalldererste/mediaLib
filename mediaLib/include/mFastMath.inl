
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
inline bool mVector::InBounds(const mVector & a) const { return DirectX::XMVector2InBounds(v, a.v); }
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
