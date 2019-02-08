#ifndef mFastMath_h__
#define mFastMath_h__

#include "mediaLib.h"

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

  mVector();
  mVector(const mVector &a);
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

  mVector  operator +    (const mVector &a) const;
  mVector  operator -    (const mVector &a) const;
  mVector  operator *    (const mVector &a) const;
  mVector  operator /    (const mVector &a) const;
  mVector& operator +=   (const mVector &a);
  mVector& operator -=   (const mVector &a);
  mVector& operator *=   (const mVector &a);
  mVector& operator /=   (const mVector &a);
  mVector  operator *    (const float_t a) const;
  mVector  operator /    (const float_t a) const;
  mVector& operator *=   (const float_t a);
  mVector& operator /=   (const float_t a);

  mVector  operator -    () const;

  explicit operator mVec2f() const;
  explicit operator mVec3f() const;
  explicit operator mVec4f() const;

  explicit operator DirectX::XMFLOAT2() const;
  explicit operator DirectX::XMFLOAT3() const;
  explicit operator DirectX::XMFLOAT4() const;

  mVec4t<bool> ComponentEquals(const mVector &a) const;
  mVector Abs() const;

  static mVector mVECTORCALL Min(const mVector &a, const mVector &b);
  static mVector mVECTORCALL Max(const mVector &a, const mVector &b);
  static mVector mVECTORCALL Lerp(const mVector &a, const mVector &b, const float_t t);
  static mVector mVECTORCALL LerpVector(const mVector &a, const mVector &b, const mVector &t);
  static mVector mVECTORCALL Barycentric(const mVector &a, const mVector &b, const mVector &c, const float_t f, const float_t g);
  static mVector mVECTORCALL BarycentricVector(const mVector &a, const mVector &b, const mVector &c, const mVector &f, const mVector &g);
  static mVector mVECTORCALL CatmullRom(const mVector &a, const mVector &b, const mVector &c, const mVector &d, const float_t f);
  static mVector mVECTORCALL CatmullRomVector(const mVector &a, const mVector &b, const mVector &c, const mVector &d, const mVector &f);
  static mVector mVECTORCALL Hermite(const mVector &v1, const mVector &t1, const mVector &v2, const mVector &t2, const float_t f);
  static mVector mVECTORCALL HermiteVector(const mVector &v1, const mVector &t1, const mVector &v2, const mVector &t2, const mVector &f);

  mVector AngleBetweenNormals2(const mVector &a) const;
  mVector AngleBetweenNormalsEst2(const mVector &a) const;
  mVector AngleBetweenVectors2(const mVector &a) const;
  mVector ClampLength2(const float_t min, const float_t max) const;
  mVector ClampLengthVectors2(const mVector &min, const mVector &max) const;
  float_t Dot2(const mVector &a) const;
  mVector Cross2(const mVector &a) const;
  bool Equals2(const mVector &a) const;
  bool NotEqualTo2(const mVector &a) const;
  bool EqualsApproximately2(const mVector &a, const mVector &epsilon) const;
  bool Greater2(const mVector &a) const;
  bool GreaterOrEqual2(const mVector &a) const;
  bool Less2(const mVector &a) const;
  bool LessOrEqual2(const mVector &a) const;
  bool InBounds2(const mVector &a) const;
  static mVector mVECTORCALL IntersectLine2(const mVector &line1Point1, const mVector &line1Point2, const mVector &line2Point1, const mVector &line2Point2);
  static float_t mVECTORCALL LinePointDistance2(const mVector &line1Point1, const mVector &line1Point2, const mVector &point);
  float_t Length2() const;
  float_t LengthEst2() const;
  float_t LengthSquared2() const;
  mVector Normalize2() const;
  mVector NormalizeEst2() const;
  mVector Orthogonal2() const;
  mVector ReciprocalLength2() const;
  mVector ReciprocalLengthEst2() const;
  static mVector mVECTORCALL Reflect2(const mVector &incident, const mVector &normal);
  static mVector mVECTORCALL Refract2(const mVector &incident, const mVector &normal, const float_t refractionIndex);
  static mVector mVECTORCALL RefractVector2(const mVector &incident, const mVector &normal, const mVector &refractionIndex);
  mVector Transform2(const mMatrix &matrix) const;
  mVector TransformCoord2(const mMatrix &matrix) const;
  mVector TransformNormal2(const mMatrix &matrix) const;

  static mResult mVECTORCALL TransformCoordStream2(OUT DirectX::XMFLOAT2 *pOutputData, const size_t outputStride, IN DirectX::XMFLOAT2 *pInputData, const size_t inputStride, const size_t inputLength, const mMatrix &matrix);
  static mResult mVECTORCALL TransformNormalStream2(OUT DirectX::XMFLOAT2 *pOutputData, const size_t outputStride, IN DirectX::XMFLOAT2 *pInputData, const size_t inputStride, const size_t inputLength, const mMatrix &matrix);

  mVector AngleBetweenNormals3(const mVector &a) const;
  mVector AngleBetweenNormalsEst3(const mVector &a) const;
  mVector AngleBetweenVectors3(const mVector &a) const;
  mVector ClampLength3(const float_t min, const float_t max) const;
  mVector ClampLengthVectors3(const mVector &min, const mVector &max) const;
  float_t Dot3(const mVector &a) const;
  mVector Cross3(const mVector &a) const;
  bool Equals3(const mVector &a) const;
  bool NotEqualTo3(const mVector &a) const;
  bool EqualsApproximately3(const mVector &a, const mVector &epsilon) const;
  bool Greater3(const mVector &a) const;
  bool GreaterOrEqual3(const mVector &a) const;
  bool Less3(const mVector &a) const;
  bool LessOrEqual3(const mVector &a) const;
  bool InBounds3(const mVector &a) const;
  static float_t mVECTORCALL LinePointDistance3(const mVector &line1Point1, const mVector &line1Point2, const mVector &point);
  float_t Length3() const;
  float_t LengthEst3() const;
  float_t LengthSquared3() const;
  mVector Normalize3() const;
  mVector NormalizeEst3() const;
  mVector Orthogonal3() const;
  mVector ReciprocalLength3() const;
  mVector ReciprocalLengthEst3() const;
  static mVector mVECTORCALL Reflect3(const mVector &incident, const mVector &normal);
  static mVector mVECTORCALL Refract3(const mVector &incident, const mVector &normal, const float_t refractionIndex);
  static mVector mVECTORCALL RefractVector3(const mVector &incident, const mVector &normal, const mVector &refractionIndex);
  mVector Transform3(const mMatrix &matrix) const;
  mVector TransformCoord3(const mMatrix &matrix) const;
  mVector TransformNormal3(const mMatrix &matrix) const;
  mVector Rotate3(const mQuaternion &quaternion) const;
  mVector RotateInverse3(const mQuaternion &quaternion) const;

  static mResult mVECTORCALL TransformCoordStream3(OUT DirectX::XMFLOAT3 *pOutputData, const size_t outputStride, IN DirectX::XMFLOAT3 *pInputData, const size_t inputStride, const size_t inputLength, const mMatrix &matrix);
  static mResult mVECTORCALL TransformNormalStream3(OUT DirectX::XMFLOAT3 *pOutputData, const size_t outputStride, IN DirectX::XMFLOAT3 *pInputData, const size_t inputStride, const size_t inputLength, const mMatrix &matrix);
  static mResult mVECTORCALL ComponentsFromNormal3(OUT mVector *pParallel, OUT mVector *pPerpendicular, const mVector &v, const mVector &normal);

  mVector AngleBetweenNormals4(const mVector &a) const;
  mVector AngleBetweenNormalsEst4(const mVector &a) const;
  mVector AngleBetweenVectors4(const mVector &a) const;
  mVector ClampLength4(const float_t min, const float_t max) const;
  mVector ClampLengthVectors4(const mVector &min, const mVector &max) const;
  float_t Dot4(const mVector &a) const;
  mVector Cross4(const mVector &a, const mVector & b) const;
  bool Equals4(const mVector &a) const;
  bool NotEqualTo4(const mVector &a) const;
  bool EqualsApproximately4(const mVector &a, const mVector &epsilon) const;
  bool Greater4(const mVector &a) const;
  bool GreaterOrEqual4(const mVector &a) const;
  bool Less4(const mVector &a) const;
  bool LessOrEqual4(const mVector &a) const;
  bool InBounds4(const mVector &a) const;
  float_t Length4() const;
  float_t LengthEst4() const;
  float_t LengthSquared4() const;
  mVector Normalize4() const;
  mVector NormalizeEst4() const;
  mVector Orthogonal4() const;
  mVector ReciprocalLength4() const;
  mVector ReciprocalLengthEst4() const;
  static mVector mVECTORCALL Reflect4(const mVector &incident, const mVector &normal);
  static mVector mVECTORCALL Refract4(const mVector &incident, const mVector &normal, const float_t refractionIndex);
  static mVector mVECTORCALL RefractVector4(const mVector &incident, const mVector &normal, const mVector &refractionIndex);
  mVector Transform4(const mMatrix &matrix) const;

  static mResult mVECTORCALL TransformStream4(OUT DirectX::XMFLOAT4 *pOutputData, const size_t outputStride, IN DirectX::XMFLOAT4 *pInputData, const size_t inputStride, const size_t inputLength, const mMatrix &matrix);
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

  mQuaternion();
  explicit mQuaternion(DirectX::XMVECTOR v);

  mQuaternion operator *(const mQuaternion &q1) const;
  mQuaternion& operator *=(const mQuaternion &q1);

  static mQuaternion mVECTORCALL BaryCentric(const mQuaternion &q0, const mQuaternion &q1, const mQuaternion &q2, const float_t f, const float_t g);
  static mQuaternion mVECTORCALL BaryCentricV(const mQuaternion &q0, const mQuaternion &q1, const mQuaternion &q2, const mVector &f, const mVector &g);
  mQuaternion Conjugate() const;
  float_t Dot(const mQuaternion &q2) const;
  bool Equals(const mQuaternion &q2) const;
  mQuaternion Exp() const;
  static mQuaternion mVECTORCALL Identity();
  mQuaternion Inverse() const;
  bool IsIdentity() const;
  float_t Length() const;
  float_t LengthSq() const;
  mQuaternion Ln() const;
  mQuaternion Multiply(const mQuaternion &q2) const;
  mQuaternion Normalize() const;
  mQuaternion NormalizeEst() const;
  bool NotEqualTo(const mQuaternion &q2) const;
  mVector ReciprocalLength() const;
  static mQuaternion mVECTORCALL RotationAxis(const mVector &axis, const float_t angle);
  static mQuaternion mVECTORCALL RotationMatrix(const mMatrix &m);
  static mQuaternion mVECTORCALL RotationNormal(const mVector &normalAxis, const float_t angle);
  static mQuaternion mVECTORCALL RotationRollPitchYaw(const float_t pitch, const float_t yaw, const float_t roll);
  static mQuaternion mVECTORCALL RotationRollPitchYawFromVector(const mVector &angles);
  static mQuaternion mVECTORCALL Slerp(const mQuaternion &q0, const mQuaternion &q1, const float_t t);
  static mQuaternion mVECTORCALL SlerpV(const mQuaternion &q0, const mQuaternion &q1, const mVector &t);
  static mQuaternion mVECTORCALL Squad(const mQuaternion &q0, const mQuaternion &q1, const mQuaternion &q2, const mQuaternion &q3, const float_t t);
  static mQuaternion mVECTORCALL SquadV(const mQuaternion &q0, const mQuaternion &q1, const mQuaternion &q2, const mQuaternion &q3, const mVector &t);
  mResult SquadSetup(OUT mVector *pA, OUT mVector *pB, OUT mVector *pC, const mQuaternion &q0, const mQuaternion &q1, const mQuaternion &q2, const mQuaternion &q3);
  mResult ToAxisAngle(OUT mVector *pAxis, OUT float_t *pAngle);
  mVec3f ToEulerAngles();
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

  mMatrix operator *(const mMatrix &q1) const;
  mMatrix& operator *=(const mMatrix &q1);
  mMatrix& operator =(const mMatrix &copy);

  static mMatrix mVECTORCALL AffineTransformation(const mVector &scaling, const mVector &rotationOrigin, const mQuaternion &rotationQuaternion, const mVector &translation);
  static mMatrix mVECTORCALL AffineTransformation2D(const mVector &scaling, const mVector &rotationOrigin, const float_t rotation, const mVector &translation);
  mResult Decompose(OUT mVector *pOutScale, OUT mQuaternion *pOutRotQuat, OUT mVector *pOutTrans) const;
  mVector Determinant() const;
  static mMatrix mVECTORCALL Identity();
  mMatrix Inverse(OUT OPTIONAL mVector *pDeterminant = nullptr) const;
  static mMatrix mVECTORCALL LookAtLH(const mVector &eyePosition, const mVector &focusPosition, const mVector &upDirection);
  static mMatrix mVECTORCALL LookAtRH(const mVector &eyePosition, const mVector &focusPosition, const mVector &upDirection);
  static mMatrix mVECTORCALL LookToLH(const mVector &eyePosition, const mVector &eyeDirection, const mVector &upDirection);
  static mMatrix mVECTORCALL LookToRH(const mVector &eyePosition, const mVector &eyeDirection, const mVector &upDirection);
  mMatrix Multiply(const mMatrix &m2) const;
  mMatrix MultiplyTranspose(const mMatrix &m2) const;
  static mMatrix mVECTORCALL OrthographicLH(const float_t viewWidth, const float_t viewHeight, const float_t nearZ, const float_t farZ);
  static mMatrix mVECTORCALL OrthographicRH(const float_t viewWidth, const float_t viewHeight, const float_t nearZ, const float_t farZ);
  static mMatrix mVECTORCALL OrthographicOffCenterLH(const float_t viewLeft, const float_t viewRight, const float_t viewBottom, const float_t viewTop, const float_t nearZ, const float_t farZ);
  static mMatrix mVECTORCALL OrthographicOffCenterRH(const float_t viewLeft, const float_t viewRight, const float_t viewBottom, const float_t viewTop, const float_t nearZ, const float_t farZ);
  static mMatrix mVECTORCALL PerspectiveLH(const float_t viewWidth, const float_t viewHeight, const float_t nearZ, const float_t farZ);
  static mMatrix mVECTORCALL PerspectiveRH(const float_t viewWidth, const float_t viewHeight, const float_t nearZ, const float_t farZ);
  static mMatrix mVECTORCALL PerspectiveFovLH(const float_t fovAngleY, const float_t aspectRatio, const float_t nearZ, const float_t farZ);
  static mMatrix mVECTORCALL PerspectiveFovRH(const float_t fovAngleY, const float_t aspectRatio, const float_t nearZ, const float_t farZ);
  static mMatrix mVECTORCALL PerspectiveOffCenterLH(const float_t viewLeft, const float_t viewRight, const float_t viewBottom, const float_t viewTop, const float_t nearZ, const float_t farZ);
  static mMatrix mVECTORCALL PerspectiveOffCenterRH(const float_t viewLeft, const float_t viewRight, const float_t viewBottom, const float_t viewTop, const float_t nearZ, const float_t farZ);
  static mMatrix mVECTORCALL Reflect(const mVector &reflectionPlane);
  static mMatrix mVECTORCALL RotationAxis(const mVector &axis, const float_t angle);
  static mMatrix mVECTORCALL RotationQuaternion(const mQuaternion &quaternion);
  static mMatrix mVECTORCALL RotationNormal(const mVector &normalAxis, const float_t angle);
  static mMatrix mVECTORCALL RotationRollPitchYaw(const float_t pitch, const float_t yaw, const float_t roll);
  static mMatrix mVECTORCALL RotationRollPitchYawFromVector(const mVector &angles);
  static mMatrix mVECTORCALL RotationX(const float_t angle);
  static mMatrix mVECTORCALL RotationY(const float_t angle);
  static mMatrix mVECTORCALL RotationZ(const float_t angle);
  static mMatrix mVECTORCALL Scale(const float_t scaleX, const float_t scaleY, const float_t scaleZ);
  static mMatrix mVECTORCALL ScalingFromVector(const mVector &scale);
  static mMatrix mVECTORCALL Shadow(const mVector &shadowPlane, const mVector &lightPosition);
  static mMatrix mVECTORCALL Transformation(const mVector &scalingOrigin, const mQuaternion &scalingOrientationQuaternion, const mVector &scaling, const mVector &rotationOrigin, const mQuaternion &rotationQuaternion, const mVector &translation);
  static mMatrix mVECTORCALL Transformation2D(const mVector &scalingOrigin, const float_t scalingOrientation, const mVector &scaling, const mVector &rotationOrigin, const float_t rotation, const mVector &translation);
  static mMatrix mVECTORCALL Translation(const float_t offsetX, const float_t offsetY, const float_t offsetZ);
  static mMatrix mVECTORCALL TranslationFromVector(const mVector &offset);
  mMatrix Transpose() const;
};

#endif // mFastMath_h__
