#include "mediaLib.h"
#include "mFastMath.h"

//////////////////////////////////////////////////////////////////////////

#ifdef GIT_BUILD // Define __M_FILE__
  #ifdef __M_FILE__
    #undef __M_FILE__
  #endif
  #define __M_FILE__ "YOugLsA2pf+bvt2o9OdpJMpHtqbPkc1GbfgvGBqGvwyDX7xUO1J2tQ8kgQWyu/LiVcVK/zTRhkFz4lWs"
#endif

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

mResult mVECTORCALL mVector::TransformStream4(OUT DirectX::XMFLOAT4 *pOutputData, const size_t outputStride, IN DirectX::XMFLOAT4 *pInputData, const size_t inputStride, const size_t inputLength, const mMatrix &matrix)
{
  mFUNCTION_SETUP();

  mERROR_IF(pOutputData == nullptr || pInputData == nullptr, mR_ArgumentNull);
  DirectX::XMVector4TransformStream(pOutputData, outputStride, pInputData, inputStride, inputLength, matrix.m);

  mRETURN_SUCCESS();
}

mMatrix mVector::OuterProduct4(const mVector a, const mVector b)
{
  mMatrix m;

  for (size_t x = 0; x < 4; x++)
    for (size_t y = 0; y < 4; y++)
      m._m[x][y] = a._v[x] * b._v[y];

  return m;
}

//////////////////////////////////////////////////////////////////////////

mResult mQuaternion::SquadSetup(OUT mVector *pA, OUT mVector *pB, OUT mVector *pC, const mQuaternion &q0, const mQuaternion &q1, const mQuaternion &q2, const mQuaternion &q3) const
{
  mFUNCTION_SETUP();

  mERROR_IF(pA == nullptr || pB == nullptr || pC == nullptr, mR_ArgumentNull);
  DirectX::XMQuaternionSquadSetup(&pA->v, &pB->v, &pC->v, q0.q, q1.q, q2.q, q3.q);

  mRETURN_SUCCESS();
}

mResult mQuaternion::ToAxisAngle(OUT mVector *pAxis, OUT float_t *pAngle) const
{
  mFUNCTION_SETUP();

  mERROR_IF(pAxis == nullptr || pAngle == nullptr, mR_ArgumentNull);
  DirectX::XMQuaternionToAxisAngle(&pAxis->v, pAngle, q);

  mRETURN_SUCCESS();
}

mResult mQuaternion::GetAverageEst(IN const mQuaternion *pValues, const size_t count, OUT mQuaternion *pAverage)
{
  if (pValues == nullptr || pAverage == nullptr)
    return mR_ArgumentNull;

  mVector avg(0, 0, 0, 0);

  for (size_t i = 0; i < count; i++)
  {
    const mQuaternion &q = pValues[i];

    if (i > 0 && q.Dot(pValues[0]) < 0.f)
      avg += -mVector(q.q);
    else
      avg += mVector(q.q);
  }

  *pAverage = mQuaternion(avg.v).Normalize();

  return mR_Success;
}

mVec3f mQuaternion::ToEulerAngles() const
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

//////////////////////////////////////////////////////////////////////////

mResult mMatrix::Decompose(OUT mVector *pOutScale, OUT mQuaternion *pOutRotQuat, OUT mVector *pOutTrans) const
{
  mFUNCTION_SETUP();

  mERROR_IF(pOutScale == nullptr || pOutRotQuat == nullptr || pOutTrans == nullptr, mR_ArgumentNull);
  mERROR_IF(DirectX::XMMatrixDecompose(&pOutScale->v, &pOutRotQuat->q, &pOutTrans->v, m), mR_InternalError);

  mRETURN_SUCCESS();
}

mMatrix mMatrix::AddComponentWise(const mMatrix &a, const mMatrix &b)
{
  mMatrix ret;

  for (size_t i = 0; i < 4; i++)
    for (size_t j = 0; j < 4; j++)
      ret._m[i][j] = a._m[i][j] + b._m[i][j];

  return ret;
}

mMatrix mMatrix::SubtractComponentWise(const mMatrix &a, const mMatrix &b)
{
  mMatrix ret;

  for (size_t i = 0; i < 4; i++)
    for (size_t j = 0; j < 4; j++)
      ret._m[i][j] = a._m[i][j] - b._m[i][j];

  return ret;
}

mMatrix mMatrix::MultiplyComponentWise(const mMatrix &a, const mMatrix &b)
{
  mMatrix ret;

  for (size_t i = 0; i < 4; i++)
    for (size_t j = 0; j < 4; j++)
      ret._m[i][j] = a._m[i][j] * b._m[i][j];

  return ret;
}

mMatrix mMatrix::DivideComponentWise(const mMatrix &a, const mMatrix &b)
{
  mMatrix ret;

  for (size_t i = 0; i < 4; i++)
    for (size_t j = 0; j < 4; j++)
      ret._m[i][j] = a._m[i][j] / b._m[i][j];

  return ret;
}
