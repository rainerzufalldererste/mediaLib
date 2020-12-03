#include "mCamera3D.h"

#ifdef GIT_BUILD // Define __M_FILE__
  #ifdef __M_FILE__
    #undef __M_FILE__
  #endif
  #define __M_FILE__ "ByyqEw/J6o3CfA41rnZdMxFit9m3xJ4SMWwxX9UnY/RgKtpDzAstQODyIN7UCRpxaHERO3pZ6zHQ5I+r"
#endif

mFUNCTION(mCamera3D_CreateWithDirection, OUT mCamera3D *pCamera, const mVec3f position, const mVec3f lookDirection, const float_t verticalFOV, const float_t aspectRatio, const float_t nearPlane, const float_t farPlane, const mVec3f up /* = mVec3f(0, 0, 1) */)
{
  mFUNCTION_SETUP();

  mERROR_IF(pCamera == nullptr, mR_ArgumentNull);

  pCamera->viewMatrix = mMatrix::LookToRH((mVector)position, (mVector)lookDirection, (mVector)up);
  pCamera->nearPlane = nearPlane;
  pCamera->farPlane = farPlane;
  pCamera->vFov = verticalFOV;
  pCamera->aspectRatio = aspectRatio;
  pCamera->projectionMatrix = mMatrix::PerspectiveFovRH(verticalFOV, aspectRatio, nearPlane, farPlane);
  pCamera->viewProjectionMatrix = pCamera->viewMatrix * pCamera->projectionMatrix;

  mRETURN_SUCCESS();
}

mFUNCTION(mCamera3D_CreateWithLookAt, OUT mCamera3D *pCamera, const mVec3f position, const mVec3f lookAt, const float_t verticalFOV, const float_t aspectRatio, const float_t nearPlane, const float_t farPlane, const mVec3f up /* = mVec3f(0, 0, 1) */)
{
  return mCamera3D_CreateWithDirection(pCamera, position, lookAt - position, verticalFOV, aspectRatio, nearPlane, farPlane, up);
}

void mCamera3D_Rotate(mCamera3D &camera, const mVec3f angles)
{
  camera.viewMatrix *= mMatrix::RotationRollPitchYawFromVector(angles.yxz());
}

void mCamera3D_Rotate(mCamera3D &camera, const mQuaternion quaternion)
{
  camera.viewMatrix *= mMatrix::RotationQuaternion(quaternion);
}

void mCamera3D_Move(mCamera3D &camera, const mVec3f movement)
{
  camera.viewMatrix *= mMatrix::TranslationFromVector(movement.xzy() * -1.f);
}

void mCamera3D_SetLookFromAt(mCamera3D &camera, const mVec3f position, const mVec3f lookAt, const mVec3f up /* = mVec3f(0, 0, 1) */)
{
  camera.viewMatrix = mMatrix::LookAtRH((mVector)position, (mVector)lookAt, (mVector)up);
}

void mCamera3D_SetLookFromTo(mCamera3D &camera, const mVec3f position, const mVec3f direction, const mVec3f up /* = mVec3f(0, 0, 1) */)
{
  camera.viewMatrix = mMatrix::LookToRH((mVector)position, (mVector)direction, (mVector)up);
}

void mCamera3D_Finalize(mCamera3D &camera)
{
  camera.viewProjectionMatrix = camera.viewMatrix * camera.projectionMatrix;
}
