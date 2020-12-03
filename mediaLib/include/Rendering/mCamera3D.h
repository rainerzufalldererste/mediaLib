#ifndef mCamera3D_h__
#define mCamera3D_h__

#include "mediaLib.h"

#ifdef GIT_BUILD // Define __M_FILE__
  #ifdef __M_FILE__
    #undef __M_FILE__
  #endif
  #define __M_FILE__ "aflhek23nNRfhAbyPuVk5QS3cSy9JDPtYXzJJoYlSVh0DEr+v5L4q/Hucl8kmj8vu9jCnOTHbdoMycRx"
#endif

struct mCamera3D
{
  float_t aspectRatio, vFov, nearPlane, farPlane;

  mMatrix viewMatrix, projectionMatrix, viewProjectionMatrix;
};

mFUNCTION(mCamera3D_CreateWithDirection, OUT mCamera3D *pCamera, const mVec3f position, const mVec3f lookDirection, const float_t verticalFOV, const float_t aspectRatio, const float_t nearPlane, const float_t farPlane, const mVec3f up = mVec3f(0, 0, 1));
mFUNCTION(mCamera3D_CreateWithLookAt, OUT mCamera3D *pCamera, const mVec3f position, const mVec3f lookAt, const float_t verticalFOV, const float_t aspectRatio, const float_t nearPlane, const float_t farPlane, const mVec3f up = mVec3f(0, 0, 1));

void mCamera3D_Rotate(mCamera3D &camera, const mVec3f angles);
void mCamera3D_Rotate(mCamera3D &camera, const mQuaternion quaternion);

void mCamera3D_Move(mCamera3D &camera, const mVec3f movement);

void mCamera3D_SetLookFromAt(mCamera3D &camera, const mVec3f position, const mVec3f lookAt, const mVec3f up = mVec3f(0, 0, 1));
void mCamera3D_SetLookFromTo(mCamera3D &camera, const mVec3f position, const mVec3f direction, const mVec3f up = mVec3f(0, 0, 1));

void mCamera3D_Finalize(mCamera3D &camera);

#endif // mCamera3D_h__
