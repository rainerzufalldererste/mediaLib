#ifndef mGameControllerState_h__
#define mGameControllerState_h__

#include "mediaLib.h"

#ifdef GIT_BUILD // Define __M_FILE__
  #ifdef __M_FILE__
    #undef __M_FILE__
  #endif
  #define __M_FILE__ "FgmgEU0Kv7tDkX5DiiDzep+9gLTx58zdCpAHCeVLCx/ynmm79YlINdxsHlRItNW98a8+xMbVPuzkZyVj"
#endif

struct mGameControllerState;

mFUNCTION(mGameControllerState_GetGameControllerCount, OUT size_t *pCount);

mFUNCTION(mGameControllerState_Create, OUT mPtr<mGameControllerState> *pControllerState, IN mAllocator *pAllocator, const size_t index);
mFUNCTION(mGameControllerState_Destroy, IN_OUT mPtr<mGameControllerState> *pControllerState);

mFUNCTION(mGameControllerState_Update, mPtr<mGameControllerState> &controllerState);

mFUNCTION(mGameControllerState_IsDisconnected, mPtr<mGameControllerState> &controllerState, OUT bool *pIsDisconnected);
mFUNCTION(mGameControllerState_IsWired, mPtr<mGameControllerState> &controllerState, OUT bool *pIsWired);
mFUNCTION(mGameControllerState_GetApproximateBatteryLevel, mPtr<mGameControllerState> &controllerState, OUT float_t *pBatteryLevel);

mFUNCTION(mGameControllerState_GetDescription, mPtr<mGameControllerState> &controllerState, OUT mString *pDescription);

mFUNCTION(mGameControllerState_GetAnalogStick, mPtr<mGameControllerState> &controllerState, const size_t index, OUT mVec2f *pValue);
mFUNCTION(mGameControllerState_GetAnalogTrigger, mPtr<mGameControllerState> &controllerState, const size_t index, OUT float_t *pValue);
mFUNCTION(mGameControllerState_GetDigitalInput, mPtr<mGameControllerState> &controllerState, const size_t index, OUT mVec2i *pValue);

enum mGameControllerState_Button
{
  mGCS_B_A,
  mGCS_B_B,
  mGCS_B_X,
  mGCS_B_Y,
  mGCS_B_LeftTrigger,
  mGCS_B_RightTrigger,
  mGCS_B_Back,
  mGCS_B_Start,
  mGCS_B_LeftAnalogStick,
  mGCS_B_RightAnalogStick,
};

mFUNCTION(mGameControllerState_GetButton, mPtr<mGameControllerState> &controllerState, const mGameControllerState_Button index, OUT bool *pValue);
mFUNCTION(mGameControllerState_GetLastButtonState, mPtr<mGameControllerState> &controllerState, const mGameControllerState_Button index, OUT bool *pValue);
mFUNCTION(mGameControllerState_GetButtonPressed, mPtr<mGameControllerState> &controllerState, const mGameControllerState_Button index, OUT bool *pValue);
mFUNCTION(mGameControllerState_GetButtonReleased, mPtr<mGameControllerState> &controllerState, const mGameControllerState_Button index, OUT bool *pValue);

mFUNCTION(mGameControllerState_Rumble, mPtr<mGameControllerState> &controllerState, const size_t durationMs, const float_t strength);
mFUNCTION(mGameControllerState_StopRumble, mPtr<mGameControllerState> &controllerState);

struct mGameControllerState_ConstantHapticEffect
{
  float_t strength = 1.f, fadeInStartStrength = 0.f, fadeOutEndStrength = 0.f;
  size_t durationMs = 125, fadeInTimeMs = 125, fadeOutTimeMs = 125;
};

mFUNCTION(mGameControllerState_PlayHapticEffect, mPtr<mGameControllerState> &controllerState, const mGameControllerState_ConstantHapticEffect &effect);

#endif // mGameControllerState_h__
