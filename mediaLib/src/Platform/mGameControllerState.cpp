#include "mGameControllerState.h"

#define DECLSPEC
#include "SDL.h"
#undef DECLSPEC

#ifdef GIT_BUILD // Define __M_FILE__
  #ifdef __M_FILE__
    #undef __M_FILE__
  #endif
  #define __M_FILE__ "vDI77C627jrHPoo/Kdm2w+LJduW5TqcTWeaxCe8V3LJCITq72hOoVf49pq4EEdcYP2kzgr9KfD5nCjvo"
#endif

constexpr size_t mGameControllerState_AnalogStickTriggerCount = 2;
constexpr size_t mGameControllerState_AnalogTriggerCount = 2;
constexpr size_t mGameControllerState_DPadCount = 1;
constexpr size_t mGameControllerState_ButtonCount = 16;

struct mGameControllerState
{
  SDL_Joystick *pJoystickHandle;
  mVec2f currentAnalogStick[mGameControllerState_AnalogStickTriggerCount];
  mVec2f lastAnalogStick[mGameControllerState_AnalogStickTriggerCount];
  float_t currentTrigger[mGameControllerState_AnalogStickTriggerCount];
  float_t lastTrigger[mGameControllerState_AnalogStickTriggerCount];
  mVec2i currentDPad[mGameControllerState_DPadCount];
  mVec2i lastDPad[mGameControllerState_DPadCount];
  bool dpadModifiesAnalogInput;
  bool isWired, isDisconnected;
  float_t batteryLevel;
  size_t ballCount, buttonCount, hatCount, axisCount;
  bool lastButtons[mGameControllerState_ButtonCount];
  bool currentButtons[mGameControllerState_ButtonCount];
  SDL_Haptic *pHapticHandle;
  bool allowRumble;
};

//////////////////////////////////////////////////////////////////////////

static mFUNCTION(mGameControllerState_Destroy_Internal, IN_OUT mGameControllerState *pControllerState);

//////////////////////////////////////////////////////////////////////////

mFUNCTION(mGameControllerState_GetGameControllerCount, OUT size_t *pCount)
{
  mFUNCTION_SETUP();

  mERROR_IF(pCount == nullptr, mR_ArgumentNull);

  const int32_t count = SDL_NumJoysticks();
  mERROR_IF(count < 0, mR_InternalError);

  *pCount = (size_t)count;

  mRETURN_SUCCESS();
}

//////////////////////////////////////////////////////////////////////////

mFUNCTION(mGameControllerState_Create, OUT mPtr<mGameControllerState> *pControllerState, IN mAllocator *pAllocator, const size_t index)
{
  mFUNCTION_SETUP();

  mERROR_IF(pControllerState == nullptr, mR_ArgumentNull);
  mERROR_IF(index > INT32_MAX, mR_ArgumentOutOfBounds);

  size_t controllerCount = 0;
  mERROR_CHECK(mGameControllerState_GetGameControllerCount(&controllerCount));
  mERROR_IF(controllerCount == 0, mR_ResourceNotFound);
  mERROR_IF(controllerCount <= index, mR_IndexOutOfBounds);

  SDL_Joystick *pJoystick = SDL_JoystickOpen((int32_t)index);
  mERROR_IF(pJoystick == nullptr, mR_ResourceNotFound);
  mDEFER_CALL_ON_ERROR(pJoystick, SDL_JoystickClose);
  mERROR_IF(SDL_FALSE == SDL_JoystickGetAttached(pJoystick), mR_ResourceNotFound);

  SDL_Haptic *pHaptic = SDL_HapticOpenFromJoystick(pJoystick);
  mDEFER_CALL_ON_ERROR(pHaptic, SDL_HapticClose);

  mDEFER_CALL_ON_ERROR(pControllerState, mSharedPointer_Destroy);
  mERROR_CHECK((mSharedPointer_Allocate<mGameControllerState>(pControllerState, pAllocator, [](mGameControllerState *pData) {mGameControllerState_Destroy_Internal(pData);}, 1)));

  mGameControllerState *pState = pControllerState->GetPointer();

  pState->pJoystickHandle = pJoystick;
  pState->pHapticHandle = pHaptic;

  pState->axisCount = mMax(0, SDL_JoystickNumAxes(pJoystick));
  pState->ballCount = mMax(0, SDL_JoystickNumBalls(pJoystick));
  pState->buttonCount = mMax(0, SDL_JoystickNumButtons(pJoystick));
  pState->hatCount = mMax(0, SDL_JoystickNumHats(pJoystick));

  pState->allowRumble = (pHaptic != nullptr && SDL_HapticRumbleInit(pHaptic) == 0);

  // Update twice to populate both current and last values.
  mERROR_CHECK(mGameControllerState_Update(*pControllerState));
  mERROR_CHECK(mGameControllerState_Update(*pControllerState));

  mRETURN_SUCCESS();
}

mFUNCTION(mGameControllerState_Destroy, IN_OUT mPtr<mGameControllerState> *pControllerState)
{
  return mSharedPointer_Destroy(pControllerState);
}

//////////////////////////////////////////////////////////////////////////

mFUNCTION(mGameControllerState_Update, mPtr<mGameControllerState> &controllerState)
{
  mFUNCTION_SETUP();

  mERROR_IF(controllerState == nullptr, mR_ArgumentNull);

  controllerState->isDisconnected = SDL_FALSE == SDL_JoystickGetAttached(controllerState->pJoystickHandle);

  if (controllerState->isDisconnected)
  {
    mZeroMemory(controllerState->lastAnalogStick, mARRAYSIZE(controllerState->lastAnalogStick));
    mZeroMemory(controllerState->currentAnalogStick, mARRAYSIZE(controllerState->currentAnalogStick));
    mZeroMemory(controllerState->lastTrigger, mARRAYSIZE(controllerState->lastTrigger));
    mZeroMemory(controllerState->currentTrigger, mARRAYSIZE(controllerState->currentTrigger));
    mZeroMemory(controllerState->lastDPad, mARRAYSIZE(controllerState->lastDPad));
    mZeroMemory(controllerState->currentDPad, mARRAYSIZE(controllerState->currentDPad));
    mZeroMemory(controllerState->lastButtons, mARRAYSIZE(controllerState->lastButtons));
    mZeroMemory(controllerState->currentButtons, mARRAYSIZE(controllerState->currentButtons));

    mRETURN_SUCCESS();
  }

  SDL_JoystickUpdate();

  const SDL_JoystickPowerLevel level = SDL_JoystickCurrentPowerLevel(controllerState->pJoystickHandle);

  switch (level)
  {
  default:
  case SDL_JOYSTICK_POWER_UNKNOWN:
  case SDL_JOYSTICK_POWER_WIRED:
    controllerState->isWired = true;
    controllerState->batteryLevel = 1.f;
    break;

  case SDL_JOYSTICK_POWER_MAX:
  case SDL_JOYSTICK_POWER_FULL: // <= 100%
    controllerState->batteryLevel = 1.f;
      break;

  case SDL_JOYSTICK_POWER_MEDIUM: // <= 70%
    controllerState->batteryLevel = .7f;
    break;

  case SDL_JOYSTICK_POWER_LOW: // <= 20%
    controllerState->batteryLevel = .2f;
    break;

  case SDL_JOYSTICK_POWER_EMPTY: // <= 5%
    controllerState->batteryLevel = .05f;
    break;
  }

  const size_t maxAxis = mMin(controllerState->axisCount, mARRAYSIZE(controllerState->currentAnalogStick) * 3);

  for (size_t i = 0; i < maxAxis; i += 3)
  {
    controllerState->lastAnalogStick[i / 3] = controllerState->currentAnalogStick[i / 3];
    controllerState->lastTrigger[i / 3] = controllerState->currentTrigger[i / 3];

    if (i + 1 < maxAxis)
      controllerState->currentAnalogStick[i / 3] = mVec2f(SDL_JoystickGetAxis(controllerState->pJoystickHandle, (int32_t)i) / (float_t)INT16_MAX, SDL_JoystickGetAxis(controllerState->pJoystickHandle, (int32_t)i + 1) / 32768.f);
    else
      controllerState->currentAnalogStick[i / 3] = mVec2f(SDL_JoystickGetAxis(controllerState->pJoystickHandle, (int32_t)i) / (float_t)INT16_MAX, 0);

    if (i + 2 < maxAxis)
      controllerState->currentTrigger[i / 3] = (SDL_JoystickGetAxis(controllerState->pJoystickHandle, (int32_t)i + 2) + (float_t)INT16_MIN) / (float_t)UINT16_MAX;
  }

  mMemcpy(controllerState->lastButtons, controllerState->currentButtons, mARRAYSIZE(controllerState->lastButtons));

  const size_t maxButton = mMin(controllerState->buttonCount, mARRAYSIZE(controllerState->currentButtons));

  for (size_t i = 0; i < maxButton; i++)
    controllerState->currentButtons[i] = (SDL_FALSE != SDL_JoystickGetButton(controllerState->pJoystickHandle, (int32_t)i));

  const size_t maxDPad = mMin(controllerState->hatCount, mARRAYSIZE(controllerState->currentAnalogStick));

  for (size_t i = 0; i < maxDPad; i += 2)
  {
    controllerState->lastDPad[i / 2] = controllerState->currentDPad[i / 2];

    const uint8_t hatValue = SDL_JoystickGetHat(controllerState->pJoystickHandle, (int32_t)i);

    controllerState->currentDPad[i / 2] = mVec2i(((hatValue & SDL_HAT_RIGHT) != 0) - ((hatValue & SDL_HAT_LEFT) != 0), ((hatValue & SDL_HAT_DOWN) != 0) - ((hatValue & SDL_HAT_UP) != 0));
  }

  mRETURN_SUCCESS();
}

mFUNCTION(mGameControllerState_IsDisconnected, mPtr<mGameControllerState> &controllerState, OUT bool *pIsDisconnected)
{
  mFUNCTION_SETUP();

  mERROR_IF(controllerState == nullptr || pIsDisconnected == nullptr, mR_ArgumentNull);

  *pIsDisconnected = controllerState->isDisconnected;

  mRETURN_SUCCESS();
}

mFUNCTION(mGameControllerState_IsWired, mPtr<mGameControllerState> &controllerState, OUT bool *pIsWired)
{
  mFUNCTION_SETUP();

  mERROR_IF(controllerState == nullptr || pIsWired == nullptr, mR_ArgumentNull);

  *pIsWired = controllerState->isWired;

  mRETURN_SUCCESS();
}

mFUNCTION(mGameControllerState_GetApproximateBatteryLevel, mPtr<mGameControllerState> &controllerState, OUT float_t *pBatteryLevel)
{
  mFUNCTION_SETUP();

  mERROR_IF(controllerState == nullptr || pBatteryLevel == nullptr, mR_ArgumentNull);

  *pBatteryLevel = controllerState->batteryLevel;

  mRETURN_SUCCESS();
}

mFUNCTION(mGameControllerState_GetDescription, mPtr<mGameControllerState> &controllerState, OUT mString *pDescription)
{
  mFUNCTION_SETUP();

  mERROR_IF(controllerState == nullptr, mR_ArgumentNull);
  mERROR_IF(pDescription == nullptr, mR_ArgumentNull);

  const char *name = SDL_JoystickName(controllerState->pJoystickHandle);

  if (name == nullptr)
    mERROR_CHECK(mString_Create(pDescription, "<UNKNOWN>", pDescription->pAllocator));
  else
    mERROR_CHECK(mString_Create(pDescription, name, pDescription->pAllocator));

  mRETURN_SUCCESS();
}

mFUNCTION(mGameControllerState_GetAnalogStick, mPtr<mGameControllerState> &controllerState, const size_t index, OUT mVec2f *pValue)
{
  mFUNCTION_SETUP();

  mERROR_IF(controllerState == nullptr || pValue == nullptr, mR_ArgumentNull);
  mERROR_IF(index >= mARRAYSIZE(controllerState->currentAnalogStick) || controllerState->axisCount / 3 + 1 <= index, mR_IndexOutOfBounds);

  *pValue = controllerState->currentAnalogStick[index];

  mRETURN_SUCCESS();
}

mFUNCTION(mGameControllerState_GetAnalogTrigger, mPtr<mGameControllerState> &controllerState, const size_t index, OUT float_t *pValue)
{
  mFUNCTION_SETUP();

  mERROR_IF(controllerState == nullptr || pValue == nullptr, mR_ArgumentNull);
  mERROR_IF(index >= mARRAYSIZE(controllerState->currentTrigger) || controllerState->axisCount / 3 + 1 <= index, mR_IndexOutOfBounds);

  *pValue = controllerState->currentTrigger[index];

  mRETURN_SUCCESS();
}

mFUNCTION(mGameControllerState_GetDigitalInput, mPtr<mGameControllerState> &controllerState, const size_t index, OUT mVec2i *pValue)
{
  mFUNCTION_SETUP();

  mERROR_IF(controllerState == nullptr || pValue == nullptr, mR_ArgumentNull);
  mERROR_IF(index >= mARRAYSIZE(controllerState->currentDPad) || controllerState->hatCount <= index, mR_IndexOutOfBounds);

  *pValue = controllerState->currentDPad[index];

  mRETURN_SUCCESS();
}

mFUNCTION(mGameControllerState_GetButton, mPtr<mGameControllerState> &controllerState, const mGameControllerState_Button index, OUT bool *pValue)
{
  mFUNCTION_SETUP();

  mERROR_IF(controllerState == nullptr || pValue == nullptr, mR_ArgumentNull);
  mERROR_IF(index >= mARRAYSIZE(controllerState->currentButtons) || index >= controllerState->buttonCount, mR_IndexOutOfBounds);

  *pValue = controllerState->currentButtons[index];

  mRETURN_SUCCESS();
}

mFUNCTION(mGameControllerState_GetLastButtonState, mPtr<mGameControllerState> &controllerState, const mGameControllerState_Button index, OUT bool *pValue)
{
  mFUNCTION_SETUP();

  mERROR_IF(controllerState == nullptr || pValue == nullptr, mR_ArgumentNull);
  mERROR_IF(index >= mARRAYSIZE(controllerState->currentButtons) || index >= controllerState->buttonCount, mR_IndexOutOfBounds);

  *pValue = controllerState->lastButtons[index];

  mRETURN_SUCCESS();
}

mFUNCTION(mGameControllerState_GetButtonPressed, mPtr<mGameControllerState> &controllerState, const mGameControllerState_Button index, OUT bool *pValue)
{
  mFUNCTION_SETUP();

  mERROR_IF(controllerState == nullptr || pValue == nullptr, mR_ArgumentNull);
  mERROR_IF(index >= mARRAYSIZE(controllerState->currentButtons) || index >= controllerState->buttonCount, mR_IndexOutOfBounds);

  *pValue = controllerState->currentButtons[index] && !controllerState->lastButtons[index];

  mRETURN_SUCCESS();
}

mFUNCTION(mGameControllerState_GetButtonReleased, mPtr<mGameControllerState> &controllerState, const mGameControllerState_Button index, OUT bool *pValue)
{
  mFUNCTION_SETUP();

  mERROR_IF(controllerState == nullptr || pValue == nullptr, mR_ArgumentNull);
  mERROR_IF(index >= mARRAYSIZE(controllerState->currentButtons) || index >= controllerState->buttonCount, mR_IndexOutOfBounds);

  *pValue = !controllerState->currentButtons[index] && controllerState->lastButtons[index];

  mRETURN_SUCCESS();
}

mFUNCTION(mGameControllerState_Rumble, mPtr<mGameControllerState> &controllerState, const size_t durationMs, const float_t strength)
{
  mFUNCTION_SETUP();

  mERROR_IF(controllerState == nullptr, mR_ArgumentNull);
  mERROR_IF(controllerState->pHapticHandle == nullptr || !controllerState->allowRumble, mR_NotSupported);

  mERROR_IF(0 != SDL_HapticRumblePlay(controllerState->pHapticHandle, strength, (uint32_t)mMin((size_t)UINT32_MAX, durationMs)), mR_InternalError);

  mRETURN_SUCCESS();
}

mFUNCTION(mGameControllerState_StopRumble, mPtr<mGameControllerState> &controllerState)
{
  mFUNCTION_SETUP();

  mERROR_IF(controllerState == nullptr, mR_ArgumentNull);
  mERROR_IF(controllerState->pHapticHandle == nullptr || !controllerState->allowRumble, mR_NotSupported);

  mERROR_IF(0 != SDL_HapticRumbleStop(controllerState->pHapticHandle), mR_InternalError);

  mRETURN_SUCCESS();
}

mFUNCTION(mGameControllerState_PlayHapticEffect, mPtr<mGameControllerState> &controllerState, const mGameControllerState_ConstantHapticEffect &effect)
{
  mFUNCTION_SETUP();

  mERROR_IF(controllerState == nullptr, mR_ArgumentNull);
  mERROR_IF(controllerState->pHapticHandle == nullptr, mR_NotSupported);

  SDL_HapticConstant effectDescription;
  mZeroMemory(&effectDescription, 1);

  effectDescription.type = SDL_HAPTIC_CONSTANT;
  effectDescription.direction.type = SDL_HAPTIC_CARTESIAN;
  effectDescription.direction.dir[1] = -1; // Coming from in front of the user. This is somewhat arbitrary.
  effectDescription.length = (uint32_t)mMin((size_t)0x7FFF, effect.durationMs); // See https://wiki.libsdl.org/SDL_HapticEffect.
  effectDescription.level = (int16_t)(effect.strength * INT16_MAX);
  effectDescription.attack_level = (int16_t)(effect.fadeInStartStrength * INT16_MAX);
  effectDescription.fade_level = (int16_t)(effect.fadeOutEndStrength * INT16_MAX);
  effectDescription.attack_length = (int16_t)mMin((size_t)0x7FFF, effect.fadeInTimeMs); // See https://wiki.libsdl.org/SDL_HapticEffect.
  effectDescription.fade_length = (int16_t)mMin((size_t)0x7FFF, effect.fadeOutTimeMs); // See https://wiki.libsdl.org/SDL_HapticEffect.

  const int32_t effectId = SDL_HapticNewEffect(controllerState->pHapticHandle, reinterpret_cast<SDL_HapticEffect *>(&effectDescription));
  mERROR_IF(effectId < 0, mR_InternalError);

  mERROR_IF(SDL_HapticRunEffect(controllerState->pHapticHandle, effectId, 1) != 0, mR_InternalError);

  mRETURN_SUCCESS();
}

//////////////////////////////////////////////////////////////////////////

static mFUNCTION(mGameControllerState_Destroy_Internal, IN_OUT mGameControllerState *pControllerState)
{
  mFUNCTION_SETUP();

  mERROR_IF(pControllerState == nullptr, mR_ArgumentNull);

  if (pControllerState->pJoystickHandle != nullptr)
    SDL_JoystickClose(pControllerState->pJoystickHandle);

  if (pControllerState->pHapticHandle)
    SDL_HapticClose(pControllerState->pHapticHandle);

  mRETURN_SUCCESS();
}
