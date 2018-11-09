#include "mKeyboardState.h"

#include "SDL.h"

mFUNCTION(mKeyboardState_Update, OUT mKeyboardState *pKeyboardState)
{
  mFUNCTION_SETUP();

  mERROR_IF(pKeyboardState == nullptr, mR_ArgumentNull);

  if (pKeyboardState->pKeys != nullptr)
  {
    mERROR_CHECK(mMemcpy(pKeyboardState->lastKeys, pKeyboardState->currentKeys, pKeyboardState->keyCount));
    pKeyboardState->hasPreviousState = true;
  }

  int keyCount = 0;
  pKeyboardState->pKeys = SDL_GetKeyboardState(&keyCount);

  pKeyboardState->keyCount = (size_t)keyCount;

  mERROR_CHECK(mMemcpy(pKeyboardState->currentKeys, pKeyboardState->pKeys, pKeyboardState->keyCount));

  mRETURN_SUCCESS();
}

bool mKeyboardState_IsKeyDown(const mKeyboardState &keyboardState, const mKey key)
{
  if (keyboardState.pKeys == nullptr || key >= keyboardState.keyCount || key >= mKey_Count || key <= mK_UNKNOWN)
    return false;

  return keyboardState.pKeys[key] == SDL_TRUE;
}

bool mKeyboardState_IsKeyUp(const mKeyboardState &keyboardState, const mKey key)
{
  if (keyboardState.pKeys == nullptr || key >= keyboardState.keyCount || key >= mKey_Count || key <= mK_UNKNOWN)
    return false;

  return keyboardState.pKeys[key] == SDL_FALSE;
}

bool mKeyboardState_KeyPress(const mKeyboardState &keyboardState, const mKey key)
{
  if (!keyboardState.hasPreviousState || key >= keyboardState.keyCount || key >= mKey_Count || key <= mK_UNKNOWN)
    return false;

  return keyboardState.currentKeys[key] == SDL_TRUE && keyboardState.lastKeys[key] == SDL_FALSE;
}

bool mKeyboardState_KeyLift(const mKeyboardState &keyboardState, const mKey key)
{
  if (!keyboardState.hasPreviousState || key >= keyboardState.keyCount || key >= mKey_Count || key <= mK_UNKNOWN)
    return false;

  return keyboardState.currentKeys[key] == SDL_FALSE && keyboardState.lastKeys[key] == SDL_TRUE;
}
