// Copyright 2018 Christoph Stiller
// 
// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files(the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions :
// 
// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
// 
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "mSemaphore.h"
#include <windows.h>
#include <IntSafe.h>

struct mSemaphore
{
  CRITICAL_SECTION criticalSection;
  CONDITION_VARIABLE conditionVariable;
  mAllocator *pAllocator;
};

mFUNCTION(mSemaphore_Destroy_Internal, IN mSemaphore *pSemaphore);

mFUNCTION(mSemaphore_Create, OUT mSemaphore **ppSemaphore, IN OPTIONAL mAllocator *pAllocator)
{
  mFUNCTION_SETUP();

  mERROR_IF(ppSemaphore == nullptr, mR_ArgumentNull);

  mDEFER_DESTRUCTION_ON_ERROR(ppSemaphore, mSetToNullptr);
  mDEFER_ON_ERROR(mAllocator_FreePtr(pAllocator, ppSemaphore));
  mERROR_CHECK(mAllocator_AllocateZero(pAllocator, ppSemaphore, 1));

  (*ppSemaphore)->pAllocator = pAllocator;

  InitializeCriticalSection(&(*ppSemaphore)->criticalSection);
  mDEFER_ON_ERROR(DeleteCriticalSection(&(*ppSemaphore)->criticalSection));

  InitializeConditionVariable(&(*ppSemaphore)->conditionVariable);

  mRETURN_SUCCESS();
}

mFUNCTION(mSemaphore_Destroy, IN_OUT mSemaphore **ppSemaphore)
{
  mFUNCTION_SETUP();

  mERROR_IF(ppSemaphore == nullptr, mR_ArgumentNull);

  mDEFER_DESTRUCTION(ppSemaphore, mSetToNullptr);
  mERROR_CHECK(mSemaphore_Destroy_Internal(*ppSemaphore));

  mAllocator *pAllocator = (*ppSemaphore)->pAllocator;
  mERROR_CHECK(mAllocator_FreePtr(pAllocator, ppSemaphore));

  mRETURN_SUCCESS();
}

mFUNCTION(mSemaphore_Lock, IN mSemaphore *pSemaphore)
{
  mFUNCTION_SETUP();

  mERROR_IF(pSemaphore == nullptr, mR_ArgumentNull);

  EnterCriticalSection(&pSemaphore->criticalSection);

  mRETURN_SUCCESS();
}

mFUNCTION(mSemaphore_Unlock, IN mSemaphore *pSemaphore)
{
  mFUNCTION_SETUP();

  mERROR_IF(pSemaphore == nullptr, mR_ArgumentNull);

  LeaveCriticalSection(&pSemaphore->criticalSection);

  mRETURN_SUCCESS();
}

mFUNCTION(mSemaphore_WakeOne, IN mSemaphore *pSemaphore)
{
  mFUNCTION_SETUP();

  mERROR_IF(pSemaphore == nullptr, mR_ArgumentNull);

  WakeConditionVariable(&pSemaphore->conditionVariable);

  mRETURN_SUCCESS();
}

mFUNCTION(mSemaphore_WakeAll, IN mSemaphore *pSemaphore)
{
  mFUNCTION_SETUP();

  mERROR_IF(pSemaphore == nullptr, mR_ArgumentNull);

  WakeAllConditionVariable(&pSemaphore->conditionVariable);

  mRETURN_SUCCESS();
}

mFUNCTION(mSemaphore_Sleep, IN mSemaphore *pSemaphore, const size_t timeoutMilliseconds /* = mSemaphore_SleepTime::mS_ST_Infinite */)
{
  mFUNCTION_SETUP();

  mERROR_IF(pSemaphore == nullptr, mR_ArgumentNull);
  mERROR_IF(timeoutMilliseconds > DWORD_MAX, mR_IndexOutOfBounds);

  const BOOL result = SleepConditionVariableCS(&pSemaphore->conditionVariable, &pSemaphore->criticalSection, (DWORD)timeoutMilliseconds);

  switch (result)
  {
  case 0:
    mRETURN_SUCCESS();

  case ERROR_TIMEOUT:
    mRETURN_RESULT(mR_Timeout);

  default:
    mRETURN_RESULT(mR_InternalError);
  }
}

mFUNCTION(mSemaphore_Destroy_Internal, IN mSemaphore *pSemaphore)
{
  mFUNCTION_SETUP();

  mERROR_IF(pSemaphore == nullptr, mR_ArgumentNull);

  // A `CONDITION_VARIABLE` does not have a destroy function. (https://stackoverflow.com/a/28981408)
  DeleteCriticalSection(&pSemaphore->criticalSection);

  mRETURN_SUCCESS();
}
