// Copyright 2018 Christoph Stiller
// 
// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files(the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions :
// 
// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
// 
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "mThreadPool.h"

struct mTask
{
  std::function<mResult(void)> function;
  std::atomic<mTask_State> state;
  mSemaphore *pSemaphore;
  mResult result;
};

mFUNCTION(mTask_Destroy_Internal, IN mTask *pTask);

mFUNCTION(mTask_Create, OUT mTask **ppTask, const std::function<mResult(void)> &function)
{
  mFUNCTION_SETUP();

  mERROR_IF(ppTask == nullptr, mR_ArgumentNull);
  mERROR_CHECK(mAllocZero(ppTask, 1));
  mERROR_CHECK(mTask_CreateInplace(*ppTask, function));

  mRETURN_SUCCESS();
}

mFUNCTION(mTask_CreateInplace, IN mTask *pTask, const std::function<mResult(void)> &function)
{
  mFUNCTION_SETUP();

  mERROR_IF(pTask == nullptr, mR_ArgumentNull);

  pTask->result = mR_Success;
  new (&pTask->function) std::function<mResult(void)>(function);
  new (&pTask->state) std::atomic<mTask_State>(mT_S_Initialized);

  mERROR_CHECK(mSemaphore_Create(&pTask->pSemaphore));

  mRETURN_SUCCESS();
}

mFUNCTION(mTask_Destroy, IN_OUT mTask **ppTask)
{
  mFUNCTION_SETUP();

  mERROR_IF(ppTask == nullptr, mR_ArgumentNull);

  mERROR_CHECK(mTask_Destroy_Internal(*ppTask));

  mRETURN_SUCCESS();
}

mFUNCTION(mTask_Join, IN mTask *pTask, const size_t timeoutMilliseconds /* = mSemaphore_SleepTime::mS_ST_Infinite */)
{
  mFUNCTION_SETUP();

  mERROR_IF(pTask == nullptr, mR_ArgumentNull);

  mRETURN_SUCCESS();
}

mFUNCTION(mTask_Execute, IN mTask *pTask)
{
  mFUNCTION_SETUP();

  mERROR_IF(pTask == nullptr, mR_ArgumentNull);

  mRETURN_SUCCESS();
}

mFUNCTION(mTask_Abort, IN mTask *pTask)
{
  mFUNCTION_SETUP();

  mERROR_IF(pTask == nullptr, mR_ArgumentNull);

  mRETURN_SUCCESS();
}

mFUNCTION(mTask_GetResult, IN mTask * pTask, OUT mResult * pResult)
{
  mFUNCTION_SETUP();

  mERROR_IF(pTask == nullptr, mR_ArgumentNull);

  mRETURN_SUCCESS();
}

mFUNCTION(mTask_GetState, IN mTask *pTask, OUT mTask_State *pTaskState)
{
  mFUNCTION_SETUP();

  mERROR_IF(pTask == nullptr, mR_ArgumentNull);

  mRETURN_SUCCESS();
}

mFUNCTION(mTask_Destroy_Internal, IN mTask *pTask)
{
  mFUNCTION_SETUP();

  mERROR_IF(pTask == nullptr, mR_ArgumentNull);

  mERROR_CHECK(mSemaphore_Destroy(&pTask->pSemaphore));
  pTask->state.~atomic();
  pTask->function.~function();

  mRETURN_SUCCESS();
}
