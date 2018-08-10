// Copyright 2018 Christoph Stiller
// 
// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files(the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions :
// 
// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
// 
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef mAllocator_h__
#define mAllocator_h__

#include "default.h"

// Parameters: 
//   OUT uint8_t **ppData: pointer to pointer to allocate to.
//   const size_t size: size of a single object.
//   const size_t count: amount of objects.
//   IN void *pUserData: associated user data with the custom allocator.
typedef mResult(mAllocator_AllocFunction) (OUT uint8_t **, const size_t, const size_t, IN void *);

// Parameters: 
//   IN uint8_t *ppData: pointer to free memory at.
//   IN void *pUserData: associated user data with the custom allocator.
typedef mResult(mAllocator_FreeFunction) (IN uint8_t *, IN void *);

// Parameters: 
//   IN_OUT uint8_t *pDestination: pointer to the destination.
//   IN_OUT uint8_t *pSource: size of a single object.
//   const size_t size: size of a single object.
//   const size_t count: amount of objects.
//   IN void *pUserData: associated user data with the custom allocator.
typedef mResult(mAllocator_MoveFunction) (IN_OUT uint8_t *, IN_OUT uint8_t *, const size_t, const size_t, IN void *);

// Parameters: 
//   IN_OUT uint8_t *pDestination: pointer to the destination.
//   IN_OUT uint8_t *pSource: size of a single object.
//   const size_t size: size of a single object.
//   const size_t count: amount of objects.
//   IN void *pUserData: associated user data with the custom allocator.
typedef mResult(mAllocator_CopyFunction) (IN_OUT uint8_t *, IN uint8_t *, const size_t, const size_t, IN void *);

// Parameters: 
//   IN void *pUserData: associated user data with the custom allocator.
typedef mResult(mAllocator_DestroyAllocator) (IN void *);

mFUNCTION(mDefaultAllocator_Alloc, OUT uint8_t **ppData, const size_t size, const size_t count, IN void *pUserData = nullptr);
mFUNCTION(mDefaultAllocator_AllocZero, OUT uint8_t **ppData, const size_t size, const size_t count, IN void *pUserData = nullptr);
mFUNCTION(mDefaultAllocator_Realloc, OUT uint8_t **ppData, const size_t size, const size_t count, IN void *pUserData = nullptr);
mFUNCTION(mDefaultAllocator_Free, OUT uint8_t *pData, IN void *pUserData = nullptr);
mFUNCTION(mDefaultAllocator_Move, IN_OUT uint8_t *pDestimation, IN uint8_t *pSource, const size_t size, const size_t count, IN void *pUserData = nullptr);
mFUNCTION(mDefaultAllocator_Copy, IN_OUT uint8_t *pDestimation, IN uint8_t *pSource, const size_t size, const size_t count, IN void *pUserData = nullptr);

struct mAllocator
{
  mAllocator_AllocFunction *pAllocate = nullptr;
  mAllocator_AllocFunction *pAllocateZero = nullptr;
  mAllocator_AllocFunction *pReallocate = nullptr;
  mAllocator_FreeFunction *pFree = nullptr;
  mAllocator_MoveFunction *pMove = nullptr;
  mAllocator_CopyFunction *pCopy = nullptr;
  mAllocator_DestroyAllocator *pDestroyAllocator = nullptr;

  void *pUserData = nullptr;
  bool initialized = false;
};

mFUNCTION(mAllocator_Create, OUT mAllocator *pAllocator, IN mAllocator_AllocFunction *pAllocFunction, IN mAllocator_FreeFunction *pFree, IN OPTIONAL mAllocator_MoveFunction *pMove = nullptr, IN OPTIONAL mAllocator_CopyFunction *pCopy = nullptr, IN OPTIONAL mAllocator_AllocFunction *pAllocZeroFunction = nullptr, IN OPTIONAL mAllocator_AllocFunction *pReallocFunction = nullptr, IN OPTIONAL mAllocator_DestroyAllocator *pDestroyAllocator = nullptr, IN OPTIONAL void *pUserData = nullptr);
mFUNCTION(mAllocator_Destoy, IN_OUT mAllocator *pAllocator);

template <typename T>
mFUNCTION(mAllocator_Allocate, IN OPTIONAL mAllocator *pAllocator, OUT T **pData, const size_t count);

template <typename T>
mFUNCTION(mAllocator_AllocateZero, IN OPTIONAL mAllocator *pAllocator, OUT T **ppData, const size_t count);

template <typename T>
mFUNCTION(mAllocator_Reallocate, IN OPTIONAL mAllocator *pAllocator, OUT T **ppData, const size_t count);

template <typename T>
mFUNCTION(mAllocator_FreePtr, IN OPTIONAL mAllocator *pAllocator, IN_OUT T **ppData);

template <typename T>
mFUNCTION(mAllocator_Free, IN OPTIONAL mAllocator *pAllocator, IN T *pData);

template <typename T>
mFUNCTION(mAllocator_Move, IN OPTIONAL mAllocator *pAllocator, IN_OUT T *pDestination, IN_OUT T *pSource, const size_t count);

template <typename T>
mFUNCTION(mAllocator_Copy, IN OPTIONAL mAllocator *pAllocator, IN_OUT T *pDestination, IN T *pSource, const size_t count);

//////////////////////////////////////////////////////////////////////////

template <typename T>
mFUNCTION(mAllocator_Allocate, IN OPTIONAL mAllocator *pAllocator, OUT T **ppData, const size_t count)
{
  mFUNCTION_SETUP();

  mERROR_IF(ppData == nullptr, mR_ArgumentNull);
  mDEFER_DESTRUCTION_ON_ERROR(ppData, mSetToNullptr);

  if (pAllocator == nullptr || !pAllocator->initialized || (pAllocator->pAllocate == nullptr && pAllocator->pAllocateZero == nullptr))
    mERROR_CHECK(mDefaultAllocator_Alloc((uint8_t **)ppData, sizeof(T), count));
  else if (pAllocator->pAllocate != nullptr)
    mERROR_CHECK(pAllocator->pAllocate((uint8_t **)ppData, sizeof(T), count, pAllocator->pUserData));
  else
    mERROR_CHECK(pAllocator->pAllocateZero((uint8_t **)ppData, sizeof(T), count, pAllocator->pUserData));

  mRETURN_SUCCESS();
}

template <typename T>
mFUNCTION(mAllocator_AllocateZero, IN OPTIONAL mAllocator *pAllocator, OUT T **ppData, const size_t count)
{
  mFUNCTION_SETUP();

  mERROR_IF(ppData == nullptr, mR_ArgumentNull);
  mDEFER_DESTRUCTION_ON_ERROR(ppData, mSetToNullptr);

  if (pAllocator == nullptr || !pAllocator->initialized || (pAllocator->pAllocate == nullptr && pAllocator->pAllocateZero == nullptr))
  {
    mERROR_CHECK(mDefaultAllocator_AllocZero((uint8_t **)ppData, sizeof(T), count));
  }
  else if (pAllocator->pAllocateZero != nullptr)
  {
    mERROR_CHECK(pAllocator->pAllocateZero((uint8_t **)ppData, sizeof(T), count, pAllocator->pUserData));
  }
  else
  {
    mERROR_CHECK(pAllocator->pAllocate((uint8_t **)ppData, sizeof(T), count, pAllocator->pUserData));
    mERROR_CHECK(mMemset(*ppData, count, 0));
  }

  mRETURN_SUCCESS();
}

template<typename T>
inline mFUNCTION(mAllocator_Reallocate, IN OPTIONAL mAllocator * pAllocator, OUT T ** ppData, const size_t count)
{
  mFUNCTION_SETUP();

  if (pAllocator == nullptr || !pAllocator->initialized || pAllocator->pReallocate == nullptr)
    mERROR_CHECK(mDefaultAllocator_Realloc((uint8_t **)ppData, sizeof(T), count));
  else
    mERROR_CHECK(pAllocator->pReallocate((uint8_t **)ppData, sizeof(T), count));

  mRETURN_SUCCESS();
}

template<typename T>
inline mFUNCTION(mAllocator_FreePtr, IN OPTIONAL mAllocator *pAllocator, IN_OUT T **ppData)
{
  mFUNCTION_SETUP();

  mERROR_IF(ppData == nullptr, mR_ArgumentNull);

  mDEFER_DESTRUCTION(ppData, mSetToNullptr);

  if (*ppData == nullptr)
    mRETURN_SUCCESS();

  if (pAllocator == nullptr || !pAllocator->initialized || pAllocator->pFree == nullptr)
    mERROR_CHECK(mDefaultAllocator_Free((uint8_t *)*ppData));
  else
    mERROR_CHECK(pAllocator->pFree((uint8_t *)*ppData, pAllocator->pUserData));

  mRETURN_SUCCESS();
}

template<typename T>
inline mFUNCTION(mAllocator_Free, IN OPTIONAL mAllocator *pAllocator, IN T *pData)
{
  mFUNCTION_SETUP();

  if (pData == nullptr)
    mRETURN_SUCCESS();

  if (pAllocator == nullptr || !pAllocator->initialized || pAllocator->pFree == nullptr)
    mERROR_CHECK(mDefaultAllocator_Free((uint8_t *)pData));
  else
    mERROR_CHECK(pAllocator->pFree((uint8_t *)pData, pAllocator->pUserData));

  mRETURN_SUCCESS();
}

template<typename T>
inline mFUNCTION(mAllocator_Move, IN OPTIONAL mAllocator *pAllocator, IN_OUT T *pDestination, IN_OUT T *pSource, const size_t count)
{
  mFUNCTION_SETUP();

  mERROR_IF(pDestination == nullptr || pSource == nullptr, mR_ArgumentNull);

  if (pAllocator == nullptr || !pAllocator->initialized || pAllocator->pMove == nullptr)
    mERROR_CHECK(mDefaultAllocator_Move((uint8_t *)pDestination, (uint8_t *)pSource, sizeof(T), count));
  else
    mERROR_CHECK(pAllocator->pMove((uint8_t *)pDestination, (uint8_t *)pSource, sizeof(T), count, pAllocator->pUserData));

  mRETURN_SUCCESS();
}

template<typename T>
inline mFUNCTION(mAllocator_Copy, IN OPTIONAL mAllocator *pAllocator, IN_OUT T *pDestination, IN T *pSource, const size_t count)
{
  mFUNCTION_SETUP();

  mERROR_IF(pDestination == nullptr || pSource == nullptr, mR_ArgumentNull);

  if (pAllocator == nullptr || !pAllocator->initialized || pAllocator->pCopy == nullptr)
    mERROR_CHECK(mDefaultAllocator_Copy((uint8_t *)pDestination, (uint8_t *)pSource, sizeof(T), count));
  else
    mERROR_CHECK(pAllocator->pOopy((uint8_t *)pDestination, (uint8_t *)pSource, sizeof(T), count, pAllocator->pUserData));

  mRETURN_SUCCESS();
}

#endif // mAllocator_h__
