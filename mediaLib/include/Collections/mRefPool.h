// Copyright 2018 Christoph Stiller
// 
// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files(the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions :
// 
// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
// 
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef mRefPool_h__
#define mRefPool_h__

#include "default.h"
#include "mPool.h"
#include "mMutex.h"

template <typename T>
struct mRefPool
{
  struct refPoolPtrData
  {
    T element;
    typename mSharedPointer<T>::PointerParams ptrParams;
    size_t index;
  };

  struct refPoolPtr
  {
    size_t dataIndex;
    mPtr<T> ptr;
  };

  mPtr<mPool<refPoolPtrData>> data;
  mPtr<mPool<refPoolPtr>> ptrs;
  mMutex *pMutex;
  mAllocator *pAllocator;
  bool keepForever;
};

template <typename T>
mFUNCTION(mRefPool_Create, mPtr<mRefPool<T>> *pRefPool, mAllocator *pAllocator, const bool keepEntriesForever = false);

template <typename T>
mFUNCTION(mRefPool_Destroy, mPtr<mRefPool<T>> *pRefPool);

template <typename T>
mFUNCTION(mRefPool_Add, mPtr<mRefPool<T>> &refPool, IN T *pItem, OUT mPtr<T> *pIndex);

template <typename T>
mFUNCTION(mRefPool_AddEmpty, mPtr<mRefPool<T>> &refPool, OUT mPtr<T> *pIndex);

template <typename T>
mFUNCTION(mRefPool_Crush, mPtr<mRefPool<T>> &refPool);

template <typename T>
mFUNCTION(mRefPool_ForEach, mPtr<mRefPool<T>> &refPool, const std::function<mResult (mPtr<T> &)> &function);

// would be handled by cpp but still nicer if explicitly defined.
template <typename T>
mFUNCTION(mDestruct, struct mRefPool<T>::refPoolPtr *pData);

#include "mRefPool.inl"

#endif // mRefPool_h__
