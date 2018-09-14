// Copyright 2018 Christoph Stiller
// 
// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files(the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions :
// 
// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
// 
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef mMutex_h__
#define mMutex_h__

#include "default.h"

struct mMutex;

mFUNCTION(mMutex_Create, OUT mMutex **ppMutex, IN OPTIONAL mAllocator *pAllocator);
mFUNCTION(mMutex_Destroy, IN_OUT mMutex **ppMutex);

mFUNCTION(mMutex_Lock, IN mMutex *pMutex);
mFUNCTION(mMutex_Unlock, IN mMutex *pMutex);

struct mRecursiveMutex;

mFUNCTION(mRecursiveMutex_Create, OUT mRecursiveMutex **ppMutex, IN OPTIONAL mAllocator *pAllocator);
mFUNCTION(mRecursiveMutex_Destroy, IN_OUT mRecursiveMutex **ppMutex);

mFUNCTION(mRecursiveMutex_Lock, IN mRecursiveMutex *pMutex);
mFUNCTION(mRecursiveMutex_Unlock, IN mRecursiveMutex *pMutex);

#endif // mMutex_h__
