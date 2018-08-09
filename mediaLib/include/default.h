// Copyright 2018 Christoph Stiller
// 
// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files(the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions :
// 
// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
// 
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifdef _HAS_EXCEPTIONS
#undef _HAS_EXCEPTIONS
#endif

#define _HAS_EXCEPTIONS 0

#ifndef _USE_MATH_DEFINES
#define _USE_MATH_DEFINES 1
#endif // !_USE_MATH_DEFINES

#include <assert.h>
#include <stdint.h>
#include <climits>
#include <math.h>
#include <functional>
#include <type_traits>
#include <inttypes.h>

#ifndef _DEPENDENCIES_DEFINED
#define _DEPENDENCIES_DEFINED

#ifndef RETURN
#define RETURN return
#endif // !RETURN

#ifndef IN
#define IN
#endif // !IN

#ifndef OUT
#define OUT
#endif // !OUT

#ifndef IN_OUT
#define IN_OUT IN OUT
#endif // !IN_OUT

#ifndef OPTIONAL
#define OPTIONAL
#endif // !OPTIONAL

#define mVECTORCALL __vectorcall
#define mINLINE __forceinline

#define mPI M_PI
#define mTWOPI 6.283185307179586476925286766559
#define mHALFPI M_PI_2
#define mQUARTERPI M_PI_4
#define mSQRT2 M_SQRT2
#define mINVSQRT2 M_SQRT1_2
#define mSQRT3 1.414213562373095048801688724209698
#define mINV_SQRT3 0.5773502691896257645091487805019574556
#define mPIf 3.141592653589793f
#define mTWOPIf 6.283185307179586f
#define mHALFPIf ((float)M_PI_2)
#define mQUARTERPIf ((float)M_PI_4)
#define mSQRT2f 1.414213562373095f
#define mINVSQRT2f 0.7071067811865475f
#define mSQRT3f 1.414213562373095f
#define mINVSQRT3f 0.57735026918962576f

#define mDEG2RAD (mPI / 180.0)
#define mDEG2RADf (mPIf / 180.0f)
#define mRAD2DEG (180.0 / mPI)
#define mRAD2DEGf (180.0f / mPIf)

#define mARRAYSIZE(arrayName) (sizeof(arrayName) / sizeof(arrayName[0]))

#define mASSERT(expression, text) do { if(!(expression)) { mPRINT("Assertion Failed: %s\n'%s'\n\nIn File '%s' : Line '%" PRIi32 "' (Function '%s')\n", #expression, text, __FILE__, __LINE__, __FUNCTION__); assert(expression); } } while (0)
#define mFAIL(text) mPRINT("Assertion Failed: '%s'\n\nIn File '%s' : Line '%" PRIi32 "' (Function '%s')\n", text, __FILE__, __LINE__, __FUNCTION__)
#define mPRINT(text, ...) printf(text, __VA_ARGS__)

template <typename T>
void mUnused(T unused)
{
  (void)unused;
}

template <typename T, typename ...Args>
void mUnused(T unused, Args && ...args)
{
  (void)unused;
  mUnused(args...);
}

#ifndef _DEBUG
#define mASSERT_DEBUG(expr, text) mASSERT(expr, text)
#define mFAIL_DEBUG(text) mFAIL(text)
#else //_DEBUG
#define mASSERT_DEBUG(expr, text) mUnused(expr, text)
#define mFAIL_DEBUG(text) mUnused(text)
#endif

#endif // !_DEPENDENCIES_DEFINED

#ifndef default_h__
#define default_h__

#include "mResult.h"
#include "mDefer.h"
#include "mMemory.h"
#include "mSharedPointer.h"
#include "mMath.h"
#include "mFastMath.h"
#include "mStaticIf.h"

#endif // default_h__
