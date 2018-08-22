// Copyright 2018 Christoph Stiller
// 
// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files(the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions :
// 
// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
// 
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef mString_h__
#define mString_h__

#include "default.h"

typedef uint32_t mchar_t;

struct mString
{
  char *pData;
  mAllocator *pAllocator;
  size_t bytes;
  size_t length;

  mString();
  mString(char *text, IN OPTIONAL mAllocator *pAllocator = nullptr);
  ~mString();

  size_t Size() const;
  size_t Length() const;

  bool IsValidUnicode() const;

  mchar_t operator[](size_t index);
};

template <size_t TCount>
mFUNCTION(mString_Create, OUT mString *pString, char text[TCount], IN OPTIONAL mAllocator *pAllocator = nullptr);

mFUNCTION(mString_Create, OUT mString *pString, char *text, IN OPTIONAL mAllocator *pAllocator = nullptr);

mFUNCTION(mString_Create, OUT mString *pString, const size_t size, char *text, IN OPTIONAL mAllocator *pAllocator = nullptr);

mFUNCTION(mString_Destroy, IN_OUT mString *pString);

mFUNCTION(mString_GetByteSize, const mString &string, OUT size_t *pSize);

mFUNCTION(mString_GetLength, const mString &string, OUT size_t *pLength);

mFUNCTION(mString_IsValidUnicode, const mString &string, OUT bool *pIsUnicode);

mFUNCTION(mString_ToWideString, const mString &string, std::wstring *pWideString);

//////////////////////////////////////////////////////////////////////////

template<size_t TCount>
inline mFUNCTION(mString_Create, OUT mString *pString, char text[TCount], IN OPTIONAL mAllocator *pAllocator /* = nullptr */)
{
  mFUNCTION_SETUP();

  mERROR_CHECK(mString_Create(pString, text, TCount, pAllocator));

  mRETURN_SUCCESS();
}

#endif // mString_h__
