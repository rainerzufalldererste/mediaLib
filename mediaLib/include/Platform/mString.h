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

typedef int32_t mchar_t;

template <size_t TCount>
mchar_t mToChar(const char c[TCount]);
mchar_t mToChar(const char *c, const size_t size);

struct mString
{
  char *text;
  mAllocator *pAllocator;
  size_t bytes;
  size_t count;
  size_t capacity;
  bool hasFailed;

  mString();

  template <size_t TSize>
  mString(const char text[TSize], IN OPTIONAL mAllocator *pAllocator = nullptr);

  mString(const char *text, const size_t size, IN OPTIONAL mAllocator *pAllocator = nullptr);
  mString(const char *text, IN OPTIONAL mAllocator *pAllocator = nullptr);

  ~mString();

  mString(const mString &copy);
  mString(mString &&move);

  mString & operator = (const mString &copy);
  mString & operator = (mString &&move);

  size_t Size() const;
  size_t Count() const;

  mchar_t operator[](const size_t index) const;

  mString operator +(const mString &s) const;
  mString operator +=(const mString &s);

  bool operator == (const mString &s) const;
  bool operator != (const mString &s) const;

  explicit operator std::string() const;
  explicit operator std::wstring() const;

  const char * c_str() const;
};

template <size_t TCount>
mFUNCTION(mString_Create, OUT mString *pString, const char text[TCount], IN OPTIONAL mAllocator *pAllocator = nullptr);

mFUNCTION(mString_Create, OUT mString *pString, const char *text, IN OPTIONAL mAllocator *pAllocator = nullptr);

mFUNCTION(mString_Create, OUT mString *pString, const char *text, const size_t size, IN OPTIONAL mAllocator *pAllocator = nullptr);

template <typename ...Args>
mFUNCTION(mString_CreateFormat, OUT mString *pString, IN OPTIONAL mAllocator *pAllocator, const char *formatString, Args&&... args);

mFUNCTION(mString_Destroy, IN_OUT mString *pString);

mFUNCTION(mString_Reserve, mString &string, const size_t size);

mFUNCTION(mString_GetByteSize, const mString &string, OUT size_t *pSize);

mFUNCTION(mString_GetCount, const mString &string, OUT size_t *pLength);

mFUNCTION(mString_ToWideString, const mString &string, std::wstring *pWideString);

mFUNCTION(mString_Substring, const mString &text, OUT mString *pSubstring, const size_t startCharacter);

mFUNCTION(mString_Substring, const mString &text, OUT mString *pSubstring, const size_t startCharacter, const size_t length);

mFUNCTION(mString_Append, mString &text, const mString &appendedText);

mFUNCTION(mString_ToDirectoryPath, OUT mString *pString, const mString &text);

mFUNCTION(mString_ToFilePath, OUT mString *pString, const mString &text);

mFUNCTION(mString_Equals, const mString &stringA, const mString &stringB, bool *pAreEqual);

//////////////////////////////////////////////////////////////////////////

template<size_t TCount>
inline mchar_t mToChar(const char c[TCount])
{
  return mToChar(c, TCount);
}

template<size_t TCount>
inline mFUNCTION(mString_Create, OUT mString *pString, const char text[TCount], IN OPTIONAL mAllocator *pAllocator /* = nullptr */)
{
  mFUNCTION_SETUP();

  mERROR_CHECK(mString_Create(pString, text, TCount, pAllocator));

  mRETURN_SUCCESS();
}

template<typename ...Args>
inline mFUNCTION(mString_CreateFormat, OUT mString *pString, IN OPTIONAL mAllocator *pAllocator, const char *formatString, Args && ...args)
{
  mFUNCTION_SETUP();

  char text[2048];
  mERROR_IF(0 > sprintf_s(text, formatString, std::forward<Args>(args)...), mR_InternalError);

  mERROR_CHECK(mString_Create(pString, text, pAllocator));

  mRETURN_SUCCESS();
}

template<size_t TSize>
inline mString::mString(const char text[TSize], IN OPTIONAL mAllocator *pAllocator /* = nullptr */) : mString(text, TSize, pAllocator)
{ }

//////////////////////////////////////////////////////////////////////////

template <size_t TCount>
struct mInplaceString
{
  char text[TCount + 1];
  size_t bytes;
  size_t count;

  const char * c_str() const;
};

template <size_t TCount>
mFUNCTION(mInplaceString_Create, OUT mInplaceString<TCount> *pStackString);

template <size_t TCount, size_t TTextCount>
mFUNCTION(mInplaceString_Create, OUT mInplaceString<TCount> *pStackString, const char text[TTextCount]);

template <size_t TCount>
mFUNCTION(mInplaceString_CreateRaw, OUT mInplaceString<TCount> *pStackString, const char *text);

template <size_t TCount>
mFUNCTION(mInplaceString_Create, OUT mInplaceString<TCount> *pStackString, const char *text, const size_t size);

template <size_t TCount, size_t TTextCount>
mFUNCTION(mInplaceString_Create, OUT mInplaceString<TCount> *pStackString, const mInplaceString<TTextCount> &text);

template <size_t TCount>
mFUNCTION(mInplaceString_Create, OUT mInplaceString<TCount> *pStackString, const mString &text);

template <size_t TCount>
mFUNCTION(mString_Create, OUT mString *pString, const mInplaceString<TCount> &stackString);

template <size_t TCount>
mFUNCTION(mInplaceString_GetByteSize, const mInplaceString<TCount> &string, OUT size_t *pSize);

template <size_t TCount>
mFUNCTION(mInplaceString_GetCount, const mInplaceString<TCount> &string, OUT size_t *pLength);

mFUNCTION(mInplaceString_GetCount_Internal, const char *text, const size_t maxSize, OUT size_t *pCount, OUT size_t *pSize);

//////////////////////////////////////////////////////////////////////////

template<size_t TCount>
inline const char * mInplaceString<TCount>::c_str() const
{
  return this->text;
}

template<size_t TCount>
inline mFUNCTION(mInplaceString_Create, OUT mInplaceString<TCount> *pStackString)
{
  mFUNCTION_SETUP();

  mERROR_IF(pStackString == nullptr, mR_ArgumentNull);

  pStackString->bytes = 1;
  pStackString->count = 1;
  mERROR_CHECK(mMemset(pStackString->text, TCount + 1, 0));

  mRETURN_SUCCESS();
}

template<size_t TCount, size_t TTextCount>
inline mFUNCTION(mInplaceString_Create, OUT mInplaceString<TCount> *pStackString, const char text[TTextCount])
{
  mFUNCTION_SETUP();

  mERROR_CHECK(mInplaceString_Create(pStackString, (char *)text, TTextCount));

  mRETURN_SUCCESS();
}

template<size_t TCount>
inline mFUNCTION(mInplaceString_CreateRaw, OUT mInplaceString<TCount> *pStackString, const char *text)
{
  mFUNCTION_SETUP();

  mERROR_IF(text == nullptr, mR_ArgumentNull);

  mERROR_CHECK(mInplaceString_Create(pStackString, text, strlen(text) + 1));

  mRETURN_SUCCESS();
}

template<size_t TCount>
inline mFUNCTION(mInplaceString_Create, OUT mInplaceString<TCount>* pStackString, const char * text, const size_t size)
{
  mFUNCTION_SETUP();

  mERROR_IF(pStackString == nullptr || text == nullptr, mR_ArgumentNull);

  size_t textSize;
  size_t textCount;

  mERROR_CHECK(mInplaceString_GetCount_Internal(text, size, &textCount, &textSize));
  mERROR_IF(textSize > TCount, mR_ArgumentOutOfBounds);

  pStackString->bytes = textSize;
  pStackString->count = textCount;
  mERROR_CHECK(mMemcpy(pStackString->text, text, textSize + 1));

  mRETURN_SUCCESS();
}

template<size_t TCount, size_t TTextCount>
inline mFUNCTION(mInplaceString_Create, OUT mInplaceString<TCount>* pStackString, const mInplaceString<TTextCount>& text)
{
  mFUNCTION_SETUP();

  mERROR_IF(pStackString == nullptr, mR_ArgumentNull);
  mERROR_IF(TCount < text.bytes, mR_ArgumentOutOfBounds);

  pStackString->bytes = text.bytes + 1;
  pStackString->count = text.count;
  mERROR_CHECK(mMemcpy(pStackString->text, text.text, pStackString->bytes + 1));

  mRETURN_SUCCESS();
}

template<size_t TCount>
inline mFUNCTION(mInplaceString_Create, OUT mInplaceString<TCount>* pStackString, const mString & text)
{
  mFUNCTION_SETUP();

  mERROR_IF(pStackString == nullptr, mR_ArgumentNull);
  mERROR_IF(TCount < text.bytes, mR_ArgumentOutOfBounds);

  pStackString->bytes = text.bytes;
  pStackString->count = text.count;
  mERROR_CHECK(mMemcpy(pStackString->text, text.text, pStackString->bytes));

  mRETURN_SUCCESS();
}

template<size_t TCount>
inline mFUNCTION(mString_Create, OUT mString * pString, const mInplaceString<TCount> &stackString)
{
  mFUNCTION_SETUP();

  mERROR_CHECK(mString_Create(pString, stackString.text, stackString.bytes));

  mRETURN_SUCCESS();
}

template<size_t TCount>
inline mFUNCTION(mInplaceString_GetByteSize, const mInplaceString<TCount>& string, OUT size_t * pSize)
{
  mFUNCTION_SETUP();

  mERROR_IF(pSize == nullptr, mR_ArgumentNull);

  *pSize = string.bytes;

  mRETURN_SUCCESS();
}

template<size_t TCount>
inline mFUNCTION(mInplaceString_GetCount, const mInplaceString<TCount>& string, OUT size_t * pLength)
{
  mFUNCTION_SETUP();

  mERROR_IF(pLength == nullptr, mR_ArgumentNull);

  *pLength = string.count;

  mRETURN_SUCCESS();
}

#endif // mString_h__
