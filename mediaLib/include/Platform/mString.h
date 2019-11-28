#ifndef mString_h__
#define mString_h__

#include "mediaLib.h"

typedef int32_t mchar_t;

template <size_t TCount>
mchar_t mToChar(const char c[TCount]);
mchar_t mToChar(IN const char *c, const size_t size);

struct mIteratedString
{
  char *character;
  mchar_t codePoint;
  size_t characterSize;
  size_t index;
  size_t offset;

  mIteratedString(char *character, const mchar_t codePoint, const size_t characterSize, const size_t index, const size_t offset);
};

struct mUtf8StringIterator
{
  char *string;
  size_t bytes;
  size_t position = 0;
  size_t charCount = 0;
  mchar_t codePoint = 0;
  ptrdiff_t characterSize = 0;

  mUtf8StringIterator(char *string);
  mUtf8StringIterator(char *string, size_t bytes);

  mUtf8StringIterator& begin();
  mUtf8StringIterator end();

  bool operator!=(const mUtf8StringIterator &b);
  mUtf8StringIterator& operator++();

  mIteratedString operator*();
};

struct mString
{
  char *text;
  mAllocator *pAllocator = &mDefaultAllocator;
  
  // Includes null terminator.
  size_t bytes;

  // Includes null terminator.
  size_t count;

  size_t capacity;
  bool hasFailed;

  mString();

  template <size_t TSize>
  mString(const char text[TSize], IN OPTIONAL mAllocator *pAllocator = nullptr);

  mString(IN const char *text, const size_t size, IN OPTIONAL mAllocator *pAllocator = nullptr);
  mString(IN const char *text, IN OPTIONAL mAllocator *pAllocator = nullptr);

  template <size_t TSize>
  mString(const wchar_t text[TSize], IN OPTIONAL mAllocator *pAllocator = nullptr);

  mString(const wchar_t *text, const size_t size, IN OPTIONAL mAllocator *pAllocator = nullptr);
  mString(const wchar_t *text, IN OPTIONAL mAllocator *pAllocator = nullptr);

  ~mString();

  mString(const mString &copy);
  mString(mString &&move);

  mString & operator = (const mString &copy);
  mString & operator = (mString &&move);

  // Size in bytes.
  // Includes null terminator.
  size_t Size() const;

  // Number of utf8 characters in this string.
  // Includes null terminator.
  size_t Count() const;

  mchar_t operator[](const size_t index) const;

  mString operator +(const mString &s) const;
  mString operator +=(const mString &s);

  bool operator == (const mString &s) const;
  bool operator != (const mString &s) const;

  mUtf8StringIterator begin() const;
  mUtf8StringIterator end() const;

  inline const char * c_str() const { return text; };

  bool StartsWith(const mString &s) const;
  bool EndsWith(const mString &s) const;

  bool Contains(const mString &s) const;
};

template <size_t TCount>
mFUNCTION(mString_Create, OUT mString *pString, const char text[TCount], IN OPTIONAL mAllocator *pAllocator = nullptr);

mFUNCTION(mString_Create, OUT mString *pString, IN const char *text, IN OPTIONAL mAllocator *pAllocator = nullptr);
mFUNCTION(mString_Create, OUT mString *pString, IN const char *text, const size_t size, IN OPTIONAL mAllocator *pAllocator = nullptr);

template <size_t TCount>
mFUNCTION(mString_Create, OUT mString *pString, const wchar_t text[TCount], IN OPTIONAL mAllocator *pAllocator = nullptr);

mFUNCTION(mString_Create, OUT mString *pString, const wchar_t *text, IN OPTIONAL mAllocator *pAllocator = nullptr);
mFUNCTION(mString_Create, OUT mString *pString, const wchar_t *text, const size_t bufferSize, IN OPTIONAL mAllocator *pAllocator = nullptr);
mFUNCTION(mString_Create, OUT mString *pString, const mString &from, IN OPTIONAL mAllocator *pAllocator = nullptr);

template <typename ...Args>
mFUNCTION(mString_CreateFormat, OUT mString *pString, IN OPTIONAL mAllocator *pAllocator, IN const char *formatString, Args&&... args);

mFUNCTION(mString_Destroy, IN_OUT mString *pString);

mFUNCTION(mString_Reserve, mString &string, const size_t size);

mFUNCTION(mString_GetByteSize, const mString &string, OUT size_t *pSize);
mFUNCTION(mString_GetCount, const mString &string, OUT size_t *pLength);

mFUNCTION(mString_ToWideString, const mString &string, OUT wchar_t *pWideString, const size_t bufferCount);
mFUNCTION(mString_ToWideString, const mString &string, OUT wchar_t *pWideString, const size_t bufferCount, OUT size_t *pWideStringCount);
mFUNCTION(mString_GetRequiredWideStringCount, const mString &string, OUT size_t *pWideStringCount);

mFUNCTION(mString_Substring, const mString &text, OUT mString *pSubstring, const size_t startCharacter);
mFUNCTION(mString_Substring, const mString &text, OUT mString *pSubstring, const size_t startCharacter, const size_t length);

mFUNCTION(mString_Append, mString &text, const mString &appendedText);
mFUNCTION(mString_Append, mString &text, const char *appendedText);
mFUNCTION(mString_Append, mString &text, const char *appendedText, const size_t size);

template <typename ...Args>
mFUNCTION(mString_AppendFormat, mString &text, const char *format, Args&&... args);

mFUNCTION(mString_AppendUnsignedInteger, mString &text, const uint64_t value);
mFUNCTION(mString_AppendInteger, mString &text, const int64_t value);
mFUNCTION(mString_AppendBool, mString &text, const bool value);
mFUNCTION(mString_AppendDouble, mString &text, const double_t value);

mFUNCTION(mString_ToDirectoryPath, OUT mString *pString, const mString &text);
mFUNCTION(mString_ToFilePath, OUT mString *pString, const mString &text);

mFUNCTION(mString_Equals, const mString &stringA, const mString &stringB, bool *pAreEqual);

mFUNCTION(mString_StartsWith, const mString &stringA, const mString &start, OUT bool *pStartsWith);
mFUNCTION(mString_EndsWith, const mString &stringA, const mString &end, OUT bool *pEndsWith);

mFUNCTION(mString_FindFirst, const mString &string, const mString &find, OUT size_t *pStartChar, OUT bool *pContained);
mFUNCTION(mString_Contains, const mString &stringA, const mString &contained, OUT bool *pContains);

mFUNCTION(mString_RemoveChar, const mString &string, const mchar_t remove, OUT mString *pResult);
mFUNCTION(mString_RemoveString, const mString &string, const mString &remove, OUT mString *pResult);

mFUNCTION(mString_Replace, const mString &string, const mString &replace, const mString &with, OUT mString *pResult);

mFUNCTION(mString_TrimStart, const mString &string, const mchar_t trimmedChar, OUT mString *pTrimmedString);
mFUNCTION(mString_TrimEnd, const mString &string, const mchar_t trimmedChar, OUT mString *pTrimmedString);

// parameters to the lambda:
//   mchar_t: utf-8 codepoint for comparisons
//   char *: start of first byte of the char
//   size_t: bytes of the utf-8 char.
mFUNCTION(mString_ForEachChar, const mString &string, const std::function<mResult(mchar_t, char *, size_t)> &function);

template <>
struct mIsTriviallyMemoryMovable<mString>
{
  static constexpr bool value = true;
};

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

template<size_t TCount>
inline mFUNCTION(mString_Create, OUT mString *pString, const wchar_t text[TCount], IN OPTIONAL mAllocator *pAllocator)
{
  mFUNCTION_SETUP();

  mERROR_CHECK(mString_Create(pString, text, TCount, pAllocator));

  mRETURN_SUCCESS();
}

template<typename ...Args>
inline mFUNCTION(mString_CreateFormat, OUT mString *pString, IN OPTIONAL mAllocator *pAllocator, IN const char *formatString, Args && ...args)
{
  mFUNCTION_SETUP();

  char text[1024 * 8];
  mERROR_IF(0 > sprintf_s(text, formatString, std::forward<Args>(args)...), mR_InternalError);

  mERROR_CHECK(mString_Create(pString, text, pAllocator));

  mRETURN_SUCCESS();
}

template <typename ...Args>
mFUNCTION(mString_AppendFormat, mString &text, const char *format, Args&&... args)
{
  mFUNCTION_SETUP();

  char appendedText[1024 * 8];
  mERROR_IF(0 > sprintf_s(appendedText, format, std::forward<Args>(args)...), mR_InternalError);

  mERROR_CHECK(mString_Append(text, appendedText, mARRAYSIZE(appendedText)));

  mRETURN_SUCCESS();
}

template<size_t TSize>
inline mString::mString(const char text[TSize], IN OPTIONAL mAllocator *pAllocator /* = nullptr */) : mString(text, TSize, pAllocator)
{ }

template<size_t TSize>
inline mString::mString(const wchar_t text[TSize], IN OPTIONAL mAllocator *pAllocator /* = nullptr */) : mString(text, TSize, pAllocator)
{ }

//////////////////////////////////////////////////////////////////////////

template <size_t TCount>
struct mInplaceString
{
  char text[TCount + 1] = "";
  size_t bytes = 0;
  size_t count = 0;

  mInplaceString() = default;

  mInplaceString(const mInplaceString<TCount> &copy);
  mInplaceString(mInplaceString<TCount> &&move);
  
  mInplaceString<TCount> & operator = (const mInplaceString<TCount> &copy);
  mInplaceString<TCount> & operator = (mInplaceString<TCount> &&move);

  const char * c_str() const;

  template <size_t TOtherCount>
  bool operator == (const mInplaceString<TOtherCount> &other) const;

  template <size_t TOtherCount>
  bool operator != (const mInplaceString<TOtherCount> &other) const;

  mUtf8StringIterator begin() const;
  mUtf8StringIterator end() const;
};

template <size_t TCount>
mFUNCTION(mInplaceString_Create, OUT mInplaceString<TCount> *pStackString);

template <size_t TCount, size_t TTextCount>
mFUNCTION(mInplaceString_Create, OUT mInplaceString<TCount> *pStackString, const char text[TTextCount]);

template <size_t TCount>
mFUNCTION(mInplaceString_CreateRaw, OUT mInplaceString<TCount> *pStackString, IN const char *text);

template <size_t TCount>
mFUNCTION(mInplaceString_Create, OUT mInplaceString<TCount> *pStackString, IN const char *text, const size_t size);

template <size_t TCount, size_t TTextCount>
mFUNCTION(mInplaceString_Create, OUT mInplaceString<TCount> *pStackString, const mInplaceString<TTextCount> &text);

template <size_t TCount>
mFUNCTION(mInplaceString_Create, OUT mInplaceString<TCount> *pStackString, const mString &text);

template <size_t TCount>
mFUNCTION(mString_Create, OUT mString *pString, const mInplaceString<TCount> &stackString, IN OPTIONAL mAllocator *pAllocator = nullptr);

template <size_t TCount>
mFUNCTION(mString_Append, mString &text, const mInplaceString<TCount> &appendedText);

template <size_t TCount>
mFUNCTION(mInplaceString_GetByteSize, const mInplaceString<TCount> &string, OUT size_t *pSize);

template <size_t TCount>
mFUNCTION(mInplaceString_GetCount, const mInplaceString<TCount> &string, OUT size_t *pLength);

mFUNCTION(mInplaceString_GetCount_Internal, IN const char *text, const size_t maxSize, OUT size_t *pCount, OUT size_t *pSize);
bool mInplaceString_StringsAreEqual_Internal(IN const char *textA, IN const char *textB, const size_t bytes, const size_t count);

template <size_t TCount>
struct mIsTriviallyMemoryMovable<mInplaceString<TCount>>
{
  static constexpr bool value = true;
};

//////////////////////////////////////////////////////////////////////////

template<size_t TCount>
inline mInplaceString<TCount>::mInplaceString(const mInplaceString<TCount> &copy) :
  bytes(copy.bytes),
  count(copy.count)
{
  mMemcpy(text, copy.text, bytes / sizeof(char));
}

template<size_t TCount>
inline mInplaceString<TCount>::mInplaceString(mInplaceString<TCount> &&move) :
  bytes(move.bytes),
  count(move.count)
{
  mMemmove(text, move.text, bytes / sizeof(char));
}

template<size_t TCount>
inline mInplaceString<TCount> & mInplaceString<TCount>::operator=(const mInplaceString<TCount> &copy)
{
  bytes = copy.bytes;
  count = copy.count;
  mMemcpy(text, copy.text, bytes / sizeof(char));

  return *this;
}

template<size_t TCount>
inline mInplaceString<TCount> & mInplaceString<TCount>::operator=(mInplaceString<TCount> &&move)
{
  bytes = move.bytes;
  count = move.count;
  mMemmove(text, move.text, bytes / sizeof(char));

  return *this;
}

template<size_t TCount>
inline const char * mInplaceString<TCount>::c_str() const
{
  return this->text;
}

template<size_t TCount>
inline mUtf8StringIterator mInplaceString<TCount>::begin() const
{
  return mUtf8StringIterator((char *)text, bytes);
}

template<size_t TCount>
inline mUtf8StringIterator mInplaceString<TCount>::end() const
{
  return mUtf8StringIterator(nullptr, 0);
}

template<size_t TCount>
template<size_t TOtherCount>
inline bool mInplaceString<TCount>::operator==(const mInplaceString<TOtherCount> &other) const
{
  if (other.bytes != bytes || other.count != this->count)
    return false;

  return mInplaceString_StringsAreEqual_Internal(text, other.text, bytes, count);
}

template<size_t TCount>
template<size_t TOtherCount>
inline bool mInplaceString<TCount>::operator!=(const mInplaceString<TOtherCount> &other) const
{
  return !(*this == other);
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
inline mFUNCTION(mInplaceString_CreateRaw, OUT mInplaceString<TCount> *pStackString, IN const char *text)
{
  mFUNCTION_SETUP();

  mERROR_IF(text == nullptr, mR_ArgumentNull);

  mERROR_CHECK(mInplaceString_Create(pStackString, text, strlen(text) + 1));

  mRETURN_SUCCESS();
}

template<size_t TCount>
inline mFUNCTION(mInplaceString_Create, OUT mInplaceString<TCount> *pStackString, IN const char *text, const size_t size)
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
inline mFUNCTION(mInplaceString_Create, OUT mInplaceString<TCount> *pStackString, const mInplaceString<TTextCount> &text)
{
  mFUNCTION_SETUP();

  mERROR_IF(pStackString == nullptr, mR_ArgumentNull);
  mERROR_IF(TCount < text.bytes, mR_ArgumentOutOfBounds);

  pStackString->bytes = text.bytes;
  pStackString->count = text.count;
  mERROR_CHECK(mMemcpy(pStackString->text, text.text, pStackString->bytes + 1));

  mRETURN_SUCCESS();
}

template<size_t TCount>
inline mFUNCTION(mInplaceString_Create, OUT mInplaceString<TCount> *pStackString, const mString &text)
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
inline mFUNCTION(mString_Create, OUT mString *pString, const mInplaceString<TCount> &stackString, IN OPTIONAL mAllocator *pAllocator /* = nullptr */)
{
  mFUNCTION_SETUP();

  mERROR_CHECK(mString_Create(pString, stackString.text, stackString.bytes, pAllocator));

  mRETURN_SUCCESS();
}

template<size_t TCount>
inline mFUNCTION(mString_Append, mString &text, const mInplaceString<TCount> &appendedText)
{
  mFUNCTION_SETUP();

  mString s;
  s.capacity = s.bytes = appendedText.bytes;
  s.count = appendedText.count;
  s.hasFailed = false;
  s.text = const_cast<char *>(appendedText.text);

  // Prevent from attempting to free `s.text`;
  mDEFER(
    s.capacity = s.count = s.bytes = 0;
    s.text = nullptr;
  );

  mERROR_CHECK(mString_Append(text, s));

  mRETURN_SUCCESS();
}

template<size_t TCount>
inline mFUNCTION(mInplaceString_GetByteSize, const mInplaceString<TCount> &string, OUT size_t *pSize)
{
  mFUNCTION_SETUP();

  mERROR_IF(pSize == nullptr, mR_ArgumentNull);

  *pSize = string.bytes;

  mRETURN_SUCCESS();
}

template<size_t TCount>
inline mFUNCTION(mInplaceString_GetCount, const mInplaceString<TCount> &string, OUT size_t *pLength)
{
  mFUNCTION_SETUP();

  mERROR_IF(pLength == nullptr, mR_ArgumentNull);

  *pLength = string.count;

  mRETURN_SUCCESS();
}

#endif // mString_h__
