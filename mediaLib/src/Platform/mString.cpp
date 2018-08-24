// Copyright 2018 Christoph Stiller
// 
// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files(the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions :
// 
// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
// 
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "mString.h"
#include "utf8proc.h"

mchar_t mToChar(char *c, const size_t size)
{
  utf8proc_int32_t codePoint;
  utf8proc_iterate((uint8_t *)c, size, &codePoint);

  return mchar_t(codePoint);
}

//////////////////////////////////////////////////////////////////////////

mString::mString() :
  text(nullptr),
  pAllocator(nullptr),
  bytes(0),
  count(0),
  capacity(0),
  hasFailed(false)
{ }

mString::mString(char * text, const size_t size, IN OPTIONAL mAllocator * pAllocator)
{
  mResult result = mR_Success;

  this->pAllocator = pAllocator;
  this->bytes = size;

  mERROR_CHECK_GOTO(mAllocator_AllocateZero(this->pAllocator, &this->text, this->bytes), result, epilogue);
  this->capacity = bytes;
  mERROR_CHECK_GOTO(mAllocator_Copy(this->pAllocator, this->text, text, this->bytes), result, epilogue);

  size_t offset = 0;
  this->count = 0;

  while (offset < bytes)
  {
    utf8proc_int32_t codePoint;
    ptrdiff_t characterSize;
    mERROR_IF_GOTO((characterSize = utf8proc_iterate((uint8_t *)this->text + offset, this->bytes - offset, &codePoint)) < 0, mR_InternalError, result, epilogue);
    mERROR_IF_GOTO(codePoint < 0, mR_InternalError, result, epilogue);
    offset += (size_t)characterSize;
    this->count++;
  }

  return;

epilogue:
  this->~mString();
  hasFailed = true;
}

mString::mString(char *text, IN OPTIONAL mAllocator *pAllocator) : mString(text, strlen(text), pAllocator)
{ }

mString::~mString()
{
  if (text != nullptr)
    mAllocator_FreePtr(pAllocator, &text);

  text = nullptr;
  pAllocator = nullptr;
  bytes = 0;
  count = 0;
  capacity = 0;
  hasFailed = false;
}

mString::mString(const mString &copy) :
  text(nullptr),
  pAllocator(nullptr),
  bytes(0),
  count(0),
  capacity(0),
  hasFailed(false)
{
  if (copy.hasFailed || copy.capacity == 0)
    goto epilogue;

  mResult result = mR_Success;

  pAllocator = copy.pAllocator;

  mERROR_CHECK_GOTO(mAllocator_AllocateZero(pAllocator, &text, copy.bytes), result, epilogue);
  capacity = bytes = copy.bytes;
  count = copy.count;

  mERROR_CHECK_GOTO(mAllocator_Copy(pAllocator, text, copy.text, bytes), result, epilogue);

  return;

epilogue:
  this->~mString();
  hasFailed = true;
}

mString::mString(mString &&move) :
  text(move.text),
  pAllocator(move.pAllocator),
  bytes(move.bytes),
  count(move.count),
  capacity(move.capacity),
  hasFailed(move.hasFailed)
{
  move.text = nullptr;
  move.count = 0;
  move.bytes = 0;
  move.pAllocator = nullptr;
  move.capacity = 0;
  move.hasFailed = false;
}

mString & mString::operator=(const mString &copy)
{
  mResult result = mR_Success;

  this->hasFailed = false;

  if (this->capacity < copy.bytes)
  {
    mERROR_CHECK_GOTO(mAllocator_Reallocate(pAllocator, &text, copy.bytes), result, epilogue);
    this->capacity = copy.bytes;
  }
  else if(copy.bytes > 0)
  {
    mERROR_CHECK_GOTO(mAllocator_Copy(pAllocator, text, copy.text, copy.bytes), result, epilogue);
  }

  this->bytes = copy.bytes;
  this->count = copy.count;

  return *this;

epilogue:
  this->hasFailed = true;
  return *this;
}

mString & mString::operator=(mString &&move)
{
  this->~mString();
  new (this) mString(std::move(move));
  return *this;
}

size_t mString::Size() const
{
  return bytes;
}

size_t mString::Count() const
{
  return count;
}

mchar_t mString::operator[](const size_t index) const
{
  utf8proc_int32_t codePoint = UTF8PROC_ERROR_INVALIDOPTS;

  if (index >= count)
    return codePoint;

  size_t offset = 0;

  for (size_t i = 0; i <= index; ++i)
  {
    const ptrdiff_t characterSize = utf8proc_iterate((uint8_t *)this->text + offset, this->bytes - offset, &codePoint);

    if (characterSize < 0 || codePoint < 0)
      return codePoint;

    offset += (size_t)characterSize;
  }

  return codePoint;
}

mString mString::operator+(const mString &s) const
{
  mString ret;
  mResult result = mR_Success;

  mERROR_CHECK_GOTO(mAllocator_AllocateZero(this->pAllocator, &ret.text, this->bytes + s.bytes - 1), result, epilogue);
  ret.capacity = this->bytes + s.bytes - 1;
  mERROR_CHECK_GOTO(mAllocator_Copy(this->pAllocator, ret.text, this->text, this->bytes - 1), result, epilogue);
  mERROR_CHECK_GOTO(mAllocator_Copy(s.pAllocator, ret.text + this->bytes, s.text, s.bytes), result, epilogue);
  ret.bytes = ret.capacity;
  ret.count = s.count + count;
  ret.pAllocator = pAllocator;

  return ret;

epilogue:
  ret = mString();
  ret.hasFailed = true;
  return ret;
}

mString mString::operator+=(const mString &s)
{
  if (hasFailed || s.hasFailed)
    return *this;

  mResult result = mR_Success;

  if (bytes > 0 && s.bytes > 0)
  {
    if (capacity < bytes + s.bytes - 1)
    {
      if (capacity * 2 >= bytes + s.bytes - 1)
      {
        const size_t newCapacity = capacity *= 2;

        mERROR_CHECK_GOTO(mAllocator_Reallocate(pAllocator, &text, newCapacity), result, epilogue);
        capacity = newCapacity;
      }
      else
      {
        const size_t newCapacity = bytes + s.bytes - 1;

        mERROR_CHECK_GOTO(mAllocator_Reallocate(pAllocator, &text, newCapacity), result, epilogue);
        capacity = newCapacity;
      }
    }
  }
  else
  {
    if (s.bytes == 0)
      return *this;
    else
      return *this = s;
  }

  count += s.count;
  bytes += s.bytes - 1;

  mERROR_CHECK_GOTO(mAllocator_Copy(pAllocator, text + bytes, s.text, s.bytes), result, epilogue);

  return *this;

epilogue:
  hasFailed = true;
  return *this;
}

mString::operator std::string() const
{
  return std::string(text);
}

mString::operator std::wstring() const
{
  wchar_t *wtext = nullptr;
  mDefer<wchar_t *> cleanup;

  if (bytes < 1024)
  {
    cleanup = mDefer_Create(mFreePtrStack, &wtext);

    if (mFAILED(mAllocStackZero(&wtext, bytes * 2)))
      goto epilogue;
  }
  else
  {
    cleanup = mDefer_Create((std::function<void(wchar_t **)>)[&](wchar_t **) { mAllocator_FreePtr(nullptr, &wtext); }, (wchar_t **)nullptr);

    if (mFAILED(mAllocator_AllocateZero(nullptr, &wtext, bytes * 2)))
      goto epilogue;
  }

  mResult result;
  size_t i = 0;
  size_t wstrIndex = 0;

  while (i < bytes)
  {
    utf8proc_int32_t codePoint;
    ptrdiff_t characterSize;
    mERROR_IF_GOTO((characterSize = utf8proc_iterate((uint8_t *)this->text + i, this->bytes - i, &codePoint)) < 0, mR_InternalError, result, epilogue);
    mERROR_IF_GOTO(codePoint < 0, mR_InternalError, result, epilogue);
    i += (size_t)characterSize;

    if (characterSize > 2)
    {
      wtext[wstrIndex++] = (wchar_t)(codePoint & 0xFFFF);
      wtext[wstrIndex++] = (wchar_t)((codePoint >> 0x10) & 0xFFFF0000);
    }
    else
    {
      wtext[wstrIndex++] = (wchar_t)codePoint;
    }
  }

  wtext[wstrIndex] = (wchar_t)0;

  return std::wstring(wtext);

epilogue:
  return std::wstring();
}

const char * mString::c_str() const
{
  return text;
}

mFUNCTION(mString_Create, OUT mString *pString, char *text, IN OPTIONAL mAllocator *pAllocator /* = nullptr */)
{
  mFUNCTION_SETUP();

  mERROR_IF(text == nullptr, mR_ArgumentNull);

  mERROR_CHECK(mString_Create(pString, text, strlen(text), pAllocator));

  mRETURN_SUCCESS();
}

mFUNCTION(mString_Create, OUT mString *pString, char *text, const size_t size, IN OPTIONAL mAllocator *pAllocator)
{
  mFUNCTION_SETUP();

  mERROR_IF(text == nullptr, mR_ArgumentNull);
  
  if (pString->pAllocator == pAllocator)
  {
    *pString = mString();

    if (pString->capacity < size)
    {
      mERROR_CHECK(mAllocator_Reallocate(pString->pAllocator, &pString->text, size));
      pString->capacity = pString->bytes = size;
    }
  }
  else
  {
    pString->~mString();
    *pString = mString();

    pString->pAllocator = pAllocator;

    mERROR_CHECK(mAllocator_AllocateZero(pAllocator, &pString->text, size));
    pString->capacity = pString->bytes = size;
  }

  size_t offset = 0;
  pString->count = 0;

  while (offset < pString->bytes)
  {
    utf8proc_int32_t codePoint;
    ptrdiff_t characterSize;
    mERROR_IF((characterSize = utf8proc_iterate((uint8_t *)pString->text + offset, pString->bytes - offset, &codePoint)) < 0, mR_InternalError);
    mERROR_IF(codePoint < 0, mR_InternalError);
    offset += (size_t)characterSize;
    pString->count++;
  }

  mRETURN_SUCCESS();
}

mFUNCTION(mString_Destroy, IN_OUT mString * pString)
{
  mFUNCTION_SETUP();

  mERROR_IF(pString == nullptr, mR_ArgumentNull);

  if (pString->text != nullptr)
    mERROR_CHECK(mAllocator_FreePtr(pString->pAllocator, &pString->text));

  pString->capacity = 0;
  pString->bytes = 0;
  pString->count = 0;
  pString->hasFailed = false;
  pString->pAllocator = nullptr;

  mRETURN_SUCCESS();
}

mFUNCTION(mString_GetByteSize, const mString &string, OUT size_t *pSize)
{
  mFUNCTION_SETUP();

  mERROR_IF(pSize == nullptr, mR_ArgumentNull);

  *pSize = string.bytes;

  mRETURN_SUCCESS();
}

mFUNCTION(mString_GetCount, const mString &string, OUT size_t *pLength)
{
  mFUNCTION_SETUP();

  mERROR_IF(pLength == nullptr, mR_ArgumentNull);

  *pLength = string.count;

  mRETURN_SUCCESS();
}

mFUNCTION(mString_ToWideString, const mString &string, std::wstring *pWideString)
{
  mFUNCTION_SETUP();

  wchar_t *wtext = nullptr;
  mDefer<wchar_t *> cleanup;

  if (string.bytes < 1024)
  {
    cleanup = mDefer_Create(mFreePtrStack, &wtext);
    mERROR_CHECK(mAllocStackZero(&wtext, string.bytes * 2));
  }
  else
  {
    cleanup = mDefer_Create((std::function<void(wchar_t **)>)[&](wchar_t **) { mAllocator_FreePtr(nullptr, &wtext); }, (wchar_t **)nullptr);
    mERROR_CHECK(mAllocator_AllocateZero(nullptr, &wtext, string.bytes * 2));
  }

  size_t offset = 0;
  size_t wstrIndex = 0;

  while (offset < string.bytes)
  {
    utf8proc_int32_t codePoint;
    ptrdiff_t characterSize;
    mERROR_IF((characterSize = utf8proc_iterate((uint8_t *)string.text + offset, string.bytes - offset, &codePoint)) < 0, mR_InternalError);
    mERROR_IF(codePoint < 0, mR_InternalError);
    offset += (size_t)characterSize;

    if (characterSize > 2)
    {
      wtext[wstrIndex++] = (wchar_t)(codePoint & 0xFFFF);
      wtext[wstrIndex++] = (wchar_t)((codePoint >> 0x10) & 0xFFFF0000);
    }
    else
    {
      wtext[wstrIndex++] = (wchar_t)codePoint;
    }
  }

  wtext[wstrIndex] = (wchar_t)0;

  *pWideString = std::wstring(wtext);

  mRETURN_SUCCESS();
}

mFUNCTION(mString_Substring, const mString & text, OUT mString * pSubstring, const size_t startCharacter)
{
  mFUNCTION_SETUP();

  mERROR_IF(pSubstring == nullptr, mR_ArgumentNull);
  mERROR_IF(startCharacter >= text.count, mR_IndexOutOfBounds);

  mERROR_CHECK(mString_Substring(text, pSubstring, startCharacter, text.count - startCharacter - 1));

  mRETURN_SUCCESS();
}

mFUNCTION(mString_Substring, const mString & text, OUT mString * pSubstring, const size_t startCharacter, const size_t length)
{
  mFUNCTION_SETUP();

  mERROR_IF(pSubstring == nullptr, mR_ArgumentNull);
  mERROR_IF(startCharacter >= text.count || startCharacter + length >= text.count, mR_IndexOutOfBounds);
  
  size_t byteOffset = 0;
  size_t character = 0;

  while (character < startCharacter)
  {
    utf8proc_int32_t codePoint;
    ptrdiff_t characterSize;
    mERROR_IF((characterSize = utf8proc_iterate((uint8_t *)text.text + byteOffset, text.bytes - byteOffset, &codePoint)) < 0, mR_InternalError);
    mERROR_IF(codePoint < 0, mR_InternalError);
    byteOffset += (size_t)characterSize;
    character++;
  }

  *pSubstring = mString();
  char utf8char[5] = { 0, 0, 0, 0, 0 };

  while (character - startCharacter < length)
  {
    utf8proc_int32_t codePoint;
    ptrdiff_t characterSize;
    mERROR_IF((characterSize = utf8proc_iterate((uint8_t *)text.text + byteOffset, text.bytes - byteOffset, &codePoint)) < 0, mR_InternalError);
    mERROR_IF(codePoint < 0, mR_InternalError);
    byteOffset += (size_t)characterSize;
    character++;
    *(utf8proc_int32_t *)utf8char = codePoint;
    mERROR_CHECK(mString_Append(*pSubstring, utf8char));
  }

  mRETURN_SUCCESS();
}

mFUNCTION(mString_Append, mString &text, const mString &appendedText)
{
  mFUNCTION_SETUP();

  if (text.bytes > 0 && appendedText.bytes > 0)
  {
    if (text.capacity < text.bytes + appendedText.bytes - 1)
    {
      if (text.capacity * 2 >= text.bytes + appendedText.bytes - 1)
      {
        const size_t newCapacity = text.capacity *= 2;

        mERROR_CHECK(mAllocator_Reallocate(text.pAllocator, &text.text, newCapacity));
        text.capacity = newCapacity;
      }
      else
      {
        const size_t newCapacity = text.bytes + appendedText.bytes - 1;

        mERROR_CHECK(mAllocator_Reallocate(text.pAllocator, &text.text, newCapacity));
        text.capacity = newCapacity;
      }
    }
  }
  else
  {
    if (appendedText.bytes == 0)
    {
      mRETURN_SUCCESS();
    }
    else
    {
      text.hasFailed = false;

      if (text.capacity < text.bytes)
      {
        mERROR_CHECK(mAllocator_Reallocate(text.pAllocator, &text.text, appendedText.bytes));
        text.capacity = appendedText.bytes;
      }
      else if (appendedText.bytes > 0)
      {
        mERROR_CHECK(mAllocator_Copy(text.pAllocator, text.text, appendedText.text, appendedText.bytes));
      }

      text.bytes = appendedText.bytes;
      text.count = appendedText.count;

      mRETURN_SUCCESS();
    }
  }

  text.count += appendedText.count;
  text.bytes += appendedText.bytes - 1;

  mERROR_CHECK(mAllocator_Copy(text.pAllocator, text.text + text.bytes, appendedText.text, appendedText.bytes));

  mRETURN_SUCCESS();
}
