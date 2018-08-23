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
  capacity(0)
{ }

mString::mString(char * text, const size_t size, IN OPTIONAL mAllocator * pAllocator)
{
  mResult result = mR_Success;

  this->pAllocator = pAllocator;
  this->bytes = size;

  mERROR_CHECK_GOTO(mAllocator_AllocateZero(this->pAllocator, &this->text, this->bytes), result, epilogue);
  this->capacity = bytes;
  mERROR_CHECK_GOTO(mAllocator_Copy(this->pAllocator, this->text, text, this->bytes), result, epilogue);

  size_t i = 0;
  this->count = 0;

  while (i < bytes)
  {
    utf8proc_int32_t codePoint;
    ptrdiff_t characterSize;
    mERROR_IF_GOTO((characterSize = utf8proc_iterate((uint8_t *)this->text + i, this->bytes - i, &codePoint)) < 0, mR_InternalError, result, epilogue);
    mERROR_IF_GOTO(codePoint < 0, mR_InternalError, result, epilogue);
    i += (size_t)characterSize;
    this->count++;
  }

  return;

epilogue:
  this->~mString();
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
}

mString::mString(const mString &copy) :
  text(nullptr),
  pAllocator(nullptr),
  bytes(0),
  count(0),
  capacity(0)
{
  mResult result = mR_Success;

  pAllocator = copy.pAllocator;

  mERROR_CHECK_GOTO(mAllocator_AllocateZero(pAllocator, &text, copy.bytes), result, epilogue);
  capacity = bytes = copy.bytes;
  count = copy.count;

  mERROR_CHECK_GOTO(mAllocator_Copy(pAllocator, text, copy.text, bytes), result, epilogue);

  return;

epilogue:
  this->~mString();
}

mString::mString(mString &&move) :
  text(move.text),
  pAllocator(move.pAllocator),
  bytes(move.bytes),
  count(move.count),
  capacity(move.capacity)
{
  move.text = nullptr;
  move.count = 0;
  move.bytes = 0;
  move.pAllocator = nullptr;
  move.capacity = 0;
}

mString & mString::operator=(const mString &copy)
{
  this->~mString();
  new (this) mString(copy);
}

mString & mString::operator=(mString &&move)
{
  this->~mString();
  new (this) mString(std::move(move));
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

mString::operator std::string() const
{
  return std::string(text);
}

mString::operator std::wstring() const
{
  wchar_t *wtext = nullptr;

  if (bytes < 1024)
  {
    mDEFER_DESTRUCTION(&wtext, mFreePtrStack);

    if (mFAILED(mAllocStackZero(&wtext, bytes * 2)))
      return std::wstring();
  }
  else
  {
    mDEFER(mAllocator_FreePtr(nullptr, &wtext));
  }
}

const char * mString::c_str() const
{
  return text;
}
