// Copyright 2018 Christoph Stiller
// 
// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files(the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions :
// 
// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
// 
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "mString.h"
#include "utf8proc.h"

mString::mString()
  : text(nullptr),
  pAllocator(nullptr),
  bytes(0),
  length(0)
{ }

mString::mString(char *text, IN OPTIONAL mAllocator *pAllocator)
{
  mResult result = mR_Success;

  this->pAllocator = pAllocator;
  this->bytes = strlen(text);

  mERROR_CHECK_GOTO(mAllocator_AllocateZero(this->pAllocator, &this->text, this->bytes), result, epilogue);
  this->capacity = bytes;
  mERROR_CHECK_GOTO(mAllocator_Copy(this->pAllocator, this->text, text, this->bytes), result, epilogue);

  size_t i = 0;
  this->length = 0;
  
  while (i < bytes)
  {
    utf8proc_int32_t codePoint;
    ptrdiff_t position;
    mERROR_IF_GOTO((position = utf8proc_iterate((uint8_t *)this->text + i, this->bytes, &codePoint)) < 0, mR_InternalError, result, epilogue);
    mERROR_IF_GOTO(codePoint < 0, mR_InternalError, result, epilogue);
    i += (size_t)position;
    this->length++;
  }

  return;

epilogue:
  this->~mString();
}

mString::~mString()
{
  if (text != nullptr)
    mAllocator_FreePtr(pAllocator, &text);

  text = nullptr;
  pAllocator = nullptr;
  bytes = 0;
  length = 0;
}

size_t mString::Size() const
{
  return bytes;
}

size_t mString::Length() const
{
  return length;
}
