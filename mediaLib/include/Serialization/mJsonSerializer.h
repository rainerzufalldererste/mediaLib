// Copyright 2018 Christoph Stiller
// 
// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files(the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions :
// 
// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
// 
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef mJsonSerializer_h__
#define mJsonSerializer_h__

#include "default.h"

struct mJsonWriter;

mFUNCTION(mJsonWriter_Create, OUT mPtr<mJsonWriter> *pJsonWriter, IN mAllocator *pAllocator);
mFUNCTION(mJsonWriter_Destroy, IN_OUT mPtr<mJsonWriter> *pJsonWriter);

mFUNCTION(mJsonWriter_StartContainer, mPtr<mJsonWriter> &jsonWriter, const mString &containerName);
mFUNCTION(mJsonWriter_EndContainer, mPtr<mJsonWriter> &jsonWriter);

mFUNCTION(mJsonWriter_WriteValue, mPtr<mJsonWriter> &jsonWriter, const mString &name, const size_t value);
mFUNCTION(mJsonWriter_WriteValue, mPtr<mJsonWriter> &jsonWriter, const mString &name, const int64_t value);
mFUNCTION(mJsonWriter_WriteValue, mPtr<mJsonWriter> &jsonWriter, const mString &name, const double_t value);
mFUNCTION(mJsonWriter_WriteValue, mPtr<mJsonWriter> &jsonWriter, const mString &name, const bool value);
mFUNCTION(mJsonWriter_WriteValue, mPtr<mJsonWriter> &jsonWriter, const mString &name, const mString &value);

#endif // mJsonSerializer_h__
