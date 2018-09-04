// Copyright 2018 Christoph Stiller
// 
// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files(the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions :
// 
// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
// 
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef mJson_h__
#define mJson_h__

#include "default.h"

struct mJsonWriter;

mFUNCTION(mJsonWriter_Create, OUT mPtr<mJsonWriter> *pJsonWriter, IN mAllocator *pAllocator);
mFUNCTION(mJsonWriter_Destroy, IN_OUT mPtr<mJsonWriter> *pJsonWriter);

mFUNCTION(mJsonWriter_BeginUnnamed, mPtr<mJsonWriter> &jsonWriter);
mFUNCTION(mJsonWriter_EndUnnamed, mPtr<mJsonWriter> &jsonWriter);

mFUNCTION(mJsonWriter_BeginArray, mPtr<mJsonWriter> &jsonWriter, const char *name);
mFUNCTION(mJsonWriter_EndArray, mPtr<mJsonWriter> &jsonWriter);

mFUNCTION(mJsonWriter_BeginNamed, mPtr<mJsonWriter> &jsonWriter, const char *name);
mFUNCTION(mJsonWriter_EndNamed, mPtr<mJsonWriter> &jsonWriter);

mFUNCTION(mJsonWriter_AddValue, mPtr<mJsonWriter> &jsonWriter, const char *name, const double_t value);
mFUNCTION(mJsonWriter_AddValue, mPtr<mJsonWriter> &jsonWriter, const char *name, const mString &value);
mFUNCTION(mJsonWriter_AddValue, mPtr<mJsonWriter> &jsonWriter, const char *name, const char *value);
mFUNCTION(mJsonWriter_AddValue, mPtr<mJsonWriter> &jsonWriter, const char *name, const bool value);
mFUNCTION(mJsonWriter_AddValue, mPtr<mJsonWriter> &jsonWriter, const char *name, nullptr_t);

mFUNCTION(mJsonWriter_AddArrayValue, mPtr<mJsonWriter> &jsonWriter, const double_t value);
mFUNCTION(mJsonWriter_AddArrayValue, mPtr<mJsonWriter> &jsonWriter, const mString &value);
mFUNCTION(mJsonWriter_AddArrayValue, mPtr<mJsonWriter> &jsonWriter, const char *value);
mFUNCTION(mJsonWriter_AddArrayValue, mPtr<mJsonWriter> &jsonWriter, const bool value);
mFUNCTION(mJsonWriter_AddArrayValue, mPtr<mJsonWriter> &jsonWriter, nullptr_t);

mFUNCTION(mJsonWriter_ToString, mPtr<mJsonWriter> &jsonWriter, OUT mString *pString);
mFUNCTION(mJsonWriter_ToFile, mPtr<mJsonWriter> &jsonWriter, const mString &filename);

struct mJsonReader;

enum mJsonReader_EntryType
{
  mJR_ET_Invalid,
  mJR_ET_Object,
  mJR_ET_Array,
  mJR_ET_Number,
  mJR_ET_String,
  mJR_ET_Bool,
  mJR_ET_Null,
};

mFUNCTION(mJsonReader_CreateFromString, OUT mPtr<mJsonReader> *pJsonReader, IN mAllocator *pAllocator, const mString &text);
mFUNCTION(mJsonReader_CreateFromFile, OUT mPtr<mJsonReader> *pJsonReader, IN mAllocator *pAllocator, const mString &filename);
mFUNCTION(mJsonReader_Destroy, IN_OUT mPtr<mJsonReader> *pJsonReader);

mFUNCTION(mJsonReader_StepIntoNamed, mPtr<mJsonReader> &jsonReader, const char *name);
mFUNCTION(mJsonReader_ExitNamed, mPtr<mJsonReader> &jsonReader);

mFUNCTION(mJsonReader_StepIntoArrayItem, mPtr<mJsonReader> &jsonReader, const size_t index);
mFUNCTION(mJsonReader_ExitArrayItem, mPtr<mJsonReader> &jsonReader);

mFUNCTION(mJsonReader_StepIntoArray, mPtr<mJsonReader> &jsonReader, const char *name);
mFUNCTION(mJsonReader_ExitArray, mPtr<mJsonReader> &jsonReader);

mFUNCTION(mJsonReader_GetArrayCount, mPtr<mJsonReader> &jsonReader, OUT size_t *pCount);

mFUNCTION(mJsonReader_ReadNamedValue, mPtr<mJsonReader> &jsonReader, const char *name, OUT mString *pString);
mFUNCTION(mJsonReader_ReadNamedValue, mPtr<mJsonReader> &jsonReader, const char *name, OUT double_t *pValue);
mFUNCTION(mJsonReader_ReadNamedValue, mPtr<mJsonReader> &jsonReader, const char *name, OUT bool *pValue);
mFUNCTION(mJsonReader_ReadNamedNull, mPtr<mJsonReader> &jsonReader, const char *name);

mFUNCTION(mJsonReader_ReadArrayValue, mPtr<mJsonReader> &jsonReader, const size_t index, OUT mString *pString);
mFUNCTION(mJsonReader_ReadArrayValue, mPtr<mJsonReader> &jsonReader, const size_t index, OUT double_t *pValue);
mFUNCTION(mJsonReader_ReadArrayValue, mPtr<mJsonReader> &jsonReader, const size_t index, OUT bool *pValue);
mFUNCTION(mJsonReader_ReadArrayNull, mPtr<mJsonReader> &jsonReader, const size_t index);

mFUNCTION(mJsonReader_GetCurrentValueType, mPtr<mJsonReader> &jsonReader, OUT mJsonReader_EntryType *pEntryType);

mFUNCTION(mJsonReader_ReadCurrentValue, mPtr<mJsonReader> &jsonReader, OUT mString *pString);
mFUNCTION(mJsonReader_ReadCurrentValue, mPtr<mJsonReader> &jsonReader, OUT double_t *pValue);
mFUNCTION(mJsonReader_ReadCurrentValue, mPtr<mJsonReader> &jsonReader, OUT bool *pValue);

mFUNCTION(mJsonReader_ArrayForEach, mPtr<mJsonReader> &jsonReader, const std::function<mResult(mPtr<mJsonReader> &, const size_t index)> &callback);

#endif // mJson_h__
