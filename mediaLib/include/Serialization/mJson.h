#ifndef mJson_h__
#define mJson_h__

#include "mediaLib.h"

struct mJsonWriter;

mFUNCTION(mJsonWriter_Create, OUT mPtr<mJsonWriter> *pJsonWriter, IN mAllocator *pAllocator);
mFUNCTION(mJsonWriter_Destroy, IN_OUT mPtr<mJsonWriter> *pJsonWriter);

mFUNCTION(mJsonWriter_BeginUnnamed, mPtr<mJsonWriter> &jsonWriter);
mFUNCTION(mJsonWriter_EndUnnamed, mPtr<mJsonWriter> &jsonWriter);

mFUNCTION(mJsonWriter_BeginArray, mPtr<mJsonWriter> &jsonWriter, const char *name);
mFUNCTION(mJsonWriter_BeginArray, mPtr<mJsonWriter> &jsonWriter);
mFUNCTION(mJsonWriter_EndArray, mPtr<mJsonWriter> &jsonWriter);

mFUNCTION(mJsonWriter_BeginNamed, mPtr<mJsonWriter> &jsonWriter, const char *name);
mFUNCTION(mJsonWriter_EndNamed, mPtr<mJsonWriter> &jsonWriter);
mFUNCTION(mJsonWriter_AddValue, mPtr<mJsonWriter> &jsonWriter, const char *name, const double_t value);
mFUNCTION(mJsonWriter_AddValue, mPtr<mJsonWriter> &jsonWriter, const char *name, const mString &value);
mFUNCTION(mJsonWriter_AddValue, mPtr<mJsonWriter> &jsonWriter, const char *name, const char *value);
mFUNCTION(mJsonWriter_AddValue, mPtr<mJsonWriter> &jsonWriter, const char *name, const bool value);
mFUNCTION(mJsonWriter_AddValue, mPtr<mJsonWriter> &jsonWriter, const char *name, nullptr_t);

mFUNCTION(mJsonWriter_AddValueX, mPtr<mJsonWriter> &jsonWriter, const char *name, const mVec2f &value);
mFUNCTION(mJsonWriter_AddValueX, mPtr<mJsonWriter> &jsonWriter, const char *name, const mVec3f &value);
mFUNCTION(mJsonWriter_AddValueX, mPtr<mJsonWriter> &jsonWriter, const char *name, const mVec4f &value);
mFUNCTION(mJsonWriter_AddValueX, mPtr<mJsonWriter> &jsonWriter, const char *name, const mVector &value);

mFUNCTION(mJsonWriter_AddArrayValue, mPtr<mJsonWriter> &jsonWriter, const double_t value);
mFUNCTION(mJsonWriter_AddArrayValue, mPtr<mJsonWriter> &jsonWriter, const mString &value);
mFUNCTION(mJsonWriter_AddArrayValue, mPtr<mJsonWriter> &jsonWriter, const char *value);
mFUNCTION(mJsonWriter_AddArrayValue, mPtr<mJsonWriter> &jsonWriter, const bool value);
mFUNCTION(mJsonWriter_AddArrayValue, mPtr<mJsonWriter> &jsonWriter, nullptr_t);

mFUNCTION(mJsonWriter_AddArrayValueX, mPtr<mJsonWriter> &jsonWriter, const mVec2f &value);
mFUNCTION(mJsonWriter_AddArrayValueX, mPtr<mJsonWriter> &jsonWriter, const mVec3f &value);
mFUNCTION(mJsonWriter_AddArrayValueX, mPtr<mJsonWriter> &jsonWriter, const mVec4f &value);
mFUNCTION(mJsonWriter_AddArrayValueX, mPtr<mJsonWriter> &jsonWriter, const mVector &value);

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
mFUNCTION(mJsonReader_ReadNamedValue, mPtr<mJsonReader> &jsonReader, const char *name, OUT mVec2f *pVec2);
mFUNCTION(mJsonReader_ReadNamedValue, mPtr<mJsonReader> &jsonReader, const char *name, OUT mVec3f *pVec3);
mFUNCTION(mJsonReader_ReadNamedValue, mPtr<mJsonReader> &jsonReader, const char *name, OUT mVec4f *pVec4);
mFUNCTION(mJsonReader_ReadNamedNull, mPtr<mJsonReader> &jsonReader, const char *name);

mFUNCTION(mJsonReader_ReadArrayValue, mPtr<mJsonReader> &jsonReader, const size_t index, OUT mString *pString);
mFUNCTION(mJsonReader_ReadArrayValue, mPtr<mJsonReader> &jsonReader, const size_t index, OUT double_t *pValue);
mFUNCTION(mJsonReader_ReadArrayValue, mPtr<mJsonReader> &jsonReader, const size_t index, OUT bool *pValue);
mFUNCTION(mJsonReader_ReadArrayValue, mPtr<mJsonReader> &jsonReader, const size_t index, OUT mVec2f *pVec2);
mFUNCTION(mJsonReader_ReadArrayValue, mPtr<mJsonReader> &jsonReader, const size_t index, OUT mVec3f *pVec3);
mFUNCTION(mJsonReader_ReadArrayValue, mPtr<mJsonReader> &jsonReader, const size_t index, OUT mVec4f *pVec4);
mFUNCTION(mJsonReader_ReadArrayNull, mPtr<mJsonReader> &jsonReader, const size_t index);

mFUNCTION(mJsonReader_GetCurrentValueType, mPtr<mJsonReader> &jsonReader, OUT mJsonReader_EntryType *pEntryType);

mFUNCTION(mJsonReader_ReadCurrentValue, mPtr<mJsonReader> &jsonReader, OUT mString *pString);
mFUNCTION(mJsonReader_ReadCurrentValue, mPtr<mJsonReader> &jsonReader, OUT double_t *pValue);
mFUNCTION(mJsonReader_ReadCurrentValue, mPtr<mJsonReader> &jsonReader, OUT bool *pValue);

mFUNCTION(mJsonReader_ArrayForEach, mPtr<mJsonReader> &jsonReader, const std::function<mResult(mPtr<mJsonReader> &, const size_t index)> &callback);

#endif // mJson_h__
