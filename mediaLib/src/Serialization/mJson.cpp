// Copyright 2018 Christoph Stiller
// 
// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files(the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions :
// 
// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
// 
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "mJson.h"
#include "mQueue.h"
#include "mFile.h"

#include "cJSON/cJSON.h"
#include "cJSON/cJSON_Utils.h"

#include "cJSON/cJSON.c"

//////////////////////////////////////////////////////////////////////////

mFUNCTION(mJson_CheckError_Internal, IN cJSON *pJson);

//////////////////////////////////////////////////////////////////////////

enum mJsonWriterEntryType : uint8_t
{
  Object,
  Array,
};

struct mJsonWriter
{
  mPtr<mQueue<cJSON *>> currentBlock;
  mPtr<mQueue<mJsonWriterEntryType>> currentType;
  bool finalized;
};

//////////////////////////////////////////////////////////////////////////

mFUNCTION(mJsonWriter_Destroy_Internal, IN mJsonWriter *pJsonWriter);
mFUNCTION(mJsonWriter_GetLastInQueue_Internal, mPtr<mJsonWriter> &jsonWriter, OUT cJSON **ppJsonElement);
mFUNCTION(mJsonWriter_GetLastInTypeQueue_Internal, mPtr<mJsonWriter> &jsonWriter, OUT mJsonWriterEntryType *pJsonElement);
mFUNCTION(mJsonWriter_PopQueue_Internal, mPtr<mJsonWriter> &jsonWriter);
mFUNCTION(mJsonWriter_PushQueue_Internal, mPtr<mJsonWriter> &jsonWriter, IN cJSON *pJsonElement, const mJsonWriterEntryType type);

//////////////////////////////////////////////////////////////////////////

mFUNCTION(mJsonWriter_Create, OUT mPtr<mJsonWriter> *pJsonWriter, IN mAllocator * pAllocator)
{
  mFUNCTION_SETUP();

  mERROR_IF(pJsonWriter == nullptr, mR_ArgumentNull);

  mERROR_CHECK(mSharedPointer_Allocate(pJsonWriter, pAllocator, (std::function<void(mJsonWriter *)>)[](mJsonWriter *pData) { mJsonWriter_Destroy_Internal(pData); }, 1));

  mERROR_CHECK(mQueue_Create(&(*pJsonWriter)->currentBlock, pAllocator));
  mERROR_CHECK(mQueue_Create(&(*pJsonWriter)->currentType, pAllocator));

  mERROR_CHECK(mJsonWriter_BeginUnnamed(*pJsonWriter));

  mRETURN_SUCCESS();
}

mFUNCTION(mJsonWriter_Destroy, IN_OUT mPtr<mJsonWriter> *pJsonWriter)
{
  mFUNCTION_SETUP();

  mERROR_CHECK(mSharedPointer_Destroy(pJsonWriter));

  mRETURN_SUCCESS();
}

mFUNCTION(mJsonWriter_BeginUnnamed, mPtr<mJsonWriter> &jsonWriter)
{
  mFUNCTION_SETUP();

  mERROR_IF(jsonWriter == nullptr, mR_ArgumentNull);
  mERROR_IF(jsonWriter->finalized, mR_ResourceStateInvalid);

  cJSON *pJson = nullptr;
  mERROR_CHECK(mJsonWriter_GetLastInQueue_Internal(jsonWriter, &pJson));

  if (pJson == nullptr)
  {
    cJSON *pChild = cJSON_CreateObject();
    mDEFER_DESTRUCTION_ON_ERROR(pChild, (std::function<void (cJSON *)>)cJSON_Delete);
    mERROR_CHECK(mJson_CheckError_Internal(pChild));

    mERROR_CHECK(mJsonWriter_PushQueue_Internal(jsonWriter, pChild, Object));
  }
  else
  {
    cJSON *pChild = cJSON_CreateObject();
    mDEFER_DESTRUCTION_ON_ERROR(pChild, (std::function<void(cJSON *)>)cJSON_Delete);
    mERROR_CHECK(mJson_CheckError_Internal(pChild));

    mJsonWriterEntryType lastType;
    mERROR_CHECK(mJsonWriter_GetLastInTypeQueue_Internal(jsonWriter, &lastType));

    mERROR_CHECK(mJsonWriter_PushQueue_Internal(jsonWriter, pChild, Object));

    switch (lastType)
    {
    case Array:
      cJSON_AddItemToArray(pJson, pChild);
      break;

    default:
      mRETURN_RESULT(mR_ResourceStateInvalid);
      break;
    }
  }

  mRETURN_SUCCESS();
}

mFUNCTION(mJsonWriter_EndUnnamed, mPtr<mJsonWriter> &jsonWriter)
{
  mFUNCTION_SETUP();

  mERROR_IF(jsonWriter == nullptr, mR_ArgumentNull);

  mERROR_CHECK(mJsonWriter_PopQueue_Internal(jsonWriter));

  mRETURN_SUCCESS();
}

mFUNCTION(mJsonWriter_BeginArray, mPtr<mJsonWriter> &jsonWriter, const char *name)
{
  mFUNCTION_SETUP();

  mERROR_IF(jsonWriter == nullptr, mR_ArgumentNull);
  mERROR_IF(jsonWriter->finalized, mR_ResourceStateInvalid);

  cJSON *pJson = nullptr;
  mERROR_CHECK(mJsonWriter_GetLastInQueue_Internal(jsonWriter, &pJson));
  mERROR_IF(pJson == nullptr, mR_NotInitialized);

  cJSON *pChild = cJSON_CreateArray();
  mDEFER_DESTRUCTION_ON_ERROR(pChild, (std::function<void(cJSON *)>)cJSON_Delete);
  mERROR_CHECK(mJson_CheckError_Internal(pChild));

  mJsonWriterEntryType lastType;
  mERROR_CHECK(mJsonWriter_GetLastInTypeQueue_Internal(jsonWriter, &lastType));

  mERROR_CHECK(mJsonWriter_PushQueue_Internal(jsonWriter, pChild, Array));

  switch (lastType)
  {
  case Object:
    mERROR_IF(name == nullptr, mR_ArgumentNull);
    cJSON_AddItemToObject(pJson, name, pChild);
    break;

  default:
    mRETURN_RESULT(mR_ResourceStateInvalid);
    break;
  }

  mRETURN_SUCCESS();
}

mFUNCTION(mJsonWriter_EndArray, mPtr<mJsonWriter> &jsonWriter)
{
  mFUNCTION_SETUP();

  mERROR_CHECK(mJsonWriter_EndUnnamed(jsonWriter));

  mRETURN_SUCCESS();
}

mFUNCTION(mJsonWriter_BeginNamed, mPtr<mJsonWriter> &jsonWriter, const char *name)
{
  mFUNCTION_SETUP();

  mERROR_IF(jsonWriter == nullptr, mR_ArgumentNull);
  mERROR_IF(jsonWriter->finalized, mR_ResourceStateInvalid);

  cJSON *pJson = nullptr;
  mERROR_CHECK(mJsonWriter_GetLastInQueue_Internal(jsonWriter, &pJson));
  mERROR_IF(pJson == nullptr, mR_NotInitialized);

  cJSON *pChild = cJSON_CreateObject();
  mDEFER_DESTRUCTION_ON_ERROR(pChild, (std::function<void(cJSON *)>)cJSON_Delete);
  mERROR_CHECK(mJson_CheckError_Internal(pChild));

  mJsonWriterEntryType lastType;
  mERROR_CHECK(mJsonWriter_GetLastInTypeQueue_Internal(jsonWriter, &lastType));

  mERROR_CHECK(mJsonWriter_PushQueue_Internal(jsonWriter, pChild, Object));

  switch (lastType)
  {
  case Object:
    mERROR_IF(name == nullptr, mR_ArgumentNull);
    cJSON_AddItemToObject(pJson, name, pChild);
    break;

  default:
    mRETURN_RESULT(mR_ResourceStateInvalid);
    break;
  }

  mRETURN_SUCCESS();
}

mFUNCTION(mJsonWriter_EndNamed, mPtr<mJsonWriter> &jsonWriter)
{
  mFUNCTION_SETUP();

  mERROR_CHECK(mJsonWriter_EndUnnamed(jsonWriter));

  mRETURN_SUCCESS();
}

mFUNCTION(mJsonWriter_AddValue, mPtr<mJsonWriter> &jsonWriter, const char *name, const double_t value)
{
  mFUNCTION_SETUP();

  mERROR_IF(jsonWriter == nullptr || name == nullptr, mR_ArgumentNull);
  mERROR_IF(jsonWriter->finalized, mR_ResourceStateInvalid);

  cJSON *pJson = nullptr;
  mERROR_CHECK(mJsonWriter_GetLastInQueue_Internal(jsonWriter, &pJson));
  mERROR_IF(pJson == nullptr, mR_NotInitialized);

  mJsonWriterEntryType lastType;
  mERROR_CHECK(mJsonWriter_GetLastInTypeQueue_Internal(jsonWriter, &lastType));

  switch (lastType)
  {
  case Object:
    mERROR_IF(nullptr == cJSON_AddNumberToObject(pJson, name, value), mR_InternalError);
    break;

  default:
    mERROR_CHECK(mJsonWriter_BeginUnnamed(jsonWriter));
    mERROR_CHECK(mJsonWriter_AddValue(jsonWriter, name, value));
    mERROR_CHECK(mJsonWriter_EndUnnamed(jsonWriter));
    break;
  }

  mRETURN_SUCCESS();
}

mFUNCTION(mJsonWriter_AddValue, mPtr<mJsonWriter> &jsonWriter, const char *name, const mString &value)
{
  mFUNCTION_SETUP();

  mERROR_IF(jsonWriter == nullptr || name == nullptr, mR_ArgumentNull);
  mERROR_IF(jsonWriter->finalized, mR_ResourceStateInvalid);

  cJSON *pJson = nullptr;
  mERROR_CHECK(mJsonWriter_GetLastInQueue_Internal(jsonWriter, &pJson));
  mERROR_IF(pJson == nullptr, mR_NotInitialized);

  mJsonWriterEntryType lastType;
  mERROR_CHECK(mJsonWriter_GetLastInTypeQueue_Internal(jsonWriter, &lastType));

  switch (lastType)
  {
  case Object:
    mERROR_IF(nullptr == cJSON_AddStringToObject(pJson, name, value.c_str()), mR_InternalError);
    break;

  default:
    mERROR_CHECK(mJsonWriter_BeginUnnamed(jsonWriter));
    mERROR_CHECK(mJsonWriter_AddValue(jsonWriter, name, value));
    mERROR_CHECK(mJsonWriter_EndUnnamed(jsonWriter));
    break;
  }

  mRETURN_SUCCESS();
}

mFUNCTION(mJsonWriter_AddValue, mPtr<mJsonWriter>& jsonWriter, const char *name, const char *value)
{
  mFUNCTION_SETUP();

  mString string;
  mERROR_CHECK(mString_Create(&string, value));

  mERROR_CHECK(mJsonWriter_AddValue(jsonWriter, name, string));

  mRETURN_SUCCESS();
}

mFUNCTION(mJsonWriter_AddValue, mPtr<mJsonWriter> &jsonWriter, const char *name, const bool value)
{
  mFUNCTION_SETUP();

  mERROR_IF(jsonWriter == nullptr || name == nullptr, mR_ArgumentNull);
  mERROR_IF(jsonWriter->finalized, mR_ResourceStateInvalid);

  cJSON *pJson = nullptr;
  mERROR_CHECK(mJsonWriter_GetLastInQueue_Internal(jsonWriter, &pJson));
  mERROR_IF(pJson == nullptr, mR_NotInitialized);

  mJsonWriterEntryType lastType;
  mERROR_CHECK(mJsonWriter_GetLastInTypeQueue_Internal(jsonWriter, &lastType));

  switch (lastType)
  {
  case Object:
    mERROR_IF(nullptr == cJSON_AddBoolToObject(pJson, name, value), mR_InternalError);
    break;

  default:
    mERROR_CHECK(mJsonWriter_BeginUnnamed(jsonWriter));
    mERROR_CHECK(mJsonWriter_AddValue(jsonWriter, name, value));
    mERROR_CHECK(mJsonWriter_EndUnnamed(jsonWriter));
    break;
  }

  mRETURN_SUCCESS();
}

mFUNCTION(mJsonWriter_AddValue, mPtr<mJsonWriter> &jsonWriter, const char *name, nullptr_t)
{
  mFUNCTION_SETUP();

  mERROR_IF(jsonWriter == nullptr || name == nullptr, mR_ArgumentNull);
  mERROR_IF(jsonWriter->finalized, mR_ResourceStateInvalid);

  cJSON *pJson = nullptr;
  mERROR_CHECK(mJsonWriter_GetLastInQueue_Internal(jsonWriter, &pJson));
  mERROR_IF(pJson == nullptr, mR_NotInitialized);

  mJsonWriterEntryType lastType;
  mERROR_CHECK(mJsonWriter_GetLastInTypeQueue_Internal(jsonWriter, &lastType));

  switch (lastType)
  {
  case Object:
    mERROR_IF(nullptr == cJSON_AddNullToObject(pJson, name), mR_InternalError);
    break;

  default:
    mERROR_CHECK(mJsonWriter_BeginUnnamed(jsonWriter));
    mERROR_CHECK(mJsonWriter_AddValue(jsonWriter, name, nullptr));
    mERROR_CHECK(mJsonWriter_EndUnnamed(jsonWriter));
    break;
  }

  mRETURN_SUCCESS();
}

mFUNCTION(mJsonWriter_AddValueX, mPtr<mJsonWriter> &jsonWriter, const char *name, const mVec2f &value)
{
  mFUNCTION_SETUP();

  mERROR_CHECK(mJsonWriter_BeginArray(jsonWriter, name));
  mDEFER(mJsonWriter_EndArray(jsonWriter));

  mERROR_CHECK(mJsonWriter_AddArrayValue(jsonWriter, value.x));
  mERROR_CHECK(mJsonWriter_AddArrayValue(jsonWriter, value.y));

  mRETURN_SUCCESS();
}

mFUNCTION(mJsonWriter_AddValueX, mPtr<mJsonWriter> &jsonWriter, const char *name, const mVec3f &value)
{
  mFUNCTION_SETUP();

  mERROR_CHECK(mJsonWriter_BeginArray(jsonWriter, name));
  mDEFER(mJsonWriter_EndArray(jsonWriter));

  mERROR_CHECK(mJsonWriter_AddArrayValue(jsonWriter, value.x));
  mERROR_CHECK(mJsonWriter_AddArrayValue(jsonWriter, value.y));
  mERROR_CHECK(mJsonWriter_AddArrayValue(jsonWriter, value.z));

  mRETURN_SUCCESS();
}

mFUNCTION(mJsonWriter_AddValueX, mPtr<mJsonWriter> &jsonWriter, const char *name, const mVec4f &value)
{
  mFUNCTION_SETUP();

  mERROR_CHECK(mJsonWriter_BeginArray(jsonWriter, name));
  mDEFER(mJsonWriter_EndArray(jsonWriter));

  mERROR_CHECK(mJsonWriter_AddArrayValue(jsonWriter, value.x));
  mERROR_CHECK(mJsonWriter_AddArrayValue(jsonWriter, value.y));
  mERROR_CHECK(mJsonWriter_AddArrayValue(jsonWriter, value.z));
  mERROR_CHECK(mJsonWriter_AddArrayValue(jsonWriter, value.w));

  mRETURN_SUCCESS();
}

mFUNCTION(mJsonWriter_AddValueX, mPtr<mJsonWriter> &jsonWriter, const char *name, const mVector &value)
{
  mFUNCTION_SETUP();

  mERROR_CHECK(mJsonWriter_BeginArray(jsonWriter, name));
  mDEFER(mJsonWriter_EndArray(jsonWriter));

  mERROR_CHECK(mJsonWriter_AddArrayValue(jsonWriter, value.x));
  mERROR_CHECK(mJsonWriter_AddArrayValue(jsonWriter, value.y));
  mERROR_CHECK(mJsonWriter_AddArrayValue(jsonWriter, value.z));
  mERROR_CHECK(mJsonWriter_AddArrayValue(jsonWriter, value.w));

  mRETURN_SUCCESS();
}

mFUNCTION(mJsonWriter_AddArrayValue, mPtr<mJsonWriter> &jsonWriter, const double_t value)
{
  mFUNCTION_SETUP();

  mERROR_IF(jsonWriter == nullptr, mR_ArgumentNull);
  mERROR_IF(jsonWriter->finalized, mR_ResourceStateInvalid);

  cJSON *pJson = nullptr;
  mERROR_CHECK(mJsonWriter_GetLastInQueue_Internal(jsonWriter, &pJson));
  mERROR_IF(pJson == nullptr, mR_NotInitialized);

  mJsonWriterEntryType lastType;
  mERROR_CHECK(mJsonWriter_GetLastInTypeQueue_Internal(jsonWriter, &lastType));

  switch (lastType)
  {
  case Array:
    mERROR_IF(nullptr == cJSON_AddNumberToObject(pJson, "", value), mR_InternalError);
    break;

  default:
    mRETURN_RESULT(mR_ResourceStateInvalid);
    break;
  }

  mRETURN_SUCCESS();
}

mFUNCTION(mJsonWriter_AddArrayValue, mPtr<mJsonWriter> &jsonWriter, const mString &value)
{
  mFUNCTION_SETUP();

  mERROR_IF(jsonWriter == nullptr, mR_ArgumentNull);
  mERROR_IF(jsonWriter->finalized, mR_ResourceStateInvalid);

  cJSON *pJson = nullptr;
  mERROR_CHECK(mJsonWriter_GetLastInQueue_Internal(jsonWriter, &pJson));
  mERROR_IF(pJson == nullptr, mR_NotInitialized);

  mJsonWriterEntryType lastType;
  mERROR_CHECK(mJsonWriter_GetLastInTypeQueue_Internal(jsonWriter, &lastType));

  switch (lastType)
  {
  case Array:
    mERROR_IF(nullptr == cJSON_AddStringToObject(pJson, "", value.c_str()), mR_InternalError);
    break;

  default:
    mRETURN_RESULT(mR_ResourceStateInvalid);
    break;
  }

  mRETURN_SUCCESS();
}

mFUNCTION(mJsonWriter_AddArrayValue, mPtr<mJsonWriter> &jsonWriter, const char *value)
{
  mFUNCTION_SETUP();

  mString string;
  mERROR_CHECK(mString_Create(&string, value));

  mERROR_CHECK(mJsonWriter_AddArrayValue(jsonWriter, string));

  mRETURN_SUCCESS();
}

mFUNCTION(mJsonWriter_AddArrayValue, mPtr<mJsonWriter> &jsonWriter, const bool value)
{
  mFUNCTION_SETUP();

  mERROR_IF(jsonWriter == nullptr, mR_ArgumentNull);
  mERROR_IF(jsonWriter->finalized, mR_ResourceStateInvalid);

  cJSON *pJson = nullptr;
  mERROR_CHECK(mJsonWriter_GetLastInQueue_Internal(jsonWriter, &pJson));
  mERROR_IF(pJson == nullptr, mR_NotInitialized);

  mJsonWriterEntryType lastType;
  mERROR_CHECK(mJsonWriter_GetLastInTypeQueue_Internal(jsonWriter, &lastType));

  switch (lastType)
  {
  case Array:
    mERROR_IF(nullptr == cJSON_AddBoolToObject(pJson, "", value), mR_InternalError);
    break;

  default:
    mRETURN_RESULT(mR_ResourceStateInvalid);
    break;
  }

  mRETURN_SUCCESS();
}

mFUNCTION(mJsonWriter_AddArrayValue, mPtr<mJsonWriter> &jsonWriter, nullptr_t)
{
  mFUNCTION_SETUP();

  mERROR_IF(jsonWriter == nullptr, mR_ArgumentNull);
  mERROR_IF(jsonWriter->finalized, mR_ResourceStateInvalid);

  cJSON *pJson = nullptr;
  mERROR_CHECK(mJsonWriter_GetLastInQueue_Internal(jsonWriter, &pJson));
  mERROR_IF(pJson == nullptr, mR_NotInitialized);

  mJsonWriterEntryType lastType;
  mERROR_CHECK(mJsonWriter_GetLastInTypeQueue_Internal(jsonWriter, &lastType));

  switch (lastType)
  {
  case Array:
    mERROR_IF(nullptr == cJSON_AddNullToObject(pJson, ""), mR_InternalError);
    break;

  default:
    mRETURN_RESULT(mR_ResourceStateInvalid);
    break;
  }

  mRETURN_SUCCESS();
}

mFUNCTION(mJsonWriter_AddArrayValueX, mPtr<mJsonWriter> &jsonWriter, const mVec2f &value)
{
  mFUNCTION_SETUP();

  mERROR_CHECK(mJsonWriter_BeginUnnamed(jsonWriter));
  mDEFER(mJsonWriter_EndUnnamed(jsonWriter));

  mERROR_CHECK(mJsonWriter_AddArrayValue(jsonWriter, value.x));
  mERROR_CHECK(mJsonWriter_AddArrayValue(jsonWriter, value.y));

  mRETURN_SUCCESS();
}

mFUNCTION(mJsonWriter_AddArrayValueX, mPtr<mJsonWriter> &jsonWriter, const mVec3f &value)
{
  mFUNCTION_SETUP();

  mERROR_CHECK(mJsonWriter_BeginUnnamed(jsonWriter));
  mDEFER(mJsonWriter_EndUnnamed(jsonWriter));

  mERROR_CHECK(mJsonWriter_AddArrayValue(jsonWriter, value.x));
  mERROR_CHECK(mJsonWriter_AddArrayValue(jsonWriter, value.y));
  mERROR_CHECK(mJsonWriter_AddArrayValue(jsonWriter, value.z));

  mRETURN_SUCCESS();
}

mFUNCTION(mJsonWriter_AddArrayValueX, mPtr<mJsonWriter> &jsonWriter, const mVec4f &value)
{
  mFUNCTION_SETUP();

  mERROR_CHECK(mJsonWriter_BeginUnnamed(jsonWriter));
  mDEFER(mJsonWriter_EndUnnamed(jsonWriter));

  mERROR_CHECK(mJsonWriter_AddArrayValue(jsonWriter, value.x));
  mERROR_CHECK(mJsonWriter_AddArrayValue(jsonWriter, value.y));
  mERROR_CHECK(mJsonWriter_AddArrayValue(jsonWriter, value.z));
  mERROR_CHECK(mJsonWriter_AddArrayValue(jsonWriter, value.w));

  mRETURN_SUCCESS();
}

mFUNCTION(mJsonWriter_AddArrayValueX, mPtr<mJsonWriter> &jsonWriter, const mVector &value)
{
  mFUNCTION_SETUP();

  mERROR_CHECK(mJsonWriter_BeginUnnamed(jsonWriter));
  mDEFER(mJsonWriter_EndUnnamed(jsonWriter));

  mERROR_CHECK(mJsonWriter_AddArrayValue(jsonWriter, value.x));
  mERROR_CHECK(mJsonWriter_AddArrayValue(jsonWriter, value.y));
  mERROR_CHECK(mJsonWriter_AddArrayValue(jsonWriter, value.z));
  mERROR_CHECK(mJsonWriter_AddArrayValue(jsonWriter, value.w));

  mRETURN_SUCCESS();
}

mFUNCTION(mJsonWriter_ToString, mPtr<mJsonWriter> &jsonWriter, OUT mString *pString)
{
  mFUNCTION_SETUP();

  mERROR_IF(jsonWriter == nullptr || pString == nullptr, mR_ArgumentNull);

  cJSON *pJson = nullptr;
  mERROR_CHECK(mQueue_PeekFront(jsonWriter->currentBlock, &pJson));

  char *text = cJSON_Print(pJson);
  mERROR_IF(text == nullptr, mR_InternalError);
  mDEFER(cJSON_free(text));

  mERROR_CHECK(mString_Create(pString, text));

  mRETURN_SUCCESS();
}

mFUNCTION(mJsonWriter_ToFile, mPtr<mJsonWriter> &jsonWriter, const mString &filename)
{
  mFUNCTION_SETUP();

  mERROR_IF(jsonWriter == nullptr, mR_ArgumentNull);

  cJSON *pJson = nullptr;
  mERROR_CHECK(mQueue_PeekFront(jsonWriter->currentBlock, &pJson));

  char *text = cJSON_Print(pJson);
  mERROR_IF(text == nullptr, mR_InternalError);
  mDEFER(cJSON_free(text));

  mString fileContents;
  mERROR_CHECK(mJsonWriter_ToString(jsonWriter, &fileContents));

  mERROR_CHECK(mFile_WriteAllText(filename, text));

  mRETURN_SUCCESS();
}

//////////////////////////////////////////////////////////////////////////

mFUNCTION(mJsonWriter_Destroy_Internal, IN mJsonWriter *pJsonWriter)
{
  mFUNCTION_SETUP();

  mERROR_IF(pJsonWriter == nullptr, mR_ArgumentNull);

  size_t count = 0;
  mERROR_CHECK(mQueue_GetCount(pJsonWriter->currentBlock, &count));

  if (count > 0)
  {
    cJSON *pJson = nullptr;
    mERROR_CHECK(mQueue_PopFront(pJsonWriter->currentBlock, &pJson));

    cJSON_Delete(pJson);
  }

  mERROR_CHECK(mQueue_Destroy(&pJsonWriter->currentBlock));
  mERROR_CHECK(mQueue_Destroy(&pJsonWriter->currentType));

  mRETURN_SUCCESS();
}

mFUNCTION(mJsonWriter_GetLastInQueue_Internal, mPtr<mJsonWriter> &jsonWriter, OUT cJSON **ppJsonElement)
{
  mFUNCTION_SETUP();

  size_t count = 0;
  mERROR_CHECK(mQueue_GetCount(jsonWriter->currentBlock, &count));

  if (count == 0)
    *ppJsonElement = nullptr;
  else
    mERROR_CHECK(mQueue_PeekBack(jsonWriter->currentBlock, ppJsonElement));

  mRETURN_SUCCESS();
}

mFUNCTION(mJsonWriter_GetLastInTypeQueue_Internal, mPtr<mJsonWriter> &jsonWriter, OUT mJsonWriterEntryType *pJsonElement)
{
  mFUNCTION_SETUP();

  size_t count = 0;
  mERROR_CHECK(mQueue_GetCount(jsonWriter->currentType, &count));

  if (count == 0)
    *pJsonElement = Object;
  else
    mERROR_CHECK(mQueue_PeekBack(jsonWriter->currentType, pJsonElement));

  mRETURN_SUCCESS();
}

mFUNCTION(mJsonWriter_PopQueue_Internal, mPtr<mJsonWriter> &jsonWriter)
{
  mFUNCTION_SETUP();

  size_t count = 0;
  mERROR_CHECK(mQueue_GetCount(jsonWriter->currentType, &count));

  if (count <= 1)
  {
    jsonWriter->finalized = true;
    mRETURN_SUCCESS();
  }

  cJSON *pJson;
  mERROR_CHECK(mQueue_PopBack(jsonWriter->currentBlock, &pJson));

  mJsonWriterEntryType entrytype;
  mERROR_CHECK(mQueue_PopBack(jsonWriter->currentType, &entrytype));

  mRETURN_SUCCESS();
}

mFUNCTION(mJsonWriter_PushQueue_Internal, mPtr<mJsonWriter> &jsonWriter, IN cJSON *pJsonElement, const mJsonWriterEntryType type)
{
  mFUNCTION_SETUP();

  mERROR_CHECK(mQueue_PushBack(jsonWriter->currentBlock, pJsonElement));
  mERROR_CHECK(mQueue_PushBack(jsonWriter->currentType, type));

  mRETURN_SUCCESS();
}

//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////

struct mJsonReader
{
  mPtr<mQueue<cJSON *>> currentBlock;
};

mFUNCTION(mJsonReader_Create_Internal, OUT mPtr<mJsonReader> *pJsonReader, IN mAllocator *pAllocator);
mFUNCTION(mJsonReader_Destroy_Internal, IN_OUT mJsonReader *pJsonReader);
mFUNCTION(mJsonReader_PeekLast_Internal, mPtr<mJsonReader> &jsonReader, OUT cJSON **ppJson);
mFUNCTION(mJsonReader_PopLast_Internal, mPtr<mJsonReader> &jsonReader);
mFUNCTION(mJsonReader_PushValue_Internal, mPtr<mJsonReader> &jsonReader, IN cJSON *pJson);

//////////////////////////////////////////////////////////////////////////

mFUNCTION(mJsonReader_CreateFromString, OUT mPtr<mJsonReader> *pJsonReader, IN mAllocator *pAllocator, const mString &text)
{
  mFUNCTION_SETUP();

  mERROR_CHECK(mJsonReader_Create_Internal(pJsonReader, pAllocator));

  cJSON *pJson = cJSON_Parse(text.c_str());
  mERROR_CHECK(mJson_CheckError_Internal(pJson));
  mERROR_CHECK(mJsonReader_PushValue_Internal(*pJsonReader, pJson));

  mRETURN_SUCCESS();
}

mFUNCTION(mJsonReader_CreateFromFile, OUT mPtr<mJsonReader> *pJsonReader, IN mAllocator *pAllocator, const mString &filename)
{
  mFUNCTION_SETUP();

  mString fileContents;
  mERROR_CHECK(mFile_ReadAllText(filename, pAllocator, &fileContents));

  mERROR_CHECK(mJsonReader_CreateFromString(pJsonReader, pAllocator, fileContents));

  mRETURN_SUCCESS();
}

mFUNCTION(mJsonReader_Destroy, IN_OUT mPtr<mJsonReader> *pJsonReader)
{
  mFUNCTION_SETUP();

  mERROR_IF(pJsonReader == nullptr, mR_ArgumentNull);

  mERROR_CHECK(mSharedPointer_Destroy(pJsonReader));

  mRETURN_SUCCESS();
}

mFUNCTION(mJsonReader_StepIntoNamed, mPtr<mJsonReader> &jsonReader, const char *name)
{
  mFUNCTION_SETUP();

  mERROR_IF(jsonReader == nullptr || name == nullptr, mR_ArgumentNull);

  cJSON *pJson;
  mERROR_CHECK(mJsonReader_PeekLast_Internal(jsonReader, &pJson));

  cJSON *pChild = cJSON_GetObjectItemCaseSensitive(pJson, name);
  mERROR_CHECK(mJson_CheckError_Internal(pChild));
  mERROR_CHECK(mJsonReader_PushValue_Internal(jsonReader, pChild));

  mRETURN_SUCCESS();
}

mFUNCTION(mJsonReader_ExitNamed, mPtr<mJsonReader> &jsonReader)
{
  mFUNCTION_SETUP();

  mERROR_IF(jsonReader == nullptr, mR_ArgumentNull);

  mERROR_CHECK(mJsonReader_PopLast_Internal(jsonReader));

  mRETURN_SUCCESS();
}

mFUNCTION(mJsonReader_StepIntoArrayItem, mPtr<mJsonReader>& jsonReader, const size_t index)
{
  mFUNCTION_SETUP();

  mERROR_IF(jsonReader == nullptr, mR_ArgumentNull);

  cJSON *pJson;
  mERROR_CHECK(mJsonReader_PeekLast_Internal(jsonReader, &pJson));

  cJSON *pChild = cJSON_GetArrayItem(pJson, (int)index);
  mERROR_CHECK(mJson_CheckError_Internal(pChild));
  mERROR_CHECK(mJsonReader_PushValue_Internal(jsonReader, pChild));

  mRETURN_SUCCESS();
}

mFUNCTION(mJsonReader_ExitArrayItem, mPtr<mJsonReader>& jsonReader)
{
  mFUNCTION_SETUP();

  mERROR_IF(jsonReader == nullptr, mR_ArgumentNull);

  mERROR_CHECK(mJsonReader_PopLast_Internal(jsonReader));

  mRETURN_SUCCESS();
}

mFUNCTION(mJsonReader_StepIntoArray, mPtr<mJsonReader> &jsonReader, const char *name)
{
  mFUNCTION_SETUP();

  mERROR_CHECK(mJsonReader_StepIntoNamed(jsonReader, name));

  mRETURN_SUCCESS();
}

mFUNCTION(mJsonReader_ExitArray, mPtr<mJsonReader> &jsonReader)
{
  mFUNCTION_SETUP();

  mERROR_CHECK(mJsonReader_ExitNamed(jsonReader));

  mRETURN_SUCCESS();
}

mFUNCTION(mJsonReader_GetArrayCount, mPtr<mJsonReader> &jsonReader, OUT size_t *pCount)
{
  mFUNCTION_SETUP();

  mERROR_IF(jsonReader == nullptr || pCount == nullptr, mR_ArgumentNull);

  cJSON *pJson;
  mERROR_CHECK(mJsonReader_PeekLast_Internal(jsonReader, &pJson));

  mERROR_IF(!cJSON_IsArray(pJson), mR_ResourceStateInvalid);
  
  *pCount = cJSON_GetArraySize(pJson);

  mRETURN_SUCCESS();
}

mFUNCTION(mJsonReader_ReadNamedValue, mPtr<mJsonReader> &jsonReader, const char *name, OUT mString *pString)
{
  mFUNCTION_SETUP();

  mERROR_IF(jsonReader == nullptr || name == nullptr || pString == nullptr, mR_ArgumentNull);

  cJSON *pJson;
  mERROR_CHECK(mJsonReader_PeekLast_Internal(jsonReader, &pJson));

  cJSON *pChild = cJSON_GetObjectItemCaseSensitive(pJson, name);
  mERROR_CHECK(mJson_CheckError_Internal(pChild));

  mERROR_IF(!cJSON_IsString(pChild) || pChild->valuestring == nullptr, mR_ResourceNotFound);

  mERROR_CHECK(mString_Create(pString, pChild->valuestring, nullptr));

  mRETURN_SUCCESS();
}

mFUNCTION(mJsonReader_ReadNamedValue, mPtr<mJsonReader> &jsonReader, const char *name, OUT double_t *pValue)
{
  mFUNCTION_SETUP();

  mERROR_IF(jsonReader == nullptr || name == nullptr || pValue == nullptr, mR_ArgumentNull);

  cJSON *pJson;
  mERROR_CHECK(mJsonReader_PeekLast_Internal(jsonReader, &pJson));

  cJSON *pChild = cJSON_GetObjectItemCaseSensitive(pJson, name);
  mERROR_CHECK(mJson_CheckError_Internal(pChild));

  mERROR_IF(!cJSON_IsNumber(pChild), mR_ResourceNotFound);

  *pValue = pChild->valuedouble;

  mRETURN_SUCCESS();
}

mFUNCTION(mJsonReader_ReadNamedValue, mPtr<mJsonReader> &jsonReader, const char *name, OUT bool *pValue)
{
  mFUNCTION_SETUP();

  mERROR_IF(jsonReader == nullptr || name == nullptr || pValue == nullptr, mR_ArgumentNull);

  cJSON *pJson;
  mERROR_CHECK(mJsonReader_PeekLast_Internal(jsonReader, &pJson));

  cJSON *pChild = cJSON_GetObjectItemCaseSensitive(pJson, name);
  mERROR_CHECK(mJson_CheckError_Internal(pChild));

  mERROR_IF(!cJSON_IsBool(pChild), mR_ResourceNotFound);

  *pValue = cJSON_IsTrue(pChild) != 0;

  mRETURN_SUCCESS();
}

mFUNCTION(mJsonReader_ReadNamedNull, mPtr<mJsonReader> &jsonReader, const char *name)
{
  mFUNCTION_SETUP();

  mERROR_IF(jsonReader == nullptr || name == nullptr, mR_ArgumentNull);

  cJSON *pJson;
  mERROR_CHECK(mJsonReader_PeekLast_Internal(jsonReader, &pJson));

  cJSON *pChild = cJSON_GetObjectItemCaseSensitive(pJson, name);
  mERROR_CHECK(mJson_CheckError_Internal(pChild));

  mERROR_IF(!cJSON_IsNull(pChild), mR_ResourceNotFound);

  mRETURN_SUCCESS();
}

mFUNCTION(mJsonReader_ReadArrayValue, mPtr<mJsonReader> &jsonReader, const size_t index, OUT mString *pString)
{
  mFUNCTION_SETUP();

  mERROR_IF(jsonReader == nullptr || pString == nullptr, mR_ArgumentNull);
  mERROR_IF(index > INT_MAX, mR_ArgumentOutOfBounds);

  cJSON *pJson;
  mERROR_CHECK(mJsonReader_PeekLast_Internal(jsonReader, &pJson));

  cJSON *pChild = cJSON_GetArrayItem(pJson, (int)index);
  mERROR_CHECK(mJson_CheckError_Internal(pChild));

  mERROR_IF(!cJSON_IsString(pChild) || pChild->valuestring == nullptr, mR_ResourceNotFound);

  mERROR_CHECK(mString_Create(pString, pChild->valuestring, nullptr));

  mRETURN_SUCCESS();
}

mFUNCTION(mJsonReader_ReadArrayValue, mPtr<mJsonReader> &jsonReader, const size_t index, OUT double_t *pValue)
{
  mFUNCTION_SETUP();

  mERROR_IF(jsonReader == nullptr || pValue == nullptr, mR_ArgumentNull);
  mERROR_IF(index > INT_MAX, mR_ArgumentOutOfBounds);

  cJSON *pJson;
  mERROR_CHECK(mJsonReader_PeekLast_Internal(jsonReader, &pJson));

  cJSON *pChild = cJSON_GetArrayItem(pJson, (int)index);
  mERROR_CHECK(mJson_CheckError_Internal(pChild));

  mERROR_IF(!cJSON_IsNumber(pChild), mR_ResourceNotFound);

  *pValue = pChild->valuedouble;

  mRETURN_SUCCESS();
}

mFUNCTION(mJsonReader_ReadArrayValue, mPtr<mJsonReader> &jsonReader, const size_t index, OUT bool *pValue)
{
  mFUNCTION_SETUP();

  mERROR_IF(jsonReader == nullptr || pValue == nullptr, mR_ArgumentNull);
  mERROR_IF(index > INT_MAX, mR_ArgumentOutOfBounds);

  cJSON *pJson;
  mERROR_CHECK(mJsonReader_PeekLast_Internal(jsonReader, &pJson));

  cJSON *pChild = cJSON_GetArrayItem(pJson, (int)index);
  mERROR_CHECK(mJson_CheckError_Internal(pChild));

  mERROR_IF(!cJSON_IsBool(pChild), mR_ResourceNotFound);

  *pValue = cJSON_IsTrue(pChild) != 0;

  mRETURN_SUCCESS();
}

mFUNCTION(mJsonReader_ReadArrayNull, mPtr<mJsonReader> &jsonReader, const size_t index)
{
  mFUNCTION_SETUP();

  mERROR_IF(jsonReader == nullptr, mR_ArgumentNull);
  mERROR_IF(index > INT_MAX, mR_ArgumentOutOfBounds);

  cJSON *pJson;
  mERROR_CHECK(mJsonReader_PeekLast_Internal(jsonReader, &pJson));

  cJSON *pChild = cJSON_GetArrayItem(pJson, (int)index);
  mERROR_CHECK(mJson_CheckError_Internal(pChild));

  mERROR_IF(!cJSON_IsNull(pChild), mR_ResourceNotFound);

  mRETURN_SUCCESS();
}

mFUNCTION(mJsonReader_GetCurrentValueType, mPtr<mJsonReader> &jsonReader, OUT mJsonReader_EntryType *pEntryType)
{
  mFUNCTION_SETUP();

  mERROR_IF(jsonReader == nullptr || pEntryType == nullptr, mR_ArgumentNull);

  cJSON *pJson;
  mERROR_CHECK(mJsonReader_PeekLast_Internal(jsonReader, &pJson));

  if (cJSON_IsObject(pJson))
    *pEntryType = mJR_ET_Object;
  else if (cJSON_IsArray(pJson))
    *pEntryType = mJR_ET_Array;
  else if (cJSON_IsNumber(pJson))
    *pEntryType = mJR_ET_Number;
  else if (cJSON_IsString(pJson))
    *pEntryType = mJR_ET_String;
  else if (cJSON_IsBool(pJson))
    *pEntryType = mJR_ET_Bool;
  else if (cJSON_IsNull(pJson))
    *pEntryType = mJR_ET_Null;
  else
    *pEntryType = mJR_ET_Invalid;

  mRETURN_SUCCESS();
}

mFUNCTION(mJsonReader_ReadCurrentValue, mPtr<mJsonReader> &jsonReader, OUT mString *pString)
{
  mFUNCTION_SETUP();

  mERROR_IF(jsonReader == nullptr || pString == nullptr, mR_ArgumentNull);

  cJSON *pJson;
  mERROR_CHECK(mJsonReader_PeekLast_Internal(jsonReader, &pJson));

  mERROR_IF(!cJSON_IsString(pJson) || pJson->valuestring == nullptr, mR_ResourceIncompatible);

  mERROR_CHECK(mString_Create(pString, pJson->valuestring));

  mRETURN_SUCCESS();
}

mFUNCTION(mJsonReader_ReadCurrentValue, mPtr<mJsonReader>& jsonReader, OUT double_t * pValue)
{
  mFUNCTION_SETUP();

  mERROR_IF(jsonReader == nullptr || pValue == nullptr, mR_ArgumentNull);

  cJSON *pJson;
  mERROR_CHECK(mJsonReader_PeekLast_Internal(jsonReader, &pJson));

  mERROR_IF(!cJSON_IsNumber(pJson), mR_ResourceIncompatible);
  
  *pValue = pJson->valuedouble;

  mRETURN_SUCCESS();
}

mFUNCTION(mJsonReader_ReadCurrentValue, mPtr<mJsonReader>& jsonReader, OUT bool * pValue)
{
  mFUNCTION_SETUP();

  mERROR_IF(jsonReader == nullptr || pValue == nullptr, mR_ArgumentNull);

  cJSON *pJson;
  mERROR_CHECK(mJsonReader_PeekLast_Internal(jsonReader, &pJson));

  mERROR_IF(!cJSON_IsBool(pJson), mR_ResourceIncompatible);

  *pValue = cJSON_IsTrue(pJson) != 0;

  mRETURN_SUCCESS();
}

mFUNCTION(mJsonReader_ArrayForEach, mPtr<mJsonReader> &jsonReader, const std::function<mResult(mPtr<mJsonReader> &, const size_t index)> &callback)
{
  mFUNCTION_SETUP();

  mERROR_IF(jsonReader == nullptr || callback == nullptr, mR_ArgumentNull);

  cJSON *pJson;
  mERROR_CHECK(mJsonReader_PeekLast_Internal(jsonReader, &pJson));

  size_t index = 0;
  cJSON *pElement = nullptr;
  cJSON_ArrayForEach(pElement, pJson)
  {
    mERROR_CHECK(mJsonReader_PushValue_Internal(jsonReader, pElement));
    mDEFER(mJsonReader_PopLast_Internal(jsonReader));

    mERROR_CHECK(callback(jsonReader, index));

    ++index;
  }

  mRETURN_SUCCESS();
}

//////////////////////////////////////////////////////////////////////////

mFUNCTION(mJsonReader_Create_Internal, OUT mPtr<mJsonReader> *pJsonReader, IN mAllocator *pAllocator)
{
  mFUNCTION_SETUP();

  mERROR_IF(pJsonReader == nullptr, mR_ArgumentNull);

  mERROR_CHECK(mSharedPointer_Allocate(pJsonReader, pAllocator, (std::function<void(mJsonReader *)>)[](mJsonReader *pData) { mJsonReader_Destroy_Internal(pData); }, 1));

  mERROR_CHECK(mQueue_Create(&(*pJsonReader)->currentBlock, pAllocator));

  mRETURN_SUCCESS();
}

mFUNCTION(mJsonReader_Destroy_Internal, IN_OUT mJsonReader *pJsonReader)
{
  mFUNCTION_SETUP();

  mERROR_IF(pJsonReader == nullptr, mR_ArgumentNull);

  size_t count = 0;
  mERROR_CHECK(mQueue_GetCount(pJsonReader->currentBlock, &count));

  if (count > 0)
  {
    cJSON *pJson = nullptr;
    mERROR_CHECK(mQueue_PopFront(pJsonReader->currentBlock, &pJson));

    cJSON_Delete(pJson);
  }

  mERROR_CHECK(mQueue_Destroy(&pJsonReader->currentBlock));

  mRETURN_SUCCESS();
}

mFUNCTION(mJsonReader_PeekLast_Internal, mPtr<mJsonReader> &jsonReader, OUT cJSON **ppJson)
{
  mFUNCTION_SETUP();

  size_t count = 0;
  mERROR_CHECK(mQueue_GetCount(jsonReader->currentBlock, &count));

  if (count == 0)
    mRETURN_RESULT(mR_ResourceStateInvalid);

  mERROR_CHECK(mQueue_PeekBack(jsonReader->currentBlock, ppJson));

  mRETURN_SUCCESS();
}

mFUNCTION(mJsonReader_PopLast_Internal, mPtr<mJsonReader>& jsonReader)
{
  mFUNCTION_SETUP();

  size_t count = 0;
  mERROR_CHECK(mQueue_GetCount(jsonReader->currentBlock, &count));

  if (count <= 1)
    mRETURN_SUCCESS();

  cJSON *pJson;
  mERROR_CHECK(mQueue_PopBack(jsonReader->currentBlock, &pJson));

  mRETURN_SUCCESS();
}

mFUNCTION(mJsonReader_PushValue_Internal, mPtr<mJsonReader> &jsonReader, IN cJSON *pJson)
{
  mFUNCTION_SETUP();

  mERROR_CHECK(mQueue_PushBack(jsonReader->currentBlock, pJson));

  mRETURN_SUCCESS();
}

//////////////////////////////////////////////////////////////////////////

mFUNCTION(mJson_CheckError_Internal, IN cJSON *pJson)
{
  mFUNCTION_SETUP();

  if (pJson == nullptr)
  {
    const char *error = cJSON_GetErrorPtr();

    if (error != nullptr)
      mPRINT_ERROR("mJsonReader Error: %s\n", error);

    mRETURN_RESULT(mR_ResourceNotFound);
  }

  mRETURN_SUCCESS();
}
