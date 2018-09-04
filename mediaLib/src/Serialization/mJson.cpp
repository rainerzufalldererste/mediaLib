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
    const auto __defer__ = mDefer_Create<cJSON>(cJSON_Delete, pChild, &mSTDRESULT);
    mERROR_IF(pChild == nullptr, mR_InternalError);

    mERROR_CHECK(mJsonWriter_PushQueue_Internal(jsonWriter, pChild, Object));
  }
  else
  {
    cJSON *pChild = cJSON_CreateObject();
    const auto __defer__ = mDefer_Create<cJSON>(cJSON_Delete, pChild, &mSTDRESULT);
    mERROR_IF(pChild == nullptr, mR_InternalError);

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
  const auto __defer__ = mDefer_Create<cJSON>(cJSON_Delete, pChild, &mSTDRESULT);
  mERROR_IF(pChild == nullptr, mR_InternalError);

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
  const auto __defer__ = mDefer_Create<cJSON>(cJSON_Delete, pChild, &mSTDRESULT);
  mERROR_IF(pChild == nullptr, mR_InternalError);

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

  mString fileContents;
  mERROR_CHECK(mJsonWriter_ToString(jsonWriter, &fileContents));

  mERROR_CHECK(mFile_WriteAllText(filename, fileContents));

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
    cJSON *pJson;
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
