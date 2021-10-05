#include "mBlobWriter.h"
#include "mBlobReader.h"

#include "mQueue.h"

#ifdef GIT_BUILD // Define __M_FILE__
  #ifdef __M_FILE__
    #undef __M_FILE__
  #endif
  #define __M_FILE__ "pmPali6eQDBvND+Ml/YWp3RqvJuGPXw2wrhukhwKy5fXQ/wJUKUdLPWqUcH0c9N4grr2zeQ/vJXDmIxu"
#endif

constexpr size_t mBlob_Version = 1;

enum mBlob_Type : uint8_t
{
  _mB_T_Invalid,

  mB_T_Container,
  mB_T_Array,
  mB_T_BaseTypeArray,
  mB_T_Empty,
  mB_T_UInt8,
  mB_T_UInt32,
  mB_T_Int32,
  mB_T_UInt64,
  mB_T_Int64,
  mB_T_Float,
  mB_T_Double,
  mB_T_Bool,
  mB_T_UInt64_2,
  mB_T_Int64_2,
  mB_T_Float_2,
  mB_T_Float_3,
  mB_T_Float_4,
  mB_T_Double_2,
  mB_T_Double_3,
  mB_T_Double_4,
  mB_T_Text,
  mB_T_RawData,

  _mB_T_Count,
};

#pragma pack(push, 1)
struct mBlob_BaseType
{
  mBlob_Type type;
};
#pragma pack(pop)

#pragma pack(push, 1)
struct mBlob_Container : mBlob_BaseType
{
  size_t containerSizeBytes;
};
#pragma pack(pop)

#pragma pack(push, 1)
struct mBlob_Header
{
  size_t version;
  mBlob_Container container;
};
#pragma pack(pop)

#pragma pack(push, 1)
struct mBlob_Array : mBlob_Container
{
  size_t count;
};
#pragma pack(pop)

#pragma pack(push, 1)
struct mBlob_BaseTypeArray : mBlob_Container
{
  mBlob_Type elementType;
  size_t count;
};
#pragma pack(pop)

#pragma pack(push, 1)
template <typename T>
struct mBlob_Value : mBlob_BaseType
{
  T value;
};
#pragma pack(pop)

#pragma pack(push, 1)
struct mBlob_Text : mBlob_BaseType
{
  size_t length;
};
#pragma pack(pop)

#pragma pack(push, 1)
struct mBlob_RawData : mBlob_BaseType
{
  size_t length;
};
#pragma pack(pop)

struct mBlobWriter
{
  mUniqueContainer<mQueue<size_t>> parentContainerOffsets;
  size_t size, capacity;
  uint8_t *pData;
  mAllocator *pAllocator;
};

//////////////////////////////////////////////////////////////////////////

void mBlobWriter_Destroy_Internal(IN_OUT mBlobWriter *pWriter);
mFUNCTION(mBlobWriter_EndVariousContainers, mPtr<mBlobWriter> &writer, const mBlob_Type expectedType);

//////////////////////////////////////////////////////////////////////////

template <typename T>
inline mFUNCTION(mBlobWriter_WriteRaw_Internal, mPtr<mBlobWriter> &writer, const T &value)
{
  mFUNCTION_SETUP();

  if (writer->capacity < writer->size + sizeof(T))
  {
    const size_t newCapacity = ((writer->capacity + sizeof(T)) * 2 + 1023) & ~(size_t)1023;
    
    mERROR_CHECK(mAllocator_Reallocate(writer->pAllocator, &writer->pData, newCapacity));

    writer->capacity = newCapacity;
  }

  mMemcpy(reinterpret_cast<T *>(writer->pData + writer->size), &value, 1);

  writer->size += sizeof(T);

  mRETURN_SUCCESS();
}

template <typename T>
inline mFUNCTION(mBlobWriter_WriteData_Internal, mPtr<mBlobWriter> &writer, const T *pValue, const size_t count)
{
  mFUNCTION_SETUP();

  if (writer->capacity < writer->size + sizeof(T) * count)
  {
    const size_t newCapacity = ((writer->capacity + sizeof(T) * count) * 2 + 1023) & ~(size_t)1023;
    
    mERROR_CHECK(mAllocator_Reallocate(writer->pAllocator, &writer->pData, newCapacity));

    writer->capacity = newCapacity;
  }

  mMemcpy(reinterpret_cast<T *>(writer->pData + writer->size), pValue, count);

  writer->size += sizeof(T) * count;

  mRETURN_SUCCESS();
}

template <typename T>
inline mFUNCTION(mBlobWriter_BeginContainer_Internal, mPtr<mBlobWriter> &writer, const T &container)
{
  mFUNCTION_SETUP();

  mERROR_IF(writer == nullptr, mR_ArgumentNull);

  size_t parentOffset;
  mERROR_CHECK(mQueue_PeekBack(writer->parentContainerOffsets, &parentOffset));

  mBlob_Container *pParent = reinterpret_cast<mBlob_Container *>(writer->pData + parentOffset);

  switch (pParent->type)
  {
  case mB_T_Container:
  case mB_T_Array:
    break;

  case mB_T_BaseTypeArray:
    mRETURN_RESULT(mR_ResourceIncompatible);

  default:
    mRETURN_RESULT(mR_ResourceInvalid);
  }

  const size_t offsetBefore = writer->size;

  // This may reallocated `writer->pData`.
  mERROR_CHECK(mBlobWriter_WriteRaw_Internal(writer, container));

  mERROR_CHECK(mQueue_PushBack(writer->parentContainerOffsets, offsetBefore));

  pParent = reinterpret_cast<mBlob_Container *>(writer->pData + parentOffset);

  if (pParent->type == mB_T_Array)
    static_cast<mBlob_Array *>(pParent)->count++;

  mRETURN_SUCCESS();
}

template <typename T>
inline mFUNCTION(mBlobWriter_AddValue_Internal, mPtr<mBlobWriter> &writer, const T &value, const mBlob_Type type)
{
  mFUNCTION_SETUP();

  mERROR_IF(writer == nullptr, mR_ArgumentNull);

  size_t parentOffset;
  mERROR_CHECK(mQueue_PeekBack(writer->parentContainerOffsets, &parentOffset));

  mBlob_Container *pParent = reinterpret_cast<mBlob_Container *>(writer->pData + parentOffset);

  switch (pParent->type)
  {
  case mB_T_Container:
  case mB_T_Array:
  case mB_T_BaseTypeArray:
    break;

  default:
    mRETURN_RESULT(mR_ResourceInvalid);
  }

  if (pParent->type == mB_T_BaseTypeArray)
  {
    mBlob_BaseTypeArray *pBaseTypeArray = static_cast<mBlob_BaseTypeArray *>(pParent);

    if (pBaseTypeArray->elementType == mB_T_Empty)
      pBaseTypeArray->elementType = type;
    else
      mERROR_IF(pBaseTypeArray->elementType != type, mR_ResourceIncompatible);

    // This may reallocated `writer->pData`.
    mERROR_CHECK(mBlobWriter_WriteRaw_Internal(writer, value));

    pBaseTypeArray = reinterpret_cast<mBlob_BaseTypeArray *>(writer->pData + parentOffset);
    pBaseTypeArray->containerSizeBytes += sizeof(T);
    pBaseTypeArray->count++;
  }
  else
  {
    mBlob_Value<T> blobValue;
    blobValue.type = type;
    blobValue.value = value;

    // This may reallocated `writer->pData`.
    mERROR_CHECK(mBlobWriter_WriteRaw_Internal(writer, blobValue));

    pParent = reinterpret_cast<mBlob_Container *>(writer->pData + parentOffset);
    pParent->containerSizeBytes += sizeof(blobValue);

    switch (pParent->type)
    {
    case mB_T_Array:
      static_cast<mBlob_Array *>(pParent)->count++;
      break;
    }
  }

  mRETURN_SUCCESS();
}

//////////////////////////////////////////////////////////////////////////

mFUNCTION(mBlobWriter_Create, OUT mPtr<mBlobWriter> *pWriter, IN mAllocator *pAllocator)
{
  mFUNCTION_SETUP();

  mERROR_IF(pWriter == nullptr, mR_ArgumentNull);

  mDEFER_CALL_ON_ERROR(pWriter, mSharedPointer_Destroy);
  mERROR_CHECK((mSharedPointer_Allocate<mBlobWriter>(pWriter, pAllocator, mBlobWriter_Destroy_Internal, 1)));

  mBlobWriter *pInstance = pWriter->GetPointer();

  pInstance->pAllocator = pAllocator;
  mERROR_CHECK(mQueue_Create(&pInstance->parentContainerOffsets, pAllocator));

  mBlob_Header header;
  header.version = mBlob_Version;
  header.container.type = mB_T_Container;
  header.container.containerSizeBytes = sizeof(header.container);

  // This will allocate `pInstance->pData`.
  mERROR_CHECK(mBlobWriter_WriteRaw_Internal(*pWriter, header));

  mERROR_CHECK(mQueue_PushBack(pInstance->parentContainerOffsets, offsetof(mBlob_Header, container)));

  mRETURN_SUCCESS();
}

mFUNCTION(mBlobWriter_Destroy, IN_OUT mPtr<mBlobWriter> *pWriter)
{
  return mSharedPointer_Destroy(pWriter);
}

mFUNCTION(mBlobWriter_GetData, mPtr<mBlobWriter> &writer, OUT const uint8_t **ppData, OUT size_t *pBytes)
{
  mFUNCTION_SETUP();

  mERROR_IF(writer == nullptr || ppData == nullptr || pBytes == nullptr, mR_ArgumentNull);

  // End all Containers.
  for (const auto &_offset : writer->parentContainerOffsets->IterateReverse())
  {
    mBlob_Container *pContainer = reinterpret_cast<mBlob_Container *>(writer->pData + _offset);

    if (_offset == offsetof(mBlob_Header, container))
      break;

    switch (pContainer->type)
    {
    case mB_T_Container:
    case mB_T_Array:
    case mB_T_BaseTypeArray:
      break;

    default:
      mRETURN_RESULT(mR_ResourceInvalid);
    }

    mERROR_CHECK(mBlobWriter_EndVariousContainers(writer, pContainer->type));
  }

  *ppData = writer->pData;
  *pBytes = writer->size;

  mRETURN_SUCCESS();
}

mFUNCTION(mBlobWriter_BeginContainer, mPtr<mBlobWriter> &writer)
{
  mBlob_Container container;
  container.type = mB_T_Container;
  container.containerSizeBytes = sizeof(container);

  return mBlobWriter_BeginContainer_Internal(writer, container);
}

mFUNCTION(mBlobWriter_BeginArray, mPtr<mBlobWriter> &writer)
{
  mBlob_Array container;
  container.type = mB_T_Array;
  container.containerSizeBytes = sizeof(container);
  container.count = 0;

  return mBlobWriter_BeginContainer_Internal(writer, container);
}

mFUNCTION(mBlobWriter_BeginBaseTypeArray, mPtr<mBlobWriter> &writer)
{
  mBlob_BaseTypeArray container;
  container.type = mB_T_BaseTypeArray;
  container.containerSizeBytes = sizeof(container);
  container.count = 0;
  container.elementType = mB_T_Empty;

  return mBlobWriter_BeginContainer_Internal(writer, container);
}

mFUNCTION(mBlobWriter_EndContainer, mPtr<mBlobWriter> &writer)
{
  return mBlobWriter_EndVariousContainers(writer, mB_T_Container);
}

mFUNCTION(mBlobWriter_EndArray, mPtr<mBlobWriter> &writer)
{
  return mBlobWriter_EndVariousContainers(writer, mB_T_Array);
}

mFUNCTION(mBlobWriter_EndBaseTypeArray, mPtr<mBlobWriter> &writer)
{
  return mBlobWriter_EndVariousContainers(writer, mB_T_BaseTypeArray);
}

mFUNCTION(mBlobWriter_AddValue, mPtr<mBlobWriter> &writer, const uint8_t value)
{
  return mBlobWriter_AddValue_Internal(writer, value, mB_T_UInt8);
}

mFUNCTION(mBlobWriter_AddValue, mPtr<mBlobWriter> &writer, const uint32_t value)
{
  return mBlobWriter_AddValue_Internal(writer, value, mB_T_UInt32);
}

mFUNCTION(mBlobWriter_AddValue, mPtr<mBlobWriter> &writer, const int32_t value)
{
  return mBlobWriter_AddValue_Internal(writer, value, mB_T_Int32);
}

mFUNCTION(mBlobWriter_AddValue, mPtr<mBlobWriter> &writer, const uint64_t value)
{
  return mBlobWriter_AddValue_Internal(writer, value, mB_T_UInt64);
}

mFUNCTION(mBlobWriter_AddValue, mPtr<mBlobWriter> &writer, const int64_t value)
{
  return mBlobWriter_AddValue_Internal(writer, value, mB_T_Int64);
}

mFUNCTION(mBlobWriter_AddValue, mPtr<mBlobWriter> &writer, const float_t value)
{
  return mBlobWriter_AddValue_Internal(writer, value, mB_T_Float);
}

mFUNCTION(mBlobWriter_AddValue, mPtr<mBlobWriter> &writer, const double_t value)
{
  return mBlobWriter_AddValue_Internal(writer, value, mB_T_Double);
}

mFUNCTION(mBlobWriter_AddValue, mPtr<mBlobWriter> &writer, const bool value)
{
  return mBlobWriter_AddValue_Internal(writer, value, mB_T_Bool);
}

mFUNCTION(mBlobWriter_AddValue, mPtr<mBlobWriter> &writer, const mVec2s value)
{
  return mBlobWriter_AddValue_Internal(writer, value, mB_T_UInt64_2);
}

mFUNCTION(mBlobWriter_AddValue, mPtr<mBlobWriter> &writer, const mVec2i value)
{
  return mBlobWriter_AddValue_Internal(writer, value, mB_T_Int64_2);
}

mFUNCTION(mBlobWriter_AddValue, mPtr<mBlobWriter> &writer, const mVec2f value)
{
  return mBlobWriter_AddValue_Internal(writer, value, mB_T_Float_2);
}

mFUNCTION(mBlobWriter_AddValue, mPtr<mBlobWriter> &writer, const mVec3f value)
{
  return mBlobWriter_AddValue_Internal(writer, value, mB_T_Float_3);
}

mFUNCTION(mBlobWriter_AddValue, mPtr<mBlobWriter> &writer, const mVec4f value)
{
  return mBlobWriter_AddValue_Internal(writer, value, mB_T_Float_4);
}

mFUNCTION(mBlobWriter_AddValue, mPtr<mBlobWriter> &writer, const mVec2d value)
{
  return mBlobWriter_AddValue_Internal(writer, value, mB_T_Double_2);
}

mFUNCTION(mBlobWriter_AddValue, mPtr<mBlobWriter> &writer, const mVec3d value)
{
  return mBlobWriter_AddValue_Internal(writer, value, mB_T_Double_3);
}

mFUNCTION(mBlobWriter_AddValue, mPtr<mBlobWriter> &writer, const mVec4d value)
{
  return mBlobWriter_AddValue_Internal(writer, value, mB_T_Double_4);
}

mFUNCTION(mBlobWriter_AddValue, mPtr<mBlobWriter> &writer, const mString &text)
{
  mFUNCTION_SETUP();

  mERROR_IF(text.hasFailed, mR_ResourceStateInvalid);

  mERROR_CHECK(mBlobWriter_AddValue(writer, text.c_str(), text.bytes));

  mRETURN_SUCCESS();
}

mFUNCTION(mBlobWriter_AddValue, mPtr<mBlobWriter> &writer, const char *text, const size_t length)
{
  mFUNCTION_SETUP();

  mERROR_IF(writer == nullptr, mR_ArgumentNull);
  mERROR_IF(text == nullptr, mR_ArgumentNull);

  size_t parentOffset;
  mERROR_CHECK(mQueue_PeekBack(writer->parentContainerOffsets, &parentOffset));

  mBlob_Container *pParent = reinterpret_cast<mBlob_Container *>(writer->pData + parentOffset);

  switch (pParent->type)
  {
  case mB_T_Container:
  case mB_T_Array:
    break;

  case mB_T_BaseTypeArray: // Base Type arrays cannot contain variable length data.
  default:
    mRETURN_RESULT(mR_ResourceInvalid);
  }

  mBlob_Text blobText;
  blobText.type = mB_T_Text;
  blobText.length = length;

  // This may reallocated `writer->pData`.
  mERROR_CHECK(mBlobWriter_WriteRaw_Internal(writer, blobText));
  mERROR_CHECK(mBlobWriter_WriteData_Internal(writer, text, length));

  pParent = reinterpret_cast<mBlob_Container *>(writer->pData + parentOffset);
  pParent->containerSizeBytes += sizeof(blobText) + length;

  switch (pParent->type)
  {
  case mB_T_Array:
    static_cast<mBlob_Array *>(pParent)->count++;
    break;
  }

  mRETURN_SUCCESS();
}

mFUNCTION(mBlobWriter_AddValue, mPtr<mBlobWriter> &writer, const uint8_t *pData, const size_t bytes)
{
  mFUNCTION_SETUP();

  mERROR_IF(writer == nullptr, mR_ArgumentNull);
  mERROR_IF(pData == nullptr, mR_ArgumentNull);

  size_t parentOffset;
  mERROR_CHECK(mQueue_PeekBack(writer->parentContainerOffsets, &parentOffset));

  mBlob_Container *pParent = reinterpret_cast<mBlob_Container *>(writer->pData + parentOffset);

  switch (pParent->type)
  {
  case mB_T_Container:
  case mB_T_Array:
    break;

  case mB_T_BaseTypeArray: // Base Type arrays cannot contain variable length data.
  default:
    mRETURN_RESULT(mR_ResourceInvalid);
  }

  mBlob_RawData blobText;
  blobText.type = mB_T_RawData;
  blobText.length = bytes;

  // This may reallocated `writer->pData`.
  mERROR_CHECK(mBlobWriter_WriteRaw_Internal(writer, blobText));
  mERROR_CHECK(mBlobWriter_WriteData_Internal(writer, pData, bytes));

  pParent = reinterpret_cast<mBlob_Container *>(writer->pData + parentOffset);
  pParent->containerSizeBytes += sizeof(blobText) + bytes;

  switch (pParent->type)
  {
  case mB_T_Array:
    static_cast<mBlob_Array *>(pParent)->count++;
    break;
  }

  mRETURN_SUCCESS();
}

//////////////////////////////////////////////////////////////////////////

void mBlobWriter_Destroy_Internal(IN_OUT mBlobWriter *pWriter)
{
  if (pWriter == nullptr)
    return;

  mQueue_Destroy(&pWriter->parentContainerOffsets);
  mAllocator_FreePtr(pWriter->pAllocator, &pWriter->pData);
}

mFUNCTION(mBlobWriter_EndVariousContainers, mPtr<mBlobWriter> &writer, const mBlob_Type expectedType)
{
  mFUNCTION_SETUP();

  mERROR_IF(writer == nullptr, mR_ArgumentNull);

  size_t parentOffset;
  mERROR_CHECK(mQueue_PeekBack(writer->parentContainerOffsets, &parentOffset));

  // If someone attempts to leave the header container.
  mERROR_IF(parentOffset == offsetof(mBlob_Header, container), mR_ResourceNotFound);

  mBlob_Container *pContainer = reinterpret_cast<mBlob_Container *>(writer->pData + parentOffset);

  mERROR_IF(pContainer->type != expectedType, mR_ResourceIncompatible);

  // Remove container from the parent container stack.
  mERROR_CHECK(mQueue_PopBack(writer->parentContainerOffsets, &parentOffset));

  // Now, let's have a look at it's parent container.
  mERROR_CHECK(mQueue_PeekBack(writer->parentContainerOffsets, &parentOffset));

  mBlob_Container *pParent = reinterpret_cast<mBlob_Container *>(writer->pData + parentOffset);

  switch (pParent->type)
  {
  case mB_T_Container:
  case mB_T_Array:
    break;

  case mB_T_BaseTypeArray: // Base type arrays cannot contain containers, so I have no idea how you'd end up here, but that's certainly very corrupted.
  default:
    mRETURN_RESULT(mR_ResourceInvalid);
  }

  pParent->containerSizeBytes += pContainer->containerSizeBytes;
  mASSERT_DEBUG(writer->size == pParent->containerSizeBytes + parentOffset, "Invalid Offset");

  mRETURN_SUCCESS();
}

//////////////////////////////////////////////////////////////////////////

struct mBlobReader
{
  mUniqueContainer<mQueue<size_t>> parentContainerOffsets;
  size_t position, size;
  uint8_t *pData;
  mAllocator *pAllocator;
};

//////////////////////////////////////////////////////////////////////////

void mBlobReder_Destroy_Internal(mBlobReader *pReader);

//////////////////////////////////////////////////////////////////////////

template <typename T>
inline mFUNCTION(mBlobReader_ReadValue_Internal, mPtr<mBlobReader> &reader, OUT T *pValue, const mBlob_Type expectedType)
{
  mFUNCTION_SETUP();

  mERROR_IF(reader == nullptr || pValue == nullptr, mR_ArgumentNull);

  size_t parentOffset;
  mERROR_CHECK(mQueue_PeekBack(reader->parentContainerOffsets, &parentOffset));

  const mBlob_Container *pParent = reinterpret_cast<const mBlob_Container *>(reader->pData + parentOffset);

  switch (pParent->type)
  {
  case mB_T_Container:
  case mB_T_Array:
  case mB_T_BaseTypeArray:
    break;

  default:
    mRETURN_RESULT(mR_ResourceInvalid);
  }

  const size_t afterContainerPosition = pParent->containerSizeBytes + parentOffset;

  mERROR_IF(reader->position >= afterContainerPosition, mR_IndexOutOfBounds);

  if (pParent->type == mB_T_BaseTypeArray)
  {
    const mBlob_BaseTypeArray *pArray = static_cast<const mBlob_BaseTypeArray *>(pParent);

    mERROR_IF(pArray->elementType != expectedType, mR_ResourceIncompatible);

    *pValue = *reinterpret_cast<const T *>(reader->pData + reader->position);
    reader->position += sizeof(T);
  }
  else
  {
    const mBlob_BaseType *pBlobBaseType = reinterpret_cast<const mBlob_BaseType *>(reader->pData + reader->position);

    mERROR_IF(expectedType != pBlobBaseType->type, mR_ResourceIncompatible);

    *pValue = static_cast<const mBlob_Value<T> *>(pBlobBaseType)->value;
    reader->position += sizeof(mBlob_Value<T>);
  }

  mRETURN_SUCCESS();
}

//////////////////////////////////////////////////////////////////////////

mFUNCTION(mBlobReader_Create, OUT mPtr<mBlobReader> *pReader, IN mAllocator *pAllocator, IN const uint8_t *pBlobData, const size_t blobSize)
{
  mFUNCTION_SETUP();

  mERROR_IF(pReader == nullptr || pBlobData == nullptr, mR_ArgumentNull);
  mERROR_IF(blobSize < sizeof(mBlob_Header), mR_ResourceInvalid);

  mDEFER_CALL_ON_ERROR(pReader, mSharedPointer_Destroy);
  mERROR_CHECK((mSharedPointer_Allocate<mBlobReader>(pReader, pAllocator, mBlobReder_Destroy_Internal, 1)));

  mBlobReader *pInstance = pReader->GetPointer();

  pInstance->pAllocator = pAllocator;
  pInstance->size = blobSize;

  mERROR_CHECK(mQueue_Create(&pInstance->parentContainerOffsets, pAllocator));

  mERROR_CHECK(mAllocator_Allocate(pInstance->pAllocator, &pInstance->pData, blobSize));
  mERROR_CHECK(mMemcpy(pInstance->pData, pBlobData, blobSize));

  mBlob_Header *pHeader = reinterpret_cast<mBlob_Header *>(pInstance->pData);

  mERROR_IF(pHeader->version > mBlob_Version, mR_ResourceIncompatible);
  mERROR_IF(pHeader->container.type != mB_T_Container, mR_ResourceIncompatible);
  mERROR_IF(pHeader->container.containerSizeBytes < blobSize - offsetof(mBlob_Header, container), mR_ResourceInvalid);

  mERROR_CHECK(mQueue_PushBack(pInstance->parentContainerOffsets, offsetof(mBlob_Header, container)));

  pInstance->position = sizeof(mBlob_Header);

  mRETURN_SUCCESS();
}

mFUNCTION(mBlobReader_Destroy, IN_OUT mPtr<mBlobReader> *pReader)
{
  return mSharedPointer_Destroy(pReader);
}

mFUNCTION(mBlobReader_StepIntoContainer, mPtr<mBlobReader> &reader)
{
  mFUNCTION_SETUP();

  mERROR_IF(reader == nullptr, mR_ArgumentNull);

  size_t parentOffset;
  mERROR_CHECK(mQueue_PeekBack(reader->parentContainerOffsets, &parentOffset));

  const mBlob_Container *pParent = reinterpret_cast<const mBlob_Container *>(reader->pData + parentOffset);

  switch (pParent->type)
  {
  case mB_T_Container:
  case mB_T_Array:
    break;

  case mB_T_BaseTypeArray: // Base Type arrays cannot contain containers.
  default:
    mRETURN_RESULT(mR_ResourceInvalid);
  }

  const size_t afterContainerPosition = pParent->containerSizeBytes + parentOffset;

  mERROR_IF(reader->position >= afterContainerPosition, mR_IndexOutOfBounds);

  const mBlob_BaseType *pBlobBaseType = reinterpret_cast<const mBlob_BaseType *>(reader->pData + reader->position);

  size_t containerSize = 0;

  switch (pBlobBaseType->type)
  {
  case mB_T_Container:
    containerSize = sizeof(mBlob_Container);
    break;

  case mB_T_Array:
    containerSize = sizeof(mBlob_Array);
    break;

  case mB_T_BaseTypeArray:
    containerSize = sizeof(mBlob_BaseTypeArray);
    break;

  default:
    mRETURN_RESULT(mR_ResourceIncompatible);
  }

  mERROR_CHECK(mQueue_PushBack(reader->parentContainerOffsets, reader->position));

  reader->position += containerSize;

  mRETURN_SUCCESS();
}

mFUNCTION(mBlobReader_ExitContainer, mPtr<mBlobReader> &reader)
{
  mFUNCTION_SETUP();

  mERROR_IF(reader == nullptr, mR_ArgumentNull);

  size_t parentOffset;
  mERROR_CHECK(mQueue_PeekBack(reader->parentContainerOffsets, &parentOffset));

  mERROR_IF(parentOffset == offsetof(mBlob_Header, container), mR_ResourceNotFound);

  const mBlob_Container *pContainer = reinterpret_cast<const mBlob_Container *>(reader->pData + parentOffset);

  switch (pContainer->type)
  {
  case mB_T_Container:
  case mB_T_Array:
  case mB_T_BaseTypeArray:
    break;

  default:
    mRETURN_RESULT(mR_ResourceInvalid);
  }

  mERROR_CHECK(mQueue_PopBack(reader->parentContainerOffsets, &parentOffset));

  reader->position = parentOffset + pContainer->containerSizeBytes;

  mRETURN_SUCCESS();
}

mFUNCTION(mBlobReader_GetArrayCount, mPtr<mBlobReader> &reader, OUT size_t *pCount)
{
  mFUNCTION_SETUP();

  mERROR_IF(reader == nullptr || pCount == nullptr, mR_ArgumentNull);

  size_t parentOffset;
  mERROR_CHECK(mQueue_PeekBack(reader->parentContainerOffsets, &parentOffset));

  const mBlob_Container *pParent = reinterpret_cast<const mBlob_Container *>(reader->pData + parentOffset);

  switch (pParent->type)
  {
  case mB_T_Array:
  {
    const mBlob_Array *pArray = static_cast<const mBlob_Array *>(pParent);

    *pCount = pArray->count;

    break;
  }

  case mB_T_BaseTypeArray:
  {
    const mBlob_BaseTypeArray *pArray = static_cast<const mBlob_BaseTypeArray *>(pParent);

    *pCount = pArray->count;

    break;
  }

  case mB_T_Container:
  default:
    mRETURN_RESULT(mR_ResourceInvalid);
  }

  mRETURN_SUCCESS();
}

mFUNCTION(mBlobReader_SkipValue, mPtr<mBlobReader> &reader)
{
  mFUNCTION_SETUP();

  mERROR_IF(reader == nullptr, mR_ArgumentNull);

  size_t parentOffset;
  mERROR_CHECK(mQueue_PeekBack(reader->parentContainerOffsets, &parentOffset));

  const mBlob_Container *pParent = reinterpret_cast<const mBlob_Container *>(reader->pData + parentOffset);

  switch (pParent->type)
  {
  case mB_T_Container:
  case mB_T_Array:
  case mB_T_BaseTypeArray:
    break;

  default:
    mRETURN_RESULT(mR_ResourceInvalid);
  }

  const size_t afterContainerPosition = pParent->containerSizeBytes + parentOffset;

  mERROR_IF(reader->position >= afterContainerPosition, mR_IndexOutOfBounds);

  if (pParent->type == mB_T_BaseTypeArray)
  {
    const mBlob_BaseTypeArray *pArray = static_cast<const mBlob_BaseTypeArray *>(pParent);

    switch (pArray->elementType)
    {
    case mB_T_Empty:
      mRETURN_RESULT(mR_IndexOutOfBounds);

    case mB_T_Container:
    case mB_T_Array:
    case mB_T_BaseTypeArray:
    case mB_T_Text:
    case mB_T_RawData:
    default:
      mRETURN_RESULT(mR_ResourceInvalid); // Base Type arrays cannot contain any variable length data.

    case mB_T_UInt8: reader->position += sizeof(uint8_t); break;
    case mB_T_UInt32: reader->position += sizeof(uint32_t); break;
    case mB_T_Int32: reader->position += sizeof(int32_t); break;
    case mB_T_UInt64: reader->position += sizeof(uint64_t); break;
    case mB_T_Int64: reader->position += sizeof(int64_t); break;
    case mB_T_Float: reader->position += sizeof(float_t); break;
    case mB_T_Double: reader->position += sizeof(double_t); break;
    case mB_T_Bool: reader->position += sizeof(bool); break;
    case mB_T_UInt64_2: reader->position += sizeof(mVec2s); break;
    case mB_T_Int64_2: reader->position += sizeof(mVec2i); break;
    case mB_T_Float_2: reader->position += sizeof(mVec2f); break;
    case mB_T_Float_3: reader->position += sizeof(mVec3f); break;
    case mB_T_Float_4: reader->position += sizeof(mVec4f); break;
    case mB_T_Double_2: reader->position += sizeof(mVec2d); break;
    case mB_T_Double_3: reader->position += sizeof(mVec3d); break;
    case mB_T_Double_4: reader->position += sizeof(mVec4d); break;
    }
  }
  else
  {
    const mBlob_BaseType *pBlobBaseType = reinterpret_cast<const mBlob_BaseType *>(reader->pData + reader->position);

    switch (pBlobBaseType->type)
    {
    case mB_T_Container:
    case mB_T_Array:
    case mB_T_BaseTypeArray:
    {
      const mBlob_Container *pValue = static_cast<const mBlob_Container *>(pBlobBaseType);

      reader->position += pValue->containerSizeBytes;

      break;
    }

    case mB_T_Text:
    {
      const mBlob_Text *pValue = static_cast<const mBlob_Text *>(pBlobBaseType);

      reader->position += sizeof(mBlob_Text) + pValue->length;

      break;
    }

    case mB_T_RawData:
    {
      const mBlob_RawData *pValue = static_cast<const mBlob_RawData *>(pBlobBaseType);

      reader->position += sizeof(mBlob_RawData) + pValue->length;

      break;
    }

    case mB_T_UInt8: reader->position += sizeof(mBlob_Value<uint8_t>); break;
    case mB_T_UInt32: reader->position += sizeof(mBlob_Value<uint32_t>); break;
    case mB_T_Int32: reader->position += sizeof(mBlob_Value<int32_t>); break;
    case mB_T_UInt64: reader->position += sizeof(mBlob_Value<uint64_t>); break;
    case mB_T_Int64: reader->position += sizeof(mBlob_Value<int64_t>); break;
    case mB_T_Float: reader->position += sizeof(mBlob_Value<float_t>); break;
    case mB_T_Double: reader->position += sizeof(mBlob_Value<double_t>); break;
    case mB_T_Bool: reader->position += sizeof(mBlob_Value<bool>); break;
    case mB_T_UInt64_2: reader->position += sizeof(mBlob_Value<mVec2s>); break;
    case mB_T_Int64_2: reader->position += sizeof(mBlob_Value<mVec2i>); break;
    case mB_T_Float_2: reader->position += sizeof(mBlob_Value<mVec2f>); break;
    case mB_T_Float_3: reader->position += sizeof(mBlob_Value<mVec3f>); break;
    case mB_T_Float_4: reader->position += sizeof(mBlob_Value<mVec4f>); break;
    case mB_T_Double_2: reader->position += sizeof(mBlob_Value<mVec2d>); break;
    case mB_T_Double_3: reader->position += sizeof(mBlob_Value<mVec3d>); break;
    case mB_T_Double_4: reader->position += sizeof(mBlob_Value<mVec4d>); break;

    default:
      mRETURN_RESULT(mR_ResourceInvalid);
    }
  }

  mRETURN_SUCCESS();
}

mFUNCTION(mBlobReader_ResetToContainerFront, mPtr<mBlobReader> &reader)
{
  mFUNCTION_SETUP();

  mERROR_IF(reader == nullptr, mR_ArgumentNull);

  size_t parentOffset;
  mERROR_CHECK(mQueue_PeekBack(reader->parentContainerOffsets, &parentOffset));

  const mBlob_Container *pParent = reinterpret_cast<const mBlob_Container *>(reader->pData + parentOffset);

  size_t containerSize = 0;

  switch (pParent->type)
  {
  case mB_T_Container:
    containerSize = sizeof(mBlob_Container);
    break;

  case mB_T_Array:
    containerSize = sizeof(mBlob_Array);
    break;

  case mB_T_BaseTypeArray:
    containerSize = sizeof(mBlob_BaseTypeArray);
    break;

  default:
    mRETURN_RESULT(mR_ResourceInvalid);
  }

  reader->position = parentOffset + containerSize;

  mRETURN_SUCCESS();
}

mFUNCTION(mBlobReader_ReadValue, mPtr<mBlobReader> &reader, OUT uint8_t *pValue)
{
  return mBlobReader_ReadValue_Internal(reader, pValue, mB_T_UInt8);
}

mFUNCTION(mBlobReader_ReadValue, mPtr<mBlobReader> &reader, OUT uint32_t *pValue)
{
  return mBlobReader_ReadValue_Internal(reader, pValue, mB_T_UInt32);
}

mFUNCTION(mBlobReader_ReadValue, mPtr<mBlobReader> &reader, OUT int32_t *pValue)
{
  return mBlobReader_ReadValue_Internal(reader, pValue, mB_T_Int32);
}

mFUNCTION(mBlobReader_ReadValue, mPtr<mBlobReader> &reader, OUT uint64_t *pValue)
{
  return mBlobReader_ReadValue_Internal(reader, pValue, mB_T_UInt64);
}

mFUNCTION(mBlobReader_ReadValue, mPtr<mBlobReader> &reader, OUT int64_t *pValue)
{
  return mBlobReader_ReadValue_Internal(reader, pValue, mB_T_Int64);
}

mFUNCTION(mBlobReader_ReadValue, mPtr<mBlobReader> &reader, OUT float_t *pValue)
{
  return mBlobReader_ReadValue_Internal(reader, pValue, mB_T_Float);
}

mFUNCTION(mBlobReader_ReadValue, mPtr<mBlobReader> &reader, OUT double_t *pValue)
{
  return mBlobReader_ReadValue_Internal(reader, pValue, mB_T_Double);
}

mFUNCTION(mBlobReader_ReadValue, mPtr<mBlobReader> &reader, OUT bool *pValue)
{
  return mBlobReader_ReadValue_Internal(reader, pValue, mB_T_Bool);
}

mFUNCTION(mBlobReader_ReadValue, mPtr<mBlobReader> &reader, OUT mVec2s *pValue)
{
  return mBlobReader_ReadValue_Internal(reader, pValue, mB_T_UInt64_2);
}

mFUNCTION(mBlobReader_ReadValue, mPtr<mBlobReader> &reader, OUT mVec2i *pValue)
{
  return mBlobReader_ReadValue_Internal(reader, pValue, mB_T_Int64_2);
}

mFUNCTION(mBlobReader_ReadValue, mPtr<mBlobReader> &reader, OUT mVec2f *pValue)
{
  return mBlobReader_ReadValue_Internal(reader, pValue, mB_T_Float_2);
}

mFUNCTION(mBlobReader_ReadValue, mPtr<mBlobReader> &reader, OUT mVec3f *pValue)
{
  return mBlobReader_ReadValue_Internal(reader, pValue, mB_T_Float_3);
}

mFUNCTION(mBlobReader_ReadValue, mPtr<mBlobReader> &reader, OUT mVec4f *pValue)
{
  return mBlobReader_ReadValue_Internal(reader, pValue, mB_T_Float_4);
}

mFUNCTION(mBlobReader_ReadValue, mPtr<mBlobReader> &reader, OUT mVec2d *pValue)
{
  return mBlobReader_ReadValue_Internal(reader, pValue, mB_T_Double_2);
}

mFUNCTION(mBlobReader_ReadValue, mPtr<mBlobReader> &reader, OUT mVec3d *pValue)
{
  return mBlobReader_ReadValue_Internal(reader, pValue, mB_T_Double_3);
}

mFUNCTION(mBlobReader_ReadValue, mPtr<mBlobReader> &reader, OUT mVec4d *pValue)
{
  return mBlobReader_ReadValue_Internal(reader, pValue, mB_T_Double_4);
}

mFUNCTION(mBlobReader_ReadValue, mPtr<mBlobReader> &reader, OUT mString *pString)
{
  mFUNCTION_SETUP();

  mERROR_IF(pString == nullptr, mR_ArgumentNull);

  const char *text = nullptr;
  size_t length = 0;

  mERROR_CHECK(mBlobReader_ReadValue(reader, &text, &length));
  mERROR_CHECK(mString_Create(pString, text, length, pString->pAllocator));

  mRETURN_SUCCESS();
}

mFUNCTION(mBlobReader_ReadValue, mPtr<mBlobReader> &reader, OUT const char **pText, OUT size_t *pLength)
{
  mFUNCTION_SETUP();

  mERROR_IF(reader == nullptr, mR_ArgumentNull);
  mERROR_IF(pText == nullptr || pLength == nullptr, mR_ArgumentNull);

  size_t parentOffset;
  mERROR_CHECK(mQueue_PeekBack(reader->parentContainerOffsets, &parentOffset));

  const mBlob_Container *pParent = reinterpret_cast<const mBlob_Container *>(reader->pData + parentOffset);

  switch (pParent->type)
  {
  case mB_T_Container:
  case mB_T_Array:
    break;

  case mB_T_BaseTypeArray: // Base Type arrays cannot contain containers.
  default:
    mRETURN_RESULT(mR_ResourceInvalid);
  }

  const size_t afterContainerPosition = pParent->containerSizeBytes - parentOffset;

  mERROR_IF(reader->position >= afterContainerPosition, mR_IndexOutOfBounds);

  const mBlob_BaseType *pBlobBaseType = reinterpret_cast<const mBlob_BaseType *>(reader->pData + reader->position);

  mERROR_IF(pBlobBaseType->type != mB_T_Text, mR_ResourceIncompatible);

  const mBlob_Text *pTextBlob = static_cast<const mBlob_Text *>(pBlobBaseType);

  *pLength = pTextBlob->length;
  *pText = reinterpret_cast<const char *>(pTextBlob) + sizeof(mBlob_Text);

  reader->position += sizeof(mBlob_Text) + pTextBlob->length;

  mRETURN_SUCCESS();
}

mFUNCTION(mBlobReader_ReadValue, mPtr<mBlobReader> &reader, OUT const uint8_t **ppData, OUT size_t *pBytes)
{
  mFUNCTION_SETUP();

  mERROR_IF(reader == nullptr, mR_ArgumentNull);
  mERROR_IF(ppData == nullptr || pBytes == nullptr, mR_ArgumentNull);

  size_t parentOffset;
  mERROR_CHECK(mQueue_PeekBack(reader->parentContainerOffsets, &parentOffset));

  const mBlob_Container *pParent = reinterpret_cast<const mBlob_Container *>(reader->pData + parentOffset);

  switch (pParent->type)
  {
  case mB_T_Container:
  case mB_T_Array:
    break;

  case mB_T_BaseTypeArray: // Base Type arrays cannot contain containers.
  default:
    mRETURN_RESULT(mR_ResourceInvalid);
  }

  const size_t afterContainerPosition = pParent->containerSizeBytes - parentOffset;

  mERROR_IF(reader->position >= afterContainerPosition, mR_IndexOutOfBounds);

  const mBlob_BaseType *pBlobBaseType = reinterpret_cast<const mBlob_BaseType *>(reader->pData + reader->position);

  mERROR_IF(pBlobBaseType->type != mB_T_RawData, mR_ResourceIncompatible);

  const mBlob_RawData *pDataBlob = static_cast<const mBlob_RawData *>(pBlobBaseType);

  *pBytes = pDataBlob->length;
  *ppData = reinterpret_cast<const uint8_t *>(pDataBlob) + sizeof(mBlob_RawData);

  reader->position += sizeof(mBlob_RawData) + pDataBlob->length;

  mRETURN_SUCCESS();
}

//////////////////////////////////////////////////////////////////////////

void mBlobReder_Destroy_Internal(mBlobReader *pReader)
{
  if (pReader == nullptr)
    return;

  mQueue_Destroy(&pReader->parentContainerOffsets);
  mAllocator_FreePtr(pReader->pAllocator, &pReader->pData);
}
