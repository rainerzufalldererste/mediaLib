#include "mAttribute.h"

#ifdef GIT_BUILD // Define __M_FILE__
  #ifdef __M_FILE__
    #undef __M_FILE__
  #endif
  #define __M_FILE__ "yrJzBu1GHHvRdWuSQL3UuZATjv2SSi1bfo9l1SiL8qxtE+C35WJSU4WRX/pnRF2Bdl5qWCH7Xogfh++2"
#endif

mFUNCTION(mDestruct, IN_OUT mAttribute *pAttribute)
{
  if (pAttribute != nullptr)
  {
    if (pAttribute->type == mA_T_String || pAttribute->type == mA_T_SkipString)
    {
      mDestruct(pAttribute->String.pValue);
      mAllocator_FreePtr(&mDefaultAllocator, &pAttribute->String.pValue);
    }
    
    pAttribute->type = mA_T_Null;
  }

  return mR_Success;
}

//////////////////////////////////////////////////////////////////////////

mFUNCTION(mAttributeStore_Create, OUT mAttributeStore *pAttributeStore, IN mAllocator *pAllocator)
{
  mFUNCTION_SETUP();

  mERROR_IF(pAttributeStore == nullptr, mR_ArgumentNull);

  mERROR_CHECK(mQueue_Create(&pAttributeStore->attributes, pAllocator));

  mRETURN_SUCCESS();
}

mFUNCTION(mAttributeStore_Destroy, IN_OUT mAttributeStore *pAttributeStore)
{
  mFUNCTION_SETUP();

  mERROR_IF(pAttributeStore == nullptr, mR_ArgumentNull);

  mERROR_CHECK(mQueue_Destroy(&pAttributeStore->attributes));

  mRETURN_SUCCESS();
}

//////////////////////////////////////////////////////////////////////////

mFUNCTION(mAttributeCollection_Create, OUT mAttributeCollection *pAttributeCollection, IN mAttributeStore *pAttributeStore)
{
  mFUNCTION_SETUP();

  mERROR_IF(pAttributeCollection == nullptr, mR_ArgumentNull);

  pAttributeCollection->pStore = pAttributeStore;
  pAttributeCollection->attributeStartIndex = (size_t)-1;

  mRETURN_SUCCESS();
}

mFUNCTION(mAttributeCollection_CreateIterator, IN mAttributeCollection *pAttributeCollection, OUT mAttributeIterator *pAttributeIterator)
{
  mFUNCTION_SETUP();

  mERROR_IF(pAttributeCollection == nullptr || pAttributeIterator == nullptr, mR_ArgumentNull);

  if (pAttributeCollection->attributeStartIndex == (size_t)-1)
  {
    mAttribute attribute;
    attribute.type = mA_T_Object;
    attribute.Object.parentIndex = (uint32_t)-1;
    attribute.Object.skip = 1;

    mAttribute emptyObject;
    emptyObject.type = mA_T_EmptyObject;
    
    mERROR_CHECK(mQueue_PushBack(pAttributeCollection->pStore->attributes, std::move(attribute)));
    mERROR_CHECK(mQueue_PushBack(pAttributeCollection->pStore->attributes, std::move(emptyObject)));

    mERROR_CHECK(mQueue_GetCount(pAttributeCollection->pStore->attributes, &pAttributeCollection->attributeStartIndex));
    pAttributeCollection->attributeStartIndex -= 2;
  }

  pAttributeIterator->currentAttributeIndex = pAttributeIterator->currentContainerAttributeIndex = pAttributeCollection->attributeStartIndex;
  pAttributeIterator->objectIndex = (size_t)-1;
  pAttributeIterator->pStore = pAttributeCollection->pStore;

  mRETURN_SUCCESS();
}

mFUNCTION(mAttributeIterator_Reset, IN_OUT mAttributeIterator *pAttributeIterator)
{
  mFUNCTION_SETUP();

  mERROR_IF(pAttributeIterator == nullptr, mR_ArgumentNull);

  pAttributeIterator->currentAttributeIndex = pAttributeIterator->currentContainerAttributeIndex;
  pAttributeIterator->objectIndex = (size_t)-1;

  mRETURN_SUCCESS();
}

mFUNCTION(mAttributeIterator_Next, IN_OUT mAttributeIterator *pAttributeIterator)
{
  mFUNCTION_SETUP();

  mERROR_IF(pAttributeIterator == nullptr, mR_ArgumentNull);

  mAttribute *pObject = nullptr;
  mERROR_CHECK(mQueue_PointerAt(pAttributeIterator->pStore->attributes, pAttributeIterator->currentContainerAttributeIndex, &pObject));

  mERROR_IF(pObject->type != mA_T_Object, mR_ResourceInvalid);
  mERROR_IF(pObject->Object.size <= pAttributeIterator->objectIndex + 1, mR_IndexOutOfBounds);

  if (pAttributeIterator->objectIndex != (size_t)-1)
  {
    mAttribute *pCurrent = nullptr;
    mERROR_CHECK(mQueue_PointerAt(pAttributeIterator->pStore->attributes, pAttributeIterator->currentAttributeIndex, &pCurrent));

    if ((pCurrent->type & (mA_T_InactiveValue - 1)) == mA_T_Object)
      pAttributeIterator->currentAttributeIndex += pCurrent->Object.skip;
  }

  pAttributeIterator->objectIndex++;
  pAttributeIterator->currentAttributeIndex++;

  mAttribute *pCurrent = nullptr;

  while (true)
  {
    mERROR_CHECK(mQueue_PointerAt(pAttributeIterator->pStore->attributes, pAttributeIterator->currentAttributeIndex, &pCurrent));

    if (pCurrent->type == mA_T_SkipObject)
    {
      pAttributeIterator->currentAttributeIndex += pCurrent->Object.skip + 1;
    }
    else if ((pCurrent->type & mA_T_InactiveValue) == mA_T_InactiveValue)
    {
      pAttributeIterator->currentAttributeIndex++;
    }
    else if (pCurrent->type == mA_T_SkipTo)
    {
      pAttributeIterator->currentAttributeIndex = pCurrent->SkipTo.nextAttributeIndex;
    }
    else if (pCurrent->type == mA_T_EmptyObject)
    {
      mASSERT_DEBUG(false, "Should've been rejected at size already.");
      mRETURN_RESULT(mR_ResourceInvalid);
    }
    else
    {
      break;
    }
  }

  mRETURN_SUCCESS();
}

mFUNCTION(mAttributeIterator_Find, IN_OUT mAttributeIterator *pAttributeIterator, const mString &name)
{
  mFUNCTION_SETUP();

  mERROR_IF(pAttributeIterator == nullptr, mR_ArgumentNull);
  mERROR_IF(name.hasFailed || name.count <= 1, mR_InvalidParameter);
  mERROR_IF(name.bytes > mAttribute_NameLength, mR_InvalidParameter);

  mERROR_CHECK(mAttributeIterator_Reset(pAttributeIterator));

  mAttribute *pObject = nullptr;
  mERROR_CHECK(mQueue_PointerAt(pAttributeIterator->pStore->attributes, pAttributeIterator->currentContainerAttributeIndex, &pObject));

  mERROR_IF(pObject->type != mA_T_Object, mR_ResourceInvalid);

  for (size_t i = 0; i < pObject->Object.size; i++)
  {
    mERROR_CHECK(mAttributeIterator_Next(pAttributeIterator));

    mAttribute *pCurrent = nullptr;
    mERROR_CHECK(mQueue_PointerAt(pAttributeIterator->pStore->attributes, pAttributeIterator->currentAttributeIndex, &pCurrent));

    if (pCurrent->name == name)
      mRETURN_SUCCESS();
  }

  mRETURN_RESULT(mR_ResourceNotFound);
}

mFUNCTION(mAttributeIterator_Find, IN_OUT mAttributeIterator *pAttributeIterator, const mInplaceString<mAttribute_NameLength> &name)
{
  mFUNCTION_SETUP();

  mERROR_IF(pAttributeIterator == nullptr, mR_ArgumentNull);
  mERROR_IF(name.count <= 1, mR_InvalidParameter);
  mERROR_IF(name.bytes > mAttribute_NameLength, mR_InvalidParameter);

  mERROR_CHECK(mAttributeIterator_Reset(pAttributeIterator));

  mAttribute *pObject = nullptr;
  mERROR_CHECK(mQueue_PointerAt(pAttributeIterator->pStore->attributes, pAttributeIterator->currentContainerAttributeIndex, &pObject));

  mERROR_IF(pObject->type != mA_T_Object, mR_ResourceInvalid);

  for (size_t i = 0; i < pObject->Object.size; i++)
  {
    mERROR_CHECK(mAttributeIterator_Next(pAttributeIterator));

    mAttribute *pCurrent = nullptr;
    mERROR_CHECK(mQueue_PointerAt(pAttributeIterator->pStore->attributes, pAttributeIterator->currentAttributeIndex, &pCurrent));

    if (pCurrent->name == name)
      mRETURN_SUCCESS();
  }

  mRETURN_RESULT(mR_ResourceNotFound);
}

mFUNCTION(mAttributeIterator_AddAttribute_Internal, IN_OUT mAttributeIterator *pAttributeIterator, mAttribute &&attribute)
{
  mFUNCTION_SETUP();

  mERROR_IF(pAttributeIterator == nullptr, mR_ArgumentNull);

  size_t count;
  mERROR_CHECK(mQueue_GetCount(pAttributeIterator->pStore->attributes, &count));
  
  const size_t originalCount = count;

  mAttribute *pObject = nullptr;
  mERROR_CHECK(mQueue_PointerAt(pAttributeIterator->pStore->attributes, pAttributeIterator->currentContainerAttributeIndex, &pObject));

  mERROR_IF(pObject->type != mA_T_Object, mR_ResourceInvalid);
  mERROR_IF(pObject->Object.size == UINT32_MAX, mR_IndexOutOfBounds);

  mERROR_CHECK(mAttributeIterator_Reset(pAttributeIterator));

  while (pAttributeIterator->objectIndex + 1 != pObject->Object.size)
  {
    mERROR_CHECK(mAttributeIterator_Next(pAttributeIterator));
    
    mAttribute *pCurrent = nullptr;
    mERROR_CHECK(mQueue_PointerAt(pAttributeIterator->pStore->attributes, pAttributeIterator->currentAttributeIndex, &pCurrent));

    mERROR_IF(pCurrent->name == attribute.name, mR_ResourceAlreadyExists);
  }

  // If the object might be empty.
  if (pObject->Object.skip == 1 && pObject->Object.size == 0)
  {
    mAttribute *pCurrent = nullptr;
    mERROR_CHECK(mQueue_PointerAt(pAttributeIterator->pStore->attributes, pAttributeIterator->currentContainerAttributeIndex + 1, &pCurrent));

    if (pCurrent->type == mA_T_EmptyObject)
    {
      *pCurrent = std::move(attribute);
      
      pObject->Object.size++;
      pObject->Object.added++;

      pAttributeIterator->objectIndex = 0;

      // No need to bubble up skip, because we didn't add anything.

      mRETURN_SUCCESS();
    }
  }

  // If the last element of the object is currently at the end of the AttributeStore and all elements are stored in order.
  if (pAttributeIterator->currentContainerAttributeIndex + pObject->Object.skip + 1 == count)
  {
    mERROR_CHECK(mQueue_PushBack(pAttributeIterator->pStore->attributes, std::move(attribute)));
    mERROR_CHECK(mQueue_PointerAt(pAttributeIterator->pStore->attributes, pAttributeIterator->currentContainerAttributeIndex, &pObject));

    pObject->Object.size++;
    pObject->Object.skip++;
    pObject->Object.added++;

    pAttributeIterator->currentAttributeIndex = count;
    pAttributeIterator->objectIndex++;
  }
  else
  {
    // If the last element of the object is currently at the end of the AttributeStore.
    if (pAttributeIterator->currentAttributeIndex + 1 == count)
    {
      mERROR_CHECK(mQueue_PushBack(pAttributeIterator->pStore->attributes, std::move(attribute)));
      mERROR_CHECK(mQueue_PointerAt(pAttributeIterator->pStore->attributes, pAttributeIterator->currentContainerAttributeIndex, &pObject));

      pObject->Object.size++;
      pObject->Object.added++;

      pAttributeIterator->currentAttributeIndex = count;
      pAttributeIterator->objectIndex++;
    }
    else
    {
      // Move previously last one and replace with `SkipTo`.
      mAttribute *pPreviouslyLastOne = nullptr;
      mERROR_CHECK(mQueue_PointerAt(pAttributeIterator->pStore->attributes, pAttributeIterator->currentAttributeIndex, &pPreviouslyLastOne));

      mAttribute previousAttribute = std::move(*pPreviouslyLastOne);

      pPreviouslyLastOne->type = mA_T_SkipTo;
      pPreviouslyLastOne->SkipTo.nextAttributeIndex = count;
      count++;

      // If that was an object, add a `SkipTo` to the first element.
      if ((previousAttribute.type & (mA_T_InactiveValue - 1)) == mA_T_Object)
      {
        previousAttribute.Object.skip = 1;

        mERROR_CHECK(mQueue_PushBack(pAttributeIterator->pStore->attributes, std::move(previousAttribute)));

        mAttribute skipTo;
        skipTo.type = mA_T_SkipTo;
        skipTo.SkipTo.nextAttributeIndex = pAttributeIterator->currentAttributeIndex + 1;

        mERROR_CHECK(mQueue_PushBack(pAttributeIterator->pStore->attributes, std::move(attribute)));

        count++;
      }
      else
      {
        mERROR_CHECK(mQueue_PushBack(pAttributeIterator->pStore->attributes, std::move(previousAttribute)));
      }

      // Add new value.
      mERROR_CHECK(mQueue_PushBack(pAttributeIterator->pStore->attributes, std::move(attribute)));
      mERROR_CHECK(mQueue_PointerAt(pAttributeIterator->pStore->attributes, pAttributeIterator->currentContainerAttributeIndex, &pObject));

      pObject->Object.size++;
      pObject->Object.added++;

      pAttributeIterator->currentAttributeIndex = count;
      pAttributeIterator->objectIndex++;
    }
  }

  uint32_t parentIndex = pObject->Object.parentIndex;

  // Bubble up skip.
  while (parentIndex != (uint32_t)-1)
  {
    mAttribute *pParent = nullptr;
    mERROR_CHECK(mQueue_PointerAt(pAttributeIterator->pStore->attributes, parentIndex, &pParent));

    mERROR_IF(pParent->type != mA_T_Object, mR_ResourceInvalid);

    if (parentIndex + pParent->Object.skip + 1 == originalCount)
      pParent->Object.skip += (uint32_t)(count - originalCount + 1);
    else
      break; // If that's not the case for one of the parents, than it'll also not be true for all of its parents.

    parentIndex = pParent->Object.parentIndex;
  }

  mRETURN_SUCCESS();
}

mFUNCTION(mAttributeIterator_AddUInt64, IN_OUT mAttributeIterator *pAttributeIterator, const mString &name, const uint64_t value)
{
  mFUNCTION_SETUP();

  mERROR_IF(pAttributeIterator == nullptr, mR_ArgumentNull);
  mERROR_IF(name.hasFailed || name.count <= 1, mR_InvalidParameter);
  mERROR_IF(name.bytes > mAttribute_NameLength, mR_InvalidParameter);

  mAttribute attribute;
  mERROR_CHECK(mInplaceString_Create(&attribute.name, name));
  attribute.type = mA_T_UInt64;
  attribute.UInt64.value = value;

  mERROR_CHECK(mAttributeIterator_AddAttribute_Internal(pAttributeIterator, std::move(attribute)));

  mRETURN_SUCCESS();
}

mFUNCTION(mAttributeIterator_AddInt64, IN_OUT mAttributeIterator *pAttributeIterator, const mString &name, const int64_t value)
{
  mFUNCTION_SETUP();

  mERROR_IF(pAttributeIterator == nullptr, mR_ArgumentNull);
  mERROR_IF(name.hasFailed || name.count <= 1, mR_InvalidParameter);
  mERROR_IF(name.bytes > mAttribute_NameLength, mR_InvalidParameter);

  mAttribute attribute;
  mERROR_CHECK(mInplaceString_Create(&attribute.name, name));
  attribute.type = mA_T_Int64;
  attribute.Int64.value = value;

  mERROR_CHECK(mAttributeIterator_AddAttribute_Internal(pAttributeIterator, std::move(attribute)));

  mRETURN_SUCCESS();
}

mFUNCTION(mAttributeIterator_AddBool, IN_OUT mAttributeIterator *pAttributeIterator, const mString &name, const bool value)
{
  mFUNCTION_SETUP();

  mERROR_IF(pAttributeIterator == nullptr, mR_ArgumentNull);
  mERROR_IF(name.hasFailed || name.count <= 1, mR_InvalidParameter);
  mERROR_IF(name.bytes > mAttribute_NameLength, mR_InvalidParameter);

  mAttribute attribute;
  mERROR_CHECK(mInplaceString_Create(&attribute.name, name));
  attribute.type = mA_T_Bool;
  attribute.Bool.value = value;

  mERROR_CHECK(mAttributeIterator_AddAttribute_Internal(pAttributeIterator, std::move(attribute)));

  mRETURN_SUCCESS();
}

mFUNCTION(mAttributeIterator_AddFloat64, IN_OUT mAttributeIterator *pAttributeIterator, const mString &name, const double_t value)
{
  mFUNCTION_SETUP();

  mERROR_IF(pAttributeIterator == nullptr, mR_ArgumentNull);
  mERROR_IF(name.hasFailed || name.count <= 1, mR_InvalidParameter);
  mERROR_IF(name.bytes > mAttribute_NameLength, mR_InvalidParameter);

  mAttribute attribute;
  mERROR_CHECK(mInplaceString_Create(&attribute.name, name));
  attribute.type = mA_T_Float64;
  attribute.Float64.value = value;

  mERROR_CHECK(mAttributeIterator_AddAttribute_Internal(pAttributeIterator, std::move(attribute)));

  mRETURN_SUCCESS();
}

mFUNCTION(mAttributeIterator_AddVector2, IN_OUT mAttributeIterator *pAttributeIterator, const mString &name, const mVec2f value)
{
  mFUNCTION_SETUP();

  mERROR_IF(pAttributeIterator == nullptr, mR_ArgumentNull);
  mERROR_IF(name.hasFailed || name.count <= 1, mR_InvalidParameter);
  mERROR_IF(name.bytes > mAttribute_NameLength, mR_InvalidParameter);

  mAttribute attribute;
  mERROR_CHECK(mInplaceString_Create(&attribute.name, name));
  attribute.type = mA_T_Vector2;
  attribute.Vector2.value = value;

  mERROR_CHECK(mAttributeIterator_AddAttribute_Internal(pAttributeIterator, std::move(attribute)));

  mRETURN_SUCCESS();
}

mFUNCTION(mAttributeIterator_AddVector3, IN_OUT mAttributeIterator *pAttributeIterator, const mString &name, const mVec3f value)
{
  mFUNCTION_SETUP();

  mERROR_IF(pAttributeIterator == nullptr, mR_ArgumentNull);
  mERROR_IF(name.hasFailed || name.count <= 1, mR_InvalidParameter);
  mERROR_IF(name.bytes > mAttribute_NameLength, mR_InvalidParameter);

  mAttribute attribute;
  mERROR_CHECK(mInplaceString_Create(&attribute.name, name));
  attribute.type = mA_T_Vector3;
  attribute.Vector3.value = value;

  mERROR_CHECK(mAttributeIterator_AddAttribute_Internal(pAttributeIterator, std::move(attribute)));

  mRETURN_SUCCESS();
}

mFUNCTION(mAttributeIterator_AddVector4, IN_OUT mAttributeIterator *pAttributeIterator, const mString &name, const mVec4f value)
{
  mFUNCTION_SETUP();

  mERROR_IF(pAttributeIterator == nullptr, mR_ArgumentNull);
  mERROR_IF(name.hasFailed || name.count <= 1, mR_InvalidParameter);
  mERROR_IF(name.bytes > mAttribute_NameLength, mR_InvalidParameter);

  mAttribute attribute;
  mERROR_CHECK(mInplaceString_Create(&attribute.name, name));
  attribute.type = mA_T_Vector4;
  attribute.Vector4.value = value;

  mERROR_CHECK(mAttributeIterator_AddAttribute_Internal(pAttributeIterator, std::move(attribute)));

  mRETURN_SUCCESS();
}

mFUNCTION(mAttributeIterator_AddString, IN_OUT mAttributeIterator *pAttributeIterator, const mString &name, const mString &value)
{
  mFUNCTION_SETUP();

  mERROR_IF(pAttributeIterator == nullptr, mR_ArgumentNull);
  mERROR_IF(name.hasFailed || name.count <= 1, mR_InvalidParameter);
  mERROR_IF(name.bytes > mAttribute_NameLength, mR_InvalidParameter);

  mString *pString = nullptr;
  mDEFER_ON_ERROR(mAllocator_FreePtr(&mDefaultAllocator, &pString));
  mERROR_CHECK(mAllocator_AllocateZero(&mDefaultAllocator, &pString, 1));

  mDEFER_ON_ERROR(mDestruct(pString));
  mERROR_CHECK(mString_Create(pString, value, &mDefaultAllocator));

  mAttribute attribute;
  mERROR_CHECK(mInplaceString_Create(&attribute.name, name));
  attribute.type = mA_T_String;
  attribute.String.pValue = pString;

  mERROR_CHECK(mAttributeIterator_AddAttribute_Internal(pAttributeIterator, std::move(attribute)));

  mRETURN_SUCCESS();
}

mFUNCTION(mAttributeIterator_AddObject, IN_OUT mAttributeIterator *pAttributeIterator, const mString &name)
{
  mFUNCTION_SETUP();

  mERROR_IF(pAttributeIterator == nullptr, mR_ArgumentNull);
  mERROR_IF(name.hasFailed || name.count <= 1, mR_InvalidParameter);
  mERROR_IF(name.bytes > mAttribute_NameLength, mR_InvalidParameter);
  
  // Add Object Attribute.
  {
    mAttribute attribute;
    mERROR_CHECK(mInplaceString_Create(&attribute.name, name));
    attribute.type = mA_T_Object;
    attribute.Object.parentIndex = (uint32_t)pAttributeIterator->currentContainerAttributeIndex;

    mERROR_CHECK(mAttributeIterator_AddAttribute_Internal(pAttributeIterator, std::move(attribute)));
  }

  size_t count; // Don't modify this. Will be needed to enter the object later.
  mERROR_CHECK(mQueue_GetCount(pAttributeIterator->pStore->attributes, &count));

  // Add Empty Object Attribute.
  {
    mAttribute attribute;
    attribute.type = mA_T_EmptyObject;

    mERROR_CHECK(mQueue_PushBack(pAttributeIterator->pStore->attributes, std::move(attribute)));
    
    mAttribute *pObject = nullptr;
    mERROR_CHECK(mQueue_PointerAt(pAttributeIterator->pStore->attributes, count - 1, &pObject));

    mASSERT_DEBUG(pObject->type == mA_T_Object, "Object should have been added to end.");

    pObject->Object.skip = 1;

    uint32_t parentIndex = pObject->Object.parentIndex;

    // Bubble up skip.
    while (parentIndex != (uint32_t)-1)
    {
      mAttribute *pParent = nullptr;
      mERROR_CHECK(mQueue_PointerAt(pAttributeIterator->pStore->attributes, parentIndex, &pParent));

      mERROR_IF(pParent->type != mA_T_Object, mR_ResourceInvalid);

      if (parentIndex + pParent->Object.skip + 1 == count)
        pParent->Object.skip++;
      else
        break; // If that's not the case for one of the parents, than it'll also not be true for all of its parents.

      parentIndex = pParent->Object.parentIndex;
    }
  }

  // Enter Object Attribute.
  {
    pAttributeIterator->currentAttributeIndex = pAttributeIterator->currentContainerAttributeIndex = count - 1;
    pAttributeIterator->objectIndex = (size_t)-1;
  }

  mRETURN_SUCCESS();
}

mFUNCTION(mAttributeIterator_LeaveObject, IN_OUT mAttributeIterator *pAttributeIterator)
{
  mFUNCTION_SETUP();

  mERROR_IF(pAttributeIterator == nullptr, mR_ArgumentNull);

  mAttribute *pObject = nullptr;
  mERROR_CHECK(mQueue_PointerAt(pAttributeIterator->pStore->attributes, pAttributeIterator->currentContainerAttributeIndex, &pObject));

  mERROR_IF(pObject->Object.parentIndex == (uint32_t)-1, mR_ResourceIncompatible);
  mERROR_IF(pObject->type != mA_T_Object, mR_ResourceInvalid);

  pAttributeIterator->currentContainerAttributeIndex = pObject->Object.parentIndex;

  mRETURN_SUCCESS();
}

mFUNCTION(mAttributeIterator_GetCurrentAttribute_Internal, IN mAttributeIterator *pAttributeIterator, OUT mAttribute **ppAttribute)
{
  mFUNCTION_SETUP();

  if (pAttributeIterator->objectIndex == (uint32_t)-1)
  {
    mAttribute *pObject = nullptr;
    mERROR_CHECK(mQueue_PointerAt(pAttributeIterator->pStore->attributes, pAttributeIterator->currentContainerAttributeIndex, &pObject));

    mERROR_IF(pObject->type != mA_T_Object, mR_ResourceInvalid);
    mERROR_IF(pObject->Object.size == 0, mR_IndexOutOfBounds);

    mERROR_CHECK(mAttributeIterator_Next(pAttributeIterator));
  }

  mERROR_CHECK(mQueue_PointerAt(pAttributeIterator->pStore->attributes, pAttributeIterator->currentAttributeIndex, ppAttribute));

  mRETURN_SUCCESS();
}

mFUNCTION(mAttributeIterator_GetCurrentAttributeName, IN mAttributeIterator *pAttributeIterator, OUT mString *pName)
{
  mFUNCTION_SETUP();

  mERROR_IF(pAttributeIterator == nullptr || pName == nullptr, mR_ArgumentNull);

  mAttribute *pCurrent = nullptr;
  mERROR_CHECK(mAttributeIterator_GetCurrentAttribute_Internal(pAttributeIterator, &pCurrent));

  mERROR_CHECK(mString_Create(pName, pCurrent->name, pName->pAllocator));

  mRETURN_SUCCESS();
}

mFUNCTION(mAttributeIterator_GetCurrentAttributeName, IN mAttributeIterator *pAttributeIterator, OUT mInplaceString<mAttribute_NameLength> *pName)
{
  mFUNCTION_SETUP();

  mERROR_IF(pAttributeIterator == nullptr || pName == nullptr, mR_ArgumentNull);

  mAttribute *pCurrent = nullptr;
  mERROR_CHECK(mAttributeIterator_GetCurrentAttribute_Internal(pAttributeIterator, &pCurrent));
  
  mERROR_CHECK(mInplaceString_Create(pName, pCurrent->name));

  mRETURN_SUCCESS();
}

mFUNCTION(mAttributeIterator_GetCurrentAttributeType, IN mAttributeIterator *pAttributeIterator, OUT mAttribute_Type *pType)
{
  mFUNCTION_SETUP();

  mERROR_IF(pAttributeIterator == nullptr || pType == nullptr, mR_ArgumentNull);

  mAttribute *pCurrent = nullptr;
  mERROR_CHECK(mAttributeIterator_GetCurrentAttribute_Internal(pAttributeIterator, &pCurrent));

  *pType = pCurrent->type;

  mRETURN_SUCCESS();
}

mFUNCTION(mAttributeIterator_GetCurrentValue, IN mAttributeIterator *pAttributeIterator, OUT uint64_t *pValue)
{
  mFUNCTION_SETUP();

  mERROR_IF(pAttributeIterator == nullptr || pValue == nullptr, mR_ArgumentNull);

  mAttribute *pCurrent = nullptr;
  mERROR_CHECK(mAttributeIterator_GetCurrentAttribute_Internal(pAttributeIterator, &pCurrent));

  mERROR_IF(pCurrent->type != mA_T_UInt64, mR_ResourceIncompatible);

  *pValue = pCurrent->UInt64.value;

  mRETURN_SUCCESS();
}

mFUNCTION(mAttributeIterator_GetCurrentValue, IN mAttributeIterator *pAttributeIterator, OUT int64_t *pValue)
{
  mFUNCTION_SETUP();

  mERROR_IF(pAttributeIterator == nullptr || pValue == nullptr, mR_ArgumentNull);

  mAttribute *pCurrent = nullptr;
  mERROR_CHECK(mAttributeIterator_GetCurrentAttribute_Internal(pAttributeIterator, &pCurrent));

  mERROR_IF(pCurrent->type != mA_T_Int64, mR_ResourceIncompatible);

  *pValue = pCurrent->Int64.value;

  mRETURN_SUCCESS();
}

mFUNCTION(mAttributeIterator_GetCurrentValue, IN mAttributeIterator *pAttributeIterator, OUT bool *pValue)
{
  mFUNCTION_SETUP();

  mERROR_IF(pAttributeIterator == nullptr || pValue == nullptr, mR_ArgumentNull);

  mAttribute *pCurrent = nullptr;
  mERROR_CHECK(mAttributeIterator_GetCurrentAttribute_Internal(pAttributeIterator, &pCurrent));

  mERROR_IF(pCurrent->type != mA_T_Bool, mR_ResourceIncompatible);

  *pValue = pCurrent->Bool.value;

  mRETURN_SUCCESS();
}

mFUNCTION(mAttributeIterator_GetCurrentValue, IN mAttributeIterator *pAttributeIterator, OUT double_t *pValue)
{
  mFUNCTION_SETUP();

  mERROR_IF(pAttributeIterator == nullptr || pValue == nullptr, mR_ArgumentNull);

  mAttribute *pCurrent = nullptr;
  mERROR_CHECK(mAttributeIterator_GetCurrentAttribute_Internal(pAttributeIterator, &pCurrent));

  mERROR_IF(pCurrent->type != mA_T_Float64, mR_ResourceIncompatible);

  *pValue = pCurrent->Float64.value;

  mRETURN_SUCCESS();
}

mFUNCTION(mAttributeIterator_GetCurrentValue, IN mAttributeIterator *pAttributeIterator, OUT float_t *pValue)
{
  mFUNCTION_SETUP();

  mERROR_IF(pAttributeIterator == nullptr || pValue == nullptr, mR_ArgumentNull);

  mAttribute *pCurrent = nullptr;
  mERROR_CHECK(mAttributeIterator_GetCurrentAttribute_Internal(pAttributeIterator, &pCurrent));

  mERROR_IF(pCurrent->type != mA_T_Float64, mR_ResourceIncompatible);

  *pValue = (float_t)pCurrent->Float64.value;

  mRETURN_SUCCESS();
}

mFUNCTION(mAttributeIterator_GetCurrentValue, IN mAttributeIterator *pAttributeIterator, OUT mVec2f *pValue)
{
  mFUNCTION_SETUP();

  mERROR_IF(pAttributeIterator == nullptr || pValue == nullptr, mR_ArgumentNull);

  mAttribute *pCurrent = nullptr;
  mERROR_CHECK(mAttributeIterator_GetCurrentAttribute_Internal(pAttributeIterator, &pCurrent));

  mERROR_IF(pCurrent->type != mA_T_Vector2, mR_ResourceIncompatible);

  *pValue = pCurrent->Vector2.value;

  mRETURN_SUCCESS();
}

mFUNCTION(mAttributeIterator_GetCurrentValue, IN mAttributeIterator *pAttributeIterator, OUT mVec3f *pValue)
{
  mFUNCTION_SETUP();

  mERROR_IF(pAttributeIterator == nullptr || pValue == nullptr, mR_ArgumentNull);

  mAttribute *pCurrent = nullptr;
  mERROR_CHECK(mAttributeIterator_GetCurrentAttribute_Internal(pAttributeIterator, &pCurrent));

  mERROR_IF(pCurrent->type != mA_T_Vector3, mR_ResourceIncompatible);

  *pValue = pCurrent->Vector3.value;

  mRETURN_SUCCESS();
}

mFUNCTION(mAttributeIterator_GetCurrentValue, IN mAttributeIterator *pAttributeIterator, OUT mVec4f *pValue)
{
  mFUNCTION_SETUP();

  mERROR_IF(pAttributeIterator == nullptr || pValue == nullptr, mR_ArgumentNull);

  mAttribute *pCurrent = nullptr;
  mERROR_CHECK(mAttributeIterator_GetCurrentAttribute_Internal(pAttributeIterator, &pCurrent));

  mERROR_IF(pCurrent->type != mA_T_Vector4, mR_ResourceIncompatible);

  *pValue = pCurrent->Vector4.value;

  mRETURN_SUCCESS();
}

mFUNCTION(mAttributeIterator_GetCurrentValue, IN mAttributeIterator *pAttributeIterator, OUT mString *pValue)
{
  mFUNCTION_SETUP();

  mERROR_IF(pAttributeIterator == nullptr || pValue == nullptr, mR_ArgumentNull);

  mAttribute *pCurrent = nullptr;
  mERROR_CHECK(mAttributeIterator_GetCurrentAttribute_Internal(pAttributeIterator, &pCurrent));

  mERROR_IF(pCurrent->type != mA_T_String, mR_ResourceIncompatible);

  mERROR_CHECK(mString_Create(pValue, *pCurrent->String.pValue, pValue->pAllocator));

  mRETURN_SUCCESS();
}

mFUNCTION(mAttributeIterator_GetCurrentObject, IN mAttributeIterator *pAttributeIterator, OUT mAttributeIterator *pChildAttributeIterator)
{
  mFUNCTION_SETUP();

  mERROR_IF(pAttributeIterator == nullptr || pChildAttributeIterator == nullptr, mR_ArgumentNull);

  mAttribute *pCurrent = nullptr;
  mERROR_CHECK(mAttributeIterator_GetCurrentAttribute_Internal(pAttributeIterator, &pCurrent));

  mERROR_IF(pCurrent->type != mA_T_Object, mR_ResourceIncompatible);

  mAttributeIterator out;
  out.objectIndex = (size_t)-1;
  out.currentAttributeIndex = out.currentContainerAttributeIndex = pAttributeIterator->currentAttributeIndex;
  out.pStore = pAttributeIterator->pStore;

  *pChildAttributeIterator = std::move(out);

  mRETURN_SUCCESS();
}

mFUNCTION(mAttributeIterator_EnterCurrentObject, IN mAttributeIterator *pAttributeIterator)
{
  return mAttributeIterator_GetCurrentObject(pAttributeIterator, pAttributeIterator);
}

mFUNCTION(mAttributeIterator_DeactivateCurrentValue, IN mAttributeIterator *pAttributeIterator)
{
  mFUNCTION_SETUP();

  mERROR_IF(pAttributeIterator == nullptr, mR_ArgumentNull);

  mAttribute *pCurrent = nullptr;
  mERROR_CHECK(mAttributeIterator_GetCurrentAttribute_Internal(pAttributeIterator, &pCurrent));

  mASSERT_DEBUG(pCurrent->type > mA_T_Null || pCurrent->type < _mA_T_ActiveValueCount, "This type of Element should not be reachable.");

  pCurrent->type = (mAttribute_Type)(pCurrent->type | mA_T_InactiveValue);

  mAttribute *pObject = nullptr;
  mERROR_CHECK(mQueue_PointerAt(pAttributeIterator->pStore->attributes, pAttributeIterator->currentContainerAttributeIndex, &pObject));

  mERROR_IF(pObject->type != mA_T_Object, mR_ResourceIncompatible);

  pObject->Object.size--;
  pAttributeIterator->objectIndex--;

  if (pObject->Object.size <= pAttributeIterator->objectIndex + 1)
    mERROR_CHECK(mAttributeIterator_Reset(pAttributeIterator));
  else
    mERROR_CHECK(mAttributeIterator_Next(pAttributeIterator));

  mRETURN_SUCCESS();
}

mFUNCTION(mAttributeIterator_ReactivateValueFromGlobalHandle, IN mAttributeIterator *pAttributeIterator, const mAttributeValueHandle handle)
{
  mFUNCTION_SETUP();

  mERROR_IF(pAttributeIterator == nullptr, mR_ArgumentNull);

  size_t handleIndex = handle;

  mERROR_CHECK(mAttributeIterator_Reset(pAttributeIterator));

  mAttribute *pObject = nullptr;
  mERROR_CHECK(mQueue_PointerAt(pAttributeIterator->pStore->attributes, pAttributeIterator->currentContainerAttributeIndex, &pObject));

  mERROR_IF(pObject->type != mA_T_Object, mR_ResourceInvalid);

  mASSERT_DEBUG(pAttributeIterator->objectIndex == (size_t)-1, "Iterator should've been reset");

  pAttributeIterator->objectIndex++;
  pAttributeIterator->currentAttributeIndex++;

  mAttribute *pCurrent = nullptr;

  size_t index = 0;

  while (index < pObject->Object.added)
  {
    mERROR_CHECK(mQueue_PointerAt(pAttributeIterator->pStore->attributes, pAttributeIterator->currentAttributeIndex, &pCurrent));

    if (pCurrent->type == mA_T_SkipObject)
    {
      if (pAttributeIterator->currentAttributeIndex == handleIndex)
      {
        const mResult result = mSILENCE_ERROR(mAttributeIterator_Find(pAttributeIterator, pCurrent->name));
        mERROR_IF(mR_ResourceNotFound != result, mSUCCEEDED(result) ? mR_ResourceAlreadyExists : result);

        pCurrent->type = (mAttribute_Type)(pCurrent->type & ~mA_T_InactiveValue);
        pObject->Object.size++;

        mRETURN_SUCCESS();
      }

      pAttributeIterator->currentAttributeIndex += pCurrent->Object.skip + 1;
      index++;
    }
    else if ((pCurrent->type & mA_T_InactiveValue) == mA_T_InactiveValue)
    {
      if (pAttributeIterator->currentAttributeIndex == handleIndex)
      {
        const mResult result = mSILENCE_ERROR(mAttributeIterator_Find(pAttributeIterator, pCurrent->name));
        mERROR_IF(mR_ResourceNotFound != result, mSUCCEEDED(result) ? mR_ResourceAlreadyExists : result);

        pCurrent->type = (mAttribute_Type)(pCurrent->type & ~mA_T_InactiveValue);
        pObject->Object.size++;

        mRETURN_SUCCESS();
      }

      mERROR_IF(pAttributeIterator->currentAttributeIndex == handleIndex, mR_Success);

      pAttributeIterator->currentAttributeIndex++;
      index++;
    }
    else if (pCurrent->type == mA_T_SkipTo)
    {
      if (pAttributeIterator->currentAttributeIndex == handleIndex)
        handleIndex = pCurrent->SkipTo.nextAttributeIndex;

      pAttributeIterator->currentAttributeIndex = pCurrent->SkipTo.nextAttributeIndex;
    }
    else if (pCurrent->type == mA_T_EmptyObject)
    {
      mASSERT_DEBUG(false, "Should've been rejected at size already.");
      mRETURN_RESULT(mR_ResourceInvalid);
    }
    else if (pCurrent->type == mA_T_Object)
    {
      mERROR_IF(pAttributeIterator->currentAttributeIndex == handleIndex, mR_Success);

      pAttributeIterator->currentAttributeIndex += pCurrent->Object.skip + 1;
      pAttributeIterator->objectIndex++;
      index++;
    }
    else
    {
      mERROR_IF(pAttributeIterator->currentAttributeIndex == handleIndex, mR_Success);

      pAttributeIterator->currentAttributeIndex++;
      pAttributeIterator->objectIndex++;
      index++;
    }
  }

  mERROR_CHECK(mAttributeIterator_Reset(pAttributeIterator));

  mRETURN_RESULT(mR_ResourceNotFound);
}

mFUNCTION(mAttributeIterator_GetCurrentValueGlobalHandle, IN mAttributeIterator *pAttributeIterator, OUT mAttributeValueHandle *pHandle)
{
  mFUNCTION_SETUP();

  mERROR_IF(pAttributeIterator == nullptr || pHandle == nullptr, mR_ArgumentNull);

  if (pAttributeIterator->objectIndex == (size_t)-1)
  {
    mAttribute *pObject = nullptr;
    mERROR_CHECK(mQueue_PointerAt(pAttributeIterator->pStore->attributes, pAttributeIterator->currentContainerAttributeIndex, &pObject));

    mERROR_IF(pObject->type != mA_T_Object, mR_ResourceInvalid);
    mERROR_IF(pObject->Object.size == 0, mR_IndexOutOfBounds);

    mERROR_CHECK(mAttributeIterator_Next(pAttributeIterator));
  }

  *pHandle = pAttributeIterator->currentAttributeIndex;

  mRETURN_SUCCESS();
}

mFUNCTION(mAttributeIterator_ReplaceCurrentValueWithUInt64, IN_OUT mAttributeIterator *pAttributeIterator, const uint64_t value)
{
  mFUNCTION_SETUP();

  mERROR_IF(pAttributeIterator == nullptr, mR_ArgumentNull);

  mAttribute *pCurrent = nullptr;
  mERROR_CHECK(mAttributeIterator_GetCurrentAttribute_Internal(pAttributeIterator, &pCurrent));

  if ((pCurrent->type & (mA_T_InactiveValue - 1)) == mA_T_String)
    mERROR_CHECK(mDestruct(pCurrent));
  else
    mERROR_IF((pCurrent->type & (mA_T_InactiveValue - 1)) == mA_T_Object, mR_ResourceIncompatible);

  pCurrent->type = mA_T_UInt64;
  pCurrent->UInt64.value = value;

  mRETURN_SUCCESS();
}

mFUNCTION(mAttributeIterator_ReplaceCurrentValueWithInt64, IN_OUT mAttributeIterator *pAttributeIterator, const int64_t value)
{
  mFUNCTION_SETUP();

  mERROR_IF(pAttributeIterator == nullptr, mR_ArgumentNull);

  mAttribute *pCurrent = nullptr;
  mERROR_CHECK(mAttributeIterator_GetCurrentAttribute_Internal(pAttributeIterator, &pCurrent));

  if ((pCurrent->type & (mA_T_InactiveValue - 1)) == mA_T_String)
    mERROR_CHECK(mDestruct(pCurrent));
  else
    mERROR_IF((pCurrent->type & (mA_T_InactiveValue - 1)) == mA_T_Object, mR_ResourceIncompatible);

  pCurrent->type = mA_T_Int64;
  pCurrent->Int64.value = value;

  mRETURN_SUCCESS();
}

mFUNCTION(mAttributeIterator_ReplaceCurrentValueWithBool, IN_OUT mAttributeIterator *pAttributeIterator, const bool value)
{
  mFUNCTION_SETUP();

  mERROR_IF(pAttributeIterator == nullptr, mR_ArgumentNull);

  mAttribute *pCurrent = nullptr;
  mERROR_CHECK(mAttributeIterator_GetCurrentAttribute_Internal(pAttributeIterator, &pCurrent));

  if ((pCurrent->type & (mA_T_InactiveValue - 1)) == mA_T_String)
    mERROR_CHECK(mDestruct(pCurrent));
  else
    mERROR_IF((pCurrent->type & (mA_T_InactiveValue - 1)) == mA_T_Object, mR_ResourceIncompatible);

  pCurrent->type = mA_T_Bool;
  pCurrent->Bool.value = value;

  mRETURN_SUCCESS();
}

mFUNCTION(mAttributeIterator_ReplaceCurrentValueWithFloat64, IN_OUT mAttributeIterator *pAttributeIterator, const double_t value)
{
  mFUNCTION_SETUP();

  mERROR_IF(pAttributeIterator == nullptr, mR_ArgumentNull);

  mAttribute *pCurrent = nullptr;
  mERROR_CHECK(mAttributeIterator_GetCurrentAttribute_Internal(pAttributeIterator, &pCurrent));

  if ((pCurrent->type & (mA_T_InactiveValue - 1)) == mA_T_String)
    mERROR_CHECK(mDestruct(pCurrent));
  else
    mERROR_IF((pCurrent->type & (mA_T_InactiveValue - 1)) == mA_T_Object, mR_ResourceIncompatible);

  pCurrent->type = mA_T_Float64;
  pCurrent->Float64.value = value;

  mRETURN_SUCCESS();
}

mFUNCTION(mAttributeIterator_ReplaceCurrentValueWithVector2, IN_OUT mAttributeIterator *pAttributeIterator, const mVec2f value)
{
  mFUNCTION_SETUP();

  mERROR_IF(pAttributeIterator == nullptr, mR_ArgumentNull);

  mAttribute *pCurrent = nullptr;
  mERROR_CHECK(mAttributeIterator_GetCurrentAttribute_Internal(pAttributeIterator, &pCurrent));

  if ((pCurrent->type & (mA_T_InactiveValue - 1)) == mA_T_String)
    mERROR_CHECK(mDestruct(pCurrent));
  else
    mERROR_IF((pCurrent->type & (mA_T_InactiveValue - 1)) == mA_T_Object, mR_ResourceIncompatible);

  pCurrent->type = mA_T_Vector2;
  pCurrent->Vector2.value = value;

  mRETURN_SUCCESS();
}

mFUNCTION(mAttributeIterator_ReplaceCurrentValueWithVector3, IN_OUT mAttributeIterator *pAttributeIterator, const mVec3f value)
{
  mFUNCTION_SETUP();

  mERROR_IF(pAttributeIterator == nullptr, mR_ArgumentNull);

  mAttribute *pCurrent = nullptr;
  mERROR_CHECK(mAttributeIterator_GetCurrentAttribute_Internal(pAttributeIterator, &pCurrent));

  if ((pCurrent->type & (mA_T_InactiveValue - 1)) == mA_T_String)
    mERROR_CHECK(mDestruct(pCurrent));
  else
    mERROR_IF((pCurrent->type & (mA_T_InactiveValue - 1)) == mA_T_Object, mR_ResourceIncompatible);

  pCurrent->type = mA_T_Vector3;
  pCurrent->Vector3.value = value;

  mRETURN_SUCCESS();
}

mFUNCTION(mAttributeIterator_ReplaceCurrentValueWithVector4, IN_OUT mAttributeIterator *pAttributeIterator, const mVec4f value)
{
  mFUNCTION_SETUP();

  mERROR_IF(pAttributeIterator == nullptr, mR_ArgumentNull);

  mAttribute *pCurrent = nullptr;
  mERROR_CHECK(mAttributeIterator_GetCurrentAttribute_Internal(pAttributeIterator, &pCurrent));

  if ((pCurrent->type & (mA_T_InactiveValue - 1)) == mA_T_String)
    mERROR_CHECK(mDestruct(pCurrent));
  else
    mERROR_IF((pCurrent->type & (mA_T_InactiveValue - 1)) == mA_T_Object, mR_ResourceIncompatible);

  pCurrent->type = mA_T_Vector4;
  pCurrent->Vector4.value = value;

  mRETURN_SUCCESS();
}

mFUNCTION(mAttributeIterator_ReplaceCurrentValueWithString, IN_OUT mAttributeIterator *pAttributeIterator, const mString &value)
{
  mFUNCTION_SETUP();

  mERROR_IF(pAttributeIterator == nullptr, mR_ArgumentNull);

  mAttribute *pCurrent = nullptr;
  mERROR_CHECK(mAttributeIterator_GetCurrentAttribute_Internal(pAttributeIterator, &pCurrent));

  mERROR_IF((pCurrent->type & (mA_T_InactiveValue - 1)) == mA_T_Object, mR_ResourceIncompatible);

  if ((pCurrent->type & (mA_T_InactiveValue - 1)) != mA_T_String)
  {
    pCurrent->type = mA_T_String;
    mERROR_CHECK(mAllocator_AllocateZero(&mDefaultAllocator, &pCurrent->String.pValue, 1));
  }

  mERROR_CHECK(mString_Create(pCurrent->String.pValue, value, pCurrent->String.pValue->pAllocator));

  mRETURN_SUCCESS();
}

