template <typename T>
mFUNCTION(mChunkedArray_Destroy_Internal, IN mChunkedArray<T> *pChunkedArray);

template <typename T>
mFUNCTION(mChunkedArray_Grow_Internal, mPtr<mChunkedArray<T>> &chunkedArray);

//////////////////////////////////////////////////////////////////////////

template <typename T>
mFUNCTION(mChunkedArray_Create, OUT mPtr<mChunkedArray<T>> *pChunkedArray, IN mAllocator *pAllocator, const size_t blockSize /* = 64 */)
{
  mFUNCTION_SETUP();

  mERROR_IF(pChunkedArray == nullptr, mR_ArgumentNull);
  mERROR_IF(blockSize == 0, mR_ArgumentOutOfBounds);

  mERROR_CHECK(mSharedPointer_Allocate(pChunkedArray, pAllocator, (std::function<void(mChunkedArray<T> *)>) [](mChunkedArray<T> *pData) { mChunkedArray_Destroy_Internal(pData); }, 1));

  (*pChunkedArray)->pAllocator = pAllocator;
  (*pChunkedArray)->blockSize = blockSize;
  new (&(*pChunkedArray)->destructionFunction) std::function<mResult(T *)>(nullptr);

  mRETURN_SUCCESS();
}

template <typename T>
mFUNCTION(mChunkedArray_Destroy, IN_OUT mPtr<mChunkedArray<T>> *pChunkedArray)
{
  mFUNCTION_SETUP();

  mERROR_IF(pChunkedArray == nullptr, mR_ArgumentNull);

  mERROR_CHECK(mSharedPointer_Destroy(pChunkedArray));

  mRETURN_SUCCESS();
}

template <typename T>
mFUNCTION(mChunkedArray_Push, mPtr<mChunkedArray<T>> &chunkedArray, IN T *pItem, OUT OPTIONAL size_t *pIndex) 
{
  mFUNCTION_SETUP();

  mERROR_IF(chunkedArray == nullptr || pItem == nullptr, mR_ArgumentNull);

  if (chunkedArray->pBlocks == nullptr || chunkedArray->itemCapacity <= chunkedArray->itemCount)
  {
    mERROR_CHECK(mChunkedArray_PushBack(chunkedArray, pItem));

    if (pIndex != nullptr)
      *pIndex = chunkedArray->itemCount - 1;

    mRETURN_SUCCESS();
  }

  size_t index = 0;

  for (size_t blockIndex = 0; blockIndex < chunkedArray->blockCount; ++blockIndex)
  {
    mChunkedArray<T>::mChunkedArrayBlock *pBlock = &chunkedArray->pBlocks[blockIndex];

    if (pBlock->blockStartIndex == 0 && pBlock->blockEndIndex == pBlock->blockSize)
    {
      index += pBlock->blockSize;
      continue;
    }

    if (pBlock->blockStartIndex > 0)
    {
      --pBlock->blockStartIndex;
      index += pBlock->blockStartIndex;

      new (&pBlock->pData[pBlock->blockStartIndex]) T (*pItem);

      break;
    }
    else
    {
      index += pBlock->blockEndIndex;

      new (&pBlock->pData[pBlock->blockEndIndex]) T(*pItem);

      ++pBlock->blockEndIndex;

      break;
    }
  }

  if (pIndex != nullptr)
    *pIndex = index;

  ++chunkedArray->itemCount;

  mRETURN_SUCCESS();
}

template <typename T>
mFUNCTION(mChunkedArray_PushBack, mPtr<mChunkedArray<T>> &chunkedArray, IN T *pItem)
{
  mFUNCTION_SETUP();

  mERROR_IF(chunkedArray == nullptr || pItem == nullptr, mR_ArgumentNull);

  if (chunkedArray->pBlocks == nullptr || chunkedArray->pBlocks[chunkedArray->blockCount - 1].blockEndIndex >= chunkedArray->blockSize)
    mERROR_CHECK(mChunkedArray_Grow_Internal(chunkedArray));

  new (&chunkedArray->pBlocks[chunkedArray->blockCount - 1].pData[chunkedArray->pBlocks[chunkedArray->blockCount - 1].blockEndIndex]) T(*pItem);
  ++chunkedArray->pBlocks[chunkedArray->blockCount - 1].blockEndIndex;
  ++chunkedArray->itemCount;

  mRETURN_SUCCESS();
}

template <typename T>
mFUNCTION(mChunkedArray_PushBack, mPtr<mChunkedArray<T>> &chunkedArray, IN T &&item)
{
  mFUNCTION_SETUP();

  mERROR_IF(chunkedArray == nullptr, mR_ArgumentNull);

  if (chunkedArray->pBlocks == nullptr || chunkedArray->pBlocks[chunkedArray->blockCount - 1].blockEndIndex >= chunkedArray->blockSize)
    mERROR_CHECK(mChunkedArray_Grow_Internal(chunkedArray));

  new (&chunkedArray->pBlocks[chunkedArray->blockCount - 1].pData[chunkedArray->pBlocks[chunkedArray->blockCount - 1].blockEndIndex]) T(std::move(item));
  ++chunkedArray->pBlocks[chunkedArray->blockCount - 1].blockEndIndex;
  ++chunkedArray->itemCount;

  mRETURN_SUCCESS();
}

template <typename T>
mFUNCTION(mChunkedArray_GetCount, mPtr<mChunkedArray<T>> &chunkedArray, OUT size_t *pCount)
{
  mFUNCTION_SETUP();

  mERROR_IF(chunkedArray == nullptr || pCount == nullptr, mR_ArgumentNull);

  *pCount = chunkedArray->itemCount;

  mRETURN_SUCCESS();
}

template <typename T>
mFUNCTION(mChunkedArray_PeekAt, mPtr<mChunkedArray<T>> &chunkedArray, const size_t index, OUT T *pItem)
{
  mFUNCTION_SETUP();

  mERROR_IF(chunkedArray == nullptr || pItem == nullptr, mR_ArgumentNull);

  T *pOutItem = nullptr;
  mERROR_CHECK(mChunkedArray_PointerAt(chunkedArray, index, &pOutItem));

  *pItem = *pOutItem;

  mRETURN_SUCCESS();
}

template <typename T>
mFUNCTION(mChunkedArray_PopAt, mPtr<mChunkedArray<T>> &chunkedArray, const size_t index, OUT T *pItem)
{
  mFUNCTION_SETUP();

  mERROR_IF(chunkedArray == nullptr || pItem == nullptr, mR_ArgumentNull);
  mERROR_IF(chunkedArray->itemCount <= index, mR_IndexOutOfBounds);

  size_t currentIndex = 0;

  for (size_t blockIndex = 0; blockIndex < chunkedArray->blockCount; ++blockIndex)
  {
    mChunkedArray<T>::mChunkedArrayBlock *pBlock = &chunkedArray->pBlocks[blockIndex];

    const size_t blockSize = pBlock->blockEndIndex - pBlock->blockStartIndex;

    if (currentIndex + blockSize <= index)
    {
      currentIndex += blockSize;
      continue;
    }

    size_t innerItemIndex = index - currentIndex;

    *pItem = std::move(pBlock->pData[pBlock->blockStartIndex + innerItemIndex]);

    innerItemIndex += pBlock->blockStartIndex;

    if (innerItemIndex - pBlock->blockStartIndex <= pBlock->blockEndIndex - innerItemIndex)
    {
      if (innerItemIndex - pBlock->blockStartIndex > 0)
        mERROR_CHECK(mAllocator_Move(chunkedArray->pAllocator, &pBlock->pData[pBlock->blockStartIndex + 1], &pBlock->pData[pBlock->blockStartIndex], innerItemIndex - pBlock->blockStartIndex));

      ++pBlock->blockStartIndex;
    }
    else
    {
      if (pBlock->blockEndIndex - innerItemIndex - 1 > 0)
        mERROR_CHECK(mAllocator_Move(chunkedArray->pAllocator, &pBlock->pData[innerItemIndex], &pBlock->pData[innerItemIndex + 1], pBlock->blockEndIndex - innerItemIndex - 1));

      --pBlock->blockEndIndex;
    }

    // Cleanup if block empty.
    if (pBlock->blockStartIndex == pBlock->blockEndIndex)
    {
      mERROR_CHECK(mAllocator_FreePtr(chunkedArray->pAllocator, &pBlock->pData));

      if (blockIndex < chunkedArray->blockCount - 1)
        mERROR_CHECK(mAllocator_Move(chunkedArray->pAllocator, pBlock, pBlock + 1, chunkedArray->blockCount - blockIndex - 1));

      --chunkedArray->blockCount;
      pBlock = nullptr;
    }

    --chunkedArray->itemCount;
    break;
  }

  mRETURN_SUCCESS();
}

template <typename T>
mFUNCTION(mChunkedArray_PointerAt, mPtr<mChunkedArray<T>> &chunkedArray, const size_t index, OUT T **ppItem)
{
  mFUNCTION_SETUP();

  mERROR_IF(chunkedArray == nullptr || ppItem == nullptr, mR_ArgumentNull);
  mERROR_IF(chunkedArray->itemCount <= index, mR_IndexOutOfBounds);

  size_t currentIndex = 0;

  for (size_t blockIndex = 0; blockIndex < chunkedArray->blockCount; ++blockIndex)
  {
    mChunkedArray<T>::mChunkedArrayBlock *pBlock = &chunkedArray->pBlocks[blockIndex];

    const size_t blockSize = pBlock->blockEndIndex - pBlock->blockStartIndex;

    if (currentIndex + blockSize <= index)
    {
      currentIndex += blockSize;
      continue;
    }

    *ppItem = &pBlock->pData[pBlock->blockStartIndex + index - currentIndex];
    break;
  }

  mRETURN_SUCCESS();
}

template<typename T>
inline mFUNCTION(mChunkedArray_SetDestructionFunction, mPtr<mChunkedArray<T>> &chunkedArray, const std::function<mResult(T *)> &destructionFunction)
{
  mFUNCTION_SETUP();

  mERROR_IF(chunkedArray == nullptr || ppItem == nullptr, mR_ArgumentNull);

  chunkedArray->destructionFunction = destructionFunction;

  mRETURN_SUCCESS();
}

//////////////////////////////////////////////////////////////////////////

template<typename T>
inline mFUNCTION(mChunkedArray_Destroy_Internal, IN mChunkedArray<T>* pChunkedArray)
{
  mFUNCTION_SETUP();

  mERROR_IF(pChunkedArray == nullptr, mR_ArgumentNull);
  
  for (size_t i = 0; i < pChunkedArray->blockCount; ++i)
  {
    for (size_t index = pChunkedArray->pBlocks[i].blockStartIndex; index < pChunkedArray->pBlocks[i].blockEndIndex; ++index)
    {
      if (pChunkedArray->destructionFunction)
        mERROR_CHECK(pChunkedArray->destructionFunction(&pChunkedArray->pBlocks[i].pData[index]));
      else
        mERROR_CHECK(mDestruct(&pChunkedArray->pBlocks[i].pData[index]));
    }

    mERROR_CHECK(mAllocator_FreePtr(pChunkedArray->pAllocator, &pChunkedArray->pBlocks[i].pData));
  }

  pChunkedArray->destructionFunction.~function();

  mERROR_CHECK(mAllocator_FreePtr(pChunkedArray->pAllocator, &pChunkedArray->pBlocks));

  mRETURN_SUCCESS();
}

template<typename T>
inline mFUNCTION(mChunkedArray_Grow_Internal, mPtr<mChunkedArray<T>> &chunkedArray)
{
  mFUNCTION_SETUP();

  if (chunkedArray->pBlocks == nullptr)
  {
    const size_t newBlockCapacity = chunkedArray->blockCount + 1;
    mERROR_CHECK(mAllocator_AllocateZero(chunkedArray->pAllocator, &chunkedArray->pBlocks, newBlockCapacity));
    chunkedArray->blockCapacity = newBlockCapacity;
  }
  else if (chunkedArray->blockCapacity > chunkedArray->blockCount)
  {
    // No need to allocate.
  }
  else if (chunkedArray->itemCapacity == chunkedArray->itemCount)
  {
    const size_t newBlockCapacity = chunkedArray->blockCapacity * 2;
    mERROR_CHECK(mAllocator_Reallocate(chunkedArray->pAllocator, &chunkedArray->pBlocks, newBlockCapacity));
    chunkedArray->blockCapacity = newBlockCapacity;
  }
  else
  {
    mRETURN_SUCCESS();
  }

  ++chunkedArray->blockCount;
  mDEFER_ON_ERROR(--chunkedArray->blockCount);
  mERROR_CHECK(mAllocator_AllocateZero(chunkedArray->pAllocator, &chunkedArray->pBlocks[chunkedArray->blockCount - 1].pData, chunkedArray->blockSize));
  chunkedArray->pBlocks[chunkedArray->blockCount - 1].blockSize = chunkedArray->blockSize;
  chunkedArray->pBlocks[chunkedArray->blockCount - 1].blockStartIndex = 0;
  chunkedArray->pBlocks[chunkedArray->blockCount - 1].blockEndIndex = 0;
  chunkedArray->itemCapacity += chunkedArray->blockSize;

  mRETURN_SUCCESS();
}
