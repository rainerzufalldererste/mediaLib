#ifndef mStaticArray_h__
#define mStaticArray_h__

template <typename T, size_t TCount>
struct mStaticArray 
{
  T pData[TCount];

  mStaticArray();
  mStaticArray(const mStaticArray<T, TCount> &copy);
  mStaticArray(mStaticArray<T, TCount> &&move);

  mStaticArray<T, TCount> & operator =(const mStaticArray<T, TCount> &copy);
  mStaticArray<T, TCount> & operator =(mStaticArray<T, TCount> &&move);

  T operator[](const size_t index);
  T At(const size_t index);
  T* PointerAt(const size_t index);

  size_t Count() const;
};

//////////////////////////////////////////////////////////////////////////

template<typename T, size_t TCount>
inline mStaticArray<T, TCount>::mStaticArray()
{
  mMemset((T *)pData, Count());
}

template<typename T, size_t TCount>
inline mStaticArray<T, TCount>::mStaticArray(const mStaticArray<T, TCount> &copy)
{
  mMemcpy((T *)pData, (T *)copy.pData, Count());
}

template<typename T, size_t TCount>
inline mStaticArray<T, TCount>::mStaticArray(mStaticArray<T, TCount>&& move)
{
  mMemcpy((T *)pData, (T *)copy.pData, Count());
}

template<typename T, size_t TCount>
inline mStaticArray<T, TCount>& mStaticArray<T, TCount>::operator=(const mStaticArray<T, TCount> &copy)
{
  mMemcpy((T *)pData, (T *)copy.pData, Count());
}

template<typename T, size_t TCount>
inline mStaticArray<T, TCount>& mStaticArray<T, TCount>::operator=(mStaticArray<T, TCount> &&move)
{
  mMemcpy((T *)pData, (T *)copy.pData, Count());
}

template<typename T, size_t TCount>
inline T mStaticArray<T, TCount>::operator[](const size_t index)
{
  return pData[index];
}

template<typename T, size_t TCount>
inline T mStaticArray<T, TCount>::At(const size_t index)
{
  return pData[index];
}

template<typename T, size_t TCount>
inline T * mStaticArray<T, TCount>::PointerAt(const size_t index)
{
  return pData + index;
}

template<typename T, size_t TCount>
inline size_t mStaticArray<T, TCount>::Count() const
{
  return TCount;
}

#endif // mStaticArray_h__
