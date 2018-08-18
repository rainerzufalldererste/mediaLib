// Copyright 2018 Christoph Stiller
// 
// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files(the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions :
// 
// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
// 
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

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
inline mStaticArray<T, TCount>& mStaticArray<T, TCount>::operator=(const mStaticArray<T, TCount>& copy)
{
  mMemcpy((T *)pData, (T *)copy.pData, Count());
}

template<typename T, size_t TCount>
inline mStaticArray<T, TCount>& mStaticArray<T, TCount>::operator=(mStaticArray<T, TCount>&& move)
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
