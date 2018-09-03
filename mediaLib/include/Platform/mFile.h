// Copyright 2018 Christoph Stiller
// 
// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files(the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions :
// 
// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
// 
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef mFile_h__
#define mFile_h__

#include "default.h"
#include "mArray.h"

enum mFile_Encoding
{
  mF_E_ASCII,
};

mFUNCTION(mFile_Exists, const std::wstring &filename, OUT bool *pExists);

mFUNCTION(mFile_ReadAllBytes, const std::wstring &filename, IN OPTIONAL mAllocator *pAllocator, OUT mArray<uint8_t> *pBytes);
mFUNCTION(mFile_ReadAllText, const std::wstring &filename, IN OPTIONAL mAllocator *pAllocator, OUT std::string *pText, const mFile_Encoding encoding = mF_E_ASCII);

mFUNCTION(mFile_WriteAllBytes, const std::wstring &filename, mArray<uint8_t> &bytes);
mFUNCTION(mFile_WriteAllText, const std::wstring &filename, const std::string &text, const mFile_Encoding encoding = mF_E_ASCII);

template <typename T>
mFUNCTION(mFile_ReadAllItems, const std::wstring &filename, IN OPTIONAL mAllocator *pAllocator, OUT mArray<T> *pData);

template <typename T>
mFUNCTION(mFile_ReadRaw, const std::wstring &filename, OUT T **ppData, IN mAllocator *pAllocator, OUT size_t *pCount);

template <typename T>
mFUNCTION(mFile_WriteRaw, const std::wstring &filename, IN T *pData, const size_t count);

//////////////////////////////////////////////////////////////////////////

template<typename T>
inline mFUNCTION(mFile_ReadAllItems, const std::wstring &filename, IN OPTIONAL mAllocator *pAllocator, OUT mArray<T> *pData)
{
  mFUNCTION_SETUP();

  mERROR_IF(pData == nullptr, mR_ArgumentNull);

  if(pData->pData)
    mERROR_CHECK(mArray_Destroy(pData));

  pData->pAllocator = pAllocator;

  mERROR_CHECK(mFile_ReadRaw(filename, &pData->pData, pAllocator, &pData->count));

  mRETURN_SUCCESS();
}

template<typename T>
inline mFUNCTION(mFile_ReadRaw, const std::wstring &filename, OUT T **ppData, IN mAllocator *pAllocator, OUT size_t *pCount)
{
  mFUNCTION_SETUP();

  mERROR_IF(ppData == nullptr || pCount == nullptr, mR_ArgumentNull);

  FILE *pFile = _wfopen(filename.c_str(), L"r");
  mDEFER(if (pFile) { fclose(pFile); });
  mERROR_IF(pFile == nullptr, mR_ResourceNotFound);

  mERROR_IF(0 != fseek(pFile, 0, SEEK_END), mR_InternalError);
  
  const size_t length = ftell(pFile);
  const size_t count = length / sizeof(T);

  mERROR_IF(0 != fseek(pFile, 0, SEEK_SET), mR_InternalError);

  mERROR_CHECK(mAllocator_Allocate(pAllocator, (uint8_t **)ppData, length + 1));
  const size_t readLength = fread(*ppData, 1, length, pFile);

  *pCount = readLength / sizeof(T);

  mRETURN_SUCCESS();
}

template<typename T>
inline mFUNCTION(mFile_WriteRaw, const std::wstring &filename, IN T *pData, const size_t count)
{
  mFUNCTION_SETUP();

  mERROR_IF(pData == nullptr, mR_ArgumentNull);

  FILE *pFile = _wfopen(filename.c_str(), L"w");
  mDEFER(if (pFile) fclose(pFile););
  mERROR_IF(pFile == nullptr, mR_ResourceNotFound);

  const size_t writeCount = fwrite(pData, sizeof(T), count, pFile);

  mERROR_IF(writeCount != count, mR_InternalError);

  mRETURN_SUCCESS();
}

#endif // mFile_h__
