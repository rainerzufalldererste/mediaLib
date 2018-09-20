// Copyright 2018 Christoph Stiller
// 
// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files(the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions :
// 
// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
// 
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "mFile.h"
#include <sys\stat.h>
#include <shobjidl.h>

mFUNCTION(mFile_Exists, const std::wstring &filename, OUT bool *pExists)
{
  mFUNCTION_SETUP();

  mERROR_IF(pExists == nullptr, mR_ArgumentNull);

  struct _stat buffer;

  *pExists = (_wstat(filename.c_str(), &buffer) == 0);

  mRETURN_SUCCESS();
}

mFUNCTION(mFile_ReadAllBytes, const std::wstring & filename, IN OPTIONAL mAllocator * pAllocator, OUT mArray<uint8_t>* pBytes)
{
  mFUNCTION_SETUP();

  mERROR_CHECK(mFile_ReadAllItems(filename, pAllocator, pBytes));

  mRETURN_SUCCESS();
}

mFUNCTION(mFile_ReadAllText, const std::wstring & filename, IN OPTIONAL mAllocator * pAllocator, OUT std::string * pText, const mFile_Encoding /* encoding = mF_E_ASCII */)
{
  mFUNCTION_SETUP();

  mERROR_IF(pText == nullptr, mR_ArgumentNull);

  size_t count = 0;
  char *text = nullptr;

  mDEFER(mAllocator_FreePtr(nullptr, &text));
  mERROR_CHECK(mFile_ReadRaw(filename, &text, pAllocator, &count));
  text[count] = '\0';
  *pText = std::string(text);
  mERROR_CHECK(mAllocator_FreePtr(pAllocator, &text));

  mRETURN_SUCCESS();
}

mFUNCTION(mFile_ReadAllText, const mString & filename, IN OPTIONAL mAllocator * pAllocator, OUT mString * pText, const mFile_Encoding /* encoding = mF_E_ASCII */)
{
  mFUNCTION_SETUP();

  mERROR_IF(pText == nullptr, mR_ArgumentNull);

  std::wstring wstring;
  mERROR_CHECK(mString_ToWideString(filename, &wstring));

  size_t count = 0;
  char *text = nullptr;

  mDEFER(mAllocator_FreePtr(nullptr, &text));
  mERROR_CHECK(mFile_ReadRaw(wstring, &text, pAllocator, &count));
  text[count] = '\0';
  *pText = mString(text);
  mERROR_CHECK(mAllocator_FreePtr(pAllocator, &text));

  mRETURN_SUCCESS();
}

mFUNCTION(mFile_WriteAllBytes, const std::wstring & filename, mArray<uint8_t> &bytes)
{
  mFUNCTION_SETUP();

  mERROR_CHECK(mFile_WriteRaw(filename, bytes.pData, bytes.count));

  mRETURN_SUCCESS();
}

mFUNCTION(mFile_WriteAllText, const std::wstring & filename, const std::string &text, const mFile_Encoding /* encoding = mF_E_ASCII */)
{
  mFUNCTION_SETUP();

  mERROR_CHECK(mFile_WriteRaw(filename, text.c_str(), text.length()));

  mRETURN_SUCCESS();
}

mFUNCTION(mFile_WriteAllText, const mString &filename, const mString &text, const mFile_Encoding /* encoding = mF_E_ASCII */)
{
  mFUNCTION_SETUP();

  std::wstring wstring;
  mERROR_CHECK(mString_ToWideString(filename, &wstring));

  mERROR_CHECK(mFile_WriteRaw(wstring, text.c_str(), text.bytes - 1));

  mRETURN_SUCCESS();
}

mFUNCTION(mFile_CreateDirectory, const mString &folderPath)
{
  mFUNCTION_SETUP();

  mERROR_IF(folderPath.hasFailed, mR_InvalidParameter);

  mString path;
  mERROR_CHECK(mString_ToDirectoryPath(&path, folderPath));

  mString pathWithoutLastSlash;
  mERROR_CHECK(mString_Substring(path, &pathWithoutLastSlash, 0, path.Count() - 2));

  std::wstring directoryName;
  mERROR_CHECK(mString_ToWideString(pathWithoutLastSlash, &directoryName));

  if (0 == CreateDirectoryW(directoryName.c_str(), NULL))
  {
    const DWORD error = GetLastError();

    switch (error)
    {
    case ERROR_ALREADY_EXISTS:
      break;

    case ERROR_PATH_NOT_FOUND:
      mERROR_IF(true, mR_ResourceNotFound);
      break;

    default:
      mERROR_IF(true, mR_InternalError);
      break;
    }
  }

  mRETURN_SUCCESS();
}

HRESULT CreateAndInitializeFileOperation(REFIID riid, void **ppFileOperation)
{
  *ppFileOperation = nullptr;

  // Create the IFileOperation object
  IFileOperation *pfo;

  HRESULT hr = CoCreateInstance(__uuidof(FileOperation), NULL, CLSCTX_ALL, IID_PPV_ARGS(&pfo));

  if (SUCCEEDED(hr))
  {
    // Set the operation flags. Turn off all UI from being shown to the user during the operation. This includes error, confirmation and progress dialogs.
    hr = pfo->SetOperationFlags(FOF_NO_UI);

    if (SUCCEEDED(hr))
    {
      hr = pfo->QueryInterface(riid, ppFileOperation);
    }

    pfo->Release();
  }

  return hr;
}

mFUNCTION(mFile_DeleteFolder, const mString &folderPath)
{
  mFUNCTION_SETUP();

  mERROR_IF(folderPath.hasFailed, mR_InvalidParameter);

  mString path;
  mERROR_CHECK(mString_ToDirectoryPath(&path, folderPath));

  mString pathWithoutLastSlash;
  mERROR_CHECK(mString_Substring(path, &pathWithoutLastSlash, 0, path.Count() - 2));

  std::wstring directoryName;
  mERROR_CHECK(mString_ToWideString(pathWithoutLastSlash, &directoryName));

  HRESULT hr = S_OK;

  wchar_t absolutePath[1024 * 4];
  DWORD length = GetFullPathNameW(directoryName.c_str(), mARRAYSIZE(absolutePath), absolutePath, nullptr);
  mUnused(length);

  IShellItem *pItem = nullptr;
  mERROR_IF(FAILED(hr = SHCreateItemFromParsingName(absolutePath, nullptr, IID_PPV_ARGS(&pItem))), mR_InternalError);
  mDEFER(pItem->Release());

  IFileOperation *pFileOperation = nullptr;
  mERROR_IF(FAILED(hr = CreateAndInitializeFileOperation(IID_PPV_ARGS(&pFileOperation))), mR_InternalError);
  mDEFER(pFileOperation->Release());

  mERROR_IF(FAILED(hr = pFileOperation->DeleteItem(pItem, nullptr)), mR_InternalError);
  mERROR_IF(FAILED(hr = pFileOperation->PerformOperations()), mR_InternalError);

  mRETURN_SUCCESS();
}
