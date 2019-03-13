#include "mSystemDialogue.h"
#include "mSoftwareWindow.h"
#include "mHardwareWindow.h"

#include "SDL_syswm.h"

#include <shlobj.h>

mFUNCTION(mSystemDialogue_OpenFile_Internal, HWND window, const mString &headlineString, mPtr<mQueue<mKeyValuePair<mString, mString>>> &fileTypeExtentionPairs, OUT bool *pCanceled, OUT mString *pFileName);
mFUNCTION(mSystemDialogue_OpenFile_SdlWindowInternal, SDL_Window *pWindow, const mString &headlineString, mPtr<mQueue<mKeyValuePair<mString, mString>>> &fileTypeExtentionPairs, OUT bool *pCanceled, OUT mString *pFileName);
mFUNCTION(mSystemDialogue_SelectDirectory_SdlWindowInternal, SDL_Window *pWindow, const mString &headlineString, OUT OPTIONAL bool *pCanceled, OUT mString *pSelectedDirectory);

//////////////////////////////////////////////////////////////////////////

mFUNCTION(mSystemDialogue_OpenFile, const mString &headlineString, mPtr<mQueue<mKeyValuePair<mString, mString>>> &fileTypeNameExtentionPairs, OUT bool *pCanceled, OUT mString *pOpenedFile)
{
  mFUNCTION_SETUP();

  mERROR_CHECK(mSystemDialogue_OpenFile_Internal(nullptr, headlineString, fileTypeNameExtentionPairs, pCanceled, pOpenedFile));

  mRETURN_SUCCESS();
}

mFUNCTION(mSystemDialogue_OpenFile, mPtr<mSoftwareWindow> &window, const mString &headlineString, mPtr<mQueue<mKeyValuePair<mString, mString>>> &fileTypeNameExtentionPairs, OUT bool *pCanceled, OUT mString *pOpenedFile)
{
  mFUNCTION_SETUP();

  mERROR_IF(window == nullptr, mR_ArgumentNull);

  SDL_Window *pSdlWindow;
  mERROR_CHECK(mSoftwareWindow_GetSdlWindowPtr(window, &pSdlWindow));

  mERROR_CHECK(mSystemDialogue_OpenFile_SdlWindowInternal(pSdlWindow, headlineString, fileTypeNameExtentionPairs, pCanceled, pOpenedFile));

  mRETURN_SUCCESS();
}

mFUNCTION(mSystemDialogue_OpenFile, mPtr<mHardwareWindow> &window, const mString &headlineString, mPtr<mQueue<mKeyValuePair<mString, mString>>> &fileTypeNameExtentionPairs, OUT bool *pCanceled, OUT mString *pOpenedFile)
{
  mFUNCTION_SETUP();

  mERROR_IF(window == nullptr, mR_ArgumentNull);

  SDL_Window *pSdlWindow;
  mERROR_CHECK(mHardwareWindow_GetSdlWindowPtr(window, &pSdlWindow));

  mERROR_CHECK(mSystemDialogue_OpenFile_SdlWindowInternal(pSdlWindow, headlineString, fileTypeNameExtentionPairs, pCanceled, pOpenedFile));

  mRETURN_SUCCESS();
}

mFUNCTION(mSystemDialogue_SelectDirectory, mPtr<mSoftwareWindow> &window, const mString &headlineString, OUT OPTIONAL bool *pCanceled, OUT mString *pSelectedDirectory)
{
  mFUNCTION_SETUP();

  mERROR_IF(window == nullptr, mR_ArgumentNull);

  SDL_Window *pSdlWindow;
  mERROR_CHECK(mSoftwareWindow_GetSdlWindowPtr(window, &pSdlWindow));

  mERROR_CHECK(mSystemDialogue_SelectDirectory_SdlWindowInternal(pSdlWindow, headlineString, pCanceled, pSelectedDirectory));

  mRETURN_SUCCESS();
}

mFUNCTION(mSystemDialogue_SelectDirectory, mPtr<mHardwareWindow> &window, const mString &headlineString, OUT OPTIONAL bool *pCanceled, OUT mString *pSelectedDirectory)
{
  mFUNCTION_SETUP();

  mERROR_IF(window == nullptr, mR_ArgumentNull);

  SDL_Window *pSdlWindow;
  mERROR_CHECK(mHardwareWindow_GetSdlWindowPtr(window, &pSdlWindow));

  mERROR_CHECK(mSystemDialogue_SelectDirectory_SdlWindowInternal(pSdlWindow, headlineString, pCanceled, pSelectedDirectory));

  mRETURN_SUCCESS();
}

//////////////////////////////////////////////////////////////////////////

mFUNCTION(mSystemDialogue_OpenFile_SdlWindowInternal, SDL_Window *pWindow, const mString &headlineString, mPtr<mQueue<mKeyValuePair<mString, mString>>> &fileTypeNameExtentionPairs, OUT bool *pCanceled, OUT mString *pOpenedFile)
{
  mFUNCTION_SETUP();

  SDL_SysWMinfo wmInfo;
  SDL_VERSION(&wmInfo.version);
  mERROR_IF(SDL_FALSE == SDL_GetWindowWMInfo(pWindow, &wmInfo), mR_NotSupported);

  HWND hwnd = wmInfo.info.win.window;
  mERROR_IF(hwnd == nullptr, mR_NotSupported);

  mERROR_CHECK(mSystemDialogue_OpenFile_Internal(hwnd, headlineString, fileTypeNameExtentionPairs, pCanceled, pOpenedFile));

  mRETURN_SUCCESS();
}

mFUNCTION(mSystemDialogue_OpenFile_Internal, HWND window, const mString &headlineString, mPtr<mQueue<mKeyValuePair<mString, mString>>> &fileTypeExtentionPairs, OUT bool *pCanceled, OUT mString *pFileName)
{
  mFUNCTION_SETUP();

  size_t fileTypeCount = 0;
  mERROR_CHECK(mQueue_GetCount(fileTypeExtentionPairs, &fileTypeCount));
  size_t extentionDescriptorLength = 1;

  for (size_t i = 0; i < fileTypeCount; i++)
  {
    mKeyValuePair<mString, mString> *pKVPair = nullptr;
    mERROR_CHECK(mQueue_PointerAt(fileTypeExtentionPairs, i, &pKVPair));

    extentionDescriptorLength += pKVPair->key.bytes;
    extentionDescriptorLength += pKVPair->value.bytes;
  }

  wchar_t *pExtentions = nullptr;
  mDEFER_CALL(&pExtentions, mFreePtrStack);
  mERROR_CHECK(mAllocStackZero(&pExtentions, extentionDescriptorLength));

  size_t offset = 0;

  for (size_t i = 0; i < fileTypeCount; i++)
  {
    mKeyValuePair<mString, mString> *pKVPair = nullptr;
    mERROR_CHECK(mQueue_PointerAt(fileTypeExtentionPairs, i, &pKVPair));

    std::wstring keyw, valuew;
    mERROR_CHECK(mString_ToWideString(pKVPair->key, &keyw));
    mERROR_CHECK(mString_ToWideString(pKVPair->value, &valuew));

    mMemcpy(pExtentions + offset, keyw.c_str(), keyw.length());
    offset += keyw.length() + 1;
    mMemcpy(pExtentions + offset, valuew.c_str(), valuew.length());
    offset += valuew.length() + 1;
  }

  std::wstring headline;
  mERROR_CHECK(mString_ToWideString(headlineString, &headline));

  wchar_t fileName[MAX_PATH] = { 0 };

  OPENFILENAMEW dialogueOptions;
  mMemset(&dialogueOptions, 1);

  dialogueOptions.lStructSize = sizeof(dialogueOptions);
  dialogueOptions.hwndOwner = window;
  dialogueOptions.nMaxFile = mARRAYSIZE(fileName);
  dialogueOptions.lpstrFile = fileName;
  dialogueOptions.nFileExtension = 0;
  dialogueOptions.lpstrFilter = pExtentions;
  dialogueOptions.nFilterIndex = 1;
  dialogueOptions.lpstrTitle = headline.c_str();
  dialogueOptions.Flags = OFN_ENABLESIZING | OFN_EXPLORER | OFN_FILEMUSTEXIST | OFN_NOCHANGEDIR;

  const BOOL result = GetOpenFileNameW(&dialogueOptions);

  DWORD error = GetLastError();

  mERROR_IF(error != NO_ERROR, mR_InternalError);

  *pCanceled = !result;

  if (result != 0)
    mERROR_CHECK(mString_Create(pFileName, fileName, mARRAYSIZE(fileName)));

  mRETURN_SUCCESS();
}

mFUNCTION(mSystemDialogue_SelectDirectory_SdlWindowInternal, SDL_Window *pWindow, const mString &headlineString, OUT OPTIONAL bool *pCanceled, OUT mString *pSelectedDirectory)
{
  mFUNCTION_SETUP();

  SDL_SysWMinfo wmInfo;
  SDL_VERSION(&wmInfo.version);
  mERROR_IF(SDL_FALSE == SDL_GetWindowWMInfo(pWindow, &wmInfo), mR_NotSupported);

  HWND hwnd = wmInfo.info.win.window;
  mERROR_IF(hwnd == nullptr, mR_NotSupported);

  std::wstring headline;
  mERROR_CHECK(mString_ToWideString(headlineString, &headline));

  wchar_t folderName[MAX_PATH] = { 0 };

  BROWSEINFOW browseWindowOptions;
  mERROR_CHECK(mZeroMemory(&browseWindowOptions));

  browseWindowOptions.hwndOwner = hwnd;
  browseWindowOptions.lpszTitle = (wchar_t *)headline.c_str();
  browseWindowOptions.pszDisplayName = folderName;
  browseWindowOptions.ulFlags = BIF_VALIDATE | BIF_USENEWUI | BIF_RETURNONLYFSDIRS;

  const PIDLIST_ABSOLUTE result = SHBrowseForFolderW(&browseWindowOptions);
  
  // Free memory used.
  mDEFER(
    if (result != nullptr)
    {
      IMalloc *pImalloc = nullptr;

      if (SUCCEEDED(SHGetMalloc(&pImalloc)))
      {
        pImalloc->Free(result);
        pImalloc->Release();
      }
    }
  );

  const DWORD error = GetLastError();

  mERROR_IF(error != NO_ERROR, mR_InternalError);

  if (pCanceled != nullptr)
    *pCanceled = (result == nullptr);

  
  if (result != nullptr)
  {
    wchar_t path[MAX_PATH] = { 0 };

    mERROR_IF(FALSE == SHGetPathFromIDListW(result, path), mR_InternalError);
    mERROR_CHECK(mString_Create(pSelectedDirectory, path, mARRAYSIZE(path)));
  }

  mRETURN_SUCCESS();
}
