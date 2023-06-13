#include "mSystemError.h"
#include "mHardwareWindow.h"
#include "mSoftwareWindow.h"
#include "SDL_syswm.h"

mFUNCTION(mSystemError_ShowMessageBoxSDL_Internal, const mSystemError_Type type, SDL_Window *pWindow, const mString &title, const mString &text, const mSystemError_MessageBoxButton buttons, mSystemError_MessageBoxResponse *pResponse, const size_t defaultButtonIndex, const mSystemError_Authority authority);
mFUNCTION(mSystemError_ShowMessageBox_Internal, const mSystemError_Type type, HWND window, const mString &title, const mString &text, const mSystemError_MessageBoxButton buttons, mSystemError_MessageBoxResponse *pResponse, const size_t defaultButtonIndex, const mSystemError_Authority authority);

mFUNCTION(mSystemError_PlaySound, const mSystemError_Type type)
{
  mFUNCTION_SETUP();

  switch (type)
  {
  case mSE_T_Information:
    MessageBeep(MB_ICONINFORMATION);
    break;

  case mSE_T_Warning:
    MessageBeep(MB_ICONWARNING);
    break;

  case mSE_T_Error:
    MessageBeep(MB_ICONERROR);
    break;

  case mSE_T_Default:
  default:
    MessageBeep(MB_ICONERROR);
    break;
  }

  mRETURN_SUCCESS();
}

mFUNCTION(mSystemError_ShowMessageBox, const mSystemError_Type type, const mString & title, const mString & text, const mSystemError_MessageBoxButton buttons, mSystemError_MessageBoxResponse *pResponse, const size_t defaultButtonIndex /* = 0 */, const mSystemError_Authority authority /* = mSE_A_Task */)
{
  mFUNCTION_SETUP();

  mERROR_CHECK(mSystemError_ShowMessageBox_Internal(type, nullptr, title, text, buttons, pResponse, defaultButtonIndex, authority));

  mRETURN_SUCCESS();
}

mFUNCTION(mSystemError_ShowMessageBox, mPtr<mHardwareWindow> &window, const mSystemError_Type type, const mString & title, const mString & text, const mSystemError_MessageBoxButton buttons, mSystemError_MessageBoxResponse *pResponse, const size_t defaultButtonIndex /* = 0 */, const mSystemError_Authority authority /* = mSE_A_Window */)
{
  mFUNCTION_SETUP();

  mERROR_IF(window == nullptr, mR_ArgumentNull);

  SDL_Window *pSdlWindow;
  mERROR_CHECK(mHardwareWindow_GetSdlWindowPtr(window, &pSdlWindow));

  mERROR_CHECK(mSystemError_ShowMessageBoxSDL_Internal(type, pSdlWindow, title, text, buttons, pResponse, defaultButtonIndex, authority));

  mRETURN_SUCCESS();
}

mFUNCTION(mSystemError_ShowMessageBox, mPtr<mSoftwareWindow> &window, const mSystemError_Type type, const mString & title, const mString & text, const mSystemError_MessageBoxButton buttons, mSystemError_MessageBoxResponse *pResponse, const size_t defaultButtonIndex /* = 0 */, const mSystemError_Authority authority /* = mSE_A_Window */)
{
  mFUNCTION_SETUP();

  mERROR_IF(window == nullptr, mR_ArgumentNull);

  SDL_Window *pSdlWindow;
  mERROR_CHECK(mSoftwareWindow_GetSdlWindowPtr(window, &pSdlWindow));

  mERROR_CHECK(mSystemError_ShowMessageBoxSDL_Internal(type, pSdlWindow, title, text, buttons, pResponse, defaultButtonIndex, authority));

  mRETURN_SUCCESS();
}

//////////////////////////////////////////////////////////////////////////

mFUNCTION(mSystemError_ShowMessageBoxSDL_Internal, const mSystemError_Type type, SDL_Window *pWindow, const mString & title, const mString & text, const mSystemError_MessageBoxButton buttons, mSystemError_MessageBoxResponse * pResponse, const size_t defaultButtonIndex, const mSystemError_Authority authority)
{
  mFUNCTION_SETUP();

  SDL_SysWMinfo wmInfo;
  SDL_VERSION(&wmInfo.version);
  mERROR_IF(SDL_FALSE == SDL_GetWindowWMInfo(pWindow, &wmInfo), mR_NotSupported);

  HWND hwnd = wmInfo.info.win.window;
  mERROR_IF(hwnd == nullptr, mR_NotSupported);

  mERROR_CHECK(mSystemError_ShowMessageBox_Internal(type, hwnd, title, text, buttons, pResponse, defaultButtonIndex, authority));

  mRETURN_SUCCESS();
}

mFUNCTION(mSystemError_ShowMessageBox_Internal, const mSystemError_Type type, HWND window, const mString &title, const mString &text, const mSystemError_MessageBoxButton buttons, mSystemError_MessageBoxResponse *pResponse, const size_t defaultButtonIndex, const mSystemError_Authority authority)
{
  mFUNCTION_SETUP();

  UINT mbmode = 0;

  switch (type)
  {
  case mSE_T_Information:
    mbmode |= MB_ICONINFORMATION;
    break;

  case mSE_T_Warning:
    mbmode |= MB_ICONWARNING;
    break;

  case mSE_T_Error:
    mbmode |= MB_ICONWARNING;
    break;

  case mSE_T_Question:
    mbmode |= MB_ICONQUESTION;
    break;

  case mSE_T_Default:
  default:
    break;
  }

  switch (buttons)
  {
  case mSE_MBB_Abort_Retry_Ignore:
    mbmode |= MB_ABORTRETRYIGNORE;
    break;

  case mSE_MBB_Cancel_Try_Continue:
    mbmode |= MB_CANCELTRYCONTINUE;
    break;

  case mSE_MBB_Help:
    mbmode |= MB_HELP;
    break;

  case mSE_MBB_OK_Cancel:
    mbmode |= MB_OKCANCEL;
    break;

  case mSE_MBB_Retry_Cancel:
    mbmode |= MB_RETRYCANCEL;
    break;

  case mSE_MBB_Yes_No:
    mbmode |= MB_YESNO;
    break;

  case mSE_MBB_Yes_No_Cancel:
    mbmode |= MB_YESNOCANCEL;
    break;

  case mSE_MBB_OK:
  default:
    mbmode |= MB_OK;
    break;
  }

  switch (defaultButtonIndex)
  {
  case 1:
    mbmode |= MB_DEFBUTTON2;
    break;

  case 2:
    mbmode |= MB_DEFBUTTON3;
    break;

  case 3:
    mbmode |= MB_DEFBUTTON4;
    break;

  case 0:
  default:
    mbmode |= MB_DEFBUTTON1;
    break;
  }

  switch (authority)
  {
  case mSE_A_Task:
    mbmode |= MB_TASKMODAL;
    break;

  case mSE_A_System:
    mbmode |= MB_SYSTEMMODAL;
    break;

  case mSE_A_Window:
  default:

    if (window != nullptr)
      mbmode |= MB_APPLMODAL;
    else
      mbmode |= MB_TASKMODAL;

    break;
  }

  const int result = MessageBoxExW(window, ((std::wstring)text).c_str(), ((std::wstring)title).c_str(), mbmode, MAKELANGID(LANG_NEUTRAL, SUBLANG_NEUTRAL));

  if (pResponse != nullptr)
  {
    switch (result)
    {
    case IDOK:
      *pResponse = mSE_MBR_Ok;
      break;

    case IDABORT:
      *pResponse = mSE_MBR_Abort;
      break;

    case IDYES:
      *pResponse = mSE_MBR_Ok;
      break;

    case IDNO:
      *pResponse = mSE_MBR_No;
      break;

    case IDCONTINUE:
      *pResponse = mSE_MBR_Continue;
      break;

    case IDIGNORE:
      *pResponse = mSE_MBR_Ignore;
      break;

    case IDRETRY:
    case IDTRYAGAIN:
      *pResponse = mSE_MBR_Retry;
      break;

    case IDCANCEL:
    default:
      *pResponse = mSE_MBR_Cancel;
      break;
    }
  }

  mRETURN_SUCCESS();
}