// Copyright 2018 Christoph Stiller
// 
// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files(the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions :
// 
// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
// 
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef mSystemError_h__
#define mSystemError_h__

#include "default.h"

struct mHardwareWindow;
struct mSoftwareWindow;

enum mSystemError_Type
{
  mSE_T_Default,
  mSE_T_Information,
  mSE_T_Warning,
  mSE_T_Error,
  mSE_T_Question,
};

enum mSystemError_MessageBoxButton
{
  mSE_MBB_OK,
  mSE_MBB_Abort_Retry_Ignore,
  mSE_MBB_Cancel_Try_Continue,
  mSE_MBB_Help,
  mSE_MBB_OK_Cancel,
  mSE_MBB_Retry_Cancel,
  mSE_MBB_Yes_No,
  mSE_MBB_Yes_No_Cancel,
};

enum mSystemError_Authority
{
  // User cannot interact with the specified window (if available) until the message box has been responded to.
  mSE_A_Window,

  // User cannot interact with all windows of the specified task (if available) until the message box has been responded to.
  mSE_A_Task,

  // Only for critical or damaging errors to the system.
  mSE_A_System,
};

enum mSystemError_MessageBoxResponse
{
  mSE_MBR_Ok,
  mSE_MBR_Cancel,
  mSE_MBR_Yes,
  mSE_MBR_No,
  mSE_MBR_Abort,
  mSE_MBR_Continue,
  mSE_MBR_Ignore,
  mSE_MBR_Retry,
};

mFUNCTION(mSystemError_PlaySound, const mSystemError_Type type);

mFUNCTION(mSystemError_ShowMessageBox, const mSystemError_Type type, const mString &title, const mString &text, const mSystemError_MessageBoxButton buttons, mSystemError_MessageBoxResponse *pResponse, const size_t defaultButtonIndex = 0, const mSystemError_Authority authority = mSE_A_Task);
mFUNCTION(mSystemError_ShowMessageBox, mPtr<mHardwareWindow> &window, const mSystemError_Type type, const mString &title, const mString &text, const mSystemError_MessageBoxButton buttons, mSystemError_MessageBoxResponse *pResponse, const size_t defaultButtonIndex = 0, const mSystemError_Authority authority = mSE_A_Window);
mFUNCTION(mSystemError_ShowMessageBox, mPtr<mSoftwareWindow> &window, const mSystemError_Type type, const mString &title, const mString &text, const mSystemError_MessageBoxButton buttons, mSystemError_MessageBoxResponse *pResponse, const size_t defaultButtonIndex = 0, const mSystemError_Authority authority = mSE_A_Window);

#endif // mSystemError_h__
