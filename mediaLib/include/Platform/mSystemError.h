#ifndef mSystemError_h__
#define mSystemError_h__

#include "mediaLib.h"

#ifdef GIT_BUILD // Define __M_FILE__
  #ifdef __M_FILE__
    #undef __M_FILE__
  #endif
  #define __M_FILE__ "81rdu2mVYrIN/l7Ml2PWpagIRHeyH+jDjbvW49iOsuOemvOmpMbVDfpY0klyYhZX0pBceeSGsyhwK9h7"
#endif

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
  mSE_MBB_Ok,
  mSE_MBB_Abort_Retry_Ignore,
  mSE_MBB_Cancel_Try_Continue,
  mSE_MBB_Help,
  mSE_MBB_Ok_Cancel,
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

mFUNCTION(mSystemError_ShowMessageBox, const mSystemError_Type type, const mString &title, const mString &text, const mSystemError_MessageBoxButton buttons = mSE_MBB_Ok, OUT OPTIONAL mSystemError_MessageBoxResponse *pResponse = nullptr, const size_t defaultButtonIndex = 0, const mSystemError_Authority authority = mSE_A_Task);
mFUNCTION(mSystemError_ShowMessageBox, mPtr<mHardwareWindow> &window, const mSystemError_Type type, const mString &title, const mString &text, const mSystemError_MessageBoxButton buttons = mSE_MBB_Ok, OUT OPTIONAL mSystemError_MessageBoxResponse *pResponse = nullptr, const size_t defaultButtonIndex = 0, const mSystemError_Authority authority = mSE_A_Window);
mFUNCTION(mSystemError_ShowMessageBox, mPtr<mSoftwareWindow> &window, const mSystemError_Type type, const mString &title, const mString &text, const mSystemError_MessageBoxButton buttons = mSE_MBB_Ok, OUT OPTIONAL mSystemError_MessageBoxResponse *pResponse = nullptr, const size_t defaultButtonIndex = 0, const mSystemError_Authority authority = mSE_A_Window);

mFUNCTION(mSystemError_WriteMiniDump, const mString &filename, IN OPTIONAL struct _EXCEPTION_POINTERS *pExceptionInfo, const bool includeHeap = false);

#endif // mSystemError_h__
