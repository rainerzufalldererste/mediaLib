#ifndef mToastNotification_h__
#define mToastNotification_h__

#include "mediaLib.h"

#ifdef GIT_BUILD // Define __M_FILE__
  #ifdef __M_FILE__
    #undef __M_FILE__
  #endif
  #define __M_FILE__ "8d0ZwOGZcIfJGxRZuJIjZAQ4VyKOk60iWDYUi7SijXTSfNb6hpguhnczrzj/Cs0bkWndtCsKumalya/g"
#endif

enum mToastNotification_Result
{
  mTN_R_NotPerformed,
  
  mTN_R_NotActivated,
  mTN_R_Activated,
  mTN_R_ActivatedWithActionID,
  mTN_R_Failed,
  mTN_R_UserCanceled,
  mTN_R_TimedOut,
  mTN_R_ApplicationHidden,
  
  _mTN_R_Count
};

struct mToastNotification;

mFUNCTION(mToastNotification_SetGlobalApplicationName, const mString &appName);

mFUNCTION(mToastNotification_Create, OUT mPtr<mToastNotification> *pNotification, IN mAllocator *pAllocator, const mString &headline, const mString &text, OPTIONAL const mString &imageOrEmpty = "");
mFUNCTION(mToastNotification_Destroy, IN_OUT mPtr<mToastNotification> *pNotification);

mFUNCTION(mToastNotification_SetSilent, mPtr<mToastNotification> &notification);
mFUNCTION(mToastNotification_AddAction, mPtr<mToastNotification> &notification, const mString &actionText);

mFUNCTION(mToastNotification_Show, mPtr<mToastNotification> &notification, OUT OPTIONAL mToastNotification_Result *pResult = nullptr, OUT OPTIONAL size_t *pActionIndex = nullptr);

#endif // mToastNotification_h__
