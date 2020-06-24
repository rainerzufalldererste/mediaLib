#include "mToastNotification.h"

#include "mFile.h"

#pragma warning (push, 0)
#include "WinToast/src/wintoastlib.h"
#include "WinToast/src/wintoastlib.cpp"
#pragma warning (pop)

#ifdef GIT_BUILD // Define __M_FILE__
  #ifdef __M_FILE__
    #undef __M_FILE__
  #endif
  #define __M_FILE__ "Aw4wtyyhOhrGFjqV1TypUEpJOQJyL/knbz6fWQkCdpBj4+iqqriPfK+/BsAqDiw09mAeuSsq9iHCZXFb"
#endif

struct mToastNotification
{
  WinToastTemplate toastTemplate;
  mToastNotification_Result result;
  size_t actionId;
};

mFUNCTION(mToastNotification_Create, OUT mPtr<mToastNotification> *pNotification, IN mAllocator *pAllocator, const mString &headline, const mString &text, OPTIONAL const mString &imageOrEmpty /* = "" */)
{
  mFUNCTION_SETUP();

  mERROR_IF(!WinToast::isCompatible(), mR_ResourceIncompatible);
  mERROR_IF(pNotification == nullptr, mR_ArgumentNull);

  mAllocator *pTempAllocator = &mDefaultTempAllocator;

  mString appName;
  mERROR_CHECK(mFile_GetCurrentApplicationFilePath(&appName));
  mERROR_CHECK(mFile_ExtractFileNameFromPath(&appName, appName, false));

  wchar_t *wAppName = nullptr;
  size_t wAppNameCount = 0;

  mERROR_CHECK(mString_GetRequiredWideStringCount(appName, &wAppNameCount));
  mDEFER(mAllocator_FreePtr(pTempAllocator, &wAppName));
  mERROR_CHECK(mAllocator_AllocateZero(pTempAllocator, &wAppName, wAppNameCount));
  mERROR_CHECK(mString_ToWideString(appName, wAppName, wAppNameCount));

  mString modelName;
  mERROR_CHECK(mString_Create(&modelName, appName));
  mERROR_CHECK(mString_AppendInteger(modelName, mGetCurrentTimeNs()));

  wchar_t *wAppModelName = nullptr;
  size_t wAppModelNameCount = 0;

  mERROR_CHECK(mString_GetRequiredWideStringCount(modelName, &wAppModelNameCount));
  mDEFER(mAllocator_FreePtr(pTempAllocator, &wAppModelName));
  mERROR_CHECK(mAllocator_AllocateZero(pTempAllocator, &wAppModelName, wAppModelNameCount));
  mERROR_CHECK(mString_ToWideString(modelName, wAppModelName, wAppModelNameCount));

  wchar_t *wHeadline = nullptr;
  size_t wHeadlineCount = 0;

  mERROR_CHECK(mString_GetRequiredWideStringCount(headline, &wHeadlineCount));
  mDEFER(mAllocator_FreePtr(pTempAllocator, &wHeadline));
  mERROR_CHECK(mAllocator_AllocateZero(pTempAllocator, &wHeadline, wHeadlineCount));
  mERROR_CHECK(mString_ToWideString(headline, wHeadline, wHeadlineCount));

  wchar_t *wText = nullptr;
  size_t wTextCount = 0;

  mERROR_CHECK(mString_GetRequiredWideStringCount(text, &wTextCount));
  mDEFER(mAllocator_FreePtr(pTempAllocator, &wText));
  mERROR_CHECK(mAllocator_AllocateZero(pTempAllocator, &wText, wTextCount));
  mERROR_CHECK(mString_ToWideString(text, wText, wTextCount));
 

  wchar_t *wImagePath = nullptr;
  size_t wImagePathCount = 0;

  mERROR_CHECK(mString_GetRequiredWideStringCount(imageOrEmpty, &wImagePathCount));
  mDEFER(mAllocator_FreePtr(pTempAllocator, &wImagePath));
  mERROR_CHECK(mAllocator_AllocateZero(pTempAllocator, &wImagePath, wImagePathCount));
  mERROR_CHECK(mString_ToWideString(imageOrEmpty, wImagePath, wImagePathCount));

  mDEFER_ON_ERROR(*pNotification = nullptr);
  mERROR_CHECK((mSharedPointer_Allocate<mToastNotification>(pNotification, pAllocator, [] (mToastNotification *pData) { mDestruct(pData); }, 1)));

  new (&(*pNotification)->toastTemplate) WinToastTemplate(wImagePathCount < 2 ? WinToastTemplate::Text02 : WinToastTemplate::ImageAndText02);

  WinToast::instance()->setAppName(wAppName);
  WinToast::instance()->setAppUserModelId(wAppModelName);

  mERROR_IF(!WinToast::instance()->initialize(), mR_ResourceIncompatible);

  if (wImagePathCount >= 2)
    (*pNotification)->toastTemplate.setImagePath(wImagePath);

  (*pNotification)->toastTemplate.setTextField(wHeadline, WinToastTemplate::FirstLine);
  (*pNotification)->toastTemplate.setTextField(wText, WinToastTemplate::SecondLine);
  
  mRETURN_SUCCESS();
}

mFUNCTION(mToastNotification_Destroy, IN_OUT mPtr<mToastNotification> *pNotification)
{
  return mSharedPointer_Destroy(pNotification);
}

mFUNCTION(mToastNotification_SetSilent, mPtr<mToastNotification> &notification)
{
  mFUNCTION_SETUP();

  mERROR_IF(notification == nullptr, mR_ArgumentNull);

  notification->toastTemplate.setAudioOption(WinToastTemplate::AudioOption::Silent);

  mRETURN_SUCCESS();
}

mFUNCTION(mToastNotification_AddAction, mPtr<mToastNotification> &notification, const mString &actionText)
{
  mFUNCTION_SETUP();

  mERROR_IF(notification == nullptr, mR_ArgumentNull);

  mAllocator *pTempAllocator = &mDefaultTempAllocator;

  wchar_t *wAction = nullptr;
  size_t wActionCount = 0;

  mERROR_CHECK(mString_GetRequiredWideStringCount(actionText, &wActionCount));
  mDEFER(mAllocator_FreePtr(pTempAllocator, &wAction));
  mERROR_CHECK(mAllocator_AllocateZero(pTempAllocator, &wAction, wActionCount));
  mERROR_CHECK(mString_ToWideString(actionText, wAction, wActionCount));

  notification->toastTemplate.addAction(wAction);

  mRETURN_SUCCESS();
}

mFUNCTION(mToastNotification_Show, mPtr<mToastNotification> &notification, OUT OPTIONAL mToastNotification_Result *pResult /* = nullptr */, OUT OPTIONAL size_t *pActionIndex /* = nullptr */)
{
  mFUNCTION_SETUP();

  mERROR_IF(notification == nullptr, mR_ArgumentNull);

  class CustomNotificationHandler : public IWinToastHandler
  {
    mPtr<mToastNotification> notification;

  public:
    explicit CustomNotificationHandler(mPtr<mToastNotification> notification) :
      notification(notification)
    { }

    void toastActivated() const
    {
      const_cast<mToastNotification *>(notification.GetPointer())->result = mTN_R_ActivatedWithActionID;
    }

    void toastActivated(int32_t actionIndex) const
    {
      const_cast<mToastNotification *>(notification.GetPointer())->result = mTN_R_ActivatedWithActionID;
      const_cast<mToastNotification *>(notification.GetPointer())->actionId = actionIndex;
    }

    void toastDismissed(WinToastDismissalReason state) const
    {
      switch (state)
      {
      case UserCanceled:
        const_cast<mToastNotification *>(notification.GetPointer())->result = mTN_R_UserCanceled;
        break;

      case TimedOut:
        const_cast<mToastNotification *>(notification.GetPointer())->result = mTN_R_TimedOut;
        exit(2);
        break;
      
      case ApplicationHidden:
        const_cast<mToastNotification *>(notification.GetPointer())->result = mTN_R_ApplicationHidden;
        break;
      
      default:
        const_cast<mToastNotification *>(notification.GetPointer())->result = mTN_R_NotActivated;
        break;
      }
    }

    void toastFailed() const {
      const_cast<mToastNotification *>(notification.GetPointer())->result = mTN_R_Failed;
    }
  };

  CustomNotificationHandler handler(notification);

  mERROR_IF(WinToast::instance()->showToast(notification->toastTemplate, &handler) < 0, mR_Failure);

  if (pResult != nullptr)
    *pResult = notification->result;

  if (pActionIndex != nullptr)
    *pActionIndex = notification->actionId;

  mRETURN_SUCCESS();
}
