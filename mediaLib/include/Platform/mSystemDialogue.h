#ifndef mSystemDialogue_h__
#define mSystemDialogue_h__

#include "mediaLib.h"
#include "mQueue.h"
#include "mKeyValuePair.h"

#ifdef GIT_BUILD // Define __M_FILE__
  #ifdef __M_FILE__
    #undef __M_FILE__
  #endif
  #define __M_FILE__ "ackNszGxN2zFyAOobNj5B28xPTlUU8MPqzDS6hbeUCZmbvdGJUTY0G8ETpZhNYf1BvpPbzjU/UATuNzG"
#endif

struct mHardwareWindow;
struct mSoftwareWindow;

mFUNCTION(mSystemDialogue_OpenFile, const mString &headlineString, mPtr<mQueue<mKeyValuePair<mString, mString>>> &fileTypeNameExtensionPairs, OUT mString *pOpenedFile, OUT bool *pCanceled, const mString &initialDirectory = "");
mFUNCTION(mSystemDialogue_OpenFile, mPtr<mSoftwareWindow> &window, const mString &headlineString, mPtr<mQueue<mKeyValuePair<mString, mString>>> &fileTypeNameExtensionPairs, OUT mString *pOpenedFile, OUT bool *pCanceled, const mString &initialDirectory = "");
mFUNCTION(mSystemDialogue_OpenFile, mPtr<mHardwareWindow> &window, const mString &headlineString, mPtr<mQueue<mKeyValuePair<mString, mString>>> &fileTypeNameExtensionPairs, OUT mString *pOpenedFile, OUT bool *pCanceled, const mString &initialDirectory = "");

mFUNCTION(mSystemDialogue_SelectDirectory, const mString &headlineString, OUT mString *pSelectedDirectory, OUT OPTIONAL bool *pCanceled);
mFUNCTION(mSystemDialogue_SelectDirectory, mPtr<mSoftwareWindow> &window, const mString &headlineString, OUT mString *pSelectedDirectory, OUT OPTIONAL bool *pCanceled);
mFUNCTION(mSystemDialogue_SelectDirectory, mPtr<mHardwareWindow> &window, const mString &headlineString, OUT mString *pSelectedDirectory, OUT OPTIONAL bool *pCanceled);

mFUNCTION(mSystemDialogue_SaveFile, const mString &headlineString, mPtr<mQueue<mKeyValuePair<mString, mString>>> &fileTypeNameExtensionPairs, OUT mString *pFilename, OUT OPTIONAL bool *pCanceled, const mString &initialDirectory = "");
mFUNCTION(mSystemDialogue_SaveFile, mPtr<mSoftwareWindow> &window, const mString &headlineString, mPtr<mQueue<mKeyValuePair<mString, mString>>> &fileTypeNameExtensionPairs, OUT mString *pFilename, OUT OPTIONAL bool *pCanceled, const mString &initialDirectory = "");
mFUNCTION(mSystemDialogue_SaveFile, mPtr<mHardwareWindow> &window, const mString &headlineString, mPtr<mQueue<mKeyValuePair<mString, mString>>> &fileTypeNameExtensionPairs, OUT mString *pFilename, OUT OPTIONAL bool *pCanceled, const mString &initialDirectory = "");

enum mSystemDialogue_WindowProgressState
{
  mSD_WPS_None,
  mSD_WPS_Error,
  mSD_WPS_Paused,
  mSD_WPS_Indeterminate // <- this state should be used if the state doesn't have a determinable amount of progress, but is working on the task.
};

mFUNCTION(mSystemDialogue_SetWindowProgressState, mPtr<mSoftwareWindow> &window, const mSystemDialogue_WindowProgressState state);
mFUNCTION(mSystemDialogue_SetWindowProgressState, mPtr<mHardwareWindow> &window, const mSystemDialogue_WindowProgressState state);
mFUNCTION(mSystemDialogue_SetWindowProgress, mPtr<mSoftwareWindow> &window, const float_t progress);
mFUNCTION(mSystemDialogue_SetWindowProgress, mPtr<mHardwareWindow> &window, const float_t progress);

mFUNCTION(mSystemDialogue_AddToRecentlyOpenedFiles, const mString &filename);
mFUNCTION(mSystemDialogue_ClearRecentlyOpenedFiles);

#endif // mSystemDialogue_h__
