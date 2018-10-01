#ifndef mSystemDialogue_h__
#define mSystemDialogue_h__

#include "mediaLib.h"
#include "mQueue.h"
#include "mKeyValuePair.h"

struct mHardwareWindow;
struct mSoftwareWindow;

mFUNCTION(mSystemDialogue_OpenFile, const mString &headlineString, mPtr<mQueue<mKeyValuePair<mString, mString>>> &fileTypeNameExtentionPairs, bool *pCanceled, mString *pOpenedFile);
mFUNCTION(mSystemDialogue_OpenFile, mPtr<mSoftwareWindow> &window, const mString &headlineString, mPtr<mQueue<mKeyValuePair<mString, mString>>> &fileTypeNameExtentionPairs, bool *pCanceled, mString *pOpenedFile);
mFUNCTION(mSystemDialogue_OpenFile, mPtr<mHardwareWindow> &window, const mString &headlineString, mPtr<mQueue<mKeyValuePair<mString, mString>>> &fileTypeNameExtentionPairs, bool *pCanceled, mString *pOpenedFile);

#endif // mSystemDialogue_h__
