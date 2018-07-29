#include "default.h"
#include "mVideoFileInputHandler.h"

int main(int, char **)
{
  mFUNCTION_SETUP();

  int k = 0;
  mPtr<int> kptr;
  mDEFER_DESTRUCTION(&kptr, mSharedPointer_Destroy);
  mERROR_CHECK(mSharedPointer_Create(&kptr, &k, mAT_ForeignRessource));

  uint32_t *pPtr = nullptr;
  mDEFER_DESTRUCTION(pPtr, mFree);
  mERROR_CHECK(mAlloc(&pPtr, 1));

  mPtr<mVideoFileInputHandler> videoInput;
  mDEFER_DESTRUCTION(&videoInput, mVideoFileInputHandler_Destroy);
  mERROR_CHECK(mVideoFileInputHandler_Create(&videoInput, L"N:\\Data\\video\\Babuji3.mp4"));

  mRETURN_SUCCESS();
}