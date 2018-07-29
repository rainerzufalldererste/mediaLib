#include "default.h"

int main(int, char **)
{
  mFUNCTION_SETUP();

  int k = 0;
  mPtr<int> kptr;
  mDEFER_DESTRUCTION(kptr, mSharedPointer_Destroy);
  mERROR_CHECK(mSharedPointer_Create(&kptr, &k, mAT_ForeignRessource));

  uint32_t *pPtr = nullptr;
  mDEFER_DESTRUCTION(pPtr, mFree);
  mERROR_CHECK(mAlloc(&pPtr, 1));

  mRETURN_SUCCESS();
}