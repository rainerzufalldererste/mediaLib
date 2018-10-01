#ifndef mDestruct_h__
#define mDestruct_h__

#include "mResult.h"

template <typename T>
mFUNCTION(mDestruct, IN T *pData);

//#define mDESTRUCT_LOG_DESTRUCTIONS

//////////////////////////////////////////////////////////////////////////

template <typename T>
inline mFUNCTION(mDestruct, IN T *pData)
{
  mFUNCTION_SETUP();

#ifdef mDESTRUCT_LOG_DESTRUCTIONS
  mLOG("Destructing resource of type %s with generic destruction function.\n", typeid(T *).name());
#endif

  if (pData != nullptr && std::is_destructible<T>::value)
    pData->~T();

  mRETURN_SUCCESS();
}

#endif // mDestruct_h__
