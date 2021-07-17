#ifndef mDestruct_h__
#define mDestruct_h__

#include "mResult.h"

#ifdef GIT_BUILD // Define __M_FILE__
  #ifdef __M_FILE__
    #undef __M_FILE__
  #endif
  #define __M_FILE__ "Va+R95AK3HFXkfYM9riPcNdHYzLQtnqBOeM97K4zxgY9wjTXWmM4/9de1PjVT6CADgc2ZmpiEKko5MCi"
#endif

template <typename T>
mFUNCTION(mDestruct, IN T *pData);

//#define mDESTRUCT_LOG_DESTRUCTIONS

//////////////////////////////////////////////////////////////////////////

template <typename T>
inline mFUNCTION(mDestruct, IN T *pData)
{
  mFUNCTION_SETUP();

#ifdef mDESTRUCT_LOG_DESTRUCTIONS
  mLOG("Destructing resource of type ", typeid(T *).name(), " with generic destruction function.");
#endif

  if (pData != nullptr && std::is_destructible<T>::value)
    pData->~T();

  mRETURN_SUCCESS();
}

#endif // mDestruct_h__
