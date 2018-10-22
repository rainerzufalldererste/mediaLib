#ifndef mKeyValuePair_h__
#define mKeyValuePair_h__

#include "mediaLib.h"

template <typename TKey, typename TValue>
struct mKeyValuePair
{
  TKey key;
  TValue value;

  mKeyValuePair() = default;
  mKeyValuePair(TKey key, TValue value) : key(key), value(value) {}
};

template <typename TKey, typename TValue>
mFUNCTION(mKeyValuePair_Create, OUT mKeyValuePair<TKey, TValue> *pPair, TKey key, TValue value);

template <typename TKey, typename TValue>
mFUNCTION(mKeyValuePair_Destroy, IN_OUT mKeyValuePair<TKey, TValue> *pPair);

//////////////////////////////////////////////////////////////////////////

template <typename TKey, typename TValue>
mFUNCTION(mKeyValuePair_Create, OUT mKeyValuePair<TKey, TValue> *pPair, TKey key, TValue value)
{
  mFUNCTION_SETUP();

  mERROR_IF(pPair == nullptr, mR_ArgumentNull);

  pPair->key = key;
  pPair->value = value;

  mRETURN_SUCCESS();
}

template<typename TKey, typename TValue>
inline mFUNCTION(mKeyValuePair_Destroy, IN_OUT mKeyValuePair<TKey, TValue> *pPair)
{
  mFUNCTION_SETUP();

  mERROR_IF(pPair == nullptr, mR_ArgumentNull);

  mERROR_CHECK(mDestruct(&pPair->key));
  mERROR_CHECK(mDestruct(&pPair->value));

  mRETURN_SUCCESS();
}

template<typename TKey, typename TValue>
mFUNCTION(mDestruct, IN_OUT mKeyValuePair<TKey, TValue> *pPair)
{
  mFUNCTION_SETUP();

  mERROR_CHECK(mKeyValuePair_Destroy(pPair));

  mRETURN_SUCCESS();
}

template <typename TKey, typename TValue>
struct mIsTriviallyMemoryMovable<mKeyValuePair<TKey, TValue>>
{
  static constexpr bool value = mIsTriviallyMemoryMovable<TKey>::value && mIsTriviallyMemoryMovable<TValue>::value;
};

#endif // mKeyValuePair_h__
