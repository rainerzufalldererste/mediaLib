#ifndef mSharedPointer_h__
#define mSharedPointer_h__

#include "mediaLib.h"

#ifdef GIT_BUILD // Define __M_FILE__
  #ifdef __M_FILE__
    #undef __M_FILE__
  #endif
  #define __M_FILE__ "n6y6J1gutmiKm+SPo3/4ax3zPKfhPWxiHShRkbnjmsiG7+QsvoNWXdOcqUtAw8WOIHpkSvoEO3usJuEJ"
#endif

struct mAllocator;

#define mSHARED_POINTER_FOREIGN_RESOURCE (reinterpret_cast<mAllocator *>((size_t)-1))
//#define mSHARED_POINTER_DEBUG_OUTPUT 1

#if defined(GIT_BUILD) && defined(mSHARED_POINTER_DEBUG_OUTPUT)
#error mSHARED_POINTER_DEBUG_OUTPUT cannot be enabled for git builds.
#endif

template <typename T>
struct mIsSharedPointer
{
  static constexpr bool value = false;
};

template <typename T>
class mSharedPointer;

template <typename T>
struct mIsSharedPointer<mSharedPointer<T>>
{
  static constexpr bool value = true;
};

template <typename T>
class mReferencePack;

template <typename T>
struct mIsSharedPointer<mReferencePack<T>>
{
  static constexpr bool value = true;
};

template <typename T>
class mUniqueContainer;

template <typename T>
struct mIsSharedPointer<mUniqueContainer<T>>
{
  static constexpr bool value = true;
};

template <typename T>
class mSharedPointer
{
public:
  mSharedPointer();
  mSharedPointer(nullptr_t);

  mSharedPointer(const mSharedPointer<T> &copy);
  mSharedPointer(mSharedPointer<T> &&move);

  template <typename T2, typename std::enable_if<!std::is_same<T, T2>::value && (std::is_base_of<T2, T>::value || std::is_base_of<T, T2>::value)>* = nullptr>
  explicit mSharedPointer(const mSharedPointer<T2> &copy);

  template <typename T2, typename std::enable_if<!std::is_same<T, T2>::value && (std::is_base_of<T2, T>::value || std::is_base_of<T, T2>::value)>* = nullptr>
  explicit mSharedPointer(mSharedPointer<T2> &&move);
  
  ~mSharedPointer();

  mSharedPointer<T>& operator = (const mSharedPointer<T> &copy);
  mSharedPointer<T>& operator = (mSharedPointer<T> &&move);

  template <typename T2, typename = std::enable_if<!std::is_same<T, T2>::value && std::is_base_of<T, T2>::value>>
  mSharedPointer<T>& operator = (const mSharedPointer<T2> &copy);

  template <typename T2, typename = std::enable_if<!std::is_same<T, T2>::value && std::is_base_of<T, T2>::value>>
  mSharedPointer<T>& operator = (mSharedPointer<T2> &&move);

  T* operator -> ();
  const T* operator -> () const;
  T& operator * ();
  const T& operator * () const;

  bool operator ==(const mSharedPointer<T> &other) const;
  bool operator ==(nullptr_t) const;
  bool operator !=(const mSharedPointer<T> &other) const;
  bool operator !=(nullptr_t) const;

  operator bool() const;
  bool operator !() const;

  template <typename T2, typename std::enable_if_t<!mIsSharedPointer<T2>::value && std::is_convertible<T2, bool>::value, int>* = nullptr>
  operator T2() const;

  const T* GetPointer() const;
  T* GetPointer();
  size_t GetReferenceCount() const;

  T *m_pData;
  
  struct PointerParams
  {
    volatile size_t referenceCount = 0;
    uint8_t freeResource : 1;
    uint8_t freeParameters : 1;
    mAllocator *pAllocator = nullptr;
    std::function<void (T*)> cleanupFunction = nullptr;
    void *pUserData = nullptr;

    PointerParams() :
      freeResource(0),
      freeParameters(0)
    { }
    
  } *m_pParams;
};

// Be very careful with this! This should not be passed to functions that do under no circumstance keep a copy of the shared pointer.
template <typename T>
class mReferencePack : public mSharedPointer<T>
{
public:
  typename mSharedPointer<T>::PointerParams m_pointerParams;

  mReferencePack()
  {
    mSharedPointer<T>::m_pParams = nullptr;
    mSharedPointer<T>::m_pData = nullptr;
    m_pointerParams.referenceCount = 1;
  }

  mReferencePack(T *pValue)
  {
    mSharedPointer<T>::m_pParams = &m_pointerParams;
    mSharedPointer<T>::m_pData = pValue;
    m_pointerParams.referenceCount = 1;
  }

  mReferencePack(T *pValue, const std::function<void(T *pData)> &cleanupFunc)
  {
    mSharedPointer<T>::m_pParams = &m_pointerParams;
    mSharedPointer<T>::m_pData = pValue;
    m_pointerParams.cleanupFunction = cleanupFunc;
    m_pointerParams.referenceCount = 1;
  }

  mReferencePack(T *pValue, std::function<void(T *pData)> &&cleanupFunc)
  {
    mSharedPointer<T>::m_pParams = &m_pointerParams;
    mSharedPointer<T>::m_pData = pValue;
    m_pointerParams.cleanupFunction = std::move(cleanupFunc);
    m_pointerParams.referenceCount = 1;
  }

  mReferencePack(mReferencePack<T> &&move) :
    m_pointerParams(std::move(move.m_pointerParams))
  {
    mSharedPointer<T>::m_pParams = &m_pointerParams;
    mSharedPointer<T>::m_pData = move.m_pData;

    mASSERT_DEBUG(m_pointerParams.referenceCount == 1, "Not all references of the mReferencePack<%s> have been returned.", typeid(T).name());

    move.m_pData = nullptr;
    move.m_pParams = nullptr;
  }

  mReferencePack<T> & operator=(mReferencePack<T> &&move)
  {
    mASSERT_DEBUG(m_pointerParams.referenceCount == 1, "Not all references of the mReferencePack<%s> have been returned.", typeid(T).name());
    mASSERT_DEBUG(move.m_pointerParams.referenceCount == 1, "Not all references of the mReferencePack<%s> have been returned.", typeid(T).name());

    // Cleanup
    {
      if (mSharedPointer<T>::m_pData != nullptr && m_pointerParams.cleanupFunction && m_pointerParams.referenceCount == 1)
        m_pointerParams.cleanupFunction(mSharedPointer<T>::m_pData);
    }

    m_pointerParams = std::move(move);
    mSharedPointer<T>::m_pParams = &m_pointerParams;
    mSharedPointer<T>::m_pData = move.m_pData;

    move.m_pData = nullptr;
    move.m_pParams = nullptr;
  }

  mReferencePack(const mReferencePack<T> &copy) = delete;
  mReferencePack<T> & operator = (const mReferencePack<T> &copy) = delete;

  operator mSharedPointer<T>()
  {
    return *static_cast<mSharedPointer<T> *>(this);
  }

  operator const mSharedPointer<T>() const
  {
    return *static_cast<mSharedPointer<T> *>(this);
  }

  mSharedPointer<T> & ToPtr()
  {
    return *static_cast<mSharedPointer<T> *>(this);
  }

  const mSharedPointer<T> & ToPtr() const
  {
    return *static_cast<mSharedPointer<T> *>(this);
  }

  ~mReferencePack<T>()
  {
    mASSERT_DEBUG(m_pointerParams.referenceCount == 1, "Not all references of the mReferencePack<%s> have been returned.", typeid(T).name());

    if (mSharedPointer<T>::m_pData != nullptr && m_pointerParams.cleanupFunction && m_pointerParams.referenceCount == 1)
    {
#ifdef mSHARED_POINTER_DEBUG_OUTPUT
      mLOG("Destroying mReferencePack<%s>. (0x%" PRIx64 ")\n", typeid(T).name(), (uint64_t)m_pData);
#endif
      m_pointerParams.cleanupFunction(mSharedPointer<T>::m_pData);
    }

    mSharedPointer<T>::m_pData = nullptr;
    mSharedPointer<T>::m_pParams = nullptr;
  }
};

template <typename T>
class mUniqueContainer : public mSharedPointer<T>
{
private:
  template <typename T = T>
  typename std::enable_if<(std::is_move_constructible<T>::value || std::is_arithmetic<T>::value) && mIsTriviallyMemoryMovable<T>::value>::type
    MoveConstructFunc(IN mUniqueContainer<T> *pMove)
  {
    if (*pMove != nullptr)
    {
      new (&m_pointerParams) mSharedPointer<T>::PointerParams(std::move(pMove->m_pointerParams));
      mSharedPointer<T>::m_pParams = &m_pointerParams;
      mSharedPointer<T>::m_pData = reinterpret_cast<T *>(m_value);

      mASSERT_DEBUG(m_pointerParams.referenceCount <= 1, "Not all references of the mUniqueContainer<%s> have been returned.", typeid(T).name());

      mMemmove(reinterpret_cast<T *>(m_value), reinterpret_cast<T *>(pMove->m_value), 1);
      pMove->m_pData = nullptr;
      pMove->m_pParams = nullptr;
    }
  }

  template <typename T = T>
  typename std::enable_if<(std::is_move_constructible<T>::value || std::is_arithmetic<T>::value) && !mIsTriviallyMemoryMovable<T>::value>::type
    MoveConstructFunc(IN mUniqueContainer<T> *pMove)
  {
    if (*pMove != nullptr)
    {
      new (&m_pointerParams) mSharedPointer<T>::PointerParams(std::move(pMove->m_pointerParams));
      mSharedPointer<T>::m_pParams = &m_pointerParams;
      new (reinterpret_cast<T *>(m_value)) T(std::move(*reinterpret_cast<T *>(pMove->m_value)));
      mSharedPointer<T>::m_pData = reinterpret_cast<T *>(m_value);

      mASSERT_DEBUG(m_pointerParams.referenceCount <= 1, "Not all references of the mUniqueContainer<%s> have been returned.", typeid(T).name());

      pMove->m_pData = nullptr;
      pMove->m_pParams = nullptr;
    }
  }

  template <typename T = T>
  typename std::enable_if<(std::is_move_assignable<T>::value || std::is_arithmetic<T>::value) && mIsTriviallyMemoryMovable<T>::value>::type
    MoveAssignFunc(IN mUniqueContainer<T> *pMove)
  {
    mASSERT_DEBUG(m_pointerParams.referenceCount <= 1, "Not all references of the mUniqueContainer<%s> have been returned.", typeid(T).name());
    mASSERT_DEBUG(pMove->m_pointerParams.referenceCount == 1, "Not all references of the mUniqueContainer<%s> have been returned.", typeid(T).name());

    // Cleanup
    {
      if (mSharedPointer<T>::m_pData != nullptr && m_pointerParams.cleanupFunction && m_pointerParams.referenceCount == 1)
        m_pointerParams.cleanupFunction(mSharedPointer<T>::m_pData);
    }

    if (*pMove != nullptr)
    {
      m_pointerParams = std::move(pMove->m_pointerParams);
      mSharedPointer<T>::m_pParams = &m_pointerParams;
      mMemmove(reinterpret_cast<T *>(m_value), reinterpret_cast<T *>(pMove->m_value), 1);
      mSharedPointer<T>::m_pData = reinterpret_cast<T *>(m_value);

      pMove->m_pData = nullptr;
      pMove->m_pParams = nullptr;
    }
    else
    {
      mSharedPointer<T>::m_pData = nullptr;
      mSharedPointer<T>::m_pParams = nullptr;
    }
  }

  template <typename T = T>
  typename std::enable_if<(std::is_move_assignable<T>::value || std::is_arithmetic<T>::value) && !mIsTriviallyMemoryMovable<T>::value>::type
    MoveAssignFunc(IN mUniqueContainer<T> *pMove)
  {
    mASSERT_DEBUG(m_pointerParams.referenceCount <= 1, "Not all references of the mUniqueContainer<%s> have been returned.", typeid(T).name());
    mASSERT_DEBUG(pMove->m_pointerParams.referenceCount == 1, "Not all references of the mUniqueContainer<%s> have been returned.", typeid(T).name());

    // Cleanup
    {
      if (mSharedPointer<T>::m_pData != nullptr && m_pointerParams.cleanupFunction && m_pointerParams.referenceCount == 1)
        m_pointerParams.cleanupFunction(mSharedPointer<T>::m_pData);
    }

    if (*pMove != nullptr)
    {
      m_pointerParams = std::move(pMove->m_pointerParams);
      mSharedPointer<T>::m_pParams = &m_pointerParams;
      *reinterpret_cast<T *>(m_value) = std::move(*reinterpret_cast<T *>(pMove->m_value));
      mSharedPointer<T>::m_pData = reinterpret_cast<T *>(m_value);

      pMove->m_pData = nullptr;
      pMove->m_pParams = nullptr;
    }
    else
    {
      mSharedPointer<T>::m_pData = nullptr;
      mSharedPointer<T>::m_pParams = nullptr;
    }
  }

public:
  typename mSharedPointer<T>::PointerParams m_pointerParams;
  uint8_t m_value[sizeof(T)]; // to not accidentally initialize a value here that has a default constructor.
  
  mUniqueContainer()
  {
    mZeroMemory(m_value, ARRAYSIZE(m_value));
    m_pointerParams.referenceCount = 1;
  }

  mUniqueContainer(nullptr_t /* null */)
  {
    mZeroMemory(m_value, ARRAYSIZE(m_value));
    m_pointerParams.referenceCount = 1;
  }

  template <typename... Args>
  mUniqueContainer(Args ...args)
  {
    mZeroMemory(m_value, ARRAYSIZE(m_value));
    new (m_value) T(args...);
    mSharedPointer<T>::m_pData = reinterpret_cast<T *>(m_value);
    mSharedPointer<T>::m_pParams = &m_pointerParams;
    m_pointerParams.referenceCount = 1;
  }

  template <typename... Args>
  static void ConstructWithCleanupFunction(mUniqueContainer<T> *pTarget, const std::function<void(T *)> &cleanupFunc, Args &&...args)
  {
    pTarget->~mUniqueContainer();
    mZeroMemory(pTarget->m_value, ARRAYSIZE(pTarget->m_value));
    new (pTarget->m_value) T(std::forward<Args>(args)...);
    pTarget->m_pointerParams.cleanupFunction = cleanupFunc;
    pTarget->mSharedPointer<T>::m_pData = reinterpret_cast<T *>(pTarget->m_value);
    pTarget->mSharedPointer<T>::m_pParams = &pTarget->m_pointerParams;
    pTarget->m_pointerParams.referenceCount = 1;
  }

  template <typename... Args>
  static void ConstructWithCleanupFunction(mUniqueContainer<T> *pTarget, std::function<void(T *)> &&cleanupFunc, Args &&...args)
  {
    pTarget->~mUniqueContainer();
    mZeroMemory(pTarget->m_value, ARRAYSIZE(pTarget->m_value));
    new (pTarget->m_value) T(std::forward<Args>(args)...);
    pTarget->m_pointerParams.cleanupFunction = std::move(cleanupFunc);
    pTarget->mSharedPointer<T>::m_pData = reinterpret_cast<T *>(pTarget->m_value);
    pTarget->mSharedPointer<T>::m_pParams = &pTarget->m_pointerParams;
    pTarget->m_pointerParams.referenceCount = 1;
  }

  static void CreateWithCleanupFunction(mUniqueContainer<T> *pTarget, const std::function<void(T *)> &cleanupFunc)
  {
    pTarget->~mUniqueContainer();
    mZeroMemory(pTarget->m_value, ARRAYSIZE(pTarget->m_value));
    pTarget->m_pointerParams.cleanupFunction = cleanupFunc;
    pTarget->mSharedPointer<T>::m_pData = reinterpret_cast<T *>(pTarget->m_value);
    pTarget->mSharedPointer<T>::m_pParams = &pTarget->m_pointerParams;
    pTarget->m_pointerParams.referenceCount = 1;
  }

  static void CreateWithCleanupFunction(mUniqueContainer<T> *pTarget, std::function<void(T *)> &&cleanupFunc)
  {
    pTarget->~mUniqueContainer();
    mZeroMemory(pTarget->m_value, ARRAYSIZE(pTarget->m_value));
    pTarget->m_pointerParams.cleanupFunction = std::move(cleanupFunc);
    pTarget->mSharedPointer<T>::m_pData = reinterpret_cast<T *>(pTarget->m_value);
    pTarget->mSharedPointer<T>::m_pParams = &pTarget->m_pointerParams;
    pTarget->m_pointerParams.referenceCount = 1;
  }

  mUniqueContainer(mUniqueContainer<T> &&move)
  {
    MoveConstructFunc(&move);
  }

  mUniqueContainer<T> & operator=(mUniqueContainer<T> &&move)
  {
    MoveAssignFunc(&move);
    return *this;
  }

  mUniqueContainer(const mUniqueContainer<T> &copy) = delete;
  mUniqueContainer<T> & operator = (const mUniqueContainer<T> &copy) = delete;

  operator mSharedPointer<T>()
  {
    return *static_cast<mSharedPointer<T> *>(this);
  }

  operator const mSharedPointer<T>() const
  {
    return *static_cast<mSharedPointer<T> *>(this);
  }

  mSharedPointer<T> & ToPtr()
  {
    return *static_cast<mSharedPointer<T> *>(this);
  }

  const mSharedPointer<T> & ToPtr() const
  {
    return *static_cast<mSharedPointer<T> *>(this);
  }

  ~mUniqueContainer<T>()
  {
    mASSERT_DEBUG(m_pointerParams.referenceCount <= 1, "Not all references of the mUniqueContainer<%s> have been returned.", typeid(T).name());

    if (mSharedPointer<T>::m_pData != nullptr && m_pointerParams.cleanupFunction)
    {
#ifdef mSHARED_POINTER_DEBUG_OUTPUT
      mLOG("Destroying mUniqueContainer<%s>. (0x%" PRIx64 ")\n", typeid(T).name(), (uint64_t)m_pData);
#endif
      m_pointerParams.cleanupFunction(reinterpret_cast<T *>(m_value));
    }

    mSharedPointer<T>::m_pData = nullptr;
    mSharedPointer<T>::m_pParams = nullptr;
  }
};

template <typename T>
using mPtr = mSharedPointer<T>;

template <typename T> void mDeinit(mPtr<T> *pParam) { if (pParam != nullptr) { mSharedPointer_Destroy(pParam); } };
template <typename T, typename ...Args> void mDeinit(mPtr<T> *pParam, Args... args) { if (pParam != nullptr) { mSharedPointer_Destroy(pParam); } mDeinit(std::forward<Args>(args)); };

template <typename T>
inline mFUNCTION(mSharedPointer_Create, OUT mSharedPointer<T> *pOutSharedPointer, IN T *pData, std::function<void(T *pData)> cleanupFunction, IN mAllocator *pAllocator)
{
  mFUNCTION_SETUP();

  mERROR_IF(pOutSharedPointer == nullptr, mR_ArgumentNull);

  if (*pOutSharedPointer != nullptr)
    *pOutSharedPointer = nullptr;

  mERROR_CHECK(mAllocator_AllocateZero(pAllocator == mSHARED_POINTER_FOREIGN_RESOURCE ? nullptr : pAllocator, &pOutSharedPointer->m_pParams, 1));

  pOutSharedPointer->m_pParams = new (pOutSharedPointer->m_pParams) typename mSharedPointer<T>::PointerParams();
  pOutSharedPointer->m_pParams->referenceCount = 1;
  pOutSharedPointer->m_pParams->pAllocator = pAllocator == mSHARED_POINTER_FOREIGN_RESOURCE ? nullptr : pAllocator;
  pOutSharedPointer->m_pParams->freeResource = (pAllocator != mSHARED_POINTER_FOREIGN_RESOURCE);
  pOutSharedPointer->m_pParams->freeParameters = true;
  pOutSharedPointer->m_pParams->cleanupFunction = cleanupFunction;
  pOutSharedPointer->m_pData = pData;
  pOutSharedPointer->m_pParams->pUserData = nullptr;

#ifdef mSHARED_POINTER_DEBUG_OUTPUT
  mLOG("Created mSharedPointer<%s>. (0x%" PRIx64 ")\n", typeid(T).name(), (uint64_t)pData);
#endif

  mRETURN_SUCCESS();
}

template <typename T>
inline mFUNCTION(mSharedPointer_Create, OUT mSharedPointer<T> *pOutSharedPointer, IN T *pData, IN mAllocator *pAllocator)
{
  mFUNCTION_SETUP();

  mERROR_CHECK(mSharedPointer_Create(pOutSharedPointer, pData, std::function<void(T *pData)>(nullptr), pAllocator));

  mRETURN_SUCCESS();
}

template <typename T>
inline mFUNCTION(mSharedPointer_Allocate, OUT mSharedPointer<T> *pOutSharedPointer, IN mAllocator *pAllocator, const size_t count = 1)
{
  mFUNCTION_SETUP();

  mERROR_IF(pOutSharedPointer == nullptr, mR_ArgumentNull);

  T *pData = nullptr;
  mDEFER(mAllocator_FreePtr(pAllocator, &pData));
  mERROR_CHECK(mAllocator_AllocateZero(pAllocator, &pData, count));

  mERROR_CHECK(mSharedPointer_Create(pOutSharedPointer, pData, pAllocator));

  pData = nullptr; // to not get released on destruction.

  mRETURN_SUCCESS();
}

template <typename T>
inline mFUNCTION(mSharedPointer_Allocate, OUT mSharedPointer<T> *pOutSharedPointer, IN mAllocator *pAllocator, const std::function<void(T *)> &function, const size_t count)
{
  mFUNCTION_SETUP();

  mERROR_IF(pOutSharedPointer == nullptr, mR_ArgumentNull);

  T *pData = nullptr;
  mDEFER(mAllocator_FreePtr(pAllocator, &pData));
  mERROR_CHECK(mAllocator_AllocateZero(pAllocator, &pData, count));

  mERROR_CHECK(mSharedPointer_Create(pOutSharedPointer, pData, function, pAllocator));

  pData = nullptr; // to not get released on destruction.

  mRETURN_SUCCESS();
}

template <typename T, typename TFlexArrayType>
inline mFUNCTION(mSharedPointer_AllocateWithFlexArray, OUT mSharedPointer<T> *pOutSharedPointer, IN mAllocator *pAllocator, const size_t flexArrayCount, const std::function<void(T *)> &function)
{
  mFUNCTION_SETUP();

  mERROR_IF(pOutSharedPointer == nullptr, mR_ArgumentNull);

  uint8_t *pData = nullptr;
  mDEFER(mAllocator_FreePtr(pAllocator, &pData));
  mERROR_CHECK(mAllocator_AllocateZero(pAllocator, &pData, sizeof(T) + sizeof(TFlexArrayType) * flexArrayCount));

  mERROR_CHECK(mSharedPointer_Create(pOutSharedPointer, reinterpret_cast<T *>(pData), function, pAllocator));

  pData = nullptr; // to not get released on destruction.

  mRETURN_SUCCESS();
}

template <typename T, typename TInherited, typename std::enable_if<std::is_base_of<T, TInherited>::value>* = nullptr>
inline mFUNCTION(mSharedPointer_AllocateInherited, OUT mSharedPointer<T> *pOutSharedPointer, IN mAllocator *pAllocator, const std::function<void(TInherited *)> &function, OUT OPTIONAL TInherited **ppInherited)
{
  mFUNCTION_SETUP();

  mERROR_IF(pOutSharedPointer == nullptr, mR_ArgumentNull);

  TInherited *pData = nullptr;
  mDEFER(mAllocator_FreePtr(pAllocator, &pData));
  mERROR_CHECK(mAllocator_AllocateZero(pAllocator, &pData, 1));

  if (function != nullptr)
    mERROR_CHECK(mSharedPointer_Create(pOutSharedPointer, static_cast<T *>(pData), (std::function<void(T *)>)[function](T *pDestructData) { function(static_cast<TInherited *>(pDestructData)); }, pAllocator));
  else
    mERROR_CHECK(mSharedPointer_Create<T>(pOutSharedPointer, static_cast<T *>(pData), nullptr, pAllocator));


  if (ppInherited != nullptr)
    *ppInherited = pData;

  pData = nullptr; // to not get released on destruction.

#ifdef mSHARED_POINTER_DEBUG_OUTPUT
  mLOG(" [was inherited from %s]\n", typeid(TInherited).name());
#endif

  mRETURN_SUCCESS();
}

template <typename T, typename TInherited, typename TFlexArrayType, typename std::enable_if<std::is_base_of<T, TInherited>::value>* = nullptr>
inline mFUNCTION(mSharedPointer_AllocateInheritedWithFlexArray, OUT mSharedPointer<T> *pOutSharedPointer, IN mAllocator *pAllocator, const size_t flexArrayCount, const std::function<void(TInherited *)> &function, OUT OPTIONAL TInherited **ppInherited)
{
  mFUNCTION_SETUP();

  mERROR_IF(pOutSharedPointer == nullptr, mR_ArgumentNull);

  uint8_t *pData = nullptr;
  mDEFER(mAllocator_FreePtr(pAllocator, &pData));
  mERROR_CHECK(mAllocator_AllocateZero(pAllocator, &pData, sizeof(TInherited) + sizeof(TFlexArrayType) * flexArrayCount));

  if (function != nullptr)
    mERROR_CHECK(mSharedPointer_Create(pOutSharedPointer, reinterpret_cast<T *>(pData), (std::function<void(T *)>)[function](T *pDestructData) { function(static_cast<TInherited *>(pDestructData)); }, pAllocator));
  else
    mERROR_CHECK(mSharedPointer_Create<T>(pOutSharedPointer, reinterpret_cast<T *>(pData), nullptr, pAllocator));
  
  if (ppInherited != nullptr)
    *ppInherited = reinterpret_cast<TInherited *>(pData);

  pData = nullptr; // to not get released on destruction.

#ifdef mSHARED_POINTER_DEBUG_OUTPUT
  mLOG(" [was inherited from %s]\n", typeid(TInherited).name());
#endif

  mRETURN_SUCCESS();
}

template <typename T>
inline mFUNCTION(mSharedPointer_AllocateWithSize, OUT mSharedPointer<T> *pOutSharedPointer, IN mAllocator *pAllocator, const size_t size, const std::function<void(T *)> &function = nullptr)
{
  mFUNCTION_SETUP();

  mERROR_IF(pOutSharedPointer == nullptr, mR_ArgumentNull);

  uint8_t *pData = nullptr;
  mDEFER(mAllocator_FreePtr(pAllocator, &pData));
  mERROR_CHECK(mAllocator_AllocateZero(pAllocator, &pData, size));

  mERROR_CHECK(mSharedPointer_Create(pOutSharedPointer, reinterpret_cast<T *>(pData), function, pAllocator));

  pData = nullptr; // to not get released on destruction.

  mRETURN_SUCCESS();
}

template <typename T>
inline mFUNCTION(mSharedPointer_CreateInplace, IN_OUT mSharedPointer<T> *pOutSharedPointer, IN typename mSharedPointer<T>::PointerParams *pPointerParams, IN T *pData, IN mAllocator *pAllocator, const std::function<void(T *)> &cleanupFunction)
{
  mFUNCTION_SETUP();

  mERROR_IF(pOutSharedPointer == nullptr || pPointerParams == nullptr || pData == nullptr, mR_ArgumentNull);

  if (*pOutSharedPointer != nullptr)
    *pOutSharedPointer = nullptr;

  pOutSharedPointer->m_pParams = pPointerParams;

  new (pOutSharedPointer->m_pParams) typename mSharedPointer<T>::PointerParams();
  pOutSharedPointer->m_pParams->referenceCount = 1;
  pOutSharedPointer->m_pParams->pAllocator = pAllocator;
  pOutSharedPointer->m_pParams->freeResource = (pAllocator != mSHARED_POINTER_FOREIGN_RESOURCE);
  pOutSharedPointer->m_pParams->cleanupFunction = cleanupFunction;
  pOutSharedPointer->m_pParams->freeParameters = false;
  pOutSharedPointer->m_pParams->pUserData = nullptr;

  pOutSharedPointer->m_pData = pData;

#ifdef mSHARED_POINTER_DEBUG_OUTPUT
  mLOG("Created mSharedPointer<%s> in place. (0x%" PRIx64 ")\n", typeid(T).name(), (uint64_t)pData);
#endif

  mRETURN_SUCCESS();
}

template <typename T>
inline mFUNCTION(mSharedPointer_Destroy, IN_OUT mSharedPointer<T> *pPointer)
{
  mFUNCTION_SETUP();
  mERROR_IF(pPointer == nullptr, mR_ArgumentNull);

  *pPointer = nullptr;

  mRETURN_SUCCESS();
}

//////////////////////////////////////////////////////////////////////////

template<typename T>
inline mSharedPointer<T>::mSharedPointer()
  : m_pData(nullptr), m_pParams(nullptr)
{ }

template<typename T>
inline mSharedPointer<T>::mSharedPointer(nullptr_t)
  : m_pData(nullptr), m_pParams(nullptr)
{ }

template<typename T>
inline mSharedPointer<T>::mSharedPointer(const mSharedPointer<T> &copy)
  : m_pData(nullptr), m_pParams(nullptr)
{
  if (copy == nullptr)
    return;

  m_pData = copy.m_pData;
  m_pParams = copy.m_pParams;

  ++m_pParams->referenceCount;
}

template<typename T>
inline mSharedPointer<T>::mSharedPointer(mSharedPointer<T> &&move)
  : m_pData(nullptr), m_pParams(nullptr)
{
  if (move == nullptr)
    return;

  m_pData = move.m_pData;
  m_pParams = move.m_pParams;

  move.m_pData = nullptr;
  move.m_pParams = nullptr;

  move.~mSharedPointer();
}

template<typename T>
template <typename T2, typename std::enable_if<!std::is_same<T, T2>::value && (std::is_base_of<T2, T>::value || std::is_base_of<T, T2>::value)>*>
inline mSharedPointer<T>::mSharedPointer(const mSharedPointer<T2> &copy)
  : m_pData(nullptr), m_pParams(nullptr)
{
  if (copy == nullptr)
    return;

  m_pData = static_cast<T *>(copy.m_pData);
  m_pParams = reinterpret_cast<decltype(m_pParams)>(copy.m_pParams);

  ++m_pParams->referenceCount;
}

template<typename T>
template <typename T2, typename std::enable_if<!std::is_same<T, T2>::value && (std::is_base_of<T2, T>::value || std::is_base_of<T, T2>::value)>*>
inline mSharedPointer<T>::mSharedPointer(mSharedPointer<T2> &&move)
  : m_pData(nullptr), m_pParams(nullptr)
{
  if (move == nullptr)
    return;

  m_pData = static_cast<T *>(move.m_pData);
  m_pParams = reinterpret_cast<decltype(m_pParams)>(move.m_pParams);

  move.m_pData = nullptr;
  move.m_pParams = nullptr;

  move.~mSharedPointer<T2>();
}

template<typename T>
inline mSharedPointer<T>::~mSharedPointer()
{
  if (*this == nullptr)
  {
    m_pData = nullptr;
    m_pParams = nullptr;

    return;
  }

  size_t referenceCount = --m_pParams->referenceCount;

  if (referenceCount == 0)
  {
#ifdef mSHARED_POINTER_DEBUG_OUTPUT
    mLOG("Destroying mSharedPointer<%s>. (0x%" PRIx64 ")\n", typeid(T).name(), (uint64_t)m_pData);
#endif

    if (m_pParams->cleanupFunction)
      m_pParams->cleanupFunction(m_pData);

    if (m_pParams)
    {
      mAllocator *pAllocator = m_pParams->pAllocator;

      if (m_pParams->freeResource)
        mAllocator_FreePtr(m_pParams->pAllocator, &m_pData);

      m_pParams->~PointerParams();

      if (m_pParams->freeParameters)
        mAllocator_FreePtr(pAllocator, &m_pParams);
    }
  }

  m_pParams = nullptr;
  m_pData = nullptr;
}

template<typename T>
inline mSharedPointer<T>& mSharedPointer<T>::operator=(const mSharedPointer<T> &copy)
{
  this->~mSharedPointer<T>();

  if (copy == nullptr)
    return *this;

  m_pData = copy.m_pData;
  m_pParams = copy.m_pParams;

  ++m_pParams->referenceCount;

  return *this;
}

template<typename T>
inline mSharedPointer<T>& mSharedPointer<T>::operator=(mSharedPointer<T> &&move)
{
  if (move != nullptr && *this != nullptr && move.m_pData == m_pData)
  {
    move.~mSharedPointer<T>();
    return *this;
  }

  this->~mSharedPointer<T>();

  if (move == nullptr)
    return *this;

  m_pData = move.m_pData;
  m_pParams = move.m_pParams;

  move.m_pData = nullptr;
  move.m_pParams = nullptr;

  return *this;
}

template<typename T>
template <typename T2, typename>
inline mSharedPointer<T>& mSharedPointer<T>::operator=(const mSharedPointer<T2> &copy)
{
  this->~mSharedPointer<T>();

  if (copy == nullptr)
    return *this;

  m_pData = static_cast<T *>(copy.m_pData);
  m_pParams = reinterpret_cast<decltype(m_pParams)>(copy.m_pParams);

  ++m_pParams->referenceCount;

  return *this;
}

template<typename T>
template <typename T2, typename>
inline mSharedPointer<T>& mSharedPointer<T>::operator=(mSharedPointer<T2> &&move)
{
  if (move != nullptr && *this != nullptr && move.m_pData == m_pData)
  {
    move.~mSharedPointer<T2>();
    return *this;
  }

  this->~mSharedPointer<T>();

  if (move == nullptr)
    return *this;

  m_pData = static_cast<T *>(move.m_pData);
  m_pParams = reinterpret_cast<decltype(m_pParams)>(move.m_pParams);

  move.m_pData = nullptr;
  move.m_pParams = nullptr;

  return *this;
}

template<typename T>
inline T* mSharedPointer<T>::operator->()
{
  return m_pData;
}

template<typename T>
inline const T* mSharedPointer<T>::operator->() const
{
  return m_pData;
}

template<typename T>
inline T& mSharedPointer<T>::operator*()
{
  return *m_pData;
}

template<typename T>
inline const T& mSharedPointer<T>::operator*() const
{
  return *m_pData;
}

template<typename T>
inline bool mSharedPointer<T>::operator==(const mSharedPointer<T> &other) const
{
  if (*this == nullptr)
  {
    if (other == nullptr)
      return true;

    return false;
  }

  if (other == nullptr)
    return false;

  if (m_pData == other.m_pData)
    return true;

  return false;
}

template<typename T>
inline bool mSharedPointer<T>::operator==(nullptr_t) const
{
  if (m_pData == nullptr || m_pParams == nullptr)
    return true;

  return false;
}

template<typename T>
inline bool mSharedPointer<T>::operator!=(const mSharedPointer<T> &other) const
{
  return !(*this == other);
}

template<typename T>
inline bool mSharedPointer<T>::operator!=(nullptr_t) const
{
  return !(*this == nullptr);
}

template<typename T>
inline mSharedPointer<T>::operator bool() const
{
  return *this != nullptr;
}

template<typename T>
inline bool mSharedPointer<T>::operator!() const
{
  return *this == nullptr;
}

template<typename T>
template<typename T2, typename std::enable_if_t<!mIsSharedPointer<T2>::value && std::is_convertible<T2, bool>::value, int>* /* = nullptr */>
inline mSharedPointer<T>::operator T2() const
{
  mSTATIC_ASSERT((std::is_same<bool, T2>::value), "mSharedPointer cannot be cast to the given type.");

  return (bool)*this;
}

template<typename T>
inline const T* mSharedPointer<T>::GetPointer() const
{
  return m_pData;
}

template<typename T>
inline T* mSharedPointer<T>::GetPointer()
{
  return m_pData;
}

template<typename T>
inline size_t mSharedPointer<T>::GetReferenceCount() const
{
  if (*this == nullptr)
    return 0;

  return m_pParams->referenceCount;
}

template<typename T>
inline mFUNCTION(mDestruct, IN mPtr<T> *pData)
{
  mFUNCTION_SETUP();

  mERROR_CHECK(mSharedPointer_Destroy(pData));

  mRETURN_SUCCESS();
}

template <typename T>
struct mIsTriviallyMemoryMovable<mPtr<T>>
{
  static constexpr bool value = true;
};

template <typename T>
struct mIsTriviallyMemoryMovable<mReferencePack<T>>
{
  static constexpr bool value = false;
};

template <typename T>
struct mIsTriviallyMemoryMovable<mUniqueContainer<T>>
{
  static constexpr bool value = false;
};

#endif // mSharedPointer_h__
