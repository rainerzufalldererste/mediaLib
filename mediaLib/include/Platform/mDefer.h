#ifndef mDefer_h__
#define mDefer_h__

#include <functional>
#include <type_traits>
#include "mResult.h"
#include "mediaLib.h"

// Attention: Because of the specific use-case of this class, the copy constructor & copy assignment operator *move* instead of copying.
template <typename T>
class mDefer
{
public:
  typedef void OnExitFuncVoid();
  typedef void OnExitFuncT(T);
  typedef mResult OnExitFuncResultVoid();
  typedef mResult OnExitFuncResultT(T);

  mDefer();
  mDefer(const std::function<void()> &onExit, const mResult *pResult = nullptr);
  mDefer(std::function<void()> &&onExit, const mResult *pResult = nullptr);
  mDefer(const std::function<void(T)> &onExit, T data, const mResult *pResult = nullptr);
  mDefer(std::function<void(T)> &&onExit, T data, const mResult *pResult = nullptr);
  mDefer(OnExitFuncVoid *pOnExit, const mResult *pResult = nullptr);
  mDefer(OnExitFuncT *pOnExit, T data, const mResult *pResult = nullptr);
  mDefer(OnExitFuncResultVoid *pOnExit, const mResult *pResult = nullptr);
  mDefer(OnExitFuncResultT *pOnExit, T data, const mResult *pResult = nullptr);

  mDefer(mDefer<T> &copy);
  mDefer(mDefer<T> &&move);

  ~mDefer();

  mDefer<T>& operator = (mDefer<T> &copy);
  mDefer<T>& operator = (mDefer<T> &&move);


private:
  enum mDeferType
  {
    mDT_None,
    mDT_LambdaVoid,
    mDT_FuctionPointerVoid,
    mDT_FuctionPointerResultVoid,
    mDT_LambdaTPtr,
    mDT_FuctionPointerTPtr,
    mDT_FuctionPointerResultTPtr,
  };

  mDeferType m_deferType;
  T m_data;
  const mResult *m_pResult;

  std::function<void()> m_onExitLV;
  std::function<void(T)> m_onExitLP;

  union
  {
    OnExitFuncVoid *m_pOnExitFV;
    OnExitFuncT *m_pOnExitFP;
    OnExitFuncResultVoid *m_pOnExitFRV;
    OnExitFuncResultT *m_pOnExitFRP;
  };
};

mDefer<size_t> mDefer_Create(const std::function<void()> &onExit, const mResult *pResult = nullptr);
mDefer<size_t> mDefer_Create(std::function<void()> &&onExit, const mResult *pResult = nullptr);

template <typename T>
mDefer<T> mDefer_Create(const std::function<void(T)> &onExit, T data, const mResult *pResult = nullptr);

template <typename T>
mDefer<T> mDefer_Create(std::function<void(T)> &&onExit, T data, const mResult *pResult = nullptr);

mDefer<size_t> mDefer_Create(mDefer<size_t>::OnExitFuncVoid *pOnExit, const mResult *pResult = nullptr);

template <typename T>
mDefer<T> mDefer_Create(typename mDefer<T>::OnExitFuncT *pOnExit, T data, const mResult *pResult = nullptr);

mDefer<size_t> mDefer_Create(mDefer<size_t>::OnExitFuncResultVoid *pOnExit, const mResult *pResult = nullptr);

template <typename T>
mDefer<T> mDefer_Create(typename mDefer<T>::OnExitFuncResultT *pOnExit, T data, const mResult *pResult = nullptr);

template<typename T>
inline mDefer<T>::mDefer()
{
  m_deferType = mDeferType::mDT_None;
  m_data = T();
  m_pResult = nullptr;
}

template<typename T>
inline mDefer<T>::mDefer(const std::function<void()> &onExit, const mResult *pResult /* = nullptr */)
{
  m_deferType = mDeferType::mDT_LambdaVoid;
  m_data = T();
  m_onExitLV = onExit;
  m_pResult = pResult;
}

template<typename T>
inline mDefer<T>::mDefer(std::function<void()> &&onExit, const mResult *pResult /* = nullptr */)
{
  m_deferType = mDeferType::mDT_LambdaVoid;
  m_data = T();
  m_onExitLV = std::move(onExit);
  m_pResult = pResult;
}

template<typename T>
inline mDefer<T>::mDefer(const std::function<void(T)> &onExit, T data, const mResult *pResult /* = nullptr */)
{
  m_deferType = mDeferType::mDT_LambdaTPtr;
  m_data = data;
  m_onExitLP = onExit;
  m_pResult = pResult;
}

template<typename T>
inline mDefer<T>::mDefer(std::function<void(T)> &&onExit, T data, const mResult *pResult /* = nullptr */)
{
  m_deferType = mDeferType::mDT_LambdaTPtr;
  m_data = data;
  m_onExitLP = std::move(onExit);
  m_pResult = pResult;
}

template<typename T>
inline mDefer<T>::mDefer(OnExitFuncVoid *pOnExit, const mResult *pResult /* = nullptr */)
{
  m_deferType = mDeferType::mDT_FuctionPointerVoid;
  m_data = T();
  m_pOnExitFV = pOnExit;
  m_pResult = pResult;
}

template<typename T>
inline mDefer<T>::mDefer(OnExitFuncT *pOnExit, T data, const mResult *pResult /* = nullptr */)
{
  m_deferType = mDeferType::mDT_FuctionPointerTPtr;
  m_data = data;
  m_pOnExitFP = pOnExit;
  m_pResult = pResult;
}

template<typename T>
inline mDefer<T>::mDefer(OnExitFuncResultVoid *pOnExit, const mResult *pResult /* = nullptr */)
{
  m_deferType = mDeferType::mDT_FuctionPointerResultVoid;
  m_data = T();
  m_pOnExitFRV = pOnExit;
  m_pResult = pResult;
}

template<typename T>
inline mDefer<T>::mDefer(OnExitFuncResultT *pOnExit, T data, const mResult *pResult /* = nullptr */)
{
  m_deferType = mDeferType::mDT_FuctionPointerResultTPtr;
  m_data = data;
  m_pOnExitFRP = pOnExit;
  m_pResult = pResult;
}

// Attention: Because of the specific use-case of this class, the copy constructor & copy assignment operator *move* instead of copying.
template<typename T>
inline mDefer<T>::mDefer(mDefer<T> &copy) :
  m_deferType(copy.m_deferType),
  m_data(copy.m_data),
  m_pResult(copy.m_pResult),
  m_onExitLP(std::move(copy.m_onExitLP)),
  m_onExitLV(std::move(copy.m_onExitLV)),
  m_pOnExitFP(copy.m_pOnExitFP),
  m_pReferenceCount(copy.m_pReferenceCount),
  m_pAllocator(copy.m_pAllocator)
{
  move.m_deferType = mDT_None;
  move.m_pResult = nullptr;
  move.m_pData = nullptr;
  move.m_onExitLP = nullptr;
  move.m_onExitLV = nullptr;
  move.m_pOnExitFP = nullptr;
}

template<typename T>
inline mDefer<T>::mDefer(mDefer<T> &&move) :
  m_deferType(move.m_deferType),
  m_data(move.m_data),
  m_pResult(move.m_pResult),
  m_onExitLP(std::move(move.m_onExitLP)),
  m_onExitLV(std::move(move.m_onExitLV)),
  m_pOnExitFP(move.m_pOnExitFP),
  m_pReferenceCount(move.m_pReferenceCount),
  m_pAllocator(move.m_pAllocator)
{
  move.m_deferType = mDT_None;
  move.m_pResult = nullptr;
  move.m_data = nullptr;
  move.m_onExitLP = nullptr;
  move.m_onExitLV = nullptr;
  move.m_pOnExitFP = nullptr;
}

template<typename T>
inline mDefer<T>::~mDefer()
{
  if (m_pResult != nullptr)
    if (mSUCCEEDED(*m_pResult))
      return;

  switch (m_deferType)
  {
  case mDefer::mDeferType::mDT_LambdaVoid:
    if (m_onExitLV)
      m_onExitLV();
    break;

  case mDefer::mDeferType::mDT_LambdaTPtr:
    if (m_onExitLP)
      m_onExitLP(m_data);
    break;

  case mDefer::mDeferType::mDT_FuctionPointerVoid:
    if (m_pOnExitFV)
      (*m_pOnExitFV)();
    break;

  case mDefer::mDeferType::mDT_FuctionPointerTPtr:
    if (m_pOnExitFP)
      (*m_pOnExitFP)(m_data);
    break;

  case mDefer::mDeferType::mDT_FuctionPointerResultVoid:
    if (m_pOnExitFRV)
      (*m_pOnExitFRV)();
    break;

  case mDefer::mDeferType::mDT_FuctionPointerResultTPtr:
    if (m_pOnExitFRP)
      (*m_pOnExitFRP)(m_data);
    break;

  case mDefer::mDeferType::mDT_None:
  default:
    break;
  }
}

// Attention: Because of the specific use-case of this class, the copy constructor & copy assignment operator *move* instead of copying.
template<typename T>
inline mDefer<T>& mDefer<T>::operator=(mDefer<T> &copy)
{
  m_deferType = move.m_deferType;
  m_data = move.m_pData;
  m_pResult = move.m_pResult;

  m_onExitLV = std::move(move.m_onExitLV);
  m_onExitLP = std::move(move.m_pOnExitFP);

  // assign union:
  m_pOnExitFP = move.m_pOnExitFP;

  move.m_deferType = mDT_None;
  move.m_pResult = nullptr;
  move.m_pData = nullptr;
  move.m_onExitLP = nullptr;
  move.m_onExitLV = nullptr;
  move.m_pOnExitFP = nullptr;

  return *this;
}

template<typename T>
inline mDefer<T>& mDefer<T>::operator = (mDefer<T> &&move)
{
  m_deferType = move.m_deferType;
  m_data = move.m_data;
  m_pResult = move.m_pResult;

  m_onExitLV = std::move(move.m_onExitLV);
  m_onExitLP = std::move(move.m_pOnExitFP);

  // assign union:
  m_pOnExitFP = move.m_pOnExitFP;

  move.m_deferType = mDT_None;
  move.m_pResult = nullptr;
  move.m_data = nullptr;
  move.m_onExitLP = nullptr;
  move.m_onExitLV = nullptr;
  move.m_pOnExitFP = nullptr;

  return *this;
}

template<typename T>
inline mDefer<T> mDefer_Create(const std::function<void(T)> &onExit, T data, const mResult *pResult /* = nullptr */)
{
  return mDefer<T>(onExit, data, pResult);
}

template<typename T>
inline mDefer<T> mDefer_Create(std::function<void(T)> &&onExit, T data, const mResult *pResult /* = nullptr */)
{
  return mDefer<T>(std::forward<std::function<void(T)>>(onExit), data, pResult);
}

template<typename T>
inline mDefer<T> mDefer_Create(typename mDefer<T>::OnExitFuncT *pOnExit, T data, const mResult *pResult /* = nullptr */)
{
  return mDefer<T>(pOnExit, data, pResult);
}

template<typename T>
inline mDefer<T> mDefer_Create(typename mDefer<T>::OnExitFuncResultT *pOnExit, T data, const mResult *pResult /* = nullptr */)
{
  return mDefer<T>(pOnExit, data, pResult);
}

inline mDefer<size_t> mDefer_Create(const std::function<void()> &onExit, const mResult *pResult /* = nullptr */)
{
  return mDefer<size_t>(onExit, pResult);
}

inline mDefer<size_t> mDefer_Create(std::function<void()> &&onExit, const mResult *pResult /* = nullptr */)
{
  return mDefer<size_t>(std::forward<std::function<void()>>(onExit), pResult);
}

inline mDefer<size_t> mDefer_Create(mDefer<size_t>::OnExitFuncVoid *pOnExit, const mResult *pResult /* = nullptr */)
{
  return mDefer<size_t>(pOnExit, pResult);
}

inline mDefer<size_t> mDefer_Create(mDefer<size_t>::OnExitFuncResultVoid *pOnExit, const mResult *pResult /* = nullptr */)
{
  return mDefer<size_t>(pOnExit, pResult);
}

#define mCOMBINE_LITERALS(x, y) x ## y
#define mCOMBINE_LITERALS_INDIRECTION(x, y) mCOMBINE_LITERALS(x, y)

#ifdef __COUNTER__
#define mDEFER(...) const auto mCOMBINE_LITERALS_INDIRECTION(__defer__, __COUNTER__) = mDefer_Create([&](){ __VA_ARGS__; })
#define mDEFER_IF(conditional, ...) const auto mCOMBINE_LITERALS_INDIRECTION(__defer__, __COUNTER__) = mDefer_Create([&](){ if (conditional) { __VA_ARGS__; } })
#define mDEFER_ON_ERROR(...) const auto mCOMBINE_LITERALS_INDIRECTION(__defer__, __COUNTER__) = mDefer_Create([&](){ { __VA_ARGS__; } }, &(mSTDRESULT))
#define mDEFER_CALL(Ressource, DestructionFunction) const auto mCOMBINE_LITERALS_INDIRECTION(__defer__, __COUNTER__) = mDefer_Create((DestructionFunction), (Ressource))
#define mDEFER_CALL_ON_ERROR(Ressource, DestructionFunction) const auto mCOMBINE_LITERALS_INDIRECTION(__defer__, __COUNTER__) = mDefer_Create((DestructionFunction), (Ressource), &(mSTDRESULT))
#else
#define mDEFER(...) const auto mCOMBINE_LITERALS_INDIRECTION(__defer__, __LINE__) = mDefer_Create([&](){ __VA_ARGS__; })
#define mDEFER_IF(conditional, ...) const auto mCOMBINE_LITERALS_INDIRECTION(__defer__, __LINE__) = mDefer_Create([&](){ if (conditional) { __VA_ARGS__; } })
#define mDEFER_ON_ERROR(...) const auto mCOMBINE_LITERALS_INDIRECTION(__defer__, __LINE__) = mDefer_Create([&](){ { __VA_ARGS__; } }, &(mSTDRESULT))
#define mDEFER_CALL(Ressource, DestructionFunction) const auto mCOMBINE_LITERALS_INDIRECTION(__defer__, __LINE__) = mDefer_Create((DestructionFunction), (Ressource))
#define mDEFER_CALL_ON_ERROR(Ressource, DestructionFunction) const auto mCOMBINE_LITERALS_INDIRECTION(__defer__, __LINE__) = mDefer_Create((DestructionFunction), (Ressource), &(mSTDRESULT))
#endif

#endif // mDefer_h__
