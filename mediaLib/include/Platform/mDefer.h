#ifndef mDefer_h__
#define mDefer_h__

#include <functional>
#include <type_traits>
#include "mResult.h"
#include "mediaLib.h"

#ifdef GIT_BUILD // Define __M_FILE__
  #ifdef __M_FILE__
    #undef __M_FILE__
  #endif
  #define __M_FILE__ "T/FEsFasx00sRo0SL4Yb/7f2aVPtzXbIJadH5wtOCbHRV3mBpnXvzlN089TV87GJwVBAE0hlj1ZbGZzy"
#endif

// Attention: Because of the specific use-case of this class, the copy constructor & copy assignment operator *move* instead of copying.

class mDefer final
{
private:
  std::function<void()> onExit;

public:
  inline mDefer() noexcept {};
  inline explicit mDefer(std::function<void()> &&onExit) noexcept : onExit(std::move(onExit)) {};

  inline mDefer(mDefer &copy) noexcept { onExit = std::move(copy.onExit); copy.onExit = nullptr; }
  inline mDefer(mDefer &&move) noexcept { onExit = std::move(move.onExit); move.onExit = nullptr; }

  inline ~mDefer() noexcept { if (onExit != nullptr) onExit(); };

  inline mDefer &operator = (mDefer &copy) noexcept { onExit = std::move(copy.onExit); copy.onExit = nullptr; return *this; }
  inline mDefer &operator = (mDefer &&move) noexcept { onExit = std::move(move.onExit); move.onExit = nullptr; return *this; }
};

class mDeferOnError final
{
private:
  std::function<void()> onExit;
  const mResult &result;

public:
  inline explicit mDeferOnError(std::function<void()> &&onExit, const mResult &result) noexcept : onExit(std::move(onExit)), result(result) {};
  inline ~mDeferOnError() noexcept { if (mFAILED(result) && onExit != nullptr) onExit(); };
};

class mDeferOnSuccess final
{
private:
  std::function<void()> onExit;
  const mResult &result;

public:
  inline explicit mDeferOnSuccess(std::function<void()> &&onExit, const mResult &result) noexcept : onExit(std::move(onExit)), result(result) {};
  inline ~mDeferOnSuccess() noexcept { if (mSUCCEEDED(result) && onExit != nullptr) onExit(); };
};

template <typename T>
class mDeferCall
{
protected:
  T value;

public:
  inline explicit mDeferCall(T value) noexcept : value(value) {};
};

template <typename T>
mDeferCall<T> mDeferCall_Create(T value)
{
  return mDeferCall<T>(value);
}

template <typename T>
class mDeferCallResult
{
protected:
  T value;
  const mResult &result;

public:
  inline explicit mDeferCallResult(T value, const mResult &result) noexcept : value(value), result(result) {};
};

template <typename T>
mDeferCallResult<T> mDeferCallResult_Create(T value, const mResult &result)
{
  return mDeferCallResult<T>(value, result);
}

template <typename T>
inline T mValueOf(T value)
{
  return value;
}

#ifdef __COUNTER__
  #define __mDEFER_CLASS_NAME__ mCONCAT_LITERALS(__defer_class__, __COUNTER__)
  #define __mDEFER_INSTANCE_NAME__ mCONCAT_LITERALS(__defer__, __COUNTER__)
#else
#define __mDEFER_CLASS_NAME__ mCONCAT_LITERALS(__defer_class__, __LINE__)
#define __mDEFER_INSTANCE_NAME__ mCONCAT_LITERALS(__defer__, __LINE__)
#endif

#define mDEFER(...) const auto __mDEFER_INSTANCE_NAME__ = mDefer([&](){ __VA_ARGS__; })
#define mDEFER_IF(conditional, ...) const auto __mDEFER_INSTANCE_NAME__ = mDefer([&](){ if (conditional) { __VA_ARGS__; }})
#define mDEFER_ON_ERROR(...) const auto __mDEFER_INSTANCE_NAME__ = mDeferOnError([&](){ __VA_ARGS__; }, (mSTDRESULT))
#define mDEFER_ON_SUCCESS(...) const auto __mDEFER_INSTANCE_NAME__ = mDeferOnSuccess([&](){ __VA_ARGS__; }, (mSTDRESULT))

#define mDEFER_NO_CAPTURE_INTERNAL(className, ...) class className final { public: inline ~ className () { __VA_ARGS__; }; } __mDEFER_INSTANCE_NAME__
#define mDEFER_NO_CAPTURE(...) mDEFER_NO_CAPTURE_INTERNAL(__mDEFER_CLASS_NAME__, __VA_ARGS__)

#define mDEFER_SINGLE_CAPTURE_INTERNAL(className, capture, ...) const class className final { private: decltype(capture) &capture; public: inline explicit className(decltype(capture) &c) : capture(c) {}; inline ~ className () { __VA_ARGS__; }; } __mDEFER_INSTANCE_NAME__ = className(capture)
#define mDEFER_SINGLE_CAPTURE(capture, ...) mDEFER_SINGLE_CAPTURE_INTERNAL(__mDEFER_CLASS_NAME__, capture, __VA_ARGS__)

#if !defined(_MSC_VER) || _MSC_VER > 1920
  #define mDEFER_CALL_INTERNAL(className, param, function) const class className final : public mDeferCall<decltype(mValueOf(param))> { public: inline className (mDeferCall<decltype(mValueOf(param))> v) : mDeferCall<decltype(mValueOf(param))>(v) {}; inline ~ className() { function(value); }; } __mDEFER_INSTANCE_NAME__ = mDeferCall_Create(param)
  #define mDEFER_CALL(param, function) mDEFER_CALL_INTERNAL(__mDEFER_CLASS_NAME__, param, function)
  
  #define mDEFER_CALL_ON_RESULT_INTERNAL(className, param, function, success) const class className final : public mDeferCallResult<decltype(mValueOf(param))> { public: inline className (mDeferCallResult<decltype(mValueOf(param))> v) : mDeferCallResult<decltype(mValueOf(param))>(v) {}; inline ~ className() { if (mSUCCEEDED(result) == (success)) function(value); }; } __mDEFER_INSTANCE_NAME__ = mDeferCallResult_Create(param, (mSTDRESULT))
  #define mDEFER_CALL_ON_ERROR(param, function) mDEFER_CALL_ON_RESULT_INTERNAL(__mDEFER_CLASS_NAME__, param, function, false)
  #define mDEFER_CALL_ON_SUCCESS(param, function) mDEFER_CALL_ON_RESULT_INTERNAL(__mDEFER_CLASS_NAME__, param, function, true)
#else
  #define mDEFER_CALL(param, function) const auto __mDEFER_INSTANCE_NAME__ = mDefer([&](){ function(param); });
  #define mDEFER_CALL_ON_ERROR(param, function) const auto __mDEFER_INSTANCE_NAME__ = mDeferOnError([&](){ function(param); }, (mSTDRESULT));
  #define mDEFER_CALL_ON_SUCCESS(param, function) const auto __mDEFER_INSTANCE_NAME__ = mDeferOnSuccess([&](){ function(param); }, (mSTDRESULT));
#endif

#endif // mDefer_h__
