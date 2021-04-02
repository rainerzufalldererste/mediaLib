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
  T ____value;

public:
  inline explicit mDeferCall(T value) noexcept : ____value(value) {};
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
  T ____value;
  const mResult &____result;

public:
  inline explicit mDeferCallResult(T value, const mResult &result) noexcept : ____value(value), ____result(result) {};
};

template <typename T>
mDeferCallResult<T> mDeferCallResult_Create(T value, const mResult &result)
{
  return mDeferCallResult<T>(value, result);
}

template <typename Tnull>
class mDeferCall0
{

};

template <typename T, typename U>
class mDeferCall_2
{
protected:
  T ____value0;
  U ____value1;

public:
  inline explicit mDeferCall_2(T value0, U value1) noexcept : ____value0(value0), ____value1(value1) {};
};

template <typename T, typename U>
mDeferCall_2<T, U> mDeferCall_2_Create(T value0, U value1)
{
  return mDeferCall_2<T, U>(value0, value1);
}

template <typename T, typename U, typename V>
class mDeferCall_3
{
protected:
  T ____value0;
  U ____value1;
  V ____value2;

public:
  inline explicit mDeferCall_3(T value0, U value1, V value2) noexcept : ____value0(value0), ____value1(value1), ____value2(value2) {};
};

template <typename T, typename U, typename V>
mDeferCall_3<T, U, V> mDeferCall_3_Create(T value0, U value1, V value2)
{
  return mDeferCall_3<T, U, V>(value0, value1, value2);
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
  #define mDEFER_CALL_INTERNAL(className, param, function) const class className final : public mDeferCall<decltype(mValueOf(param))> { public: inline className (mDeferCall<decltype(mValueOf(param))> v) : mDeferCall<decltype(mValueOf(param))>(v) {}; inline ~ className() { function(____value); }; } __mDEFER_INSTANCE_NAME__ = mDeferCall_Create(param)
  #define mDEFER_CALL(param, function) mDEFER_CALL_INTERNAL(__mDEFER_CLASS_NAME__, param, function)
  
  #define mDEFER_CALL_ON_RESULT_INTERNAL(className, param, function, success) const class className final : public mDeferCallResult<decltype(mValueOf(param))> { public: inline className (mDeferCallResult<decltype(mValueOf(param))> v) : mDeferCallResult<decltype(mValueOf(param))>(v) {}; inline ~ className() { if (mSUCCEEDED(____result) == (success)) function(____value); }; } __mDEFER_INSTANCE_NAME__ = mDeferCallResult_Create(param, (mSTDRESULT))
  #define mDEFER_CALL_ON_ERROR(param, function) mDEFER_CALL_ON_RESULT_INTERNAL(__mDEFER_CLASS_NAME__, param, function, false)
  #define mDEFER_CALL_ON_SUCCESS(param, function) mDEFER_CALL_ON_RESULT_INTERNAL(__mDEFER_CLASS_NAME__, param, function, true)

#define mDEFER_CALL_0_INTERNAL(className, function) const class className final : public mDeferCall0<int> { public: inline ~ className() { function(); }; } __mDEFER_INSTANCE_NAME__;
#define mDEFER_CALL_0(function) mDEFER_CALL_0_INTERNAL(__mDEFER_CLASS_NAME__, function)

#define mDEFER_CALL_2_INTERNAL(className, function, param0, param1) const class className final : public mDeferCall_2<decltype(mValueOf(param0)), decltype(mValueOf(param1))> { public: inline className (mDeferCall_2<decltype(mValueOf(param0)), decltype(mValueOf(param1))> v) : mDeferCall_2<decltype(mValueOf(param0)), decltype(mValueOf(param1))>(v) {}; inline ~ className() { function(____value0, ____value1); }; } __mDEFER_INSTANCE_NAME__ = mDeferCall_2_Create(param0, param1)
#define mDEFER_CALL_2(function, param0, param1) mDEFER_CALL_2_INTERNAL(__mDEFER_CLASS_NAME__, function, param0, param1)

#define mDEFER_CALL_3_INTERNAL(className, function, param0, param1, param2) const class className final : public mDeferCall_3<decltype(mValueOf(param0)), decltype(mValueOf(param1)), decltype(mValueOf(param2))> { public: inline className (mDeferCall_3<decltype(mValueOf(param0)), decltype(mValueOf(param1)), decltype(mValueOf(param2))> v) : mDeferCall_3<decltype(mValueOf(param0)), decltype(mValueOf(param1)), decltype(mValueOf(param2))>(v) {}; inline ~ className() { function(____value0, ____value1, ____value2); }; } __mDEFER_INSTANCE_NAME__ = mDeferCall_3_Create(param0, param1, param2)
#define mDEFER_CALL_3(function, param0, param1, param2) mDEFER_CALL_3_INTERNAL(__mDEFER_CLASS_NAME__, function, param0, param1, param2)
#else
  #define mDEFER_CALL(param, function) const auto __mDEFER_INSTANCE_NAME__ = mDefer([&](){ function(param); });
  #define mDEFER_CALL_ON_ERROR(param, function) const auto __mDEFER_INSTANCE_NAME__ = mDeferOnError([&](){ function(param); }, (mSTDRESULT));
  #define mDEFER_CALL_ON_SUCCESS(param, function) const auto __mDEFER_INSTANCE_NAME__ = mDeferOnSuccess([&](){ function(param); }, (mSTDRESULT));
  #define mDEFER_CALL_0(function) const auto __mDEFER_INSTANCE_NAME__ = mDefer([&](){ function(); });
  #define mDEFER_CALL_2(function, param0, param1) const auto __mDEFER_INSTANCE_NAME__ = mDefer([&](){ function(param0, param1); });
  #define mDEFER_CALL_3(function, param0, param1, param2) const auto __mDEFER_INSTANCE_NAME__ = mDefer([&](){ function(param0, param1, param2); });
#endif

#endif // mDefer_h__
