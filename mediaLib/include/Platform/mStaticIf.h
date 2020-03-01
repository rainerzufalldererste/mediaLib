// See: https://baptiste-wicht.com/posts/2015/07/simulate-static_if-with-c11c14.html

#ifndef mStaticIf_h__
#define mStaticIf_h__

#ifdef GIT_BUILD // Define __M_FILE__
  #ifdef __M_FILE__
    #undef __M_FILE__
  #endif
  #define __M_FILE__ "rRgedX5yIhMrCMBGJQKVIk+bNpsqwY/VN3Zl8Fa15AtCyYB9EBQCLGniofw8xEx2dsLUUQZsLCQspLKJ"
#endif

struct mStaticIfIdentity 
{
  template<typename T>
  inline T operator()(T &&x) const 
  {
    return std::forward<T>(x);
  }
};

template<bool conditional>
struct mStaticIfStatement 
{
  template<typename TFunction>
  inline void Then(const TFunction &f)
  {
    f(mStaticIfIdentity());
  }

  template<typename TFunction>
  inline void Else(const TFunction&)
  {
  }
};

template<>
struct mStaticIfStatement<false> {
  template<typename TFunction>
  inline void Then(const TFunction&)
  {
  }

  template<typename TFunction>
  inline void Else(const TFunction &f)
  {
    f(mStaticIfIdentity());
  }
};

template<bool conditional, typename TFunction>
inline mStaticIfStatement<conditional> mStaticIf(const TFunction &f)
{
  mStaticIfStatement<conditional> if_;
  if_.Then(f);

  return if_;
}

#endif // mStaticIf_h__
