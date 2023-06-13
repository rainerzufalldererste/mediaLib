#include "mTimeStamp.h"
#include <chrono>

#ifdef GIT_BUILD // Define __M_FILE__
  #ifdef __M_FILE__
    #undef __M_FILE__
  #endif
  #define __M_FILE__ "ssy82LrbZq+W1502ChhNtBRJuleY4x6yilFpmyTEJT5DknwgCEmCGKGp0L8cFWRkOtRW+dMFN0NpD7l2"
#endif

std::chrono::time_point<std::chrono::steady_clock> _mTimePoint_StartTime = std::chrono::high_resolution_clock::now();

mFUNCTION(mTimeStamp_Now, OUT mTimeStamp *pTimeStamp)
{
  mFUNCTION_SETUP();

  mERROR_IF(pTimeStamp == nullptr, mR_ArgumentNull);

  pTimeStamp->timePoint = (std::chrono::duration<double_t, std::micro>(std::chrono::high_resolution_clock::now() - _mTimePoint_StartTime)).count() / (1000.0 * 1000.0);

  mRETURN_SUCCESS();
}

mFUNCTION(mTimeStamp_FromSeconds, OUT mTimeStamp *pTimeStamp, const double_t seconds)
{
  mFUNCTION_SETUP();

  mERROR_IF(pTimeStamp == nullptr, mR_ArgumentNull);

  pTimeStamp->timePoint = std::chrono::duration<double_t, std::micro>(seconds * 1000.0 * 1000.0).count() / (1000.0 * 1000.0);

  mRETURN_SUCCESS();
}

mTimeStamp mTimeStamp::operator+(const mTimeStamp t) const
{
  return { timePoint + t.timePoint };
}

mTimeStamp mTimeStamp::operator-(const mTimeStamp t) const
{
  return { timePoint - t.timePoint };
}

mTimeStamp & mTimeStamp::operator+=(const mTimeStamp t)
{
  timePoint += t.timePoint;

  return *this;
}

mTimeStamp & mTimeStamp::operator-=(const mTimeStamp t)
{
  timePoint -= t.timePoint;

  return *this;
}

bool mTimeStamp::operator<(const mTimeStamp t) const
{
  return timePoint < t.timePoint;
}

bool mTimeStamp::operator<=(const mTimeStamp t) const
{
  return timePoint <= t.timePoint;
}

bool mTimeStamp::operator>(const mTimeStamp t) const
{
  return timePoint > t.timePoint;
}

bool mTimeStamp::operator>=(const mTimeStamp t) const
{
  return timePoint >= t.timePoint;
}

bool mTimeStamp::operator==(const mTimeStamp t) const
{
  return timePoint == t.timePoint;
}

bool mTimeStamp::operator!=(const mTimeStamp t) const
{
  return timePoint != t.timePoint;
}
