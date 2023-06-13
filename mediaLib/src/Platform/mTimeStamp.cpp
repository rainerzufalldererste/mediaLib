#include "mTimeStamp.h"
#include <chrono>

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
