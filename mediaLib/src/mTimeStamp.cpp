// Copyright 2018 Christoph Stiller
// 
// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files(the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions :
// 
// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
// 
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "mTimeStamp.h"
#include <chrono>

std::chrono::time_point<std::chrono::steady_clock> _mTimePoint_StartTime = std::chrono::high_resolution_clock::now();

mFUNCTION(mTimeStamp_Now, OUT mTimeStamp *pTimeStamp)
{
  mFUNCTION_SETUP();

  mERROR_IF(pTimeStamp == nullptr, mR_ArgumentNull);

  pTimeStamp->timePoint = (std::chrono::duration<double_t, std::micro>(std::chrono::high_resolution_clock::now() - _mTimePoint_StartTime)).count() / 1000.0 * 1000.0;

  mRETURN_SUCCESS();
}

mFUNCTION(mTimeStamp_FromSeconds, OUT mTimeStamp *pTimeStamp, const double_t seconds)
{
  mFUNCTION_SETUP();

  mERROR_IF(pTimeStamp == nullptr, mR_ArgumentNull);

  pTimeStamp->timePoint = std::chrono::duration<double_t, std::micro>(seconds * 1000.0 * 1000.0).count() / 1000.0 * 1000.0;

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
