#include "mDateTime.h"

#ifdef GIT_BUILD // Define __M_FILE__
  #ifdef __M_FILE__
    #undef __M_FILE__
  #endif
  #define __M_FILE__ "pbhOhr3p9kG0Qh9tcYH0rBm8oYGgufSpLTGabRKaOTWFx9ivUMGPQLbjSdfs+PkJi/se4/4sFNs2rLZY"
#endif

mFUNCTION(mGetCurrentDate, OUT mString *pDate)
{
  mFUNCTION_SETUP();

  mERROR_IF(pDate == nullptr, mR_ArgumentNull);

  SYSTEMTIME systemtime;
  GetSystemTime(&systemtime);

  wchar_t localDate[0xFF];
  localDate[0] = 0;

  mERROR_IF(0 == GetDateFormatW(LOCALE_USER_DEFAULT, DATE_SHORTDATE, &systemtime, nullptr, localDate, mARRAYSIZE(localDate)), mR_InternalError);
  mERROR_CHECK(mString_Create(pDate, localDate, mARRAYSIZE(localDate), pDate->pAllocator));

  mRETURN_SUCCESS();
}

mFUNCTION(mGetCurrentDate, OUT OPTIONAL size_t *pYear, OUT OPTIONAL size_t *pMonth, OUT OPTIONAL size_t *pDay, OUT OPTIONAL size_t *pDayOfWeek)
{
  mFUNCTION_SETUP();

  SYSTEMTIME systemtime;
  GetSystemTime(&systemtime);

  if (pYear != nullptr)
    *pYear = systemtime.wYear;

  if (pMonth != nullptr)
    *pMonth = systemtime.wMonth;

  if (pDay != nullptr)
    *pDay = systemtime.wDay;

  if (pDayOfWeek != nullptr)
    *pDayOfWeek = systemtime.wDayOfWeek;

  mRETURN_SUCCESS();
}

mFUNCTION(mGetCurrentTime, OUT mString *pTime)
{
  mFUNCTION_SETUP();

  mERROR_IF(pTime == nullptr, mR_ArgumentNull);

  SYSTEMTIME systemtime;
  GetSystemTime(&systemtime);

  wchar_t localTime[0xFF];
  localTime[0] = 0;

  mERROR_IF(0 == GetTimeFormatW(LOCALE_USER_DEFAULT, TIME_NOSECONDS, &systemtime, nullptr, localTime, mARRAYSIZE(localTime)), mR_InternalError);
  mERROR_CHECK(mString_Create(pTime, localTime, mARRAYSIZE(localTime), pTime->pAllocator));

  mRETURN_SUCCESS();
}

mFUNCTION(mGetCurrentTime, OUT OPTIONAL size_t *pHours, OUT OPTIONAL size_t *pMinutes, OUT OPTIONAL size_t *pSeconds)
{
  mFUNCTION_SETUP();

  SYSTEMTIME systemtime;
  GetSystemTime(&systemtime);

  if (pHours != nullptr)
    *pHours = systemtime.wHour;

  if (pMinutes != nullptr)
    *pMinutes = systemtime.wMinute;

  if (pSeconds != nullptr)
    *pSeconds = systemtime.wSecond;

  mRETURN_SUCCESS();
}

mFUNCTION(mGetCurrentDateTime, OUT mString *pDateTime)
{
  mFUNCTION_SETUP();

  mERROR_IF(pDateTime == nullptr, mR_ArgumentNull);

  SYSTEMTIME systemtime;
  GetSystemTime(&systemtime);

  wchar_t localDateTime[1024];
  localDateTime[0] = 0;

  size_t chars = 0;

  mERROR_IF(0 == (chars = GetDateFormatW(LOCALE_USER_DEFAULT, DATE_SHORTDATE, &systemtime, nullptr, localDateTime, mARRAYSIZE(localDateTime))), mR_InternalError);
  localDateTime[chars - 1] = ' ';
  localDateTime[chars] = '\0';
  mERROR_IF(0 == GetTimeFormatW(LOCALE_USER_DEFAULT, TIME_NOSECONDS, &systemtime, nullptr, localDateTime + chars, (int32_t)(mARRAYSIZE(localDateTime) - chars)), mR_InternalError);

  mERROR_CHECK(mString_Create(pDateTime, localDateTime, mARRAYSIZE(localDateTime), pDateTime->pAllocator));

  mRETURN_SUCCESS();
}
