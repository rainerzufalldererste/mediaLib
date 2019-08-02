#ifndef mDateTime_h__
#define mDateTime_h__

#include "mediaLib.h"

mFUNCTION(mGetCurrentDate, OUT mString *pDate);
mFUNCTION(mGetCurrentDate, OUT OPTIONAL size_t *pYear, OUT OPTIONAL size_t *pMonth, OUT OPTIONAL size_t *pDay, OUT OPTIONAL size_t *pDayOfWeek);
mFUNCTION(mGetCurrentTime, OUT mString *pTime);
mFUNCTION(mGetCurrentTime, OUT OPTIONAL size_t *pHours, OUT OPTIONAL size_t *pMinutes, OUT OPTIONAL size_t *pSeconds);
mFUNCTION(mGetCurrentDateTime, OUT mString *pDateTime);

#endif // mDateTime_h__
