#include "mProfiler.h"

#include "mJson.h"

#ifdef GIT_BUILD // Define __M_FILE__
  #ifdef __M_FILE__
    #undef __M_FILE__
  #endif
  #define __M_FILE__ "noBuHXvNSeI9Urfcw6POy6WbXJvXxN5zLhNjkiuKGcInIquYzdGVDaXF9Y33pKYU36IuPeUBIShreU/s"
#endif

mProfiler _GlobalProfiler = {};

//////////////////////////////////////////////////////////////////////////

mFUNCTION(mProfiler_Allocate, const size_t eventCount)
{
  mFUNCTION_SETUP();

  mERROR_IF((eventCount & (eventCount - 1)), mR_InvalidParameter); // Parameter is not a power of two.
  mERROR_IF(_GlobalProfiler.enabled, mR_ResourceStateInvalid);

  mProfiler_Record *pRecord = nullptr;
  mERROR_CHECK(mAlloc(&pRecord, eventCount));

  _GlobalProfiler.count = eventCount;
  _GlobalProfiler.pRecords = pRecord; // THIS IS AN INTENTIONAL MEMORY LEAK! (IN CASE ANOTHER THREAD IS STILL WRITING TO THE OLD POINTER)

  MemoryBarrier();

  mRETURN_SUCCESS();
}

mFUNCTION(mProfiler_Enable)
{
  mFUNCTION_SETUP();

  mERROR_IF(_GlobalProfiler.count == 0, mR_NotInitialized);

  _InterlockedExchange64(&_GlobalProfiler.enabled, 1);

  mRETURN_SUCCESS();
}

mFUNCTION(mProfiler_Disable)
{
  _InterlockedExchange64(&_GlobalProfiler.enabled, 0);

  return mR_Success;
}

mFUNCTION(mProfiler_Reset)
{
  mFUNCTION_SETUP();

  mERROR_IF(!_GlobalProfiler.enabled, mR_ResourceStateInvalid);

  _InterlockedExchange64(&_GlobalProfiler.index, 0);

  mRETURN_SUCCESS();
}

mFUNCTION(mProfiler_WriteToChromiumTraceEventFile, const mString &filename)
{
  mFUNCTION_SETUP();

  mERROR_IF(filename.hasFailed || filename.bytes < 2, mR_InvalidParameter);
  mERROR_IF(_GlobalProfiler.count == 0 || _GlobalProfiler.pRecords == nullptr, mR_NotInitialized);

  mPtr<mJsonWriter> jsonWriter;
  mDEFER_CALL(&jsonWriter, mJsonWriter_Destroy);
  mERROR_CHECK(mJsonWriter_Create(&jsonWriter, &mDefaultTempAllocator));

  const bool wasEnabled = _GlobalProfiler.enabled != 0;

  mDEFER(
    if (wasEnabled)
      mProfiler_Enable();
  );

  if (wasEnabled)
    mERROR_CHECK(mProfiler_Disable());

  const size_t max = mMin(mMax(_GlobalProfiler.index - 1, (int64_t)0), (int64_t)_GlobalProfiler.count);

  // Add Trace Events.
  {
    mERROR_CHECK(mJsonWriter_BeginArray(jsonWriter, "traceEvents"));

    const double_t pid = (double_t)GetCurrentProcessId();
    size_t firstTimePoint = (size_t)-1;

    for (size_t i = 0; i < max; i++)
      if (_GlobalProfiler.pRecords[i].eventTime < firstTimePoint)
        firstTimePoint = _GlobalProfiler.pRecords[i].eventTime;

    for (size_t i = 0; i < max; i++)
    {
      bool found = false;

      if (!_GlobalProfiler.pRecords[i].eventStart)
      {
        for (size_t j = 0; j < max; j++)
        {
          if (i != j && _GlobalProfiler.pRecords[j].eventStart && _GlobalProfiler.pRecords[i].eventTime > _GlobalProfiler.pRecords[j].eventTime && strncmp(_GlobalProfiler.pRecords[i].name, _GlobalProfiler.pRecords[j].name, sizeof(mProfiler_Record::name)) == 0)
          {
            found = true;
            break;
          }
        }
      }
      else
      {
        for (size_t j = 0; j < max; j++)
        {
          if (i != j && !_GlobalProfiler.pRecords[j].eventStart && _GlobalProfiler.pRecords[i].eventTime < _GlobalProfiler.pRecords[j].eventTime && strncmp(_GlobalProfiler.pRecords[i].name, _GlobalProfiler.pRecords[j].name, sizeof(mProfiler_Record::name)) == 0)
          {
            found = true;
            break;
          }
        }
      }

      if (!found)
        continue;

      mERROR_CHECK(mJsonWriter_BeginUnnamed(jsonWriter));
      mERROR_CHECK(mJsonWriter_AddValue(jsonWriter, "name", _GlobalProfiler.pRecords[i].name));
      mERROR_CHECK(mJsonWriter_AddValue(jsonWriter, "pid", pid));
      mERROR_CHECK(mJsonWriter_AddValue(jsonWriter, "tid", (double_t)_GlobalProfiler.pRecords[i].asyncOrThreadIndex));
      mERROR_CHECK(mJsonWriter_AddValue(jsonWriter, "ph", _GlobalProfiler.pRecords[i].eventStart ? "B" : "E"));
      mERROR_CHECK(mJsonWriter_AddValue(jsonWriter, "ts", (double_t)(_GlobalProfiler.pRecords[i].eventTime - firstTimePoint) * 1e-3));
      mERROR_CHECK(mJsonWriter_EndUnnamed(jsonWriter));
    }

    mERROR_CHECK(mJsonWriter_EndArray(jsonWriter));
  }

  mERROR_CHECK(mJsonWriter_AddValue(jsonWriter, "displayTimeUnit", "ms"));
  mERROR_CHECK(mJsonWriter_ToFile(jsonWriter, filename));

  mRETURN_SUCCESS();
}

//////////////////////////////////////////////////////////////////////////

void mProfiler_ReportEventStart_NoInline(const char *name, const uint32_t asyncOrThreadIndex /* = GetCurrentThreadId() */)
{
  mProfiler_ReportEventStart(name, asyncOrThreadIndex);
}

void mProfiler_ReportEventEnd_NoInline(const char *name, const uint32_t asyncOrThreadIndex /* = GetCurrentThreadId() */)
{
  mProfiler_ReportEventEnd(name, asyncOrThreadIndex);
}
