#include "mRS232Handle.h"

#if defined(mPLATFORM_WINDOWS)
#include <winioctl.h>
#include "Setupapi.h"

#pragma comment(lib, "Setupapi.lib")
#else
#include <termios.h>
#include <unistd.h>
#include <fcntl.h>
#endif

#ifdef GIT_BUILD // Define __M_FILE__
  #ifdef __M_FILE__
    #undef __M_FILE__
  #endif
  #define __M_FILE__ "7m6Hsadz75nEBQw7cp7/qUFNtXy3NBwRGm4fTrSRlxWpmev/CqiEx+KPno2PuY99wYJZfs/DGZEz2OYz"
#endif

struct mRS232Handle
{
#if defined(mPLATFORM_WINDOWS)
  HANDLE handle;
#else
  int handle;
#endif

  mAllocator *pAllocator;
};

mFUNCTION(mRS232Handle_Create, OUT mRS232Handle **ppHandle, IN mAllocator *pAllocator, const char *name, const size_t baudrate)
{
  mFUNCTION_SETUP();

  mERROR_IF(ppHandle == nullptr || name == nullptr, mR_ArgumentNull);

  mRS232Handle *pHandle = nullptr;
  mDEFER_ON_ERROR(mAllocator_FreePtr(pAllocator, &pHandle));
  mERROR_CHECK(mAllocator_AllocateZero(pAllocator, &pHandle, 1));

  pHandle->pAllocator = pAllocator;

#if defined(mPLATFORM_WINDOWS)
  // Open port.
  pHandle->handle = CreateFileA(name, GENERIC_READ | GENERIC_WRITE, 0, NULL, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);

  if (pHandle->handle == INVALID_HANDLE_VALUE)
  {
    const DWORD error = GetLastError();

    switch (error)
    {
    case ERROR_FILE_NOT_FOUND:
      mRETURN_RESULT(mR_ResourceNotFound);

    case ERROR_ACCESS_DENIED:
      mRETURN_RESULT(mR_InsufficientPrivileges);

    default:
      mRETURN_RESULT(mR_InternalError);
    }
  }

  mDEFER_CALL_ON_ERROR(pHandle->handle, CloseHandle);

  // Set default timeouts.
  mERROR_CHECK(mRS232Handle_SetTimeouts(pHandle));

  // Set Baudrate.
  {
    DCB config;
    BOOL succeeded = GetCommState(pHandle->handle, &config);

    if (!succeeded)
    {
      const DWORD error = GetLastError();
      mUnused(error);

      mRETURN_RESULT(mR_InternalError);
    }

    config.BaudRate = (DWORD)baudrate;
    succeeded = SetCommState(pHandle->handle, &config);

    if (!succeeded)
    {
      const DWORD error = GetLastError();
      mUnused(error);

      mRETURN_RESULT(mR_InternalError);
    }
  }
#else
  pHandle->handle = open(name, O_RDWR | O_NOCTTY | O_NDELAY);

  mERROR_IF(pHandle->handle < 0, mR_InternalError);
  mDEFER_CALL_ON_ERROR(pHandle->handle, close);
  mDEFER_CALL_ON_ERROR(pHandle->handle, tcdrain);

  // Set Baudrate.
  {
    struct termios config;
    tcgetattr(pHandle->handle, &config);
    cfsetospeed(&config, baudrate);
    cfsetispeed(&config, baudrate);
    tcsetattr(pHandle->handle, TCSANOW, &config);
  }
#endif

  *ppHandle = pHandle;

  mRETURN_SUCCESS();
}

mFUNCTION(mRS232Handle_Create, OUT mRS232Handle **ppHandle, IN mAllocator *pAllocator, const uint32_t port, const size_t baudrate)
{
  mFUNCTION_SETUP();

  char buffer[128] = {};
  mERROR_CHECK(mFormatTo(buffer, mARRAYSIZE(buffer), "\\\\.\\COM", port));

  mERROR_CHECK(mRS232Handle_Create(ppHandle, pAllocator, buffer, baudrate));

  mRETURN_SUCCESS();
}

mFUNCTION(mRS232Handle_Destroy, IN_OUT mRS232Handle **ppHandle)
{
  mFUNCTION_SETUP();

  mERROR_IF(ppHandle == nullptr || (*ppHandle) == nullptr, mR_ArgumentNull);

#if defined(mPLATFORM_WINDOWS)
  CloseHandle((*ppHandle)->handle);
#else
  tcdrain((*ppHandle)->handle);
  close((*ppHandle)->handle);
#endif

  mAllocator *pAllocator = (*ppHandle)->pAllocator;
  mERROR_CHECK(mAllocator_FreePtr(pAllocator, ppHandle));

  mRETURN_SUCCESS();
}

mFUNCTION(mRS232Handle_SetTimeouts, IN mRS232Handle *pHandle, const uint32_t readIntervalTimeout /* = 0xFFFFFFFF */, const uint32_t readTotalTimeoutMultiplier /* = 0 */, const uint32_t readTotalTimeoutConstant /* = 0 */, const uint32_t writeTotalTimeoutMultiplier /* = 0 */, const uint32_t writeTotalTimeoutConstant /* = 0 */)
{
  mFUNCTION_SETUP();

  mERROR_IF(pHandle == nullptr, mR_ArgumentNull);

#if defined(mPLATFORM_WINDOWS)
  COMMTIMEOUTS comTimeOut;
  comTimeOut.ReadIntervalTimeout = readIntervalTimeout;
  comTimeOut.ReadTotalTimeoutMultiplier = readTotalTimeoutMultiplier;
  comTimeOut.ReadTotalTimeoutConstant = readTotalTimeoutConstant;
  comTimeOut.WriteTotalTimeoutMultiplier = writeTotalTimeoutMultiplier;
  comTimeOut.WriteTotalTimeoutConstant = writeTotalTimeoutConstant;

  const BOOL succeeded = SetCommTimeouts(pHandle->handle, &comTimeOut);

  if (!succeeded)
  {
    const DWORD error = GetLastError();
    mUnused(error);

    mRETURN_RESULT(mR_InternalError);
  }

#else
  mUnused(readIntervalTimeout, readTotalTimeoutMultiplier, readTotalTimeoutConstant, writeTotalTimeoutMultiplier, writeTotalTimeoutConstant);
#endif

  mRETURN_SUCCESS();
}

mFUNCTION(mRS232Handle_FlushBuffers, IN mRS232Handle *pHandle, const bool flushRead /* = true */, const bool flushWrite /* = true */)
{
  mFUNCTION_SETUP();

  mERROR_IF(pHandle == nullptr, mR_ArgumentNull);

#if defined(mPLATFORM_WINDOWS)
  DWORD readFlag = 0;
  DWORD writeFlag = 0;

  if (flushRead)
    readFlag = PURGE_RXABORT | PURGE_RXCLEAR;

  if (flushWrite)
    writeFlag = PURGE_TXABORT | PURGE_TXCLEAR;

  if (flushRead || flushWrite)
  {
    if (!PurgeComm(pHandle->handle, readFlag | writeFlag))
    {
      const DWORD error = GetLastError();
      mUnused(error);

      mRETURN_RESULT(mR_InternalError);
    }
  }

#else
  if (flushRead && flushWrite)
    tcflush(pHandle->handle, TCIOFLUSH);
  else if (flushRead)
    tcflush(pHandle->handle, TCIFLUSH);
  else if (flushWrite)
    tcflush(pHandle->handle, TCOFLUSH);
#endif

  mRETURN_SUCCESS();
}

#if defined(mPLATFORM_WINDOWS)

mFUNCTION(mRS232Handle_GetPortsFromName, const char *name, OUT mPtr<mQueue<uint32_t>> *pPorts, IN mAllocator *pAllocator)
{
  mFUNCTION_SETUP();

  mERROR_IF(pPorts == nullptr || name == nullptr, mR_ArgumentNull);

  mDEFER_CALL_ON_ERROR(pPorts, mQueue_Destroy);
  mERROR_CHECK(mQueue_Create(pPorts, pAllocator));

  const GUID guid = GUID_DEVINTERFACE_SERENUM_BUS_ENUMERATOR;
  const HDEVINFO devSet = SetupDiGetClassDevsA(&guid, NULL, NULL, DIGCF_PRESENT);
  SP_DEVINFO_DATA devInfo;

  mERROR_CHECK(mMemset(&devInfo, 1));
  devInfo.cbSize = sizeof(SP_DEVINFO_DATA);

  char buffer[1024];

  DWORD idNum = 0;
  BOOL moreItems = true;
  
  while (moreItems)
  {
    moreItems = SetupDiEnumDeviceInfo(devSet, idNum, &devInfo);
    mDEFER(idNum++);
    
    if (moreItems)
    {
      BYTE friendly_name[256];
      friendly_name[0] = '\0';
      DWORD friendly_size = sizeof(friendly_name);

      if (SetupDiGetDeviceRegistryPropertyA(devSet, &devInfo, SPDRP_FRIENDLYNAME, NULL, friendly_name, friendly_size, NULL))
      {
        int32_t port = -1;
        mERROR_CHECK(mFormatTo(buffer, mARRAYSIZE(buffer), name, " (COM%d)"));
        
        // TODO: This is absolutely disgusting. This should definetely transition to using `mParseUInt`!!!
        if (0 >= sscanf_s((char *)friendly_name, buffer, &port))
          continue;

        if (port != -1)
        {
          mPRINT_DEBUG("Port ", port, ", ", reinterpret_cast<const char *>(friendly_name));
          mERROR_CHECK(mQueue_PushBack(*pPorts, (uint32_t)port));
        }
      }
    }
  }

  SetupDiDeleteDeviceInfo(devSet, &devInfo); // Comment from cutil/holo: cm, this line is problematic
  SetupDiDestroyDeviceInfoList(devSet);

  mRETURN_SUCCESS();
}

mFUNCTION(mRS232Handle_GetPortsFromVID_PID, const size_t vid, const size_t pid, OUT mPtr<mQueue<uint32_t>> *pPorts, IN mAllocator *pAllocator)
{
  mFUNCTION_SETUP();

  mERROR_IF(pPorts == nullptr, mR_ArgumentNull);

  mDEFER_CALL_ON_ERROR(pPorts, mQueue_Destroy);
  mERROR_CHECK(mQueue_Create(pPorts, pAllocator));

  char comPortName[0x100];
  mERROR_CHECK(mFormatTo(comPortName, mARRAYSIZE(comPortName), "VID_", mFX()(vid), "&PID_", mFX()(pid)));

  size_t comPortNameWCount = 0;
  wchar_t comPortNameW[0x100];
  mERROR_CHECK(mString_ToWideStringRaw(comPortName, comPortNameW, mARRAYSIZE(comPortNameW), &comPortNameWCount));

  HDEVINFO deviceInfoSet = SetupDiGetClassDevsW(nullptr, TEXT("USB"), nullptr, DIGCF_ALLCLASSES | DIGCF_PRESENT);
  mERROR_IF(deviceInfoSet == INVALID_HANDLE_VALUE, mR_InternalError);
  mDEFER_CALL(deviceInfoSet, SetupDiDestroyDeviceInfoList);

  SP_DEVINFO_DATA deviceInfoData;
  mZeroMemory(&deviceInfoData);
  deviceInfoData.cbSize = sizeof(SP_DEVINFO_DATA);

  DWORD i = 0;

  while (SetupDiEnumDeviceInfo(deviceInfoSet, i, &deviceInfoData))
  {
    i++;

    DEVPROPTYPE propertyType;
    uint8_t buffer[1024];
    DWORD size = 0;

    if (SetupDiGetDeviceRegistryPropertyW(deviceInfoSet, &deviceInfoData, SPDRP_HARDWAREID, &propertyType, buffer, sizeof(buffer), &size))
    {
      if (size <= (4 + comPortNameWCount) * sizeof(wchar_t))
        continue;
      
      if (memcmp(buffer, TEXT("USB\\"), 4) != 0)
        continue;
      
      if (memcmp(buffer + 4 * sizeof(wchar_t), comPortNameW, (comPortNameWCount - 1) * sizeof(wchar_t)) != 0)
        continue;
      
      if (reinterpret_cast<wchar_t *>(buffer)[4 + comPortNameWCount - 1] != TEXT('\0') && reinterpret_cast<wchar_t *>(buffer)[4 + comPortNameWCount - 1] != TEXT('&'))
        continue;

      HKEY deviceRegistryKey;
      deviceRegistryKey = SetupDiOpenDevRegKey(deviceInfoSet, &deviceInfoData, DICS_FLAG_GLOBAL, 0, DIREG_DEV, KEY_READ);
      mERROR_IF(deviceRegistryKey == INVALID_HANDLE_VALUE, mR_InternalError);
      mDEFER_CALL(deviceRegistryKey, RegCloseKey);

      wchar_t portName[0x100];
      DWORD portNameLength = sizeof(portName);
      DWORD dataType = 0;
   
      if (RegQueryValueExW(deviceRegistryKey, TEXT("PortName"), NULL, &dataType, reinterpret_cast<BYTE *>(portName), &portNameLength) == ERROR_SUCCESS && dataType == REG_SZ && portNameLength > 3 * sizeof(wchar_t))
      {
        if (memcmp(portName, TEXT("COM"), 3 * sizeof(wchar_t)) == 0 && mIsUInt(portName + 3))
        {
          const uint64_t port = mParseUInt(portName + 3);
          const uint32_t portU32 = (uint32_t)port;
          
          if (port < UINT32_MAX)
            mERROR_CHECK(mQueue_PushBack(*pPorts, portU32));
        }
      }
    }
  }

  mRETURN_SUCCESS();
}

#endif

mFUNCTION(mRS232Handle_Read, mRS232Handle *pHandle, OUT uint8_t *pBuffer, const size_t length, OUT OPTIONAL size_t *pBytesReadCount)
{
  mFUNCTION_SETUP();

  mERROR_IF(pHandle == nullptr || pBuffer == nullptr, mR_ArgumentNull);

#if defined(mPLATFORM_WINDOWS)
  DWORD bytesRead = 0;
  const BOOL succeeded = ReadFile(pHandle->handle, pBuffer, (DWORD)length, &bytesRead, NULL);

  if (!succeeded)
  {
    const DWORD error = GetLastError();
    mUnused(error);

    mRETURN_RESULT(mR_InternalError);
  }
  
  if (pBytesReadCount)
    *pBytesReadCount = (size_t)bytesRead;
#else
  const ssize_t bytesRead = read(pHandle->handle, pBuffer, length);

  if (bytesRead < 0)
    bytesRead = 0;

  if (pBytesReadCount)
    *pBytesReadCount = (size_t)bytesRead;
#endif

  mRETURN_SUCCESS();
}

mFUNCTION(mRS232Handle_Write, mRS232Handle *pHandle, IN const uint8_t *pBuffer, const size_t length, OUT OPTIONAL size_t *pBytesWriteCount, const size_t retryTimeMs /* = 1 */)
{
  mFUNCTION_SETUP();

  mERROR_IF(pHandle == nullptr || pBuffer == nullptr, mR_ArgumentNull);

  const size_t startTime = mGetCurrentTimeMs();

#if defined(mPLATFORM_WINDOWS)
  DWORD bytesWritten = 0;

retry:
  const BOOL succeeded = WriteFile(pHandle->handle, pBuffer, (DWORD)length, &bytesWritten, NULL);

  if (!succeeded)
  {
    const DWORD error = GetLastError();

    switch (error)
    {
    case ERROR_SEM_TIMEOUT:

      if (retryTimeMs < (mGetCurrentTimeMs() - startTime))
        mRETURN_RESULT(mR_Timeout);

      mSleep(1);

      goto retry;

    default:
      mRETURN_RESULT(mR_InternalError);
    }
  }

  if (pBytesWriteCount)
    *pBytesWriteCount = (size_t)bytesWritten;
  else
    mERROR_IF(bytesWritten != length, mR_Failure);

#else
  ssize_t bytesWritten = write(pHandle->handle, pBuffer, length);

  if (bytesWritten < 0)
    bytesWritten = 0;

  if (pBytesWriteCount)
    *pBytesWriteCount = (size_t)bytesWritten;
#endif

  mRETURN_SUCCESS();
}
