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

  mERROR_IF(pHandle->handle == INVALID_HANDLE_VALUE, mR_InternalError);

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

mFUNCTION(mRS232Handle_GetPortsFromName, const char *name, OUT mPtr<mQueue<uint32_t>> *pPorts, IN mAllocator *pAllocator)
{
  mFUNCTION_SETUP();

  mERROR_IF(pPorts == nullptr || name == nullptr, mR_ArgumentNull);

  mDEFER_CALL_ON_ERROR(pPorts, mQueue_Destroy);
  mERROR_CHECK(mQueue_Create(pPorts, pAllocator));

#if defined(mPLATFORM_WINDOWS)
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
        mERROR_CHECK(mSprintf(buffer, mARRAYSIZE(buffer), "%s (COM%%d)", name));
        
        if (0 >= sscanf_s((char *)friendly_name, buffer, &port))
          continue;

        if (port != -1)
        {
          mPRINT_DEBUG("Port %" PRIi32 ", %s\n", port, friendly_name);
          mERROR_CHECK(mQueue_PushBack(*pPorts, (uint32_t)port));
        }
      }
    }
  }

  SetupDiDeleteDeviceInfo(devSet, &devInfo); // Comment from cutil/holo: cm, this line is problematic
  SetupDiDestroyDeviceInfoList(devSet);

#else
  mRETURN_RESULT(mR_NotImplemented);
#endif

  mRETURN_SUCCESS();
}

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

mFUNCTION(mRS232Handle_Write, mRS232Handle *pHandle, IN const uint8_t *pBuffer, const size_t length, OUT OPTIONAL size_t *pBytesWriteCount)
{
  mFUNCTION_SETUP();

  mERROR_IF(pHandle == nullptr || pBuffer == nullptr, mR_ArgumentNull);

#if defined(mPLATFORM_WINDOWS)
  DWORD bytesWritten = 0;

  const BOOL succeeded = WriteFile(pHandle->handle, pBuffer, (DWORD)length, &bytesWritten, NULL);

  if (!succeeded)
  {
    const DWORD error = GetLastError();
    mUnused(error);

    mRETURN_RESULT(mR_InternalError);
  }

  if (pBytesWriteCount)
    *pBytesWriteCount = (size_t)bytesWritten;
#else
  ssize_t bytesWritten = write(pHandle->handle, pBuffer, length);

  if (bytesWritten < 0)
    bytesWritten = 0;

  if (pBytesWriteCount)
    *pBytesWriteCount = (size_t)bytesWritten;
#endif

  mRETURN_SUCCESS();
}
