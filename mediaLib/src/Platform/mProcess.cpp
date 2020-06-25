#include "mProcess.h"

#include <windows.h>
#include <psapi.h>

#ifdef GIT_BUILD // Define __M_FILE__
  #ifdef __M_FILE__
    #undef __M_FILE__
  #endif
  #define __M_FILE__ "T95CYLXKyn0HJFfznCqtXnVEVMQqgV2QCx2DMHfbOUp0JkZy1dUx9YspbezP2+U+G8QQbUUAQp61CzVt"
#endif

struct mProcess
{
  mPtr<mPipe> _stdin, _stderr, _stdout;
  HANDLE processHandle;
};

//////////////////////////////////////////////////////////////////////////

mFUNCTION(mProcess_GetRunningProcesses, OUT mPtr<mQueue<mProcessInfo>> *pProcesses, IN mAllocator *pAllocator)
{
  mFUNCTION_SETUP();

  mERROR_IF(pProcesses == nullptr, mR_ArgumentNull);

  if (*pProcesses == nullptr)
    mERROR_CHECK(mQueue_Create(pProcesses, pAllocator));
  else
    mERROR_CHECK(mQueue_Clear(*pProcesses));

  DWORD processIDs[2048];
  DWORD bytesNeeded = 0;

  if (!K32EnumProcesses(processIDs, sizeof(processIDs), &bytesNeeded))
  {
    const DWORD error = GetLastError();
    mUnused(error);

    mRETURN_RESULT(mR_InternalError);
  }

  const size_t processCount = bytesNeeded / sizeof(DWORD);

  mERROR_CHECK(mQueue_Reserve(*pProcesses, processCount));

  for (size_t i = 0; i < processCount; i++)
  {
    const DWORD processID = processIDs[i];

    if (processID == 0)
      continue;

    HANDLE processHandle = OpenProcess(PROCESS_QUERY_INFORMATION, FALSE, processID);

    if (processHandle == nullptr)
    {
      GetLastError();
      continue;
    }

    mDEFER(CloseHandle(processHandle));

    wchar_t filename[MAX_PATH];

    if (0 != K32GetModuleFileNameExW(processHandle, nullptr, filename, mARRAYSIZE(filename)))
    {
      mProcessInfo processInfo = { };
      processInfo.processId = processID;
      mERROR_CHECK(mString_Create(&processInfo.filename, filename, mARRAYSIZE(filename), pAllocator));

      mERROR_CHECK(mQueue_PushBack(*pProcesses, std::move(processInfo)));
    }
  }

  mRETURN_SUCCESS();
}

//////////////////////////////////////////////////////////////////////////

mFUNCTION(mProcess_Destroy_Internal, IN_OUT mProcess *pProcess);
mFUNCTION(mProcess_Run_Internal, const mString &executable, const mString &workingDirectory, const mString &params, OPTIONAL mPtr<mPipe> stdinPipe, OPTIONAL mPtr<mPipe> stdoutPipe, OPTIONAL mPtr<mPipe> stderrPipe, OUT OPTIONAL HANDLE *pProcessHandle, const mProcess_CreationFlags flags);

//////////////////////////////////////////////////////////////////////////

mFUNCTION(mProcess_Create, OUT mPtr<mProcess> *pProcess, IN mAllocator *pAllocator, const mString &executable, const mString &workingDirectory, const mString &params, OPTIONAL mPtr<mPipe> &stdinPipe, OPTIONAL mPtr<mPipe> &stdoutPipe, OPTIONAL mPtr<mPipe> &stderrPipe, const mProcess_CreationFlags flags /* = mP_CF_None */)
{
  mFUNCTION_SETUP();

  mERROR_IF(pProcess == nullptr, mR_ArgumentNull);

  HANDLE processHandle = nullptr;

  mERROR_CHECK(mProcess_Run_Internal(executable, workingDirectory, params, stdinPipe, stdoutPipe, stderrPipe, &processHandle, flags));

  mDEFER_ON_ERROR(CloseHandle(processHandle));

  mERROR_CHECK(mSharedPointer_Allocate<mProcess>(pProcess, pAllocator, [](mProcess *pData) { mProcess_Destroy_Internal(pData); }, 1));

  (*pProcess)->_stdin = stdinPipe;
  (*pProcess)->_stdout = stdoutPipe;
  (*pProcess)->_stderr = stderrPipe;
  (*pProcess)->processHandle = processHandle;

  mRETURN_SUCCESS();
}

mFUNCTION(mProcess_CreateFromProcessId, OUT mPtr<mProcess> *pProcess, IN mAllocator *pAllocator, const size_t processId)
{
  mFUNCTION_SETUP();

  mERROR_IF(pProcess == nullptr, mR_ArgumentNull);
  mERROR_IF(processId > MAXDWORD, mR_InvalidParameter);

  HANDLE processHandle = OpenProcess(PROCESS_QUERY_INFORMATION, FALSE, (DWORD)processId);

  if (processHandle == nullptr)
  {
    const DWORD error = GetLastError();

    switch (error)
    {
    case ERROR_NOT_FOUND:
    case ERROR_FILE_NOT_FOUND:
    case ERROR_PATH_NOT_FOUND:
      mRETURN_RESULT(mR_ResourceNotFound);
      break;

    case ERROR_ACCESS_DENIED:
      mRETURN_RESULT(mR_InsufficientPrivileges);
      break;

    default:
      mRETURN_RESULT(mR_InternalError);
    }
  }

  mDEFER_ON_ERROR(CloseHandle(processHandle));

  mERROR_CHECK(mSharedPointer_Allocate<mProcess>(pProcess, pAllocator, [](mProcess *pData) { mProcess_Destroy_Internal(pData); }, 1));

  (*pProcess)->processHandle = processHandle;

  mRETURN_SUCCESS();
}

mFUNCTION(mProcess_Destroy, IN_OUT mPtr<mProcess> *pProcess)
{
  mFUNCTION_SETUP();

  mERROR_IF(pProcess == nullptr, mR_ArgumentNull);

  mERROR_CHECK(mSharedPointer_Destroy(pProcess));

  mRETURN_SUCCESS();
}

mFUNCTION(mProcess_IsRunning, mPtr<mProcess> &process, OUT bool *pIsRunning)
{
  mFUNCTION_SETUP();

  mERROR_IF(process == nullptr || pIsRunning == nullptr, mR_ArgumentNull);

  const DWORD result = WaitForSingleObject(process->processHandle, 0);

  switch (result)
  {
  case WAIT_TIMEOUT:
    *pIsRunning = true;
    break;

  case WAIT_OBJECT_0:
    *pIsRunning = false;
    break;

  case WAIT_FAILED:
  {
    const DWORD error = GetLastError();
    mUnused(error);
  }
  // continue. (not break!)
  default:
    mRETURN_RESULT(mR_InternalError);
    break;
  }

  mRETURN_SUCCESS();
}

mFUNCTION(mProcess_GetExitCode, mPtr<mProcess> &process, OUT uint32_t *pExitCode)
{
  mFUNCTION_SETUP();

  mERROR_IF(process == nullptr || pExitCode == nullptr, mR_ArgumentNull);

  bool isRunning = false;
  mERROR_CHECK(mProcess_IsRunning(process, &isRunning));

  mERROR_IF(isRunning, mR_ResourceStateInvalid);

  DWORD exitCode;

  if (FALSE == GetExitCodeProcess(process->processHandle, &exitCode))
  {
    const DWORD error = GetLastError();
    mUnused(error);

    mRETURN_RESULT(mR_InternalError);
  }

  *pExitCode = exitCode;

  mRETURN_SUCCESS();
}

mFUNCTION(mProcess_Terminate, mPtr<mProcess> &process, const uint32_t exitCode /* = (uint32_t)-1 */)
{
  mFUNCTION_SETUP();

  mERROR_IF(process == nullptr, mR_ArgumentNull);

  if (FALSE == TerminateProcess(process->processHandle, exitCode))
  {
    const DWORD error = GetLastError();
    mUnused(error);

    mRETURN_RESULT(mR_InternalError);
  }

  mRETURN_SUCCESS();
}

mFUNCTION(mProcess_WaitForExit, mPtr<mProcess> &process, const uint32_t timeoutMs /* = (uint32_t)-1 */)
{
  mFUNCTION_SETUP();

  mERROR_IF(process == nullptr, mR_ArgumentNull);

  const DWORD result = WaitForSingleObject(process->processHandle, timeoutMs);

  switch (result)
  {
  case WAIT_TIMEOUT:
    mRETURN_RESULT(mR_Timeout);
    break;

  case WAIT_OBJECT_0:
    break;

  case WAIT_FAILED:
  {
    const DWORD error = GetLastError();
    mUnused(error);
  }
  // continue. (not break!)
  default:
    mRETURN_RESULT(mR_InternalError);
    break;
  }

  mRETURN_SUCCESS();
}

mFUNCTION(mProcess_Run, const mString &executable, const mString &workingDirectory, const mString &params, const mProcess_CreationFlags flags /* = mP_CF_None */)
{
  mFUNCTION_SETUP();

  mERROR_CHECK(mProcess_Run_Internal(executable, workingDirectory, params, mPtr<mPipe>(nullptr), mPtr<mPipe>(nullptr), mPtr<mPipe>(nullptr), nullptr, flags));

  mRETURN_SUCCESS();
}

mFUNCTION(mProcess_Run, const mString &executable, const mString &workingDirectory, const mString &params, OPTIONAL mPtr<mPipe> stdinPipe, OPTIONAL mPtr<mPipe> stdoutPipe, OPTIONAL mPtr<mPipe> stderrPipe, const mProcess_CreationFlags flags /* = mP_CF_None */)
{
  mFUNCTION_SETUP();
  
  mERROR_CHECK(mProcess_Run_Internal(executable, workingDirectory, params, stdinPipe, stdoutPipe, stderrPipe, nullptr, flags));

  mRETURN_SUCCESS();
}

mFUNCTION(mProcess_RunAsElevated, const mString &executable, const mString &workingDirectory, const mString &params)
{
  mFUNCTION_SETUP();

  mERROR_IF(executable.hasFailed || executable.bytes <= 0 || workingDirectory.hasFailed || params.hasFailed, mR_InvalidParameter);

  wchar_t appName[MAX_PATH + 1];
  wchar_t workingDir[MAX_PATH + 1];
  wchar_t commandLine[1024 * 4]; // Should probably be of size 32768 (https://docs.microsoft.com/en-us/windows/win32/api/processthreadsapi/nf-processthreadsapi-createprocessw).

  mERROR_CHECK(mString_ToWideString(executable, appName, mARRAYSIZE(appName)));
  mERROR_CHECK(mString_ToWideString(workingDirectory, workingDir, mARRAYSIZE(workingDir)));
  mERROR_CHECK(mString_ToWideString(params, commandLine, mARRAYSIZE(commandLine)));

  const HINSTANCE result = ShellExecuteW(nullptr, TEXT("runas"), appName, commandLine, workingDir, SW_SHOWNORMAL);
  mERROR_IF((int32_t)((size_t)result) <= 32, mR_Failure);

  mRETURN_SUCCESS();
}

//////////////////////////////////////////////////////////////////////////

mFUNCTION(mProcess_Destroy_Internal, IN_OUT mProcess *pProcess)
{
  mFUNCTION_SETUP();

  mERROR_IF(pProcess == nullptr, mR_ArgumentNull);

  CloseHandle(pProcess->processHandle);
  
  mERROR_CHECK(mPipe_Destroy(&pProcess->_stdin));
  mERROR_CHECK(mPipe_Destroy(&pProcess->_stdout));
  mERROR_CHECK(mPipe_Destroy(&pProcess->_stderr));

  mRETURN_SUCCESS();
}

mFUNCTION(mProcess_Run_Internal, const mString &executable, const mString &workingDirectory, const mString &params, OPTIONAL mPtr<mPipe> stdinPipe, OPTIONAL mPtr<mPipe> stdoutPipe, OPTIONAL mPtr<mPipe> stderrPipe, OUT OPTIONAL HANDLE *pProcessHandle, const mProcess_CreationFlags flags)
{
  mFUNCTION_SETUP();

  mERROR_IF(executable.hasFailed || executable.bytes <= 0 || workingDirectory.hasFailed || params.hasFailed, mR_InvalidParameter);

  wchar_t appName[MAX_PATH + 1];
  wchar_t workingDir[MAX_PATH + 1];
  wchar_t commandLine[1024 * 4]; // Theoretically should be max length 32768 according to msdn (https://docs.microsoft.com/en-us/windows/win32/api/processthreadsapi/nf-processthreadsapi-createprocessw).

  mERROR_CHECK(mString_ToWideString(executable, appName, mARRAYSIZE(appName)));
  mERROR_CHECK(mString_ToWideString(workingDirectory, workingDir, mARRAYSIZE(workingDir)));
  mERROR_CHECK(mString_ToWideString(params, commandLine, mARRAYSIZE(commandLine)));

  PROCESS_INFORMATION processInfo;
  STARTUPINFO startInfo;

  mZeroMemory(&processInfo);
  mZeroMemory(&startInfo);

  startInfo.cb = sizeof(STARTUPINFO);

  if (stdinPipe != nullptr)
    mERROR_CHECK(mPipe_GetReadHandle(stdinPipe, &startInfo.hStdInput));

  if (stdoutPipe != nullptr)
    mERROR_CHECK(mPipe_GetWriteHandle(stdoutPipe, &startInfo.hStdOutput));

  if (stderrPipe != nullptr)
    mERROR_CHECK(mPipe_GetWriteHandle(stderrPipe, &startInfo.hStdError));

  startInfo.dwFlags |= STARTF_USESTDHANDLES;

  mDEFER(
    if (pProcessHandle == nullptr)
      CloseHandle(processInfo.hProcess);
    
    CloseHandle(processInfo.hThread);
  );

  if (FALSE == CreateProcessW(appName, commandLine, nullptr, nullptr, TRUE, CREATE_NO_WINDOW * !!(flags & mP_CF_NoWindow), NULL, workingDir, &startInfo, &processInfo))
  {
    const DWORD error = GetLastError();
    mUnused(error);

    mRETURN_RESULT(mR_InternalError);
  }

  if (pProcessHandle != nullptr)
    *pProcessHandle = processInfo.hProcess;

  mRETURN_SUCCESS();
}
