#ifndef mHTTPServer_h__
#define mHTTPServer_h__

#include "mediaLib.h"
#include "mBinaryChunk.h"
#include "mQueue.h"
#include "mKeyValuePair.h"
#include "mThreadPool.h"

#ifdef GIT_BUILD // Define __M_FILE__
  #ifdef __M_FILE__
    #undef __M_FILE__
  #endif
  #define __M_FILE__ "spXOOhpyugFz66f7m8FOfXjvvk6I+8kARIXKip/Z6Nc/52neDouggGa8zEluy6tt6g72Qoe3p1JQog03"
#endif

enum mHttpResponseStatusCode
{
  mHRSC_Continue = 100, // Continue
  mHRSC_SwitchingProtocols = 101, // Switching Protocols 
  mHRSC_Processing = 102, // Processing 
  mHRSC_Ok = 200, // OK
  mHRSC_Created = 201, // Created 
  mHRSC_Accepted = 202, // Accepted
  mHRSC_NonAuthoritativeInformation = 203, // Non-Authoritative Information
  mHRSC_NoContent = 204, // No Content 
  mHRSC_ResetContent = 205, // Reset Content 
  mHRSC_PartialContent = 206, // Partial Content
  mHRSC_MultiStatus = 207, // Multi-Status
  mHRSC_AlreadyReported = 208, // Already Reported 
  mHRSC_ImUsed = 226, // IM Used 
  mHRSC_MultipleChoices = 300, // Multiple Choices 
  mHRSC_MovedPermanently = 301, // Moved Permanently
  mHRSC_Found = 302, // Found
  mHRSC_SeeOther = 303, // See Other
  mHRSC_NotModified = 304, // Not Modified
  mHRSC_UseProxy = 305, // Use Proxy
  mHRSC_TemporaryRedirect = 307, // Temporary Redirect
  mHRSC_PermanentRedirect = 308, // Permanent Redirect
  mHRSC_BadRequest = 400, // Bad Request
  mHRSC_Unauthorized = 401, // Unauthorized
  mHRSC_PaymentRequired = 402, // Payment Required 
  mHRSC_Forbidden = 403, // Forbidden
  mHRSC_NotFound = 404, // Not Found
  mHRSC_MethodNotAllowed = 405, // Method Not Allowed
  mHRSC_NotAcceptable = 406, // Not Acceptable
  mHRSC_ProxyAuthenticationRequired = 407, // Proxy Authentication Required
  mHRSC_RequestTimeout = 408, // Request Timeout
  mHRSC_Conflict = 409, // Conflict
  mHRSC_Gone = 410, // Gone 
  mHRSC_LengthRequired = 411, // Length Required
  mHRSC_PreconditionFailed = 412, // Precondition Failed 
  mHRSC_PayloadTooLarge = 413, // Payload Too Large
  mHRSC_UriTooLong = 414, // URI Too Long
  mHRSC_UnsupportedMediaType = 415, // Unsupported Media Type 
  mHRSC_RangeNotSatisfiable = 416, // Range Not Satisfiable
  mHRSC_ExpectationFailed = 417, // Expectation Failed
  mHRSC_ImATeapot = 418, // I'm a teapot
  mHRSC_MisdirectedRequest = 421, // Misdirected Request 
  mHRSC_UnprocessableEntity = 422, // Unprocessable Entity
  mHRSC_Locked = 423, // Locked
  mHRSC_FailedDependency = 424, // Failed Dependency
  mHRSC_UpgradeRequired = 426, // Upgrade Required 
  mHRSC_PreconditionRequired = 428, // Precondition Required
  mHRSC_TooManyRequests = 429, // Too Many Requests
  mHRSC_RequestHeaderFieldsTooLarge = 431, // Request Header Fields Too Large 
  mHRSC_UnavailableForLegalReasons = 451, // Unavailable For Legal Reasons
  mHRSC_InternalServerError = 500, // Internal Server Error
  mHRSC_NotImplemented = 501, // Not Implemented
  mHRSC_BadGateway = 502, // Bad Gateway
  mHRSC_ServiceUnavailable = 503, // Service Unavailable 
  mHRSC_GatewayTimeout = 504, // Gateway Timeout
  mHRSC_HttpVersionNotSupported = 505, // HTTP Version Not Supported
  mHRSC_VariantAlsoNegotiates = 506, // Variant Also Negotiates
  mHRSC_InsufficientStorage = 507, // Insufficient Storage
  mHRSC_LoopDetected = 508, // Loop Detected 
  mHRSC_NotExtended = 510, // Not Extended
  mHRSC_NetworkAuthenticationRequired = 511, // Network Authentication Required 
};

enum mHttpRequestMethod
{
  mHRM_Delete = 0, // DELETE
  mHRM_Get = 1, // GET
  mHRM_Head = 2, // HEAD
  mHRM_Post = 3, // POST
  mHRM_Put = 4, // PUT
  
  // Pathological:
  mHRM_Connect = 5, // CONNECT
  mHRM_Options = 6, // OPTIONS
  mHRM_Trace = 7, // TRACE
  
  // WebDAV:
  mHRM_Copy = 8, // COPY
  mHRM_Lock = 9, // LOCK
  mHRM_Mkcol = 10, // MKCOL
  mHRM_Move = 11, // MOVE
  mHRM_PropFind = 12, // PROPFIND
  mHRM_PropPatch = 13, // PROPPATCH
  mHRM_Search = 14, // SEARCH
  mHRM_Unlock = 15, // UNLOCK
  mHRM_Bind = 16, // BIND
  mHRM_Rebind = 17, // REBIND
  mHRM_Unbind = 18, // UNBIND
  mHRM_Acl = 19, // ACL

  // Subversion:
  mHRM_Report = 20, // REPORT
  mHRM_MkActivity = 21, // MKACTIVITY
  mHRM_Checkout = 22, // CHECKOUT
  mHRM_Merge = 23, // MERGE

  // upnp:
  mHRM_MSearch = 24, // M-SEARCH
  mHRM_Notify = 25, // NOTIFY
  mHRM_Subscribe = 26, // SUBSCRIBE
  mHRM_Unsubscribe = 27, // UNSUBSCRIBE

  // RFC-5789:
  mHRM_Patch = 28, // PATCH
  mHRM_Purge = 29, // PURGE

  // CalDAV:
  mHRM_MuCalendar = 30, // MKCALENDAR
  
  // RFC-2068, section 19.6.1.2:
  mHRM_Link = 31, // LINK
  mHRM_Unlink = 32, // UNLINK

  // icecast:
  mHRM_Source = 33, // SOURCE
};

struct mHttpRequestHandler;

struct mHttpRequest
{
  mHttpRequestMethod requestMethod;
  mUniqueContainer<mQueue<mKeyValuePair<mString, mString>>> headParameters;
  mUniqueContainer<mQueue<mKeyValuePair<mString, mString>>> postParameters;
  mUniqueContainer<mQueue<mKeyValuePair<mString, mString>>> attributes;
  mPtr<struct mTcpClient> client;
  mString url;
  mAllocator *pAllocator;
};

struct mHttpResponse
{
  mHttpResponseStatusCode statusCode;
  mPtr<mBinaryChunk> responseStream;
  mString contentType;
  OPTIONAL mString charSet;
  mUniqueContainer<mQueue<mKeyValuePair<mString, mString>>> setCookies;
  mUniqueContainer<mQueue<mKeyValuePair<mString, mString>>> attributes;
};

struct mHttpRequestHandler
{
  typedef mFUNCTION(TryHandleRequestFunc, mPtr<mHttpRequestHandler> &handler, mPtr<mHttpRequest> &request, OUT bool *pCanRespond, IN_OUT mPtr<mHttpResponse> &response);

  TryHandleRequestFunc *pHandleRequest;
};

struct mHttpErrorRequestHandler
{
  typedef mFUNCTION(HandleRequestFunc, mPtr<mHttpErrorRequestHandler> &handler, const mString &errorString, IN_OUT mPtr<mHttpResponse> &response);

  HandleRequestFunc *pHandleRequest;
};

struct mHttpServer;

mFUNCTION(mHttpServer_Create, OUT mPtr<mHttpServer> *pHttpServer, IN mAllocator *pAllocator, const uint16_t port = 80, const size_t threadCount = 8);
mFUNCTION(mHttpServer_Create, OUT mPtr<mHttpServer> *pHttpServer, IN mAllocator *pAllocator, mPtr<mTasklessThreadPool> &threadPool, const uint16_t port = 80);

mFUNCTION(mHttpServer_Destroy, IN_OUT mPtr<mHttpServer> *pHttpServer);

mFUNCTION(mHttpServer_AddRequestHandler, mPtr<mHttpServer> &httpServer, mPtr<mHttpRequestHandler> &requestHandler);
mFUNCTION(mHttpServer_SetErrorRequestHandler, mPtr<mHttpServer> &httpServer, mPtr<mHttpErrorRequestHandler> &requestHandler);

mFUNCTION(mHttpServer_Start, mPtr<mHttpServer> &httpServer);

//////////////////////////////////////////////////////////////////////////

mFUNCTION(mHttpErrorRequestHandler_CreateDefaultHandler, OUT mPtr<mHttpErrorRequestHandler> *pRequestHandler, IN mAllocator *pAllocator, const bool respondWithJson = false);

#endif, //  mHTTPServer_h__
