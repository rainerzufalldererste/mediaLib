#ifndef mHTTPServer_h__
#define mHTTPServer_h__

#include "mediaLib.h"
#include "mBinaryChunk.h"
#include "mHashMap.h"

enum HttpResponseStatusCode
{
  mHRTC_Continue = 100, // Continue
  mHRTC_SwitchingProtocols = 101, // Switching Protocols 
  mHRTC_Processing = 102, // Processing 
  mHRTC_Ok = 200, // OK
  mHRTC_Created = 201, // Created 
  mHRTC_Accepted = 202, // Accepted
  mHRTC_NonAuthoritativeInformation = 203, // Non-Authoritative Information
  mHRTC_NoContent = 204, // No Content 
  mHRTC_ResetContent = 205, // Reset Content 
  mHRTC_PartialContent = 206, // Partial Content
  mHRTC_MultiStatus = 207, // Multi-Status
  mHRTC_AlreadyReported = 208, // Already Reported 
  mHRTC_ImUsed = 226, // IM Used 
  mHRTC_MultipleChoices = 300, // Multiple Choices 
  mHRTC_MovedPermanently = 301, // Moved Permanently
  mHRTC_Found = 302, // Found
  mHRTC_SeeOther = 303, // See Other
  mHRTC_NotModified = 304, // Not Modified
  mHRTC_UseProxy = 305, // Use Proxy
  mHRTC_TemporaryRedirect = 307, // Temporary Redirect
  mHRTC_PermanentRedirect = 308, // Permanent Redirect
  mHRTC_BadRequest = 400, // Bad Request
  mHRTC_Unauthorized = 401, // Unauthorized
  mHRTC_PaymentRequired = 402, // Payment Required 
  mHRTC_Forbidden = 403, // Forbidden
  mHRTC_NotFound = 404, // Not Found
  mHRTC_MethodNotAllowed = 405, // Method Not Allowed
  mHRTC_NotAcceptable = 406, // Not Acceptable
  mHRTC_ProxyAuthenticationRequired = 407, // Proxy Authentication Required
  mHRTC_RequestTimeout = 408, // Request Timeout
  mHRTC_Conflict = 409, // Conflict
  mHRTC_Gone = 410, // Gone 
  mHRTC_LengthRequired = 411, // Length Required
  mHRTC_PreconditionFailed = 412, // Precondition Failed 
  mHRTC_PayloadTooLarge = 413, // Payload Too Large
  mHRTC_UriTooLong = 414, // URI Too Long
  mHRTC_UnsupportedMediaType = 415, // Unsupported Media Type 
  mHRTC_RangeNotSatisfiable = 416, // Range Not Satisfiable
  mHRTC_ExpectationFailed = 417, // Expectation Failed
  mHRTC_MisdirectedRequest = 421, // Misdirected Request 
  mHRTC_UnprocessableEntity = 422, // Unprocessable Entity
  mHRTC_Locked = 423, // Locked
  mHRTC_FailedDependency = 424, // Failed Dependency
  mHRTC_UpgradeRequired = 426, // Upgrade Required 
  mHRTC_PreconditionRequired = 428, // Precondition Required
  mHRTC_TooManyRequests = 429, // Too Many Requests
  mHRTC_RequestHeaderFieldsTooLarge = 431, // Request Header Fields Too Large 
  mHRTC_UnavailableForLegalReasons = 451, // Unavailable For Legal Reasons
  mHRTC_InternalServerError = 500, // Internal Server Error
  mHRTC_NotImplemented = 501, // Not Implemented
  mHRTC_BadGateway = 502, // Bad Gateway
  mHRTC_ServiceUnavailable = 503, // Service Unavailable 
  mHRTC_GatewayTimeout = 504, // Gateway Timeout
  mHRTC_HttpVersionNotSupported = 505, // HTTP Version Not Supported
  mHRTC_VariantAlsoNegotiates = 506, // Variant Also Negotiates
  mHRTC_InsufficientStorage = 507, // Insufficient Storage
  mHRTC_LoopDetected = 508, // Loop Detected 
  mHRTC_NotExtended = 510, // Not Extended
  mHRTC_NetworkAuthenticationRequired = 511, // Network Authentication Required 
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
  mPtr<mHashMap<mString, mString>> headParameters;
  mPtr<mHashMap<mString, mString>> postParameters;
  mPtr<mHashMap<mString, mString>> attributes;
  mString url;
};

struct mHttpResponse
{
  HttpResponseStatusCode statusCode;
  mPtr<mBinaryChunk> responseStream;
};

struct mHttpRequestHandler
{
  typedef mFUNCTION(TryHandleRequestFunc, mPtr<mHttpRequest> &request, OUT bool *pCanRespond, IN_OUT mPtr<mHttpResponse> &response);

  TryHandleRequestFunc *pHandleRequest;
};

struct mHttpServer;

mFUNCTION(mHttpServer_Create, OUT mPtr<mHttpServer> *pHttpServer, IN mAllocator *pAllocator, const uint16_t port = 80);

#endif, //  mHTTPServer_h__
