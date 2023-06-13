#include "mGpuCompute.h"

#include "mRenderParams.h"
#include "mTexture.h"

//////////////////////////////////////////////////////////////////////////

#define CL_USE_DEPRECATED_OPENCL_1_0_APIS
#define CL_USE_DEPRECATED_OPENCL_1_1_APIS
#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#define CL_USE_DEPRECATED_OPENCL_2_0_APIS

#define NVCL_SUPPRESS_USE_DEPRECATED_OPENCL_1_0_APIS_WARNING
#define NVCL_SUPPRESS_USE_DEPRECATED_OPENCL_1_1_APIS_WARNING
#define NVCL_SUPPRESS_USE_DEPRECATED_OPENCL_1_2_APIS_WARNING
#define NVCL_SUPPRESS_USE_DEPRECATED_OPENCL_2_0_APIS_WARNING

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#include <OpenCL/cl_gl.h>
#else
#include <CL/cl.h>
#include <CL/cl_gl.h>
#endif

#include <future>

#ifdef GIT_BUILD // Define __M_FILE__
  #ifdef __M_FILE__
    #undef __M_FILE__
  #endif
  #define __M_FILE__ "W/n1qjIQ4/gl2kRSy8h1QDwhaSEU89nHvKk1tNLYCCDbPHqp25p8SZ9QFBpFv8jA20adOdNd/WH6myft"
#endif

//////////////////////////////////////////////////////////////////////////

#define HT_CODEC_APPEND_IMPL(functionName) functionName ## _m_Impl

#define clEnqueueNDRangeKernel     HT_CODEC_APPEND_IMPL(clEnqueueNDRangeKernel    )
#define clBuildProgram             HT_CODEC_APPEND_IMPL(clBuildProgram            )
#define clEnqueueReadImage         HT_CODEC_APPEND_IMPL(clEnqueueReadImage        )
#define clReleaseCommandQueue      HT_CODEC_APPEND_IMPL(clReleaseCommandQueue     )
#define clReleaseProgram           HT_CODEC_APPEND_IMPL(clReleaseProgram          )
#define clReleaseKernel            HT_CODEC_APPEND_IMPL(clReleaseKernel           )
#define clCreateCommandQueue       HT_CODEC_APPEND_IMPL(clCreateCommandQueue      )
#define clWaitForEvents            HT_CODEC_APPEND_IMPL(clWaitForEvents           )
#define clCreateFromGLTexture      HT_CODEC_APPEND_IMPL(clCreateFromGLTexture     )
#define clFinish                   HT_CODEC_APPEND_IMPL(clFinish                  )
#define clEnqueueWriteImage        HT_CODEC_APPEND_IMPL(clEnqueueWriteImage       )
#define clGetDeviceIDs             HT_CODEC_APPEND_IMPL(clGetDeviceIDs            )
#define clReleaseContext           HT_CODEC_APPEND_IMPL(clReleaseContext          )
#define clReleaseEvent             HT_CODEC_APPEND_IMPL(clReleaseEvent            )
#define clCreateContext            HT_CODEC_APPEND_IMPL(clCreateContext           )
#define clEnqueueReleaseGLObjects  HT_CODEC_APPEND_IMPL(clEnqueueReleaseGLObjects )
#define clReleaseSampler           HT_CODEC_APPEND_IMPL(clReleaseSampler          )
#define clCreateSampler            HT_CODEC_APPEND_IMPL(clCreateSampler           )
#define clFlush                    HT_CODEC_APPEND_IMPL(clFlush                   )
#define clGetPlatformIDs           HT_CODEC_APPEND_IMPL(clGetPlatformIDs          )
#define clCreateImage2D            HT_CODEC_APPEND_IMPL(clCreateImage2D           )
#define clCreateImage3D            HT_CODEC_APPEND_IMPL(clCreateImage3D           )
#define clGetDeviceInfo            HT_CODEC_APPEND_IMPL(clGetDeviceInfo           )
#define clReleaseMemObject         HT_CODEC_APPEND_IMPL(clReleaseMemObject        )
#define clEnqueueAcquireGLObjects  HT_CODEC_APPEND_IMPL(clEnqueueAcquireGLObjects )
#define clCreateKernel             HT_CODEC_APPEND_IMPL(clCreateKernel            )
#define clCreateProgramWithSource  HT_CODEC_APPEND_IMPL(clCreateProgramWithSource )
#define clSetKernelArg             HT_CODEC_APPEND_IMPL(clSetKernelArg            )
#define clGetProgramBuildInfo      HT_CODEC_APPEND_IMPL(clGetProgramBuildInfo     )
#define clCreateBuffer             HT_CODEC_APPEND_IMPL(clCreateBuffer            )
#define clEnqueueReadBuffer        HT_CODEC_APPEND_IMPL(clEnqueueReadBuffer       )
#define clEnqueueWriteBuffer       HT_CODEC_APPEND_IMPL(clEnqueueWriteBuffer      )
#define clGetKernelWorkGroupInfo   HT_CODEC_APPEND_IMPL(clGetKernelWorkGroupInfo  )
#define clGetGLContextInfoKHR      HT_CODEC_APPEND_IMPL(clGetGLContextInfoKHR     )
#define clGetMemObjectInfo         HT_CODEC_APPEND_IMPL(clGetMemObjectInfo        )
#define clGetImageInfo             HT_CODEC_APPEND_IMPL(clGetImageInfo            )

// clEnqueueNDRangeKernel
static CL_API_ENTRY cl_int(*clEnqueueNDRangeKernel)(cl_command_queue /* command_queue */,
  cl_kernel        /* kernel */,
  cl_uint          /* work_dim */,
  const size_t *   /* global_work_offset */,
  const size_t *   /* global_work_size */,
  const size_t *   /* local_work_size */,
  cl_uint          /* num_events_in_wait_list */,
  const cl_event * /* event_wait_list */,
  cl_event *       /* event */) = nullptr CL_API_SUFFIX__VERSION_1_0;

// clBuildProgram
static CL_API_ENTRY cl_int(*clBuildProgram)(cl_program           /* program */,
  cl_uint              /* num_devices */,
  const cl_device_id * /* device_list */,
  const char *         /* options */,
  void (CL_CALLBACK *  /* pfn_notify */)(cl_program /* program */, void * /* user_data */),
  void *               /* user_data */) = nullptr CL_API_SUFFIX__VERSION_1_0;

// clEnqueueReadImage
static CL_API_ENTRY cl_int(*clEnqueueReadImage)(cl_command_queue     /* command_queue */,
  cl_mem               /* image */,
  cl_bool              /* blocking_read */,
  const size_t *       /* origin[3] */,
  const size_t *       /* region[3] */,
  size_t               /* row_pitch */,
  size_t               /* slice_pitch */,
  void *               /* ptr */,
  cl_uint              /* num_events_in_wait_list */,
  const cl_event *     /* event_wait_list */,
  cl_event *           /* event */) = nullptr CL_API_SUFFIX__VERSION_1_0;

// clReleaseCommandQueue
static CL_API_ENTRY cl_int(*clReleaseCommandQueue)(cl_command_queue /* command_queue */) = nullptr CL_API_SUFFIX__VERSION_1_0;

// clReleaseProgram
static CL_API_ENTRY cl_int(*clReleaseProgram)(cl_program /* program */) = nullptr CL_API_SUFFIX__VERSION_1_0;

// clReleaseKernel
static CL_API_ENTRY cl_int(*clReleaseKernel)(cl_kernel   /* kernel */) = nullptr CL_API_SUFFIX__VERSION_1_0;

// clCreateCommandQueue
static CL_API_ENTRY cl_command_queue(*clCreateCommandQueue)(cl_context                     /* context */,
  cl_device_id                   /* device */,
  cl_command_queue_properties    /* properties */,
  cl_int *                       /* errcode_ret */) = nullptr CL_API_SUFFIX__VERSION_1_0;

// clWaitForEvents
static CL_API_ENTRY cl_int(*clWaitForEvents)(cl_uint             /* num_events */,
  const cl_event *    /* event_list */) = nullptr CL_API_SUFFIX__VERSION_1_0;

// clCreateFromGLTexture
static CL_API_ENTRY cl_mem(*clCreateFromGLTexture)(cl_context      /* context */,
  cl_mem_flags    /* flags */,
  cl_GLenum       /* target */,
  cl_GLint        /* miplevel */,
  cl_GLuint       /* texture */,
  cl_int *        /* errcode_ret */) = nullptr CL_API_SUFFIX__VERSION_1_2;

// clFinish
static CL_API_ENTRY cl_int(*clFinish)(cl_command_queue /* command_queue */) = nullptr CL_API_SUFFIX__VERSION_1_0;

// clEnqueueWriteImage
static CL_API_ENTRY cl_int(*clEnqueueWriteImage)(cl_command_queue    /* command_queue */,
  cl_mem              /* image */,
  cl_bool             /* blocking_write */,
  const size_t *      /* origin[3] */,
  const size_t *      /* region[3] */,
  size_t              /* input_row_pitch */,
  size_t              /* input_slice_pitch */,
  const void *        /* ptr */,
  cl_uint             /* num_events_in_wait_list */,
  const cl_event *    /* event_wait_list */,
  cl_event *          /* event */) = nullptr CL_API_SUFFIX__VERSION_1_0;

// clGetDeviceIDs
static CL_API_ENTRY cl_int(*clGetDeviceIDs)(cl_platform_id   /* platform */,
  cl_device_type   /* device_type */,
  cl_uint          /* num_entries */,
  cl_device_id *   /* devices */,
  cl_uint *        /* num_devices */) = nullptr CL_API_SUFFIX__VERSION_1_0;

// clReleaseContext
static CL_API_ENTRY cl_int(*clReleaseContext)(cl_context /* context */) = nullptr CL_API_SUFFIX__VERSION_1_0;

// clReleaseEvent
static CL_API_ENTRY cl_int(*clReleaseEvent)(cl_event /* event */) = nullptr CL_API_SUFFIX__VERSION_1_0;

// clCreateContext
static CL_API_ENTRY cl_context(*clCreateContext)(const cl_context_properties * /* properties */,
  cl_uint                 /* num_devices */,
  const cl_device_id *    /* devices */,
  void (CL_CALLBACK * /* pfn_notify */)(const char *, const void *, size_t, void *),
  void *                  /* user_data */,
  cl_int *                /* errcode_ret */) = nullptr CL_API_SUFFIX__VERSION_1_0;

// clEnqueueReleaseGLObjects
static CL_API_ENTRY cl_int(*clEnqueueReleaseGLObjects)(cl_command_queue      /* command_queue */,
  cl_uint               /* num_objects */,
  const cl_mem *        /* mem_objects */,
  cl_uint               /* num_events_in_wait_list */,
  const cl_event *      /* event_wait_list */,
  cl_event *            /* event */) = nullptr CL_API_SUFFIX__VERSION_1_0;

// clReleaseSampler
static CL_API_ENTRY cl_int(*clReleaseSampler)(cl_sampler /* sampler */) = nullptr CL_API_SUFFIX__VERSION_1_0;

// clCreateSampler
static CL_API_ENTRY cl_sampler(*clCreateSampler)(cl_context          /* context */,
  cl_bool             /* normalized_coords */,
  cl_addressing_mode  /* addressing_mode */,
  cl_filter_mode      /* filter_mode */,
  cl_int *            /* errcode_ret */) = nullptr CL_API_SUFFIX__VERSION_1_0;

// clFlush
static CL_API_ENTRY cl_int(*clFlush)(cl_command_queue /* command_queue */) = nullptr CL_API_SUFFIX__VERSION_1_0;

// clGetPlatformIDs
static CL_API_ENTRY cl_int(*clGetPlatformIDs)(cl_uint          /* num_entries */,
  cl_platform_id * /* platforms */,
  cl_uint *        /* num_platforms */) = nullptr CL_API_SUFFIX__VERSION_1_0;

// clCreateImage2D
static CL_API_ENTRY CL_EXT_PREFIX__VERSION_1_1_DEPRECATED cl_mem(*clCreateImage2D)(cl_context              /* context */,
  cl_mem_flags            /* flags */,
  const cl_image_format * /* image_format */,
  size_t                  /* image_width */,
  size_t                  /* image_height */,
  size_t                  /* image_row_pitch */,
  void *                  /* host_ptr */,
  cl_int *                /* errcode_ret */) CL_EXT_SUFFIX__VERSION_1_1_DEPRECATED;

// clCreateImage3D
static CL_API_ENTRY CL_EXT_PREFIX__VERSION_1_1_DEPRECATED cl_mem(*clCreateImage3D)(cl_context              /* context */,
  cl_mem_flags            /* flags */,
  const cl_image_format * /* image_format */,
  size_t                  /* image_width */,
  size_t                  /* image_height */,
  size_t                  /* image_depth */,
  size_t                  /* image_row_pitch */,
  size_t                  /* image_slice_pitch */,
  void *                  /* host_ptr */,
  cl_int *                /* errcode_ret */) CL_EXT_SUFFIX__VERSION_1_1_DEPRECATED;

// clGetDeviceInfo
static CL_API_ENTRY cl_int(*clGetDeviceInfo)(cl_device_id    /* device */,
  cl_device_info  /* param_name */,
  size_t          /* param_value_size */,
  void *          /* param_value */,
  size_t *        /* param_value_size_ret */) = nullptr CL_API_SUFFIX__VERSION_1_0;

// clReleaseMemObject
static CL_API_ENTRY cl_int(*clReleaseMemObject)(cl_mem /* memobj */) = nullptr CL_API_SUFFIX__VERSION_1_0;

// clEnqueueAcquireGLObjects
static CL_API_ENTRY cl_int(*clEnqueueAcquireGLObjects)(cl_command_queue      /* command_queue */,
  cl_uint               /* num_objects */,
  const cl_mem *        /* mem_objects */,
  cl_uint               /* num_events_in_wait_list */,
  const cl_event *      /* event_wait_list */,
  cl_event *            /* event */) = nullptr CL_API_SUFFIX__VERSION_1_0;

// clCreateKernel
static CL_API_ENTRY cl_kernel(*clCreateKernel)(cl_program      /* program */,
  const char *    /* kernel_name */,
  cl_int *        /* errcode_ret */) = nullptr CL_API_SUFFIX__VERSION_1_0;

// clCreateProgramWithSource
static CL_API_ENTRY cl_program(*clCreateProgramWithSource)(cl_context        /* context */,
  cl_uint           /* count */,
  const char **     /* strings */,
  const size_t *    /* lengths */,
  cl_int *          /* errcode_ret */) = nullptr CL_API_SUFFIX__VERSION_1_0;

// clSetKernelArg
static CL_API_ENTRY cl_int(*clSetKernelArg)(cl_kernel    /* kernel */,
  cl_uint      /* arg_index */,
  size_t       /* arg_size */,
  const void * /* arg_value */) = nullptr CL_API_SUFFIX__VERSION_1_0;

// clGetProgramBuildInfo
static CL_API_ENTRY cl_int(*clGetProgramBuildInfo)(cl_program            /* program */,
  cl_device_id          /* device */,
  cl_program_build_info /* param_name */,
  size_t                /* param_value_size */,
  void *                /* param_value */,
  size_t *              /* param_value_size_ret */) = nullptr CL_API_SUFFIX__VERSION_1_0;

// clCreateBuffer
static CL_API_ENTRY cl_mem(*clCreateBuffer)(cl_context   /* context */,
  cl_mem_flags /* flags */,
  size_t       /* size */,
  void *       /* host_ptr */,
  cl_int *     /* errcode_ret */) = nullptr CL_API_SUFFIX__VERSION_1_0;

// clEnqueueReadBuffer
static CL_API_ENTRY cl_int(*clEnqueueReadBuffer)(cl_command_queue    /* command_queue */,
  cl_mem              /* buffer */,
  cl_bool             /* blocking_read */,
  size_t              /* offset */,
  size_t              /* size */,
  void *              /* ptr */,
  cl_uint             /* num_events_in_wait_list */,
  const cl_event *    /* event_wait_list */,
  cl_event *          /* event */) = nullptr CL_API_SUFFIX__VERSION_1_0;

// clEnqueueWriteBuffer
static CL_API_ENTRY cl_int(*clEnqueueWriteBuffer)(cl_command_queue   /* command_queue */,
  cl_mem             /* buffer */,
  cl_bool            /* blocking_write */,
  size_t             /* offset */,
  size_t             /* size */,
  const void *       /* ptr */,
  cl_uint            /* num_events_in_wait_list */,
  const cl_event *   /* event_wait_list */,
  cl_event *         /* event */) = nullptr CL_API_SUFFIX__VERSION_1_0;

// clGetKernelWorkGroupInfo
static CL_API_ENTRY cl_int(*clGetKernelWorkGroupInfo)(cl_kernel                  /* kernel */,
  cl_device_id               /* device */,
  cl_kernel_work_group_info  /* param_name */,
  size_t                     /* param_value_size */,
  void *                     /* param_value */,
  size_t *                   /* param_value_size_ret */) = nullptr CL_API_SUFFIX__VERSION_1_0;

// clGetGLContextInfoKHR
static CL_API_ENTRY cl_int(*clGetGLContextInfoKHR)(const cl_context_properties * /* properties */,
  cl_gl_context_info            /* param_name */,
  size_t                        /* param_value_size */,
  void *                        /* param_value */,
  size_t *                      /* param_value_size_ret */) = nullptr CL_API_SUFFIX__VERSION_1_0;

// clGetMemObjectInfo
static CL_API_ENTRY cl_int(*clGetMemObjectInfo)(cl_mem           /* memobj */,
  cl_mem_info      /* param_name */,
  size_t           /* param_value_size */,
  void *           /* param_value */,
  size_t *         /* param_value_size_ret */) = nullptr CL_API_SUFFIX__VERSION_1_0;

// clGetImageInfo
static CL_API_ENTRY cl_int(*clGetImageInfo)(cl_mem           /* image */,
  cl_image_info    /* param_name */,
  size_t           /* param_value_size */,
  void *           /* param_value */,
  size_t *         /* param_value_size_ret */) = nullptr CL_API_SUFFIX__VERSION_1_0;

bool mOpenCL_Initialized = false;
bool mOpenCL_Available = false;

#ifdef _WIN32
HMODULE mOpenCL_DllModule;
#else
void *mOpenCL_pDllModuleHandle;
#endif

//////////////////////////////////////////////////////////////////////////

struct mGpuComputeContext
{
  CONST_FIELD bool openGlEnabled;
  CONST_FIELD bool supportsOpenGLSharing;
  CONST_FIELD bool supportsOpenGLEvent;
  CONST_FIELD bool supportsOpenGLMsaaSharing;

  CONST_FIELD cl_context context;
  CONST_FIELD cl_command_queue commandQueue;
  CONST_FIELD cl_device_id device;
};

//////////////////////////////////////////////////////////////////////////

static void mGpuComputeContext_Destroy(mGpuComputeContext *pContext);

static mFUNCTION(mOpenCL_Initialize);
static void mOpenCL_FreeLibrary();
static mFUNCTION(mOpenCL_PixelFormatToImageFormatDataType, const mPixelFormatMapping pixelFormat, OUT cl_channel_type *pValue);
static mFUNCTION(mOpenCL_PixelFormatToImageFormatChannelOrder, const mPixelFormatMapping pixelFormat, OUT cl_channel_order *pValue);
static mFUNCTION(mOpenCL_PixelFormatToImageFormat, const mPixelFormatMapping pixelFormat, OUT cl_image_format *pImageFormat);

//////////////////////////////////////////////////////////////////////////

static mFUNCTION(mGpuComputeContext_CreateOpenCLContextNoOpenGL, IN mGpuComputeContext *pContext, IN const cl_device_id *pDeviceID, const cl_uint deviceVendorId);
static mFUNCTION(mGpuComputeContext_AllocateAndRetrieveDevice, IN mAllocator *const pAllocator, OUT cl_context_properties openGlContextProperties[7], OUT cl_device_id *pDeviceID, bool withOpenGl);

//////////////////////////////////////////////////////////////////////////

mFUNCTION(mGpuComputeContext_Create, OUT mPtr<mGpuComputeContext> *pGpuComputeContext, IN mAllocator *pAllocator, const bool enableRendererSharing)
{
  mFUNCTION_SETUP();

  mERROR_IF(pGpuComputeContext == nullptr, mR_ArgumentNull);

  mERROR_CHECK(mOpenCL_Initialize());

  cl_context_properties openGlContextProperties[7];
  cl_device_id deviceID = nullptr;
  cl_int ret = 0;

  mERROR_CHECK(mGpuComputeContext_AllocateAndRetrieveDevice(pAllocator, openGlContextProperties, &deviceID, enableRendererSharing));

  mDEFER_CALL_ON_ERROR(pGpuComputeContext, mSharedPointer_Destroy);
  mERROR_CHECK((mSharedPointer_Allocate<mGpuComputeContext>(pGpuComputeContext, pAllocator, mGpuComputeContext_Destroy, 1)));
  mGpuComputeContext *pContext = pGpuComputeContext->GetPointer();

  pContext->openGlEnabled = true;

  cl_uint deviceVendorId = 0;
  clGetDeviceInfo(deviceID, CL_DEVICE_VENDOR_ID, sizeof(deviceVendorId), &deviceVendorId, nullptr);

  if (enableRendererSharing)
  {
    char extentions[2048] = ""; // 1024 is not sufficient on some integrated Intel GPUs.
    size_t extentionLength = 0;
    ret = clGetDeviceInfo(deviceID, CL_DEVICE_EXTENSIONS, mARRAYSIZE(extentions), &extentions, &extentionLength);

    const char openglextention[] = "cl_khr_gl_sharing";
    const char opengleventextention[] = "cl_khr_gl_event";
    const char openglmsaasharingextention[] = "cl_khr_gl_msaa_sharing";

    size_t index = 0;

    while (true)
    {
      if (extentionLength - index < mARRAYSIZE(openglextention) - 1 && extentionLength - index < mARRAYSIZE(opengleventextention) - 1)
        break;

      // Look for "cl_khr_gl_sharing".
      if (!pContext->supportsOpenGLSharing && extentionLength - index >= mARRAYSIZE(openglextention) - 1)
      {
        for (size_t i = 0; i < mARRAYSIZE(openglextention) - 1; ++i)
          if (extentions[index + i] != openglextention[i])
            goto not_contained_cl_khr_gl_sharing;

        pContext->supportsOpenGLSharing = true;

      not_contained_cl_khr_gl_sharing:;
      }
      // Look for "cl_khr_gl_event".
      else if (!pContext->supportsOpenGLEvent && extentionLength - index >= mARRAYSIZE(opengleventextention) - 1)
      {
        for (size_t i = 0; i < mARRAYSIZE(opengleventextention) - 1; ++i)
          if (extentions[index + i] != opengleventextention[i])
            goto not_contained_cl_khr_gl_event;

        pContext->supportsOpenGLEvent = true;

      not_contained_cl_khr_gl_event:;
      }
      // Look for "cl_khr_gl_event".
      else if (!pContext->supportsOpenGLMsaaSharing && extentionLength - index >= mARRAYSIZE(openglmsaasharingextention) - 1)
      {
        for (size_t i = 0; i < mARRAYSIZE(openglmsaasharingextention) - 1; ++i)
          if (extentions[index + i] != openglmsaasharingextention[i])
            goto not_contained_cl_khr_gl_msaa_sharing;

        pContext->supportsOpenGLMsaaSharing = true;

      not_contained_cl_khr_gl_msaa_sharing:;
      }

      // All extentions found?
      if (pContext->supportsOpenGLEvent && pContext->supportsOpenGLSharing && pContext->supportsOpenGLMsaaSharing)
        break;

      // Move to next ' '.
      for (++index; index < extentionLength; ++index)
        if (extentions[index] == ' ')
          break;

      // Move one char further.
      ++index;

      // End of string?
      if (index + 1 >= extentionLength)
        break;
    }

    mERROR_IF(!pContext->supportsOpenGLSharing, mR_ResourceIncompatible);

    // Sometimes the OpenCL driver seems to not respond and the program just hangs forever. In this case we need to forcefully kill the thread that is requesting an OpenCL context and return an error code.
    {
      const GLenum glerror = glGetError();

      mERROR_IF(glerror != GL_NO_ERROR, mR_RenderingError);

      HGLRC hglrc;
      mERROR_CHECK(mRenderParams_GetCurrentGLContext_HGLRC(&hglrc));

      HDC hdc;
      mERROR_CHECK(mRenderParams_GetCurrentGLContext_HDC(&hdc));

      if (deviceVendorId != 0x8086) // Not Intel (because their driver would crash, yay!)
      {
        ret = CL_SUCCESS;

        const std::function<void()> &asyncTask = [&]()
        {
#ifdef _WIN32
          wglMakeCurrent(hdc, hglrc);

          pContext->context = clCreateContext(openGlContextProperties, 1, &deviceID, nullptr, nullptr, &ret);
#else
          pContext->context = clCreateContext(openGlContextProperties, 1, &deviceID, nullptr, nullptr, &ret);
#endif
        };

        std::thread contextCreationThread(asyncTask);

        auto future = std::async(std::launch::async, &std::thread::join, &contextCreationThread);

        if (future.wait_for(std::chrono::seconds(5)) == std::future_status::timeout)
        {
#ifdef _WIN32
          TerminateThread(contextCreationThread.native_handle(), 0);
#elif defined(__unix__)
          pthread_cancel(contextCreationThread.native_handle());
#else
#error "Canceling a thread is not supported for this platform." 
#endif
          mRETURN_RESULT(mR_Timeout);
        }
      }

      if (deviceVendorId == 0x8086 || ret == CL_INVALID_GL_SHAREGROUP_REFERENCE_KHR) // If Intel or failed.
      {
        if (pContext->context != nullptr)
        {
          clReleaseContext(pContext->context);
          pContext->context = nullptr;
        }

        pContext->context = clCreateContext(openGlContextProperties, 1, &deviceID, nullptr, nullptr, &ret);
      }

      mERROR_IF(!pContext->context || CL_SUCCESS != ret, mR_InternalError);
    }

    const GLenum glerror = glGetError();

    mERROR_IF(glerror != GL_NO_ERROR, mR_RenderingError);

    pContext->openGlEnabled = true;
  }
  else
  {
    mERROR_CHECK(mGpuComputeContext_CreateOpenCLContextNoOpenGL(pContext, &deviceID, deviceVendorId));
  }

  // Store DeviceID.
  {
    pContext->device = deviceID;
  }

  // Create Command Queue.
  {
    pContext->commandQueue = clCreateCommandQueue(pContext->context, pContext->device, 0, &ret);

    mERROR_IF(!pContext->commandQueue, mR_InternalError);
  }

  mRETURN_SUCCESS();
}

mFUNCTION(mGpuComputeContext_CompleteQueue, OUT mPtr<mGpuComputeContext> &gpuComputeContext)
{
  mFUNCTION_SETUP();

  mERROR_IF(gpuComputeContext == nullptr, mR_ArgumentNull);

  const cl_int ret = clFinish(gpuComputeContext->commandQueue);

  mERROR_IF(ret != CL_SUCCESS, mR_InternalError);

  mRETURN_SUCCESS();
}

mFUNCTION(mGpuComputeContext_FinalizeRenderer, OUT mPtr<mGpuComputeContext> &gpuComputeContext)
{
  mFUNCTION_SETUP();

  mERROR_IF(gpuComputeContext == nullptr, mR_ArgumentNull);
  mERROR_IF(!gpuComputeContext->openGlEnabled, mR_Success);

  mGL_DEBUG_ERROR_CHECK();
  
  glFlush();

  mGL_DEBUG_ERROR_CHECK();

  glFinish();

  mGL_ERROR_CHECK();

  mRETURN_SUCCESS();
}

//////////////////////////////////////////////////////////////////////////

static void mGpuComputeContext_Destroy(mGpuComputeContext *pContext)
{
  cl_int ret = CL_SUCCESS;

  if (pContext->commandQueue)
    ret = clReleaseCommandQueue(pContext->commandQueue);

  if (pContext->context)
    ret = clReleaseContext(pContext->context);
}

static mFUNCTION(mGpuComputeContext_AllocateAndRetrieveDevice, IN mAllocator *const pAllocator, OUT cl_context_properties openGlContextProperties[7], OUT cl_device_id *pDeviceID, bool withOpenGl)
{
  mFUNCTION_SETUP();

  cl_uint openCLPlatformCount;
  cl_platform_id *pPlatforms = nullptr;
  cl_int ret = 0;

  size_t chosenPlatformIndex = 0;
  size_t currentPlatformIndex = 0;
  size_t deviceScore = 0;

  // Get platform and device information
  ret = clGetPlatformIDs(0, nullptr, &openCLPlatformCount);

  mERROR_IF(openCLPlatformCount == 0, mR_ResourceNotFound);

  mDEFER_CALL_2(mAllocator_FreePtr, pAllocator, &pPlatforms);
  mERROR_CHECK(mAllocator_Allocate(pAllocator, &pPlatforms, openCLPlatformCount));

  mERROR_IF(CL_SUCCESS != (ret = clGetPlatformIDs(openCLPlatformCount, pPlatforms, nullptr)), mR_ResourceNotFound);

  if (withOpenGl) // OpenGL will sadly not always use the primary OpenCL device.
  {
    HGLRC hglrc;
    mERROR_CHECK(mRenderParams_GetCurrentGLContext_HGLRC(&hglrc));

    HDC hdc;
    mERROR_CHECK(mRenderParams_GetCurrentGLContext_HDC(&hdc));

    const char *glVendor = reinterpret_cast<const char *>(glGetString(GL_VENDOR));

    enum VendorMapping
    {
      VM_AMD = 0x1002,
      VM_Nvidia = 0x10DE,
      VM_Intel = 0x8086,
      VM_Other = 0x0
    } openGlVendor = VM_Other;

    if (glVendor != nullptr)
    {
      if (strstr(glVendor, "NVIDIA") != nullptr || strstr(glVendor, "Nvidia") != nullptr)
        openGlVendor = VM_Nvidia;
      else if (strstr(glVendor, "ATI") != nullptr || strstr(glVendor, "AMD") != nullptr)
        openGlVendor = VM_AMD;
      else if (strstr(glVendor, "Intel") != nullptr)
        openGlVendor = VM_Intel;
    }

    do
    {
      cl_uint openCLDeviceCount;
      cl_device_id currentDeviceID = nullptr;

      ret = clGetDeviceIDs(pPlatforms[currentPlatformIndex], CL_DEVICE_TYPE_GPU, 1, &currentDeviceID, &openCLDeviceCount);

      if (ret != CL_SUCCESS || openCLDeviceCount == 0)
        if (CL_SUCCESS != (ret = clGetDeviceIDs(pPlatforms[chosenPlatformIndex], CL_DEVICE_TYPE_ALL, 1, &currentDeviceID, &openCLDeviceCount)))
          currentDeviceID = nullptr;

      if (currentDeviceID != nullptr)
      {
        cl_uint vendorID;
        clGetDeviceInfo(currentDeviceID, CL_DEVICE_VENDOR_ID, sizeof(vendorID), &vendorID, nullptr);

        if (currentPlatformIndex == 0 || vendorID == (cl_uint)openGlVendor)
        {
          chosenPlatformIndex = currentPlatformIndex;
          *pDeviceID = currentDeviceID;
        }
      }

      ++currentPlatformIndex;

    } while (currentPlatformIndex < openCLPlatformCount);

    cl_uint openCLDeviceCount;

    ret = clGetDeviceIDs(pPlatforms[chosenPlatformIndex], CL_DEVICE_TYPE_GPU, 1, pDeviceID, &openCLDeviceCount);

    if (ret != CL_SUCCESS || openCLDeviceCount == 0)
      mERROR_IF(CL_SUCCESS != (ret = clGetDeviceIDs(pPlatforms[chosenPlatformIndex], CL_DEVICE_TYPE_ALL, 1, pDeviceID, &openCLDeviceCount)), mR_ResourceNotFound);

#ifdef _WIN32
    openGlContextProperties[0] = CL_GL_CONTEXT_KHR;
    openGlContextProperties[1] = (cl_context_properties)hglrc; // HGLRC handle
    openGlContextProperties[2] = CL_WGL_HDC_KHR;
    openGlContextProperties[3] = (cl_context_properties)hdc; // HDC handle
    openGlContextProperties[4] = CL_CONTEXT_PLATFORM;
    openGlContextProperties[5] = (cl_context_properties)pPlatforms[chosenPlatformIndex];
    openGlContextProperties[6] = 0;
#elif defined(__linux__)
    //openGlContextProperties[0] = CL_GL_CONTEXT_KHR;
    //openGlContextProperties[1] = (cl_context_properties) * (int *)pGLContext; // GLXContext
    //openGlContextProperties[2] = CL_GLX_DISPLAY_KHR;
    //openGlContextProperties[3] = (cl_context_properties) * (int *)pGLDrawingContext; // GLXDrawable
    //openGlContextProperties[4] = CL_CONTEXT_PLATFORM;
    //openGlContextProperties[5] = (cl_context_properties)pPlatforms[chosenPlatformIndex];
    //openGlContextProperties[6] = 0;
#else
#error "setting the GL context is not supported for this platform." 
#endif

    if (clGetGLContextInfoKHR != nullptr)
    {
      cl_device_id openGlDeviceId;

      mERROR_IF(CL_SUCCESS != (ret = clGetGLContextInfoKHR(openGlContextProperties, CL_CURRENT_DEVICE_FOR_GL_CONTEXT_KHR, sizeof(openGlDeviceId), &openGlDeviceId, nullptr)), mR_InternalError);

      *pDeviceID = openGlDeviceId;
    }
  }
  else
  {
    do
    {
      cl_uint openCLDeviceCount;
      cl_device_id currentDeviceID = nullptr;

      ret = clGetDeviceIDs(pPlatforms[currentPlatformIndex], CL_DEVICE_TYPE_GPU, 1, &currentDeviceID, &openCLDeviceCount);

      if (ret != CL_SUCCESS || openCLDeviceCount == 0)
        if (CL_SUCCESS != (ret = clGetDeviceIDs(pPlatforms[chosenPlatformIndex], CL_DEVICE_TYPE_ALL, 1, &currentDeviceID, &openCLDeviceCount)))
          currentDeviceID = nullptr;

      if (currentDeviceID != nullptr)
      {
        cl_uint vendorID;
        clGetDeviceInfo(currentDeviceID, CL_DEVICE_VENDOR_ID, sizeof(vendorID), &vendorID, nullptr);

        size_t currentScore = 0;

        switch (vendorID)
        {
        case 0x8086: // Intel.
        default: // Unknown.
          currentScore = 0;
          break;

        case 0x1002: // AMD / ATI.
          currentScore = 1;
          break;

        case 0x10DE: // Nvidia.
          currentScore = 2;
          break;
        }

        if (currentPlatformIndex == 0 || deviceScore < currentScore)
        {
          chosenPlatformIndex = currentPlatformIndex;
          *pDeviceID = currentDeviceID;
          deviceScore = currentScore;
        }
      }

      ++currentPlatformIndex;

    } while (currentPlatformIndex < openCLPlatformCount);
  }

  mERROR_IF(*pDeviceID == nullptr, mR_ResourceIncompatible);

  mRETURN_SUCCESS();
}

static mFUNCTION(mGpuComputeContext_CreateOpenCLContextNoOpenGL, IN mGpuComputeContext *pContext, IN const cl_device_id *pDeviceID, const cl_uint deviceVendorId)
{
  mFUNCTION_SETUP();

  cl_int ret = 0;

  if (deviceVendorId != 0x8086) // Not Intel (because their driver would crash, yay!)
  {
    // Sometimes the OpenCL driver seems to not respond and the program just hangs forever. In this case we need to forcefully kill the thread that is requesting an OpenCL context and return an error code.
    {
      std::thread contextCreationThread([&]()
        {
          pContext->context = clCreateContext(nullptr, 1, pDeviceID, nullptr, nullptr, &ret);
        });

      auto future = std::async(std::launch::async, &std::thread::join, &contextCreationThread);

      if (future.wait_for(std::chrono::seconds(5)) == std::future_status::timeout)
      {
#ifdef _WIN32
        TerminateThread(contextCreationThread.native_handle(), 0);
#elif defined(__unix__)
        pthread_cancel(contextCreationThread.native_handle());
#else
#error "Canceling a thread is not supported for this platform." 
#endif
        mRETURN_RESULT(mR_Timeout);
      }
    }
  }
  else // if Intel.
  {
    pContext->context = clCreateContext(nullptr, 1, pDeviceID, nullptr, nullptr, &ret);
  }

  mERROR_IF(ret != CL_SUCCESS || !pContext->context, mR_InternalError);

  mRETURN_SUCCESS();
}

//////////////////////////////////////////////////////////////////////////

struct mGpuComputeEvent
{
  mPtr<mGpuComputeContext> context;
  cl_event _event;
};

static void mGpuComputeEvent_Destroy(mGpuComputeEvent *pEvent);

//////////////////////////////////////////////////////////////////////////


mFUNCTION(mGpuComputeEvent_Await, mPtr<mGpuComputeEvent> &_event)
{
  mFUNCTION_SETUP();

  mERROR_IF(_event == nullptr, mR_ArgumentNull);
  mERROR_IF(_event->_event == nullptr, mR_ResourceStateInvalid);

  const cl_int ret = clWaitForEvents(1, &_event->_event);

  mERROR_IF(ret != CL_SUCCESS, mR_InternalError);

  mRETURN_SUCCESS();
}

//////////////////////////////////////////////////////////////////////////

static mFUNCTION(mGpuComputeEvent_Create, OUT mPtr<mGpuComputeEvent> *pEvent, IN mAllocator *pAllocator, mPtr<mGpuComputeContext> &context, const cl_event _event)
{
  mFUNCTION_SETUP();

  mERROR_IF(pEvent == nullptr || context == nullptr || _event == nullptr, mR_ArgumentNull);

  mDEFER_CALL_ON_ERROR(pEvent, mSharedPointer_Destroy);
  mERROR_CHECK(mSharedPointer_Allocate<mGpuComputeEvent>(pEvent, pAllocator, mGpuComputeEvent_Destroy, 1));

  (*pEvent)->context = context;
  (*pEvent)->_event = _event;

  mRETURN_SUCCESS();
}

static void mGpuComputeEvent_Destroy(mGpuComputeEvent *pEvent)
{
  if (pEvent->_event != nullptr)
  {
    const cl_int ret = clReleaseEvent(pEvent->_event);

    if (ret != CL_SUCCESS)
      mPRINT_ERROR("Failed to release OpenCL event with error code 0x", mFX()(ret), ".");

    pEvent->_event = nullptr;
  }

  mSharedPointer_Destroy(&pEvent->context);
}

//////////////////////////////////////////////////////////////////////////

enum mGpuComputeBuffer_Type
{
  mGCB_T_DataBuffer,
  mGCB_T_Texture2D,
  mGCB_T_Texture3D,
  mGCB_T_Texture2DOpenGL,
  mGCB_T_Texture3DOpenGL,
};

struct mGpuComputeBuffer
{
  CONST_FIELD mPtr<mGpuComputeContext> context;
  CONST_FIELD mGpuComputeBuffer_Type type;
  CONST_FIELD mGpuComputeBuffer_ReadWriteConfiguration rwconfig;

  CONST_FIELD cl_mem handle;

  mPixelFormatMapping pixelFormat;
  mVec3s resolution;

  CONST_FIELD mPtr<mTexture> openGlTexture2D;
  CONST_FIELD mPtr<mTexture3D> openGlTexture3D;

  bool currentlyEnqueuedToBeAcquired;
};

static void mGpuComputeBuffer_Destroy(mGpuComputeBuffer *pBuffer);

//////////////////////////////////////////////////////////////////////////

mFUNCTION(mGpuComputeBuffer_CreateDataBuffer, OUT mPtr<mGpuComputeBuffer> *pGpuBuffer, IN mAllocator *pAllocator, mPtr<mGpuComputeContext> &context, const mGpuComputeBuffer_ReadWriteConfiguration rwconfig, const size_t bytes, OPTIONAL IN const void *pDataToCopy /* = nullptr */)
{
  mFUNCTION_SETUP();

  mERROR_IF(pGpuBuffer == nullptr || context == nullptr, mR_ArgumentNull);

  mDEFER_CALL_ON_ERROR(pGpuBuffer, mSharedPointer_Destroy);
  mERROR_CHECK(mSharedPointer_Allocate<mGpuComputeBuffer>(pGpuBuffer, pAllocator, mGpuComputeBuffer_Destroy, 1));
  mGpuComputeBuffer *pBuffer = pGpuBuffer->GetPointer();

  pBuffer->context = context;
  pBuffer->type = mGCB_T_DataBuffer;
  pBuffer->resolution = mVec3s(bytes, 0, 0);
  pBuffer->pixelFormat = mPF_Monochrome8;
  pBuffer->rwconfig = rwconfig;

  mAllocator *pTempAllocator = &mDefaultTempAllocator;
  uint8_t *pTmpData = nullptr;
  mDEFER_CALL_2(mAllocator_FreePtr, pTempAllocator, &pTmpData);

  if (rwconfig == mGCB_RWC_Read && pDataToCopy == nullptr)
    mERROR_CHECK(mAllocator_Allocate(pTempAllocator, &pTmpData, bytes));

  size_t rwFlag = 0;

  switch (rwconfig)
  {
  case mGCB_RWC_Read:
    rwFlag = CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR;
    break;

  case mGCB_RWC_Write:
    rwFlag = CL_MEM_WRITE_ONLY;
    
    mERROR_IF(pDataToCopy != nullptr, mR_InvalidParameter);

    break;

  case mGCB_RWC_ReadWrite:
    rwFlag = CL_MEM_READ_WRITE;

    if (pDataToCopy != nullptr)
      rwFlag = CL_MEM_COPY_HOST_PTR;
    
    break;
  }

  const void *pPtr = reinterpret_cast<const void *>((size_t)pDataToCopy | (size_t)pTmpData); // `pTmpData` will be nullptr if pDataToCopy isn't nullptr and we require data to copy (on some drivers).

  cl_int ret = CL_SUCCESS;

  pBuffer->handle = clCreateBuffer(pBuffer->context->context, rwFlag, bytes, const_cast<void *>(pPtr), &ret); // yes, the `const_cast` is terrible, but OpenCL technically allows other parameters (`CL_MEM_USE_HOST_PTR`) that would require pPtr to not be const.

  mERROR_IF(pBuffer->handle == nullptr || ret != CL_SUCCESS, mR_InternalError);

  mRETURN_SUCCESS();
}

mFUNCTION(mGpuComputeBuffer_CreateTexture2D, OUT mPtr<mGpuComputeBuffer> *pGpuBuffer, IN mAllocator *pAllocator, mPtr<mGpuComputeContext> &context, const mGpuComputeBuffer_ReadWriteConfiguration rwconfig, const mVec2s resolution, const mPixelFormatMapping pixelFormat, OPTIONAL const void *pDataToCopy /* = nullptr */)
{
  mFUNCTION_SETUP();

  mERROR_IF(pGpuBuffer == nullptr || context == nullptr, mR_ArgumentNull);
  
  cl_image_format imageFormat;
  mERROR_CHECK(mOpenCL_PixelFormatToImageFormat(pixelFormat, &imageFormat));

  mDEFER_CALL_ON_ERROR(pGpuBuffer, mSharedPointer_Destroy);
  mERROR_CHECK(mSharedPointer_Allocate<mGpuComputeBuffer>(pGpuBuffer, pAllocator, mGpuComputeBuffer_Destroy, 1));
  mGpuComputeBuffer *pBuffer = pGpuBuffer->GetPointer();

  pBuffer->context = context;
  pBuffer->type = mGCB_T_Texture2D;
  pBuffer->resolution = mVec3s(resolution, 0);
  pBuffer->pixelFormat = pixelFormat;
  pBuffer->rwconfig = rwconfig;

  mAllocator *pTempAllocator = &mDefaultTempAllocator;
  uint8_t *pTmpData = nullptr;
  mDEFER_CALL_2(mAllocator_FreePtr, pTempAllocator, &pTmpData);

  if (rwconfig == mGCB_RWC_Read && pDataToCopy == nullptr)
  {
    size_t bytes = 0;
    mERROR_CHECK(mPixelFormat_GetSize(pixelFormat.basePixelFormat, resolution, &bytes));

    mERROR_CHECK(mAllocator_Allocate(pTempAllocator, &pTmpData, bytes));
  }

  size_t rwFlag = 0;

  switch (rwconfig)
  {
  case mGCB_RWC_Read:
    rwFlag = CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR;
    break;

  case mGCB_RWC_Write:
    rwFlag = CL_MEM_WRITE_ONLY;

    mERROR_IF(pDataToCopy != nullptr, mR_InvalidParameter);

    break;

  case mGCB_RWC_ReadWrite:
    rwFlag = CL_MEM_READ_WRITE;

    if (pDataToCopy != nullptr)
      rwFlag = CL_MEM_COPY_HOST_PTR;

    break;
  }

  const void *pPtr = reinterpret_cast<const void *>((size_t)pDataToCopy | (size_t)pTmpData); // `pTmpData` will be nullptr if pDataToCopy isn't nullptr and we require data to copy (on some drivers).

  cl_int ret = CL_SUCCESS;

  pBuffer->handle = clCreateImage2D(pBuffer->context->context, rwFlag, &imageFormat, resolution.x, resolution.y, 0, const_cast<void *>(pPtr), &ret); // yes, the `const_cast` is terrible, but OpenCL technically allows other parameters (`CL_MEM_USE_HOST_PTR`) that would require pPtr to not be const.

  mERROR_IF(pBuffer->handle == nullptr || ret != CL_SUCCESS, mR_InternalError);

  mRETURN_SUCCESS();
}

mFUNCTION(mGpuComputeBuffer_CreateTexture3D, OUT mPtr<mGpuComputeBuffer> *pGpuBuffer, IN mAllocator *pAllocator, mPtr<mGpuComputeContext> &context, const mGpuComputeBuffer_ReadWriteConfiguration rwconfig, const mVec3s resolution, const mPixelFormatMapping pixelFormat, OPTIONAL const void *pDataToCopy /* = nullptr */)
{
  mFUNCTION_SETUP();

  mERROR_IF(pGpuBuffer == nullptr || context == nullptr, mR_ArgumentNull);
  
  cl_image_format imageFormat;
  mERROR_CHECK(mOpenCL_PixelFormatToImageFormat(pixelFormat, &imageFormat));

  mDEFER_CALL_ON_ERROR(pGpuBuffer, mSharedPointer_Destroy);
  mERROR_CHECK(mSharedPointer_Allocate<mGpuComputeBuffer>(pGpuBuffer, pAllocator, mGpuComputeBuffer_Destroy, 1));
  mGpuComputeBuffer *pBuffer = pGpuBuffer->GetPointer();

  pBuffer->context = context;
  pBuffer->type = mGCB_T_Texture3D;
  pBuffer->resolution = resolution;
  pBuffer->pixelFormat = pixelFormat;
  pBuffer->rwconfig = rwconfig;

  mAllocator *pTempAllocator = &mDefaultTempAllocator;
  uint8_t *pTmpData = nullptr;
  mDEFER_CALL_2(mAllocator_FreePtr, pTempAllocator, &pTmpData);

  if (rwconfig == mGCB_RWC_Read && pDataToCopy == nullptr)
  {
    size_t bytes = 0;
    mERROR_CHECK(mPixelFormat_GetSize(pixelFormat.basePixelFormat, resolution, &bytes));

    mERROR_CHECK(mAllocator_Allocate(pTempAllocator, &pTmpData, bytes));
  }

  size_t rwFlag = 0;

  switch (rwconfig)
  {
  case mGCB_RWC_Read:
    rwFlag = CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR;
    break;

  case mGCB_RWC_Write:
    rwFlag = CL_MEM_WRITE_ONLY;

    mERROR_IF(pDataToCopy != nullptr, mR_InvalidParameter);

    break;

  case mGCB_RWC_ReadWrite:
    rwFlag = CL_MEM_READ_WRITE;

    if (pDataToCopy != nullptr)
      rwFlag = CL_MEM_COPY_HOST_PTR;

    break;
  }

  const void *pPtr = reinterpret_cast<const void *>((size_t)pDataToCopy | (size_t)pTmpData); // `pTmpData` will be nullptr if pDataToCopy isn't nullptr and we require data to copy (on some drivers).

  cl_int ret = CL_SUCCESS;

  pBuffer->handle = clCreateImage3D(pBuffer->context->context, rwFlag, &imageFormat, resolution.x, resolution.y, resolution.z, 0, 0, const_cast<void *>(pPtr), &ret); // yes, the `const_cast` is terrible, but OpenCL technically allows other parameters (`CL_MEM_USE_HOST_PTR`) that would require pPtr to not be const.

  mERROR_IF(pBuffer->handle == nullptr || ret != CL_SUCCESS, mR_InternalError);

  mRETURN_SUCCESS();
}

mFUNCTION(mGpuComputeBuffer_CreateTexture2DFromRendererTexture, OUT mPtr<mGpuComputeBuffer> *pGpuBuffer, IN mAllocator *pAllocator, mPtr<mGpuComputeContext> &context, const mGpuComputeBuffer_ReadWriteConfiguration rwconfig, mPtr<mTexture> &rendererTexture, const mPixelFormatMapping pixelFormat)
{
  mFUNCTION_SETUP();

  mERROR_IF(pGpuBuffer == nullptr || context == nullptr || rendererTexture == nullptr, mR_ArgumentNull);
  mERROR_IF(rendererTexture->uploadState != mRP_US_Ready, mR_ResourceStateInvalid);
  mERROR_IF(rendererTexture->sampleCount > 1 && !context->supportsOpenGLMsaaSharing, mR_NotSupported);

  mDEFER_CALL_ON_ERROR(pGpuBuffer, mSharedPointer_Destroy);
  mERROR_CHECK(mSharedPointer_Allocate<mGpuComputeBuffer>(pGpuBuffer, pAllocator, mGpuComputeBuffer_Destroy, 1));
  mGpuComputeBuffer *pBuffer = pGpuBuffer->GetPointer();

  pBuffer->context = context;
  pBuffer->type = mGCB_T_Texture2DOpenGL;
  pBuffer->resolution = mVec3s(rendererTexture->resolution, 0);
  pBuffer->pixelFormat = pixelFormat;
  pBuffer->openGlTexture2D = rendererTexture;
  pBuffer->rwconfig = rwconfig;

  size_t rwFlag = 0;

  switch (rwconfig)
  {
  case mGCB_RWC_Read:
    rwFlag = CL_MEM_READ_ONLY;
    break;

  case mGCB_RWC_Write:
    rwFlag = CL_MEM_WRITE_ONLY;
    break;

  case mGCB_RWC_ReadWrite:
    rwFlag = CL_MEM_READ_WRITE;
    break;
  }

  cl_int ret = CL_SUCCESS;

  pBuffer->handle = clCreateFromGLTexture(pBuffer->context->context, rwFlag, rendererTexture->sampleCount > 1 ? GL_TEXTURE_2D_MULTISAMPLE : GL_TEXTURE_2D, 0, pBuffer->openGlTexture2D->textureId, &ret);

  mERROR_IF(pBuffer->handle == nullptr || ret != CL_SUCCESS, mR_InternalError);

  mRETURN_SUCCESS();
}

mFUNCTION(mGpuComputeBuffer_CreateTexture3DFromRendererTexture, OUT mPtr<mGpuComputeBuffer> *pGpuBuffer, IN mAllocator *pAllocator, mPtr<mGpuComputeContext> &context, const mGpuComputeBuffer_ReadWriteConfiguration rwconfig, mPtr<mTexture3D> &rendererTexture, const mPixelFormatMapping pixelFormat)
{
  mFUNCTION_SETUP();

  mERROR_IF(pGpuBuffer == nullptr || context == nullptr || rendererTexture == nullptr, mR_ArgumentNull);
  mERROR_IF(rendererTexture->uploadState != mRP_US_Ready, mR_ResourceStateInvalid);

  mDEFER_CALL_ON_ERROR(pGpuBuffer, mSharedPointer_Destroy);
  mERROR_CHECK(mSharedPointer_Allocate<mGpuComputeBuffer>(pGpuBuffer, pAllocator, mGpuComputeBuffer_Destroy, 1));
  mGpuComputeBuffer *pBuffer = pGpuBuffer->GetPointer();

  pBuffer->context = context;
  pBuffer->type = mGCB_T_Texture3DOpenGL;
  pBuffer->resolution = rendererTexture->resolution;
  pBuffer->pixelFormat = pixelFormat;
  pBuffer->openGlTexture3D = rendererTexture;
  pBuffer->rwconfig = rwconfig;

  size_t rwFlag = 0;

  switch (rwconfig)
  {
  case mGCB_RWC_Read:
    rwFlag = CL_MEM_READ_ONLY;
    break;

  case mGCB_RWC_Write:
    rwFlag = CL_MEM_WRITE_ONLY;
    break;

  case mGCB_RWC_ReadWrite:
    rwFlag = CL_MEM_READ_WRITE;
    break;
  }

  cl_int ret = CL_SUCCESS;

  pBuffer->handle = clCreateFromGLTexture(pBuffer->context->context, rwFlag, GL_TEXTURE_3D, 0, pBuffer->openGlTexture2D->textureId, &ret);

  mERROR_IF(pBuffer->handle == nullptr || ret != CL_SUCCESS, mR_InternalError);

  mRETURN_SUCCESS();
}

mFUNCTION(mGpuComputeBuffer_EnqueueAcquire, mPtr<mGpuComputeBuffer> &buffer, OUT OPTIONAL mPtr<mGpuComputeEvent> *pEvent /* = nullptr */, IN OPTIONAL mAllocator *pAllocator /* = nullptr */)
{
  mFUNCTION_SETUP();

  mERROR_IF(buffer == nullptr, mR_ArgumentNull);

  if (buffer->type == mGCB_T_Texture2DOpenGL || buffer->type == mGCB_T_Texture3DOpenGL)
  {
    mERROR_IF(buffer->currentlyEnqueuedToBeAcquired, mR_ResourceStateInvalid);

    cl_event _event = nullptr;
    mDEFER_CALL_ON_ERROR(_event, clReleaseEvent);

    const cl_int ret = clEnqueueAcquireGLObjects(buffer->context->commandQueue, 1, &buffer->handle, 0, nullptr, pEvent != nullptr ? &_event : nullptr);

    mERROR_IF(CL_SUCCESS != ret, mR_InternalError);

    if (pEvent != nullptr)
      mERROR_CHECK(mGpuComputeEvent_Create(pEvent, pAllocator, buffer->context, _event));

    buffer->currentlyEnqueuedToBeAcquired = true;
  }

  mRETURN_SUCCESS();
}

mFUNCTION(mGpuComputeBuffer_EnqueueRelease, mPtr<mGpuComputeBuffer> &buffer, OUT OPTIONAL mPtr<mGpuComputeEvent> *pEvent /* = nullptr */, IN OPTIONAL mAllocator *pAllocator /* = nullptr */)
{
  mFUNCTION_SETUP();

  mERROR_IF(buffer == nullptr, mR_ArgumentNull);

  if (buffer->type == mGCB_T_Texture2DOpenGL || buffer->type == mGCB_T_Texture3DOpenGL)
  {
    mERROR_IF(!buffer->currentlyEnqueuedToBeAcquired, mR_ResourceStateInvalid);

    cl_event _event = nullptr;
    mDEFER_CALL_ON_ERROR(_event, clReleaseEvent);

    const cl_int ret = clEnqueueReleaseGLObjects(buffer->context->commandQueue, 1, &buffer->handle, 0, nullptr, pEvent != nullptr ? &_event : nullptr);

    mERROR_IF(CL_SUCCESS != ret, mR_InternalError);

    if (pEvent != nullptr)
      mERROR_CHECK(mGpuComputeEvent_Create(pEvent, pAllocator, buffer->context, _event));

    buffer->currentlyEnqueuedToBeAcquired = false;
  }

  mRETURN_SUCCESS();
}

mFUNCTION(mGpuComputeBuffer_EnqueueWriteToBuffer, mPtr<mGpuComputeBuffer> &buffer, const void *pData, const size_t size, const size_t writeOffset /* = 0 */, const bool blocking /* = false */, OUT OPTIONAL mPtr<mGpuComputeEvent> *pEvent /* = nullptr */, IN OPTIONAL mAllocator *pAllocator /* = nullptr */)
{
  mFUNCTION_SETUP();

  mERROR_IF(buffer == nullptr || pData == nullptr, mR_ArgumentNull);
  mERROR_IF(blocking && pEvent != nullptr, mR_InvalidParameter);
  mERROR_IF(buffer->type != mGCB_T_DataBuffer, mR_ResourceIncompatible);

  cl_event _event = nullptr;
  mDEFER_CALL_ON_ERROR(_event, clReleaseEvent);

  const cl_int ret = clEnqueueWriteBuffer(buffer->context->commandQueue, buffer->handle, blocking ? CL_TRUE : CL_FALSE, writeOffset, size, pData, 0, nullptr, pEvent != nullptr ? &_event : nullptr);

  mERROR_IF(CL_SUCCESS != ret, mR_InternalError);

  if (pEvent != nullptr)
    mERROR_CHECK(mGpuComputeEvent_Create(pEvent, pAllocator, buffer->context, _event));

  mRETURN_SUCCESS();
}

mFUNCTION(mGpuComputeBuffer_EnqueueWriteToTexture2D, mPtr<mGpuComputeBuffer> &buffer, mPtr<mImageBuffer> &imageBuffer, const mVec2s writeOffset /* = mVec2s(0) */, const bool blocking /* = false */, OUT OPTIONAL mPtr<mGpuComputeEvent> *pEvent /* = nullptr */, IN OPTIONAL mAllocator *pAllocator /* = nullptr */)
{
  mFUNCTION_SETUP();

  mERROR_IF(buffer == nullptr || imageBuffer == nullptr, mR_ArgumentNull);
  mERROR_IF(blocking && pEvent != nullptr, mR_InvalidParameter);
  mERROR_IF(buffer->type != mGCB_T_Texture2D && buffer->type != mGCB_T_Texture2DOpenGL, mR_ResourceIncompatible);
  mERROR_IF(buffer->type == mGCB_T_Texture2DOpenGL && !buffer->currentlyEnqueuedToBeAcquired, mR_ResourceIncompatible);
  mERROR_IF(imageBuffer->pixelFormat != buffer->pixelFormat.basePixelFormat, mR_ResourceIncompatible);
  mERROR_IF(imageBuffer->currentSize.x + writeOffset.x > buffer->resolution.x, mR_ResourceIncompatible);
  mERROR_IF(imageBuffer->currentSize.y + writeOffset.y > buffer->resolution.y, mR_ResourceIncompatible);

  size_t pixelFormatUnitSize = 1;
  mERROR_CHECK(mPixelFormat_GetUnitSize(imageBuffer->pixelFormat, &pixelFormatUnitSize));

  cl_event _event = nullptr;
  mDEFER_CALL_ON_ERROR(_event, clReleaseEvent);

  const size_t origin[3] = { writeOffset.x, writeOffset.y, 0 };
  const size_t region[3] = { imageBuffer->currentSize.x, imageBuffer->currentSize.y, 1 };

  const cl_int ret = clEnqueueWriteImage(buffer->context->commandQueue, buffer->handle, blocking ? CL_TRUE : CL_FALSE, origin, region, imageBuffer->lineStride * pixelFormatUnitSize, 0, imageBuffer->pPixels, 0, nullptr, pEvent != nullptr ? &_event : nullptr);

  mERROR_IF(CL_SUCCESS != ret, mR_InternalError);

  if (pEvent != nullptr)
    mERROR_CHECK(mGpuComputeEvent_Create(pEvent, pAllocator, buffer->context, _event));

  mRETURN_SUCCESS();
}

mFUNCTION(mGpuComputeBuffer_EnqueueWriteToTexture2D, mPtr<mGpuComputeBuffer> &buffer, const void *pData, const mPixelFormat pixelFormat, const mVec2s writeSize, const mVec2s writeOffset /* = mVec2s(0) */, const size_t lineStrideBytes /* = 0 */, const bool blocking /* = false */, OUT OPTIONAL mPtr<mGpuComputeEvent> *pEvent /* = nullptr */, IN OPTIONAL mAllocator *pAllocator /* = nullptr */)
{
  mFUNCTION_SETUP();

  mERROR_IF(buffer == nullptr || pData == nullptr, mR_ArgumentNull);
  mERROR_IF(blocking && pEvent != nullptr, mR_InvalidParameter);
  mERROR_IF(buffer->type != mGCB_T_Texture2D && buffer->type != mGCB_T_Texture2DOpenGL, mR_ResourceIncompatible);
  mERROR_IF(buffer->type == mGCB_T_Texture2DOpenGL && !buffer->currentlyEnqueuedToBeAcquired, mR_ResourceIncompatible);
  mERROR_IF(pixelFormat != buffer->pixelFormat.basePixelFormat, mR_ResourceIncompatible);
  mERROR_IF(writeSize.x + writeOffset.x > buffer->resolution.x, mR_ResourceIncompatible);
  mERROR_IF(writeSize.y + writeOffset.y > buffer->resolution.y, mR_ResourceIncompatible);

  cl_event _event = nullptr;
  mDEFER_CALL_ON_ERROR(_event, clReleaseEvent);

  const size_t origin[3] = { writeOffset.x, writeOffset.y, 0 };
  const size_t region[3] = { writeSize.x, writeSize.y, 1 };

  const cl_int ret = clEnqueueWriteImage(buffer->context->commandQueue, buffer->handle, blocking ? CL_TRUE : CL_FALSE, origin, region, lineStrideBytes, 0, pData, 0, nullptr, pEvent != nullptr ? &_event : nullptr);

  mERROR_IF(CL_SUCCESS != ret, mR_InternalError);

  if (pEvent != nullptr)
    mERROR_CHECK(mGpuComputeEvent_Create(pEvent, pAllocator, buffer->context, _event));

  mRETURN_SUCCESS();
}

mFUNCTION(mGpuComputeBuffer_EnqueueWriteToTexture3D, mPtr<mGpuComputeBuffer> &buffer, const void *pData, const mPixelFormat pixelFormat, const mVec3s writeSize, const mVec3s writeOffset /* = mVec3s(0) */, const size_t lineStrideBytes /* = 0 */, const size_t rowStrideLines /* = 0 */, const bool blocking /* = false */, OUT OPTIONAL mPtr<mGpuComputeEvent> *pEvent /* = nullptr */, IN OPTIONAL mAllocator *pAllocator /* = nullptr */)
{
  mFUNCTION_SETUP();

  mERROR_IF(buffer == nullptr || pData == nullptr, mR_ArgumentNull);
  mERROR_IF(blocking && pEvent != nullptr, mR_InvalidParameter);
  mERROR_IF(buffer->type != mGCB_T_Texture3D && buffer->type != mGCB_T_Texture3DOpenGL, mR_ResourceIncompatible);
  mERROR_IF(buffer->type == mGCB_T_Texture3DOpenGL && !buffer->currentlyEnqueuedToBeAcquired, mR_ResourceIncompatible);
  mERROR_IF(pixelFormat != buffer->pixelFormat.basePixelFormat, mR_ResourceIncompatible);
  mERROR_IF(writeSize.x + writeOffset.x > buffer->resolution.x, mR_ResourceIncompatible);
  mERROR_IF(writeSize.y + writeOffset.y > buffer->resolution.y, mR_ResourceIncompatible);
  mERROR_IF(writeSize.z + writeOffset.z > buffer->resolution.z, mR_ResourceIncompatible);

  cl_event _event = nullptr;
  mDEFER_CALL_ON_ERROR(_event, clReleaseEvent);

  const size_t origin[3] = { writeOffset.x, writeOffset.y, writeOffset.z };
  const size_t region[3] = { writeSize.x, writeSize.y, writeSize.z };

  const cl_int ret = clEnqueueWriteImage(buffer->context->commandQueue, buffer->handle, blocking ? CL_TRUE : CL_FALSE, origin, region, lineStrideBytes, rowStrideLines, pData, 0, nullptr, pEvent != nullptr ? &_event : nullptr);

  mERROR_IF(CL_SUCCESS != ret, mR_InternalError);

  if (pEvent != nullptr)
    mERROR_CHECK(mGpuComputeEvent_Create(pEvent, pAllocator, buffer->context, _event));

  mRETURN_SUCCESS();
}

mFUNCTION(mGpuComputeBuffer_EnqueueReadFromBuffer, mPtr<mGpuComputeBuffer> &buffer, OUT void *pData, const size_t size, const size_t readOffset /* = 0 */, const bool blocking /* = false */, OUT OPTIONAL mPtr<mGpuComputeEvent> *pEvent /* = nullptr */, IN OPTIONAL mAllocator *pAllocator /* = nullptr */)
{
  mFUNCTION_SETUP();

  mERROR_IF(buffer == nullptr || pData == nullptr, mR_ArgumentNull);
  mERROR_IF(blocking && pEvent != nullptr, mR_InvalidParameter);
  mERROR_IF(buffer->type != mGCB_T_DataBuffer, mR_ResourceIncompatible);

  cl_event _event = nullptr;
  mDEFER_CALL_ON_ERROR(_event, clReleaseEvent);

  const cl_int ret = clEnqueueReadBuffer(buffer->context->commandQueue, buffer->handle, blocking ? CL_TRUE : CL_FALSE, readOffset, size, pData, 0, nullptr, pEvent != nullptr ? &_event : nullptr);

  mERROR_IF(CL_SUCCESS != ret, mR_InternalError);

  if (pEvent != nullptr)
    mERROR_CHECK(mGpuComputeEvent_Create(pEvent, pAllocator, buffer->context, _event));

  mRETURN_SUCCESS();
}

mFUNCTION(mGpuComputeBuffer_EnqueueReadFromTexture2D, mPtr<mGpuComputeBuffer> &buffer, OUT void *pData, const mPixelFormat pixelFormat, const mVec2s readSize, const mVec2s readOffset /* = mVec2s(0) */, const size_t lineStrideBytes /* = 0 */, const bool blocking /* = false */, OUT OPTIONAL mPtr<mGpuComputeEvent> *pEvent /* = nullptr */, IN OPTIONAL mAllocator *pAllocator /* = nullptr */)
{
  mFUNCTION_SETUP();

  mERROR_IF(buffer == nullptr || pData == nullptr, mR_ArgumentNull);
  mERROR_IF(blocking && pEvent != nullptr, mR_InvalidParameter);
  mERROR_IF(buffer->type != mGCB_T_Texture2D && buffer->type != mGCB_T_Texture2DOpenGL, mR_ResourceIncompatible);
  mERROR_IF(buffer->type == mGCB_T_Texture2DOpenGL && !buffer->currentlyEnqueuedToBeAcquired, mR_ResourceIncompatible);
  mERROR_IF(pixelFormat != buffer->pixelFormat.basePixelFormat, mR_ResourceIncompatible);
  mERROR_IF(readSize.x + readOffset.x > buffer->resolution.x, mR_ResourceIncompatible);
  mERROR_IF(readSize.y + readOffset.y > buffer->resolution.y, mR_ResourceIncompatible);

  cl_event _event = nullptr;
  mDEFER_CALL_ON_ERROR(_event, clReleaseEvent);

  const size_t origin[3] = { readOffset.x, readOffset.y, 0 };
  const size_t region[3] = { readSize.x, readSize.y, 1 };

  const cl_int ret = clEnqueueReadImage(buffer->context->commandQueue, buffer->handle, blocking ? CL_TRUE : CL_FALSE, origin, region, lineStrideBytes, 0, pData, 0, nullptr, pEvent != nullptr ? &_event : nullptr);

  mERROR_IF(CL_SUCCESS != ret, mR_InternalError);

  if (pEvent != nullptr)
    mERROR_CHECK(mGpuComputeEvent_Create(pEvent, pAllocator, buffer->context, _event));

  mRETURN_SUCCESS();
}

mFUNCTION(mGpuComputeBuffer_EnqueueReadFromTexture3D, mPtr<mGpuComputeBuffer> &buffer, OUT void *pData, const mPixelFormat pixelFormat, const mVec3s readSize, const mVec3s readOffset /* = mVec3s(0) */, const size_t lineStrideBytes /* = 0 */, const size_t rowStrideLines /* = 0 */, const bool blocking /* = false */, OUT OPTIONAL mPtr<mGpuComputeEvent> *pEvent /* = nullptr */, IN OPTIONAL mAllocator *pAllocator /* = nullptr */)
{
  mFUNCTION_SETUP();

  mERROR_IF(buffer == nullptr || pData == nullptr, mR_ArgumentNull);
  mERROR_IF(blocking && pEvent != nullptr, mR_InvalidParameter);
  mERROR_IF(buffer->type != mGCB_T_Texture3D && buffer->type != mGCB_T_Texture3DOpenGL, mR_ResourceIncompatible);
  mERROR_IF(buffer->type == mGCB_T_Texture3DOpenGL && !buffer->currentlyEnqueuedToBeAcquired, mR_ResourceIncompatible);
  mERROR_IF(pixelFormat != buffer->pixelFormat.basePixelFormat, mR_ResourceIncompatible);
  mERROR_IF(readSize.x + readOffset.x > buffer->resolution.x, mR_ResourceIncompatible);
  mERROR_IF(readSize.y + readOffset.y > buffer->resolution.y, mR_ResourceIncompatible);
  mERROR_IF(readSize.z + readOffset.z > buffer->resolution.z, mR_ResourceIncompatible);

  cl_event _event = nullptr;
  mDEFER_CALL_ON_ERROR(_event, clReleaseEvent);

  const size_t origin[3] = { readOffset.x, readOffset.y, readOffset.z };
  const size_t region[3] = { readSize.x, readSize.y, readSize.z };

  const cl_int ret = clEnqueueReadImage(buffer->context->commandQueue, buffer->handle, blocking ? CL_TRUE : CL_FALSE, origin, region, lineStrideBytes, rowStrideLines, pData, 0, nullptr, pEvent != nullptr ? &_event : nullptr);

  mERROR_IF(CL_SUCCESS != ret, mR_InternalError);

  if (pEvent != nullptr)
    mERROR_CHECK(mGpuComputeEvent_Create(pEvent, pAllocator, buffer->context, _event));

  mRETURN_SUCCESS();
}

mFUNCTION(mGpuComputeBuffer_EnqueueReadFromBuffer, mPtr<mGpuComputeBuffer> &buffer, OUT void **ppData, IN mAllocator *pDataAllocator, OPTIONAL OUT size_t *pBytes /* = nullptr */, const size_t readOffset /* = 0 */, const bool blocking /* = false */, OUT OPTIONAL mPtr<mGpuComputeEvent> *pEvent /* = nullptr */, IN OPTIONAL mAllocator *pAllocator /* = nullptr */)
{
  mFUNCTION_SETUP();

  mERROR_IF(buffer == nullptr || ppData == nullptr, mR_ArgumentNull);

  size_t bytes = 0;
  mERROR_CHECK(mPixelFormat_GetSize(buffer->pixelFormat.basePixelFormat, mVec2s(buffer->resolution.x - readOffset, 1), &bytes));

  if (pBytes != nullptr)
    *pBytes = bytes;

  uint8_t *pData = nullptr;
  mDEFER_CALL_2(mAllocator_FreePtr, pDataAllocator, &pData);
  mERROR_CHECK(mAllocator_Allocate(pDataAllocator, &pData, bytes));

  mERROR_CHECK(mGpuComputeBuffer_EnqueueReadFromBuffer(buffer, pData, bytes, readOffset, blocking, pEvent, pAllocator));

  *ppData = pData;
  pData = nullptr; // to prevent `mAllocator_FreePtr` in `mDEFER_CALL_2`.

  mRETURN_SUCCESS();
}

mFUNCTION(mGpuComputeBuffer_EnqueueReadFromTexture2D, mPtr<mGpuComputeBuffer> &buffer, OUT void **ppData, IN mAllocator *pDataAllocator, OPTIONAL OUT size_t *pBytes /* = nullptr */, const mVec2s readOffset /* = mVec2s(0) */, const bool blocking /* = false */, OUT OPTIONAL mPtr<mGpuComputeEvent> *pEvent /* = nullptr */, IN OPTIONAL mAllocator *pAllocator /* = nullptr */)
{
  mFUNCTION_SETUP();

  mERROR_IF(buffer == nullptr || ppData == nullptr, mR_ArgumentNull);

  size_t bytes = 0;
  mERROR_CHECK(mPixelFormat_GetSize(buffer->pixelFormat.basePixelFormat, buffer->resolution.ToVector2() - readOffset, &bytes));

  uint8_t *pData = nullptr;
  mDEFER_CALL_2(mAllocator_FreePtr, pDataAllocator, &pData);
  mERROR_CHECK(mAllocator_Allocate(pDataAllocator, &pData, bytes));

  mERROR_CHECK(mGpuComputeBuffer_EnqueueReadFromTexture2D(buffer, pData, buffer->pixelFormat.basePixelFormat, buffer->resolution.ToVector2() - readOffset, readOffset, 0, blocking, pEvent, pAllocator));

  *ppData = pData;
  pData = nullptr; // to prevent `mAllocator_FreePtr` in `mDEFER_CALL_2`.

  if (pBytes != nullptr)
    *pBytes = bytes;

  mRETURN_SUCCESS();
}

mFUNCTION(mGpuComputeBuffer_EnqueueReadFromTexture3D, mPtr<mGpuComputeBuffer> &buffer, OUT void **ppData, IN mAllocator *pDataAllocator, OPTIONAL OUT size_t *pBytes /* = nullptr */, const mVec3s readOffset /* = mVec3s(0) */, const bool blocking /* = false */, OUT OPTIONAL mPtr<mGpuComputeEvent> *pEvent /* = nullptr */, IN OPTIONAL mAllocator *pAllocator /* = nullptr */)
{
  mFUNCTION_SETUP();

  mERROR_IF(buffer == nullptr || ppData == nullptr, mR_ArgumentNull);

  size_t bytes = 0;
  mERROR_CHECK(mPixelFormat_GetSize(buffer->pixelFormat.basePixelFormat, buffer->resolution - readOffset, &bytes));

  uint8_t *pData = nullptr;
  mDEFER_CALL_2(mAllocator_FreePtr, pDataAllocator, &pData);
  mERROR_CHECK(mAllocator_Allocate(pDataAllocator, &pData, bytes));

  mERROR_CHECK(mGpuComputeBuffer_EnqueueReadFromTexture3D(buffer, pData, buffer->pixelFormat.basePixelFormat, buffer->resolution - readOffset, readOffset, 0, 0, blocking, pEvent, pAllocator));

  *ppData = pData;
  pData = nullptr; // to prevent `mAllocator_FreePtr` in `mDEFER_CALL_2`.

  if (pBytes != nullptr)
    *pBytes = bytes;

  mRETURN_SUCCESS();
}

mFUNCTION(mGpuComputeBuffer_GetPixelFormat, mPtr<mGpuComputeBuffer> &buffer, OUT mPixelFormatMapping *pPixelFormat)
{
  mFUNCTION_SETUP();

  mERROR_IF(buffer == nullptr || pPixelFormat == nullptr, mR_ArgumentNull);

  *pPixelFormat = buffer->pixelFormat;

  mRETURN_SUCCESS();
}

mFUNCTION(mGpuComputeBuffer_GetSize, mPtr<mGpuComputeBuffer> &buffer, OUT size_t *pSize)
{
  mFUNCTION_SETUP();

  mERROR_IF(buffer == nullptr || pSize == nullptr, mR_ArgumentNull);
  mERROR_IF(buffer->type != mGCB_T_DataBuffer, mR_ResourceIncompatible);

  *pSize = buffer->resolution.x;

  mRETURN_SUCCESS();
}

mFUNCTION(mGpuComputeBuffer_GetSize, mPtr<mGpuComputeBuffer> &buffer, OUT mVec2s *pSize)
{
  mFUNCTION_SETUP();

  mERROR_IF(buffer == nullptr || pSize == nullptr, mR_ArgumentNull);
  mERROR_IF(buffer->type != mGCB_T_Texture2D && buffer->type != mGCB_T_Texture2DOpenGL, mR_ResourceIncompatible);

  *pSize = buffer->resolution.ToVector2();

  mRETURN_SUCCESS();
}

mFUNCTION(mGpuComputeBuffer_GetSize, mPtr<mGpuComputeBuffer> &buffer, OUT mVec3s *pSize)
{
  mFUNCTION_SETUP();

  mERROR_IF(buffer == nullptr || pSize == nullptr, mR_ArgumentNull);
  mERROR_IF(buffer->type != mGCB_T_Texture3D && buffer->type != mGCB_T_Texture3DOpenGL, mR_ResourceIncompatible);

  *pSize = buffer->resolution;

  mRETURN_SUCCESS();
}

//////////////////////////////////////////////////////////////////////////

static void mGpuComputeBuffer_Destroy(mGpuComputeBuffer *pBuffer)
{
  cl_int ret = CL_SUCCESS;

  if (pBuffer->handle != nullptr)
    ret = clReleaseMemObject(pBuffer->handle);

  mSharedPointer_Destroy(&pBuffer->openGlTexture2D);
  mSharedPointer_Destroy(&pBuffer->openGlTexture3D);
  mSharedPointer_Destroy(&pBuffer->context);
}

//////////////////////////////////////////////////////////////////////////

struct mGpuComputeSampler
{
  mPtr<mGpuComputeContext> context;
  cl_sampler handle;
};

static void mGpuComputeSampler_Destroy(IN mGpuComputeSampler *pSampler);

//////////////////////////////////////////////////////////////////////////

mFUNCTION(mGpuComputeSampler_Create, OUT mPtr<mGpuComputeSampler> *pTextureSampler, IN mAllocator *pAllocator, mPtr<mGpuComputeContext> &context, const bool normalizedCoordinates, const mRenderParams_TextureWrapMode wrapMode, const mRenderParams_TextureMagnificationFilteringMode filterMode)
{
  mFUNCTION_SETUP();

  mERROR_IF(pTextureSampler == nullptr || context == nullptr, mR_ArgumentNull);

  cl_addressing_mode addressingMode = 0;

  switch (wrapMode)
  {
  case mRP_TWM_None:
    addressingMode = CL_ADDRESS_NONE;
    break;

  case mRP_TWM_Repeat:
    addressingMode = CL_ADDRESS_REPEAT;
    break;

  case mRP_TWM_ClampToBorder:
    addressingMode = CL_ADDRESS_CLAMP;
    break;

  case mRP_TWM_ClampToEdge:
    addressingMode = CL_ADDRESS_CLAMP_TO_EDGE;
    break;

  default:
    mRETURN_RESULT(mR_NotSupported);
  }

  cl_filter_mode openclFilterMode;

  switch (filterMode)
  {
  case mRP_TMagFM_BilinearInterpolation:
    openclFilterMode = CL_FILTER_LINEAR;
    break;

  case mRP_TMagFM_NearestNeighbor:
    openclFilterMode = CL_FILTER_NEAREST;
    break;

  default:
    mRETURN_RESULT(mR_NotSupported);
  }

  mDEFER_CALL_ON_ERROR(pTextureSampler, mSharedPointer_Destroy);
  mERROR_CHECK(mSharedPointer_Allocate<mGpuComputeSampler>(pTextureSampler, pAllocator, mGpuComputeSampler_Destroy, 1));
  mGpuComputeSampler *pSampler = pTextureSampler->GetPointer();

  pSampler->context = context;

  cl_int ret = CL_SUCCESS;

  pSampler->handle = clCreateSampler(pSampler->context->context, normalizedCoordinates ? CL_TRUE : CL_FALSE, addressingMode, openclFilterMode, &ret);

  mERROR_IF(pSampler->handle == nullptr || ret != CL_SUCCESS, mR_InternalError);

  mRETURN_SUCCESS();
}

//////////////////////////////////////////////////////////////////////////

static void mGpuComputeSampler_Destroy(IN mGpuComputeSampler *pSampler)
{
  cl_int ret = CL_SUCCESS;

  if (pSampler->handle != nullptr)
    ret = clReleaseSampler(pSampler->handle);

  mSharedPointer_Destroy(&pSampler->context);
}

//////////////////////////////////////////////////////////////////////////

struct mGpuComputeKernel
{
  mPtr<mGpuComputeContext> context;
  cl_kernel handle;
};

static void mGpuComputeKernel_Destroy(IN mGpuComputeKernel *pKernel);

//////////////////////////////////////////////////////////////////////////

mFUNCTION(mGpuComputeKernel_Create, OUT mPtr<mGpuComputeKernel> *pComputeKernel, IN mAllocator *pAllocator, mPtr<mGpuComputeContext> &context, const char *kernelName, const char *source, const size_t sourceLength)
{
  mFUNCTION_SETUP();

  mERROR_IF(pComputeKernel == nullptr || context == nullptr, mR_ArgumentNull);

  cl_program program = nullptr;
  mDEFER_CALL(program, clReleaseProgram);

  cl_int ret = CL_SUCCESS;
  program = clCreateProgramWithSource(context->context, 1, &source, &sourceLength, &ret);
  mERROR_IF(program == nullptr || ret != CL_SUCCESS, mR_InternalError);

  // Build the program.
  ret = clBuildProgram(program, 1, &context->device, nullptr, nullptr, nullptr);

  if (ret == CL_BUILD_PROGRAM_FAILURE || ret == -9999)
  {
    mPRINT_ERROR("Error compiling compute program.");

#ifndef GIT_BUILD
    mPRINT_ERROR(source);
#endif

    // Determine the size of the log
    size_t logSize = 0;
    ret = clGetProgramBuildInfo(program, context->device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &logSize);
    mERROR_IF(ret != CL_SUCCESS, mR_InternalError);

    // Allocate memory for the log
    char *log = nullptr;

    mAllocator *pTempAllocator = &mDefaultTempAllocator;
    mDEFER_CALL_2(mAllocator_FreePtr, pTempAllocator, &log);
    mERROR_CHECK(mAllocator_Allocate(pTempAllocator, &log, logSize));

    // Get the log
    ret = clGetProgramBuildInfo(program, context->device, CL_PROGRAM_BUILD_LOG, logSize, log, nullptr);
    mERROR_IF(ret != CL_SUCCESS, mR_InternalError);

    // Print the log
    mPRINT_ERROR("The following error occured:");
    mPRINT_ERROR(log);
    mRETURN_RESULT(mR_ResourceInvalid);
  }

  mERROR_IF(ret != CL_SUCCESS, mR_InternalError);

  mDEFER_CALL_ON_ERROR(pComputeKernel, mSharedPointer_Destroy);
  mERROR_CHECK(mSharedPointer_Allocate<mGpuComputeKernel>(pComputeKernel, pAllocator, mGpuComputeKernel_Destroy, 1));
  mGpuComputeKernel *pKernel = pComputeKernel->GetPointer();

  pKernel->context = context;
  pKernel->handle = clCreateKernel(program, kernelName, &ret);

  mERROR_IF(pKernel->handle == nullptr || ret != CL_SUCCESS, mR_InternalError);

  mRETURN_SUCCESS();
}

mFUNCTION(mGpuComputeKernel_SetArgAtIndex, mPtr<mGpuComputeKernel> &kernel, const uint32_t index, const uint32_t val)
{
  mFUNCTION_SETUP();

  mERROR_IF(kernel == nullptr, mR_ArgumentNull);

  cl_int ret = CL_SUCCESS;

  mERROR_IF(CL_SUCCESS != (ret = clSetKernelArg(kernel->handle, (cl_uint)index, sizeof(val), &val)), mR_InternalError);

  mRETURN_SUCCESS();
}

mFUNCTION(mGpuComputeKernel_SetArgAtIndex, mPtr<mGpuComputeKernel> &kernel, const uint32_t index, const int32_t val)
{
  mFUNCTION_SETUP();

  mERROR_IF(kernel == nullptr, mR_ArgumentNull);

  cl_int ret = CL_SUCCESS;

  mERROR_IF(CL_SUCCESS != (ret = clSetKernelArg(kernel->handle, (cl_uint)index, sizeof(val), &val)), mR_InternalError);

  mRETURN_SUCCESS();
}

mFUNCTION(mGpuComputeKernel_SetArgAtIndex, mPtr<mGpuComputeKernel> &kernel, const uint32_t index, const float_t val)
{
  mFUNCTION_SETUP();

  mERROR_IF(kernel == nullptr, mR_ArgumentNull);

  cl_int ret = CL_SUCCESS;

  mERROR_IF(CL_SUCCESS != (ret = clSetKernelArg(kernel->handle, (cl_uint)index, sizeof(val), &val)), mR_InternalError);

  mRETURN_SUCCESS();
}

mFUNCTION(mGpuComputeKernel_SetArgAtIndex, mPtr<mGpuComputeKernel> &kernel, const uint32_t index, const mPtr<mGpuComputeBuffer> &buffer)
{
  mFUNCTION_SETUP();

  mERROR_IF(kernel == nullptr || buffer == nullptr, mR_ArgumentNull);

  cl_int ret = CL_SUCCESS;

  mERROR_IF(CL_SUCCESS != (ret = clSetKernelArg(kernel->handle, (cl_uint)index, sizeof(buffer->handle), &buffer->handle)), mR_InternalError);

  mRETURN_SUCCESS();
}

mFUNCTION(mGpuComputeKernel_SetArgAtIndex, mPtr<mGpuComputeKernel> &kernel, const uint32_t index, const mPtr<mGpuComputeSampler> &sampler)
{
  mFUNCTION_SETUP();

  mERROR_IF(kernel == nullptr || sampler == nullptr, mR_ArgumentNull);

  const cl_int ret = clSetKernelArg(kernel->handle, (cl_uint)index, sizeof(sampler->handle), &sampler->handle);

  if (mFAILED(ret))
  {
    switch (ret)
    {
    case CL_INVALID_MEM_OBJECT:
      mRETURN_RESULT(mR_InvalidParameter);

    default:
      mRETURN_RESULT(mR_InternalError);
    }
  }

  mRETURN_SUCCESS();
}

mFUNCTION(mGpuComputeKernel_EnqueueExecution, mPtr<mGpuComputeKernel> &kernel, const size_t globalWorkSize, const size_t localWorkSize, const size_t globalWorkOffset /* = 0 */, OUT OPTIONAL mPtr<mGpuComputeEvent> *pEvent /* = nullptr */, IN OPTIONAL mAllocator *pAllocator /* = nullptr */)
{
  mFUNCTION_SETUP();

  mERROR_IF(kernel == nullptr, mR_ArgumentNull);

  cl_event _event = nullptr;
  mDEFER_CALL_ON_ERROR(_event, clReleaseEvent);

  const cl_int ret = clEnqueueNDRangeKernel(kernel->context->commandQueue, kernel->handle, 1, &globalWorkOffset, &globalWorkSize, localWorkSize == 0 ? nullptr : &localWorkSize, 0, nullptr, pEvent != nullptr ? &_event : nullptr);

  mERROR_IF(CL_SUCCESS != ret, mR_InternalError);

  if (pEvent != nullptr)
    mERROR_CHECK(mGpuComputeEvent_Create(pEvent, pAllocator, kernel->context, _event));

  mRETURN_SUCCESS();
}

mFUNCTION(mGpuComputeKernel_EnqueueExecution, mPtr<mGpuComputeKernel> &kernel, const mVec2s globalWorkSize, const mVec2s localWorkSize, const mVec2s globalWorkOffset /* = mVec2s(0) */, OUT OPTIONAL mPtr<mGpuComputeEvent> *pEvent /* = nullptr */, IN OPTIONAL mAllocator *pAllocator /* = nullptr */)
{
  mFUNCTION_SETUP();

  mERROR_IF(kernel == nullptr, mR_ArgumentNull);

  cl_event _event = nullptr;
  mDEFER_CALL_ON_ERROR(_event, clReleaseEvent);

  const cl_int ret = clEnqueueNDRangeKernel(kernel->context->commandQueue, kernel->handle, 2, globalWorkOffset.asArray, globalWorkSize.asArray, localWorkSize == mVec2s(0) ? nullptr : localWorkSize.asArray, 0, nullptr, pEvent != nullptr ? &_event : nullptr);

  if (ret != CL_SUCCESS)
  {
    switch (ret)
    {
    case CL_INVALID_KERNEL_ARGS:
      mRETURN_RESULT(mR_InvalidParameter);

    case CL_OUT_OF_RESOURCES:
      mRETURN_RESULT(mR_ResourceBusy);

    default:
      mRETURN_RESULT(mR_InternalError);
    }
  }

  if (pEvent != nullptr)
    mERROR_CHECK(mGpuComputeEvent_Create(pEvent, pAllocator, kernel->context, _event));

  mRETURN_SUCCESS();
}

mFUNCTION(mGpuComputeKernel_EnqueueExecution, mPtr<mGpuComputeKernel> &kernel, const mVec3s globalWorkSize, const mVec3s localWorkSize, const mVec3s globalWorkOffset /* = mVec3s(0) */, OUT OPTIONAL mPtr<mGpuComputeEvent> *pEvent /* = nullptr */, IN OPTIONAL mAllocator *pAllocator /* = nullptr */)
{
  mFUNCTION_SETUP();

  mERROR_IF(kernel == nullptr, mR_ArgumentNull);

  cl_event _event = nullptr;
  mDEFER_CALL_ON_ERROR(_event, clReleaseEvent);

  const cl_int ret = clEnqueueNDRangeKernel(kernel->context->commandQueue, kernel->handle, 3, globalWorkOffset.asArray, globalWorkSize.asArray, localWorkSize == mVec3s(0) ? nullptr : localWorkSize.asArray, 0, nullptr, pEvent != nullptr ? &_event : nullptr);

  mERROR_IF(CL_SUCCESS != ret, mR_InternalError);

  if (pEvent != nullptr)
    mERROR_CHECK(mGpuComputeEvent_Create(pEvent, pAllocator, kernel->context, _event));

  mRETURN_SUCCESS();
}

//////////////////////////////////////////////////////////////////////////

static void mGpuComputeKernel_Destroy(IN mGpuComputeKernel *pKernel)
{
  cl_int ret = CL_SUCCESS;

  if (pKernel->handle != nullptr)
    ret = clReleaseKernel(pKernel->handle);

  mSharedPointer_Destroy(&pKernel->context);
}

//////////////////////////////////////////////////////////////////////////

static mFUNCTION(mOpenCL_PixelFormatToImageFormat, const mPixelFormatMapping pixelFormat, OUT cl_image_format *pImageFormat)
{
  mFUNCTION_SETUP();

  bool hasSubBuffers = true;
  mERROR_CHECK(mPixelFormat_HasSubBuffers(pixelFormat.basePixelFormat, &hasSubBuffers));
  mERROR_IF(hasSubBuffers, mR_InvalidParameter);

  mERROR_CHECK(mOpenCL_PixelFormatToImageFormatChannelOrder(pixelFormat, &pImageFormat->image_channel_order));
  mERROR_CHECK(mOpenCL_PixelFormatToImageFormatDataType(pixelFormat, &pImageFormat->image_channel_data_type));

  mRETURN_SUCCESS();
}

static mFUNCTION(mOpenCL_PixelFormatToImageFormatChannelOrder, const mPixelFormatMapping pixelFormat, OUT cl_channel_order *pValue)
{
  mFUNCTION_SETUP();

  switch (pixelFormat.basePixelFormat)
  {
  case mPF_YUV444:
  case mPF_YUV422:
  case mPF_YUV440:
  case mPF_YUV420:
  case mPF_YUV411:
    mRETURN_RESULT(mR_NotSupported);
    break;

  case mPF_Monochrome8:
  case mPF_Monochrome16:
  case mPF_Monochromef16:
  case mPF_Monochromef32:
    *pValue = CL_R;
    break;

  case mPF_R8G8:
  case mPF_R16G16:
  case mPF_Rf16Gf16:
  case mPF_Rf32Gf32:
    *pValue = CL_RG;
    break;

  case mPF_R8G8B8:
  case mPF_R16G16B16:
  case mPF_Rf16Gf16Bf16:
  case mPF_Rf32Gf32Bf32:
    *pValue = CL_RGB;
    break;

  case mPF_R8G8B8A8:
  case mPF_R16G16B16A16:
  case mPF_Rf16Gf16Bf16Af16:
  case mPF_Rf32Gf32Bf32Af32:
    *pValue = CL_RGBA;
    break;

  case mPF_B8G8R8A8:
    *pValue = CL_BGRA;
    break;

  case mPF_B8G8R8:
  default:
    mRETURN_RESULT(mR_InvalidParameter);
  }

  mRETURN_SUCCESS();
}

static mFUNCTION(mOpenCL_PixelFormatToImageFormatDataType, const mPixelFormatMapping pixelFormat, OUT cl_channel_type *pValue)
{
  mFUNCTION_SETUP();

  switch (pixelFormat.basePixelFormat)
  {
  case mPF_YUV444:
  case mPF_YUV422:
  case mPF_YUV440:
  case mPF_YUV420:
  case mPF_YUV411:
  case mPF_R8G8B8:
  case mPF_B8G8R8:
  case mPF_R16G16B16:
  case mPF_Rf16Gf16:
  case mPF_Rf32Gf32Bf32:
  default:
    mRETURN_RESULT(mR_NotSupported);
    break;

  case mPF_Monochrome8:
  case mPF_R8G8:
  case mPF_R8G8B8A8:
  case mPF_B8G8R8A8:
    if (pixelFormat.isNormalized)
    {
      if (pixelFormat.isSigned)
        *pValue = CL_SNORM_INT8;
      else
        *pValue = CL_UNORM_INT8;
    }
    else
    {
      if (pixelFormat.isSigned)
        *pValue = CL_SIGNED_INT8;
      else
        *pValue = CL_UNSIGNED_INT8;
    }
    break;

  case mPF_Monochrome16:
  case mPF_R16G16:
  case mPF_R16G16B16A16:
    if (pixelFormat.isNormalized)
    {
      if (pixelFormat.isSigned)
        *pValue = CL_SNORM_INT16;
      else
        *pValue = CL_UNORM_INT16;
    }
    else
    {
      if (pixelFormat.isSigned)
        *pValue = CL_SIGNED_INT16;
      else
        *pValue = CL_UNSIGNED_INT16;
    }
    break;

  case mPF_Monochromef16:
  case mPF_Rf16Gf16Bf16:
  case mPF_Rf16Gf16Bf16Af16:
    *pValue = CL_HALF_FLOAT;
    break;

  case mPF_Monochromef32:
  case mPF_Rf32Gf32:
  case mPF_Rf32Gf32Bf32Af32:
    *pValue = CL_FLOAT;
    break;
  }

  mRETURN_SUCCESS();
}

//////////////////////////////////////////////////////////////////////////

static void mOpenCL_FreeLibrary()
{
#ifdef _WIN32
  if (mOpenCL_DllModule != nullptr)
    FreeLibrary(mOpenCL_DllModule);
#else
  if (mOpenCL_pDllModuleHandle != nullptr)
    dlclose(mOpenCL_pDllModuleHandle);
#endif
}

static mFUNCTION(mOpenCL_Initialize)
{
  mFUNCTION_SETUP();

  if (mOpenCL_Initialized)
    mRETURN_RESULT(mOpenCL_Available ? mR_Success : mR_ResourceNotFound);

  mDEFER_ON_ERROR(mOpenCL_Available = false);

#ifdef _WIN32
  mOpenCL_DllModule = LoadLibraryW(L"OpenCL.dll");

  mERROR_IF(mOpenCL_DllModule == nullptr, mR_ResourceNotFound);

#define HT_CODEC_LOAD_FROM_DLL(symbol, symbolName) HT_CODEC_DO_LOAD_FROM_DLL(symbol, symbolName, false)
#define HT_CODEC_TRY_LOAD_FROM_DLL(symbol, symbolName) HT_CODEC_DO_LOAD_FROM_DLL(symbol, symbolName, true)

#define HT_CODEC_DO_LOAD_FROM_DLL(symbol, symbolName, optional) \
  do \
  { symbol = (decltype(symbol))GetProcAddress(mOpenCL_DllModule, symbolName); \
    mERROR_IF(!optional && symbol == nullptr, mR_ResourceNotFound); \
  } while (0)
#else
  mOpenCL_pDllModuleHandle = dlopen("OpenCL.so", RTLD_LAZY);

  mERROR_IF(mOpenCL_pDllModuleHandle == nullptr, mR_ResourceNotFound);

#define HT_CODEC_DO_LOAD_FROM_DLL(symbol, symbolName, optional) \
  do \
  { symbol = (decltype(symbol))dlsym(mOpenCL_pDllModuleHandle, symbolName); \
    mERROR_IF(!optional && symbol == nullptr, mR_ResourceNotFound); \
  } while (0)
#endif

  atexit(mOpenCL_FreeLibrary);

  // TODO: Should we do anything differently with the symbols declared in `cl_gl.h`? Are they not required to be part of the dll?

  HT_CODEC_LOAD_FROM_DLL(clEnqueueNDRangeKernel, "clEnqueueNDRangeKernel");
  HT_CODEC_LOAD_FROM_DLL(clBuildProgram, "clBuildProgram");
  HT_CODEC_LOAD_FROM_DLL(clEnqueueReadImage, "clEnqueueReadImage");
  HT_CODEC_LOAD_FROM_DLL(clReleaseCommandQueue, "clReleaseCommandQueue");
  HT_CODEC_LOAD_FROM_DLL(clReleaseProgram, "clReleaseProgram");
  HT_CODEC_LOAD_FROM_DLL(clReleaseKernel, "clReleaseKernel");
  HT_CODEC_LOAD_FROM_DLL(clCreateCommandQueue, "clCreateCommandQueue");
  HT_CODEC_LOAD_FROM_DLL(clWaitForEvents, "clWaitForEvents");
  HT_CODEC_LOAD_FROM_DLL(clCreateFromGLTexture, "clCreateFromGLTexture");
  HT_CODEC_LOAD_FROM_DLL(clFinish, "clFinish");
  HT_CODEC_LOAD_FROM_DLL(clEnqueueWriteImage, "clEnqueueWriteImage");
  HT_CODEC_LOAD_FROM_DLL(clGetDeviceIDs, "clGetDeviceIDs");
  HT_CODEC_LOAD_FROM_DLL(clReleaseContext, "clReleaseContext");
  HT_CODEC_LOAD_FROM_DLL(clReleaseEvent, "clReleaseEvent");
  HT_CODEC_LOAD_FROM_DLL(clCreateContext, "clCreateContext");
  HT_CODEC_LOAD_FROM_DLL(clEnqueueReleaseGLObjects, "clEnqueueReleaseGLObjects");
  HT_CODEC_LOAD_FROM_DLL(clReleaseSampler, "clReleaseSampler");
  HT_CODEC_LOAD_FROM_DLL(clCreateSampler, "clCreateSampler");
  HT_CODEC_LOAD_FROM_DLL(clFlush, "clFlush");
  HT_CODEC_LOAD_FROM_DLL(clGetPlatformIDs, "clGetPlatformIDs");
  HT_CODEC_LOAD_FROM_DLL(clCreateImage2D, "clCreateImage2D");
  HT_CODEC_LOAD_FROM_DLL(clCreateImage3D, "clCreateImage3D");
  HT_CODEC_LOAD_FROM_DLL(clGetDeviceInfo, "clGetDeviceInfo");
  HT_CODEC_LOAD_FROM_DLL(clReleaseMemObject, "clReleaseMemObject");
  HT_CODEC_LOAD_FROM_DLL(clEnqueueAcquireGLObjects, "clEnqueueAcquireGLObjects");
  HT_CODEC_LOAD_FROM_DLL(clCreateKernel, "clCreateKernel");
  HT_CODEC_LOAD_FROM_DLL(clCreateProgramWithSource, "clCreateProgramWithSource");
  HT_CODEC_LOAD_FROM_DLL(clSetKernelArg, "clSetKernelArg");
  HT_CODEC_LOAD_FROM_DLL(clGetProgramBuildInfo, "clGetProgramBuildInfo");
  HT_CODEC_LOAD_FROM_DLL(clCreateBuffer, "clCreateBuffer");
  HT_CODEC_LOAD_FROM_DLL(clEnqueueReadBuffer, "clEnqueueReadBuffer");
  HT_CODEC_LOAD_FROM_DLL(clEnqueueWriteBuffer, "clEnqueueWriteBuffer");
  HT_CODEC_LOAD_FROM_DLL(clGetKernelWorkGroupInfo, "clGetKernelWorkGroupInfo");

  HT_CODEC_TRY_LOAD_FROM_DLL(clGetGLContextInfoKHR, "clGetGLContextInfoKHR");
  HT_CODEC_TRY_LOAD_FROM_DLL(clGetMemObjectInfo, "clGetMemObjectInfo");
  HT_CODEC_TRY_LOAD_FROM_DLL(clGetImageInfo, "clGetImageInfo");

  mOpenCL_Initialized = true;
  mOpenCL_Available = true;

  mRETURN_SUCCESS();
}
