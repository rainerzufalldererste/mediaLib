#include "mPointRenderer.h"

#include "mShader.h"
#include "mBinaryChunk.h"
#include "mInstancedRenderDataBuffer.h"

#ifdef GIT_BUILD // Define __M_FILE__
  #ifdef __M_FILE__
    #undef __M_FILE__
  #endif
  #define __M_FILE__ "ylpx4/kST4uu8w/FEi/KckGveMOQ+YjF1otTqtkzolMlKa7m+uDQgw0sjapOkS8F7Uz7tI8vfRsejWWI"
#endif

//////////////////////////////////////////////////////////////////////////

extern const char mPointRenderer_PositionAttribute[] = "position";
extern const char mPointRenderer_ScreenPosScaleAttribute[] = "screenPosScale";
extern const char mPointRenderer_ColourAttribute[] = "colour";

struct mPointRenderer
{
  mPtr<mBinaryChunk> blob;
  mColouredPointMode mode;
  mInstancedRenderDataBuffer<mRenderDataBuffer<mRDB_FloatAttribute<2, mPointRenderer_PositionAttribute>>, mRDB_FloatAttribute<3, mPointRenderer_ScreenPosScaleAttribute>, mRDB_FloatAttribute<4, mPointRenderer_ColourAttribute>> buffer;
  bool isStarted;
};

//////////////////////////////////////////////////////////////////////////

static mFUNCTION(mPointRenderer_Destroy_Internal, IN mPointRenderer *pPointRenderer);

//////////////////////////////////////////////////////////////////////////

mFUNCTION(mPointRenderer_Create, OUT mPtr<mPointRenderer> *pPointRenderer, IN mAllocator *pAllocator)
{
  mFUNCTION_SETUP();

  mERROR_IF(pPointRenderer == nullptr, mR_ArgumentNull);

  mDEFER_CALL_ON_ERROR(pPointRenderer, mSharedPointer_Destroy);
  mERROR_CHECK(mSharedPointer_Allocate(pPointRenderer, pAllocator, (std::function<void (mPointRenderer *)>)[](mPointRenderer *pData) {mPointRenderer_Destroy_Internal(pData);}, 1));

  mERROR_CHECK(mBinaryChunk_Create(&(*pPointRenderer)->blob, pAllocator));

#if defined(mRENDERER_OPENGL)
  const char vertexShader[] = mGLSL(
    attribute vec2 position;
    attribute vec3 screenPosScale;
    attribute vec4 colour;

    uniform vec2 _invScreenSize;

    out vec4 _colour;
    out vec2 _pos;
    out float _scale;

    void main()
    {
      _colour = colour;

      vec2 pos = position;

      pos -= vec2(0.5f);
      pos *= screenPosScale.z;

      _pos = pos;
      _scale = screenPosScale.z * 0.5;

      pos += screenPosScale.xy;
      pos *= _invScreenSize;

      pos.y = 1 - pos.y;

      gl_Position = vec4(pos * 2 - 1, 1, 1);
    }
  );
  
  const char fragmentShader[] = mGLSL(
    out vec4 colour;

    in vec4 _colour;
    in vec2 _pos;
    in float _scale;

    uniform float radiusFac;

    void main()
    {
      colour = _colour;
      colour.a *= clamp(_scale * radiusFac - length(_pos), 0, 1);
    }
  );

  mERROR_CHECK(mInstancedRenderDataBuffer_Create(&(*pPointRenderer)->buffer, pAllocator, vertexShader, fragmentShader));

  const float_t vertexData[] = { 0, 0, 1, 0, 0, 1, 1, 0, 1, 1, 0, 1, 0, 0 };
  mERROR_CHECK(mInstancedRenderDataBuffer_SetInstancedVertexBuffer((*pPointRenderer)->buffer, vertexData, mARRAYSIZE(vertexData)));
  mERROR_CHECK(mInstancedRenderDataBuffer_SetInstancedVertexCount((*pPointRenderer)->buffer, 6));
  mERROR_CHECK(mInstancedRenderDataBuffer_SetVertexRenderMode((*pPointRenderer)->buffer, mRP_VRM_TriangleList));

  (*pPointRenderer)->isStarted = false;

#else
  mRETURN_RESULT(mR_NotImplemented);
#endif

  mGL_DEBUG_ERROR_CHECK();

  mRETURN_SUCCESS();
}

mFUNCTION(mPointRenderer_Destroy, IN_OUT mPtr<mPointRenderer> *pPointRenderer)
{
  mFUNCTION_SETUP();

  mERROR_IF(pPointRenderer == nullptr, mR_ArgumentNull);

  mERROR_CHECK(mSharedPointer_Destroy(pPointRenderer));

  mRETURN_SUCCESS();
}

mFUNCTION(mPointRenderer_Begin, mPtr<mPointRenderer> &pointRenderer, const mColouredPointMode mode /* = mCPM_Circle */)
{
  mFUNCTION_SETUP();

  mERROR_IF(pointRenderer == nullptr, mR_ArgumentNull);

  mERROR_CHECK(mBinaryChunk_ResetWrite(pointRenderer->blob));

  pointRenderer->mode = mode;
  pointRenderer->isStarted = true;

  mGL_DEBUG_ERROR_CHECK();

  mRETURN_SUCCESS();
}

mFUNCTION(mPointRenderer_End, mPtr<mPointRenderer> &pointRenderer)
{
  mFUNCTION_SETUP();

  mERROR_IF(pointRenderer == nullptr, mR_ArgumentNull);

  pointRenderer->isStarted = false;

  if (pointRenderer->mode == mCPM_Circle)
    mERROR_CHECK(mRenderParams_SetBlendingEnabled(true));

  mPtr<mShader> &shader = pointRenderer->buffer.renderDataBuffer.shader;

  const size_t count = pointRenderer->blob->writeBytes / (sizeof(float_t) * (2 + 4 + 1));
  
  static_assert(sizeof(mVec3f) == sizeof(float_t) * 3, "Invalid Assuption.");

  mERROR_CHECK(mBinaryChunk_WriteData(pointRenderer->blob, mVec3f()));
  mERROR_CHECK(mBinaryChunk_WriteData(pointRenderer->blob, mVec4f()));
  
  mERROR_CHECK(mInstancedRenderDataBuffer_SetInstanceBuffer(pointRenderer->buffer, pointRenderer->blob->pData, pointRenderer->blob->writeBytes));
  mERROR_CHECK(mInstancedRenderDataBuffer_SetInstanceCount(pointRenderer->buffer, count));

  mERROR_CHECK(mShader_Bind(shader));
  mERROR_CHECK(mShader_SetUniform(shader, "_invScreenSize", mVec2f(1.f) / mRenderParams_CurrentRenderResolutionF));
  mERROR_CHECK(mShader_SetUniform(shader, "radiusFac", pointRenderer->mode == mCPM_Circle ? 1.f : 2.f));

  mERROR_CHECK(mInstancedRenderDataBuffer_Draw(pointRenderer->buffer));

  mRETURN_SUCCESS();
}

mFUNCTION(mPointRenderer_DrawPoint, mPtr<mPointRenderer> &pointRenderer, const mColouredPoint &point)
{
  mFUNCTION_SETUP();

  mERROR_IF(pointRenderer == nullptr, mR_ArgumentNull);
  mERROR_IF(!pointRenderer->isStarted, mR_ResourceStateInvalid);

  mERROR_CHECK(mBinaryChunk_WriteData(pointRenderer->blob, point.position));
  mERROR_CHECK(mBinaryChunk_WriteData(pointRenderer->blob, point.size));
  mERROR_CHECK(mBinaryChunk_WriteData(pointRenderer->blob, point.colour));

  mGL_DEBUG_ERROR_CHECK();

  mRETURN_SUCCESS();
}

//////////////////////////////////////////////////////////////////////////

static mFUNCTION(mPointRenderer_Destroy_Internal, IN mPointRenderer *pPointRenderer)
{
  mFUNCTION_SETUP();

  mERROR_IF(pPointRenderer == nullptr, mR_ArgumentNull);

  mERROR_CHECK(mBinaryChunk_Destroy(&pPointRenderer->blob));
  mERROR_CHECK(mInstancedRenderDataBuffer_Destroy(&pPointRenderer->buffer));

  mRETURN_SUCCESS();
}
