#include "mPointRenderer.h"

#define mPOINT_RENDERER_POSITION_ATTRIBUTE "_position0"
#define mPOINT_RENDERER_COLOUR_ATTRIBUTE "_colour0"
#define mPOINT_RENDERER_SCREEN_SIZE_ATTRIBUTE "_scale0"

#ifdef GIT_BUILD // Define __M_FILE__
  #ifdef __M_FILE__
    #undef __M_FILE__
  #endif
  #define __M_FILE__ "ylpx4/kST4uu8w/FEi/KckGveMOQ+YjF1otTqtkzolMlKa7m+uDQgw0sjapOkS8F7Uz7tI8vfRsejWWI"
#endif

mFUNCTION(mPointRenderer_Destroy_Internal, IN mPointRenderer *pPointRenderer);

//////////////////////////////////////////////////////////////////////////

mColouredPoint::mColouredPoint() :
  position(),
  colour(1),
  size(1)
{ }

mColouredPoint::mColouredPoint(const mVec2f position, const mVec4f colour, const float_t size) :
  position(position),
  colour(colour),
  size(size)
{ }

//////////////////////////////////////////////////////////////////////////

mFUNCTION(mPointRenderer_Create, OUT mPtr<mPointRenderer> *pPointRenderer, IN mAllocator *pAllocator)
{
  mFUNCTION_SETUP();

  mERROR_IF(pPointRenderer == nullptr, mR_ArgumentNull);

  mERROR_CHECK(mSharedPointer_Allocate(pPointRenderer, pAllocator, (std::function<void (mPointRenderer *)>)[](mPointRenderer *pData) {mPointRenderer_Destroy_Internal(pData);}, 1));

#if defined(mRENDERER_OPENGL)
  const char vertexShader[] = mGLSL(
    in float data;

    uniform vec2 _position0;
    uniform vec2 _scale0;

    void main()
    {
      vec2 position = _position0 / _scale0;
      position.y = 1 - position.y;

      gl_Position = vec4(position * 2 - 1, 0, 1);
    }
  );
  
  const char fragmentShader[] = mGLSL(
    out vec4 colour;
    
    uniform vec4 _colour0;

    void main()
    {
      colour = _colour0;
    }
  );

  mERROR_CHECK(mShader_Create(&(*pPointRenderer)->shader, vertexShader, fragmentShader));

  mGL_DEBUG_ERROR_CHECK();

  const float_t buffer[] = { 0 };

  glGenBuffers(1, &(*pPointRenderer)->vbo);
  glBindBuffer(GL_ARRAY_BUFFER, (*pPointRenderer)->vbo);
  glBufferData(GL_ARRAY_BUFFER, sizeof(buffer), buffer, GL_STATIC_DRAW);

  mGL_DEBUG_ERROR_CHECK();

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

#if defined(mRENDERER_OPENGL)
  switch (mode)
  {
  case mCPM_Square:
    mERROR_CHECK(mRenderParams_SetBlendingEnabled(false));
    glDisable(GL_POINT_SMOOTH);
    break;

  case mCPM_Circle:
    mERROR_CHECK(mRenderParams_SetBlendingEnabled(true));
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glEnable(GL_POINT_SMOOTH);
    break;

  default:
    mRETURN_RESULT(mR_InvalidParameter);
  }

  glBindBuffer(GL_ARRAY_BUFFER, pointRenderer->vbo);

  glEnableVertexAttribArray((GLuint)0);
  glVertexAttribPointer((GLuint)0, (GLint)1, GL_FLOAT, GL_FALSE, (GLsizei)sizeof(float_t), (const void *)0);

  mERROR_CHECK(mShader_Bind(pointRenderer->shader));
  mERROR_CHECK(mShader_SetUniform(pointRenderer->shader, mPOINT_RENDERER_SCREEN_SIZE_ATTRIBUTE, mRenderParams_CurrentRenderResolutionF));

#else
  mRETURN_RESULT(mR_NotImplemented);
#endif

  mGL_DEBUG_ERROR_CHECK();

  mRETURN_SUCCESS();
}

mFUNCTION(mPointRenderer_End, mPtr<mPointRenderer> &)
{
  mFUNCTION_SETUP();

  glDisable(GL_POINT_SMOOTH);

  mRETURN_SUCCESS();
}

mFUNCTION(mPointRenderer_DrawPoint, mPtr<mPointRenderer> &pointRenderer, const mColouredPoint &point)
{
  mFUNCTION_SETUP();

  glPointSize(point.size);

  mERROR_CHECK(mShader_SetUniform(pointRenderer->shader, mPOINT_RENDERER_POSITION_ATTRIBUTE, point.position));
  mERROR_CHECK(mShader_SetUniform(pointRenderer->shader, mPOINT_RENDERER_COLOUR_ATTRIBUTE, point.colour));

  glDrawArrays(GL_POINTS, 0, 1);

  mGL_DEBUG_ERROR_CHECK();

  mRETURN_SUCCESS();
}

//////////////////////////////////////////////////////////////////////////

mFUNCTION(mPointRenderer_Destroy_Internal, IN mPointRenderer *pPointRenderer)
{
  mFUNCTION_SETUP();

  mERROR_IF(pPointRenderer == nullptr, mR_ArgumentNull);

  mERROR_CHECK(mShader_Destroy(&pPointRenderer->shader));

#if defined(mRENDERER_OPENGL)
  glDeleteBuffers(1, &pPointRenderer->vbo);
#endif

  mRETURN_SUCCESS();
}
