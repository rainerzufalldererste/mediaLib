#ifndef mScreenQuad_h__
#define mScreenQuad_h__

#include "mRenderParams.h"
#include "mShader.h"

// mScreenQuad.
// Your fragment shader requires inputs for the amount of textures specified called `_texCoord%TextureIndex%` (`_texCoord0`, `_texCoord1`, ...).

#ifdef GIT_BUILD // Define __M_FILE__
  #ifdef __M_FILE__
    #undef __M_FILE__
  #endif
  #define __M_FILE__ "3qJ1wCVEmddupAsjv9rYJvu/WdQsbycV1t2xTV4Ak1R9fAlXBIb2/b7J3qTFMRUSLiIlSjg5zpbUdAsY"
#endif

struct mScreenQuad
{
  mPtr<mShader> shader;
#if defined(mRENDERER_OPENGL)
  GLuint vao, vbo;
#endif
};

mFUNCTION(mScreenQuad_Create, OUT mPtr<mScreenQuad> *pScreenQuad, IN mAllocator *pAllocator, const mString &fragmentShader, const size_t textureCount = 1);

#define mScreenQuad_TextureName "_texture0"
#define mScreenQuad_TextureSampleCountName "_texture0sampleCount"

// texture is called `_texture0`.
mFUNCTION(mScreenQuad_Create, OUT mPtr<mScreenQuad> *pScreenQuad, IN mAllocator *pAllocator);

// texture is called `_texture0`. sample count is called `_texture0sampleCount`.
mFUNCTION(mScreenQuad_CreateForMultisampleTexture, OUT mPtr<mScreenQuad> *pScreenQuad, IN mAllocator *pAllocator);
mFUNCTION(mScreenQuad_CreateFrom, OUT mPtr<mScreenQuad> *pScreenQuad, IN mAllocator *pAllocator, const mString &fragmentShaderPath, const size_t textureCount = 1);

mFUNCTION(mScreenQuad_Destroy, IN_OUT mPtr<mScreenQuad> *pScreenQuad);
mFUNCTION(mScreenQuad_Render, mPtr<mScreenQuad> &screenQuad);

#endif // mScreenQuad_h__
