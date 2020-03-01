#ifndef mObjReader_h__
#define mObjReader_h__

#include "mediaLib.h"
#include "mQueue.h"

#ifdef GIT_BUILD // Define __M_FILE__
  #ifdef __M_FILE__
    #undef __M_FILE__
  #endif
  #define __M_FILE__ "XN2nShS29YBk0Na+wPwsZWcUxaHnSiNk6JxnE9csHucEOVrAEGoYYN7ZMuXQMprRg8nrXGpARLF5bUfC"
#endif

struct mObjTriangleVertexInfo
{
  mVec3f position;
  mVec3f textureCoord;
  mVec3f normal;
  mVec3f colour;

  uint8_t hasTextureCoord : 1;
  uint8_t hasNormal : 1;
  uint8_t hasColour : 1;
};

struct mObjVertexInfo
{
  mVec3f position;
  mVec3f colour;
  uint8_t hasColour : 1;
};

struct mObjInfo
{
  uint8_t hasTextureCoordinates : 1,
    hasColours : 1,
    hasNormals : 1,
    hasVertices : 1,
    hasLines : 1,
    hasTriangles : 1,
    smoothShading : 1;

  mPtr<mQueue<mTriangle<mObjTriangleVertexInfo>>> triangles;
  mPtr<mQueue<mLine<mObjVertexInfo>>> lines;
  mPtr<mQueue<mObjVertexInfo>> vertices;
};

mFUNCTION(mObjInfo_Destroy, IN_OUT mObjInfo *pObjInfo);

typedef size_t mObjParseParam;

enum mObjParseParam_
{
  mOPP_Default = 0,
  mOPP_KeepVertices = 1 << 0,
};

mFUNCTION(mObjReader_Parse, const char *contents, const size_t size, IN mAllocator *pAllocator, OUT mObjInfo *pObjInfo, const mObjParseParam parseMode = mOPP_Default);
mFUNCTION(mObjReader_ParseFromFile, const mString &filename, IN mAllocator *pAllocator, OUT mObjInfo *pObjInfo, const mObjParseParam parseMode = mOPP_Default);

#endif // mObjReader_h__
