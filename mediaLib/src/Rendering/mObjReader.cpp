#include "mObjReader.h"

#include "mFile.h"

mFUNCTION(mObjInfo_Destroy, IN_OUT mObjInfo *pObjInfo)
{
  mFUNCTION_SETUP();

  mERROR_IF(pObjInfo == nullptr, mR_Success);

  mERROR_CHECK(mSharedPointer_Destroy(&pObjInfo->triangles));
  mERROR_CHECK(mSharedPointer_Destroy(&pObjInfo->lines));
  mERROR_CHECK(mSharedPointer_Destroy(&pObjInfo->vertices));

  mRETURN_SUCCESS();
}

mFUNCTION(mObjReader_Parse, const char *contents, const size_t size, IN mAllocator *pAllocator, OUT mObjInfo *pObjInfo, const mObjParseParam parseMode /* = mOPP_Default */)
{
  mFUNCTION_SETUP();

  mERROR_IF(contents == nullptr || pObjInfo == nullptr, mR_ArgumentNull);
  mERROR_IF(size == 0, mR_ArgumentNull);

  pObjInfo->hasColours = false;
  pObjInfo->hasTextureCoordinates = false;
  pObjInfo->hasNormals = false;
  pObjInfo->hasVertices = false;
  pObjInfo->hasLines = false;
  pObjInfo->hasTriangles = false;
  pObjInfo->smoothShading = false;

  pObjInfo->triangles = nullptr;
  pObjInfo->lines = nullptr;
  pObjInfo->vertices = nullptr;

  mPtr<mQueue<mObjVertexInfo>> vertices;
  mDEFER_CALL(&vertices, mQueue_Destroy);
  mERROR_CHECK(mQueue_Create(&vertices, !!(parseMode & mOPP_KeepVertices) ? pAllocator : &mDefaultTempAllocator));

  if (!!(parseMode & mOPP_KeepVertices))
    pObjInfo->vertices = vertices;

  mPtr<mQueue<mVec3f>> textureCoordinates;
  mDEFER_CALL(&textureCoordinates, mQueue_Destroy);
  mERROR_CHECK(mQueue_Create(&textureCoordinates, &mDefaultTempAllocator));

  mPtr<mQueue<mVec3f>> normals;
  mDEFER_CALL(&normals, mQueue_Destroy);
  mERROR_CHECK(mQueue_Create(&normals, &mDefaultTempAllocator));

  mERROR_CHECK(mQueue_Create(&pObjInfo->triangles, pAllocator));
  mERROR_CHECK(mQueue_Create(&pObjInfo->lines, pAllocator));

  const char *text = contents;
  bool inString = true;

  while (inString)
  {
    switch (*text)
    {
    case '\0':
    {
      inString = false;
      break;
    }

    case 'v':
    case 'V':
    {
      ++text;

      switch (*text)
      {
      case '\0':
      {
        inString = false;
        break;
      }

      case ' ':
      {
        ++text;

        while (*text == ' ')
          ++text;

        float_t floats[4 + 3];
        size_t index = 0;

        // read vertex position.
        // read potential colors.

        while (index < mARRAYSIZE(floats))
        {
          if ((*text < '0' || *text > '9') && *text != '-' && *text != '.')
            break;

          floats[index] = (float_t)mParseFloat(text, &text);

          while (*text == ' ')
            ++text;

          ++index;
        }

        mObjVertexInfo vertex;

        switch (index)
        {
          case 3:
          case 4:
            vertex.position = mVec3f(floats[0], floats[1], floats[2]);
            vertex.hasColour = false;
            break;

          case 6:
            vertex.position = mVec3f(floats[0], floats[1], floats[2]);
            vertex.colour = mVec3f(floats[3], floats[4], floats[5]);
            vertex.hasColour = true;
            pObjInfo->hasColours = true;
            break;

          case 7: // position with w component.
            vertex.position = mVec3f(floats[0], floats[1], floats[2]);
            vertex.colour = mVec3f(floats[4], floats[5], floats[6]);
            vertex.hasColour = true;
            pObjInfo->hasColours = true;
            break;

          default:
            mRETURN_RESULT(mR_ResourceInvalid);
        }

        mERROR_CHECK(mQueue_PushBack(vertices, std::move(vertex)));
        
        pObjInfo->hasVertices = true;

        text = mMin(strchr(text, '\r') - 1, strchr(text, '\n') - 1) + 2;

        if (text == (const char *)1)
        {
          inString = false;
          break;
        }

        break;
      }

      case 't':
      case 'T':
      {
        ++text;
        mERROR_IF(*text != ' ', mR_ResourceInvalid);
        ++text;

        while (*text == ' ')
          ++text;

        float_t floats[3] = { 0, 0, 0 };
        size_t index = 0;

        // read texture coordinate.

        while (index < mARRAYSIZE(floats))
        {
          if ((*text < '0' || *text > '9') && *text != '-' && *text != '.')
            break;

          floats[index] = (float_t)mParseFloat(text, &text);

          while (*text == ' ')
            ++text;

          ++index;
        }

        mERROR_IF(index < 1, mR_ResourceInvalid);
        mERROR_CHECK(mQueue_PushBack(textureCoordinates, mVec3f(floats[0], floats[1], floats[2])));

        pObjInfo->hasTextureCoordinates = true;

        break;
      }

      case 'n':
      case 'N':
      {
        ++text;
        mERROR_IF(*text != ' ', mR_ResourceInvalid);
        ++text;

        while (*text == ' ')
          ++text;

        float_t floats[3];
        size_t index = 0;

        // read normal.

        while (index < mARRAYSIZE(floats))
        {
          if ((*text < '0' || *text > '9') && *text != '-' && *text != '.')
            break;

          floats[index] = (float_t)mParseFloat(text, &text);

          while (*text == ' ')
            ++text;

          ++index;
        }

        mERROR_IF(index != 3, mR_ResourceInvalid);

        mERROR_CHECK(mQueue_PushBack(normals, mVec3f(floats[0], floats[1], floats[2])));
        pObjInfo->hasNormals = true;

        break;
      }

      default:
      {
        mRETURN_RESULT(mR_ResourceInvalid);
        break;
      }
      }

      break;
    }

    case 'l':
    case 'L':
    {
      ++text;
      mERROR_IF(*text != ' ', mR_ResourceInvalid);
      ++text;

      while (*text == ' ')
        ++text;

      size_t count;
      mERROR_CHECK(mQueue_GetCount(vertices, &count));

      mERROR_IF((*text < '0' || *text > '9') && *text != '-' && *text != '.', mR_ResourceInvalid);

      int64_t index = mParseInt(text, &text);
      mObjVertexInfo lastVertex;
      mObjVertexInfo currentVertex;

      if (index < 0)
        mERROR_CHECK(mQueue_PeekAt(vertices, (size_t)((int64_t)count + index), &currentVertex));
      else
        mERROR_CHECK(mQueue_PeekAt(vertices, (size_t)(index - 1), &currentVertex));

      while (*text == ' ')
        ++text;

      while (true)
      {
        if ((*text < '0' || *text > '9') && *text != '-' && *text != '.')
          break;

        lastVertex = currentVertex;
        index = mParseInt(text, &text);

        if (index < 0)
          mERROR_CHECK(mQueue_PeekAt(vertices, (size_t)((int64_t)count + index), &currentVertex));
        else
          mERROR_CHECK(mQueue_PeekAt(vertices, (size_t)(index - 1), &currentVertex));

        mERROR_CHECK(mQueue_PushBack(pObjInfo->lines, mLine<mObjVertexInfo>(lastVertex, currentVertex)));
        pObjInfo->hasLines = true;

        while (*text == ' ')
          ++text;
      }

      text = mMin(strchr(text, '\r') - 1, strchr(text, '\n') - 1) + 2;

      if (text == (const char *)1)
      {
        inString = false;
        break;
      }

      break;
    }

    case 'f':
    case 'F':
    {
      ++text;
      mERROR_IF(*text != ' ', mR_ResourceInvalid);
      ++text;

      while (*text == ' ')
        ++text;

      mObjTriangleVertexInfo firstVertex, previousVertex, currentVertex;
      size_t index = 0;

      mObjVertexInfo vertex;

      size_t vertexCount;
      size_t texCoordCount;
      size_t normalCount;

      mERROR_CHECK(mQueue_GetCount(vertices, &vertexCount));
      mERROR_CHECK(mQueue_GetCount(textureCoordinates, &texCoordCount));
      mERROR_CHECK(mQueue_GetCount(normals, &normalCount));

      while (true)
      {
        if ((*text < '0' || *text > '9') && *text != '-' && *text != '.')
          break;

        previousVertex = currentVertex;
        mERROR_CHECK(mZeroMemory(&currentVertex));

        const int64_t vertexIndex = mParseInt(text, &text);

        if (vertexIndex < 0)
          mERROR_CHECK(mQueue_PeekAt(vertices, (size_t)((int64_t)vertexCount + vertexIndex), &vertex));
        else
          mERROR_CHECK(mQueue_PeekAt(vertices, (size_t)(vertexIndex - 1), &vertex));

        currentVertex.hasColour = vertex.hasColour;
        currentVertex.position = vertex.position;
        currentVertex.colour = vertex.colour;

        if (*text == ' ' || *text == '\r' || *text == '\n' || *text == '\t')
          goto end_vertex;
        
        mERROR_IF(*text != '/', mR_ResourceInvalid);

        ++text;

        if (*text != '/')
        {
          mERROR_IF((*text < '0' || *text > '9') && *text != '-' && *text != '.', mR_ResourceInvalid);

          const int64_t texCoordIndex = mParseInt(text, &text);

          if (texCoordIndex < 0)
            mERROR_CHECK(mQueue_PeekAt(textureCoordinates, (size_t)((int64_t)texCoordCount + texCoordIndex), &currentVertex.textureCoord));
          else
            mERROR_CHECK(mQueue_PeekAt(textureCoordinates, (size_t)(texCoordIndex - 1), &currentVertex.textureCoord));

          currentVertex.hasTextureCoord = true;

          if (*text == ' ' || *text == '\r' || *text == '\n' || *text == '\t')
            goto end_vertex;

          mERROR_IF(*text != '/', mR_ResourceInvalid);
        }

        ++text;

        mERROR_IF((*text < '0' || *text > '9') && *text != '-' && *text != '.', mR_ResourceInvalid);

        const int64_t normalIndex = mParseInt(text, &text);

        if (normalIndex < 0)
          mERROR_CHECK(mQueue_PeekAt(normals, (size_t)((int64_t)normalCount + normalIndex), &currentVertex.normal));
        else
          mERROR_CHECK(mQueue_PeekAt(normals, (size_t)(normalIndex - 1), &currentVertex.normal));

        currentVertex.hasNormal = true;

      end_vertex:
        ++index;

        if (index > 2)
        {
          mERROR_CHECK(mQueue_PushBack(pObjInfo->triangles, mTriangle<mObjTriangleVertexInfo>(firstVertex, previousVertex, currentVertex)));
          
          pObjInfo->hasTriangles = true;
        }
        else if (index == 1)
        {
          firstVertex = currentVertex;
        }

        while (*text == ' ')
          ++text;
      }

      text = mMin(strchr(text, '\r') - 1, strchr(text, '\n') - 1) + 2;

      if (text == (const char *)1)
      {
        inString = false;
        break;
      }

      break;
    }

    case 's':
    case 'S':
    {
      ++text;
      mERROR_IF(*text != ' ', mR_ResourceInvalid);
      ++text;

      while (*text == ' ')
        ++text;

      const char *endChar = mMin(strchr(text, '\r') - 1, mMin(strchr(text, '\n') - 1, strchr(text, ' ') - 1)) + 2;

      if (endChar == (const char *)1)
      {
        inString = false;
        break;
      }

      switch (endChar - text)
      {
      case 2:
      {
        if (*text == '0')
          pObjInfo->smoothShading = false;
        else if (*text == '1')
          pObjInfo->smoothShading = true;
        else
          mRETURN_RESULT(mR_ResourceInvalid);
      
        break;
      }

      case 3:
      {
        mERROR_IF(memcmp(text, "on", 2) != 0 && memcmp(text, "ON", 2) != 0 && memcmp(text, "On", 2) != 0, mR_ResourceInvalid);
        pObjInfo->smoothShading = true;
        break;
      }

      case 4:
      {
        mERROR_IF(memcmp(text, "off", 3) != 0 && memcmp(text, "OFF", 3) != 0 && memcmp(text, "Off", 3) != 0, mR_ResourceInvalid);
        pObjInfo->smoothShading = false;
        break;
      }

      default:
      {
        mRETURN_RESULT(mR_ResourceInvalid);
        break;
      }
      }

      text = mMin(strchr(text, '\r') - 1, strchr(text, '\n') - 1) + 2;

      if (text == (const char *)1)
      {
        inString = false;
        break;
      }

      break;
    }

    case 'u':
    case 'U':
    case 'o':
    case 'O':
    case 'm':
    case 'M':
    case '#':
    {
      ++text;

      // usemtl, comments, objects, mtllib.

      text = mMin(strchr(text, '\r') - 1, strchr(text, '\n') - 1) + 2;
      
      if (text == (const char *)1)
      {
        inString = false;
        break;
      }

      break;
    }

    case '\r':
    case '\n':
    {
      ++text;
      break;
    }

    default:
    {
      mRETURN_RESULT(mR_ResourceInvalid);
      break;
    }
    }
  }

  if (!pObjInfo->hasVertices && pObjInfo->vertices == nullptr)
    pObjInfo->vertices = vertices;

  mRETURN_SUCCESS();
}

mFUNCTION(mObjReader_ParseFromFile, const mString &filename, IN mAllocator *pAllocator, OUT mObjInfo *pObjInfo, const mObjParseParam parseMode /* = mOPP_Default */)
{
  mFUNCTION_SETUP();

  mERROR_IF(pObjInfo == nullptr, mR_ArgumentNull);
  mERROR_IF(filename.hasFailed, mR_InvalidParameter);

  char *text = nullptr;
  size_t size = 0;
  mAllocator *pTempAllocator = &mDefaultTempAllocator;

  mDEFER(mAllocator_FreePtr(pTempAllocator, &text));
  mERROR_CHECK(mFile_ReadRaw(filename, &text, pTempAllocator, &size));

  mERROR_CHECK(mObjReader_Parse(text, size, pAllocator, pObjInfo, parseMode));

  mRETURN_SUCCESS();
}
