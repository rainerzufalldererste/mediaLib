#include "mTestLib.h"
#include "mObjReader.h"

mTEST(mObjReader, ReadObjString)
{
  mTEST_ALLOCATOR_SETUP();

  mObjInfo info;
  mDEFER_CALL(&info, mObjInfo_Destroy);

  info.smoothShading = false;

  const char testObj[] = R"OBJ(
o  someobject  

# List of geometric vertices, with (x, y, z, [w]) coordinates, w is optional and defaults to 1.0.
v 0.123 0.234 0.345 1
v 0.234 0.345 0.456 
v  0.345 0.456 0.567  1.0 0.5 0
v 0.456 0.567   0.678 1 0 .33 -0.66e1
v 1e-1 0.1e1 -0.e-1 1
v 0.000000000001 10000000000.0 -1

# List of texture coordinates, in (u, [v, w]) coordinates, these will vary between 0 and 1, v and w are optional and default to 0.
vt 0.500
vt  0.2500  0.5
vt 0.12500 .25 0.5   

# List of vertex normals in (x,y,z) form; normals might not be unit vectors.
vn 0.0 0.000 1.
vn .01    -.1 .0
vn   1.1 1234567890. 0.0

# Polygonal face element
f 1 2 3
f  1 2 3  -3
f 2/1 3/2 4/3
f 5/3/1 6/-2/2 1/1/3  
f 1//1    2//-1 3//2

# Line element
l  1 2 3   -3 1   
l -6 2

s On
)OBJ";

  mTEST_ASSERT_SUCCESS(mObjReader_Parse(testObj, sizeof(testObj), pAllocator, &info, mOPP_KeepVertices));

  mTEST_ASSERT_TRUE((bool)info.hasVertices);
  mTEST_ASSERT_TRUE((bool)info.hasLines);
  mTEST_ASSERT_TRUE((bool)info.hasColours);
  mTEST_ASSERT_TRUE((bool)info.hasTextureCoordinates);
  mTEST_ASSERT_TRUE((bool)info.hasNormals);
  mTEST_ASSERT_TRUE((bool)info.hasTriangles);
  mTEST_ASSERT_TRUE((bool)info.smoothShading);

  mTEST_ASSERT_EQUAL(info.vertices->count, 6);
  mTEST_ASSERT_EQUAL(info.triangles->count, 6);
  mTEST_ASSERT_EQUAL(info.lines->count, 5);

  mTEST_ASSERT_FLOAT_EQUALS(0.123f, (*info.vertices)[0].position.x);
  mTEST_ASSERT_FLOAT_EQUALS(0.234f, (*info.vertices)[0].position.y);
  mTEST_ASSERT_FLOAT_EQUALS(0.345f, (*info.vertices)[0].position.z);
  mTEST_ASSERT_FALSE((bool)(*info.vertices)[0].hasColour);

  mTEST_ASSERT_FLOAT_EQUALS(0.234f, (*info.vertices)[1].position.x);
  mTEST_ASSERT_FLOAT_EQUALS(0.345f, (*info.vertices)[1].position.y);
  mTEST_ASSERT_FLOAT_EQUALS(0.456f, (*info.vertices)[1].position.z);
  mTEST_ASSERT_FALSE((bool)(*info.vertices)[1].hasColour);

  mTEST_ASSERT_FLOAT_EQUALS(0.345f, (*info.vertices)[2].position.x);
  mTEST_ASSERT_FLOAT_EQUALS(0.456f, (*info.vertices)[2].position.y);
  mTEST_ASSERT_FLOAT_EQUALS(0.567f, (*info.vertices)[2].position.z);
  mTEST_ASSERT_TRUE((bool)(*info.vertices)[2].hasColour);
  mTEST_ASSERT_FLOAT_EQUALS(1.f, (*info.vertices)[2].colour.x);
  mTEST_ASSERT_FLOAT_EQUALS(0.5f, (*info.vertices)[2].colour.y);
  mTEST_ASSERT_FLOAT_EQUALS(0.f, (*info.vertices)[2].colour.z);

  mTEST_ASSERT_FLOAT_EQUALS(0.456f, (*info.vertices)[3].position.x);
  mTEST_ASSERT_FLOAT_EQUALS(0.567f, (*info.vertices)[3].position.y);
  mTEST_ASSERT_FLOAT_EQUALS(0.678f, (*info.vertices)[3].position.z);
  mTEST_ASSERT_TRUE((bool)(*info.vertices)[3].hasColour);
  mTEST_ASSERT_FLOAT_EQUALS(0.f, (*info.vertices)[3].colour.x);
  mTEST_ASSERT_FLOAT_EQUALS(.33f, (*info.vertices)[3].colour.y);
  mTEST_ASSERT_FLOAT_EQUALS(-.66e1f, (*info.vertices)[3].colour.z);

  mTEST_ASSERT_FLOAT_EQUALS(1e-1f, (*info.vertices)[4].position.x);
  mTEST_ASSERT_FLOAT_EQUALS(0.1e1f, (*info.vertices)[4].position.y);
  mTEST_ASSERT_FLOAT_EQUALS(-0.e-1f, (*info.vertices)[4].position.z);
  mTEST_ASSERT_FALSE((bool)(*info.vertices)[4].hasColour);

  mTEST_ASSERT_FLOAT_EQUALS(0.000000000001f, (*info.vertices)[5].position.x);
  mTEST_ASSERT_FLOAT_EQUALS(10000000000.0f, (*info.vertices)[5].position.y);
  mTEST_ASSERT_FLOAT_EQUALS(-1.f, (*info.vertices)[5].position.z);
  mTEST_ASSERT_FALSE((bool)(*info.vertices)[5].hasColour);

  mTEST_ASSERT_EQUAL((*info.triangles)[0].position0.position, (*info.vertices)[0].position);
  mTEST_ASSERT_EQUAL((*info.triangles)[0].position0.hasColour, (*info.vertices)[0].hasColour);
  mTEST_ASSERT_FALSE((bool)(*info.triangles)[0].position0.hasNormal);
  mTEST_ASSERT_FALSE((bool)(*info.triangles)[0].position0.hasTextureCoord);

  mTEST_ASSERT_EQUAL((*info.triangles)[0].position1.position, (*info.vertices)[1].position);
  mTEST_ASSERT_EQUAL((*info.triangles)[0].position1.hasColour, (*info.vertices)[1].hasColour);
  mTEST_ASSERT_FALSE((bool)(*info.triangles)[0].position1.hasNormal);
  mTEST_ASSERT_FALSE((bool)(*info.triangles)[0].position1.hasTextureCoord);

  mTEST_ASSERT_EQUAL((*info.triangles)[0].position2.position, (*info.vertices)[2].position);
  mTEST_ASSERT_EQUAL((*info.triangles)[0].position2.hasColour, (*info.vertices)[2].hasColour);
  mTEST_ASSERT_EQUAL((*info.triangles)[0].position2.colour, (*info.vertices)[2].colour);
  mTEST_ASSERT_FALSE((bool)(*info.triangles)[0].position2.hasNormal);
  mTEST_ASSERT_FALSE((bool)(*info.triangles)[0].position2.hasTextureCoord);

  mTEST_ASSERT_EQUAL((*info.triangles)[1].position0.position, (*info.vertices)[0].position);
  mTEST_ASSERT_EQUAL((*info.triangles)[1].position0.hasColour, (*info.vertices)[0].hasColour);
  mTEST_ASSERT_FALSE((bool)(*info.triangles)[1].position0.hasNormal);
  mTEST_ASSERT_FALSE((bool)(*info.triangles)[1].position0.hasTextureCoord);

  mTEST_ASSERT_EQUAL((*info.triangles)[1].position1.position, (*info.vertices)[1].position);
  mTEST_ASSERT_EQUAL((*info.triangles)[1].position1.hasColour, (*info.vertices)[1].hasColour);
  mTEST_ASSERT_FALSE((bool)(*info.triangles)[1].position1.hasNormal);
  mTEST_ASSERT_FALSE((bool)(*info.triangles)[1].position1.hasTextureCoord);

  mTEST_ASSERT_EQUAL((*info.triangles)[1].position2.position, (*info.vertices)[2].position);
  mTEST_ASSERT_EQUAL((*info.triangles)[1].position2.hasColour, (*info.vertices)[2].hasColour);
  mTEST_ASSERT_EQUAL((*info.triangles)[1].position2.colour, (*info.vertices)[2].colour);
  mTEST_ASSERT_FALSE((bool)(*info.triangles)[1].position2.hasNormal);
  mTEST_ASSERT_FALSE((bool)(*info.triangles)[1].position2.hasTextureCoord);

  mTEST_ASSERT_EQUAL((*info.triangles)[2].position0.position, (*info.vertices)[0].position);
  mTEST_ASSERT_EQUAL((*info.triangles)[2].position0.hasColour, (*info.vertices)[0].hasColour);
  mTEST_ASSERT_FALSE((bool)(*info.triangles)[2].position0.hasNormal);
  mTEST_ASSERT_FALSE((bool)(*info.triangles)[2].position0.hasTextureCoord);

  mTEST_ASSERT_EQUAL((*info.triangles)[2].position1.position, (*info.vertices)[2].position);
  mTEST_ASSERT_EQUAL((*info.triangles)[2].position1.hasColour, (*info.vertices)[2].hasColour);
  mTEST_ASSERT_EQUAL((*info.triangles)[2].position1.colour, (*info.vertices)[2].colour);
  mTEST_ASSERT_FALSE((bool)(*info.triangles)[2].position1.hasNormal);
  mTEST_ASSERT_FALSE((bool)(*info.triangles)[2].position1.hasTextureCoord);

  mTEST_ASSERT_EQUAL((*info.triangles)[2].position2.position, (*info.vertices)[3].position);
  mTEST_ASSERT_EQUAL((*info.triangles)[2].position2.hasColour, (*info.vertices)[3].hasColour);
  mTEST_ASSERT_EQUAL((*info.triangles)[2].position2.colour, (*info.vertices)[3].colour);
  mTEST_ASSERT_FALSE((bool)(*info.triangles)[2].position2.hasNormal);
  mTEST_ASSERT_FALSE((bool)(*info.triangles)[2].position2.hasTextureCoord);

  mTEST_ASSERT_EQUAL((*info.triangles)[3].position0.position, (*info.vertices)[1].position);
  mTEST_ASSERT_EQUAL((*info.triangles)[3].position0.hasColour, (*info.vertices)[1].hasColour);
  mTEST_ASSERT_FALSE((bool)(*info.triangles)[3].position0.hasNormal);
  mTEST_ASSERT_TRUE((bool)(*info.triangles)[3].position0.hasTextureCoord);
  mTEST_ASSERT_FLOAT_EQUALS(0.5f, (*info.triangles)[3].position0.textureCoord.x);
  mTEST_ASSERT_FLOAT_EQUALS(0.f, (*info.triangles)[3].position0.textureCoord.y);
  mTEST_ASSERT_FLOAT_EQUALS(0.f, (*info.triangles)[3].position0.textureCoord.z);

  mTEST_ASSERT_EQUAL((*info.triangles)[3].position1.position, (*info.vertices)[2].position);
  mTEST_ASSERT_EQUAL((*info.triangles)[3].position1.hasColour, (*info.vertices)[2].hasColour);
  mTEST_ASSERT_EQUAL((*info.triangles)[3].position1.colour, (*info.vertices)[2].colour);
  mTEST_ASSERT_FALSE((bool)(*info.triangles)[3].position1.hasNormal);
  mTEST_ASSERT_TRUE((bool)(*info.triangles)[3].position1.hasTextureCoord);
  mTEST_ASSERT_FLOAT_EQUALS(0.25f, (*info.triangles)[3].position1.textureCoord.x);
  mTEST_ASSERT_FLOAT_EQUALS(0.5f, (*info.triangles)[3].position1.textureCoord.y);
  mTEST_ASSERT_FLOAT_EQUALS(0.f, (*info.triangles)[3].position1.textureCoord.z);

  mTEST_ASSERT_EQUAL((*info.triangles)[3].position2.position, (*info.vertices)[3].position);
  mTEST_ASSERT_EQUAL((*info.triangles)[3].position2.hasColour, (*info.vertices)[3].hasColour);
  mTEST_ASSERT_EQUAL((*info.triangles)[3].position2.colour, (*info.vertices)[3].colour);
  mTEST_ASSERT_FALSE((bool)(*info.triangles)[3].position2.hasNormal);
  mTEST_ASSERT_TRUE((bool)(*info.triangles)[3].position2.hasTextureCoord);
  mTEST_ASSERT_FLOAT_EQUALS(0.125f, (*info.triangles)[3].position2.textureCoord.x);
  mTEST_ASSERT_FLOAT_EQUALS(0.25f, (*info.triangles)[3].position2.textureCoord.y);
  mTEST_ASSERT_FLOAT_EQUALS(0.5f, (*info.triangles)[3].position2.textureCoord.z);

  mTEST_ASSERT_EQUAL((*info.triangles)[4].position0.position, (*info.vertices)[4].position);
  mTEST_ASSERT_EQUAL((*info.triangles)[4].position0.hasColour, (*info.vertices)[4].hasColour);
  mTEST_ASSERT_TRUE((bool)(*info.triangles)[4].position0.hasNormal);
  mTEST_ASSERT_TRUE((bool)(*info.triangles)[4].position0.hasTextureCoord);
  mTEST_ASSERT_FLOAT_EQUALS(0.125f, (*info.triangles)[4].position0.textureCoord.x);
  mTEST_ASSERT_FLOAT_EQUALS(0.25f, (*info.triangles)[4].position0.textureCoord.y);
  mTEST_ASSERT_FLOAT_EQUALS(0.5f, (*info.triangles)[4].position0.textureCoord.z);
  mTEST_ASSERT_FLOAT_EQUALS(0.0f, (*info.triangles)[4].position0.normal.x);
  mTEST_ASSERT_FLOAT_EQUALS(0.000f, (*info.triangles)[4].position0.normal.y);
  mTEST_ASSERT_FLOAT_EQUALS(1.f, (*info.triangles)[4].position0.normal.z);

  mTEST_ASSERT_EQUAL((*info.triangles)[4].position1.position, (*info.vertices)[5].position);
  mTEST_ASSERT_EQUAL((*info.triangles)[4].position1.hasColour, (*info.vertices)[5].hasColour);
  mTEST_ASSERT_TRUE((bool)(*info.triangles)[4].position1.hasNormal);
  mTEST_ASSERT_TRUE((bool)(*info.triangles)[4].position1.hasTextureCoord);
  mTEST_ASSERT_FLOAT_EQUALS(0.25f, (*info.triangles)[4].position1.textureCoord.x);
  mTEST_ASSERT_FLOAT_EQUALS(0.5f, (*info.triangles)[4].position1.textureCoord.y);
  mTEST_ASSERT_FLOAT_EQUALS(0.f, (*info.triangles)[4].position1.textureCoord.z);
  mTEST_ASSERT_FLOAT_EQUALS(.01f, (*info.triangles)[4].position1.normal.x);
  mTEST_ASSERT_FLOAT_EQUALS(-.1f, (*info.triangles)[4].position1.normal.y);
  mTEST_ASSERT_FLOAT_EQUALS(.0f, (*info.triangles)[4].position1.normal.z);

  mTEST_ASSERT_EQUAL((*info.triangles)[4].position2.position, (*info.vertices)[0].position);
  mTEST_ASSERT_EQUAL((*info.triangles)[4].position2.hasColour, (*info.vertices)[0].hasColour);
  mTEST_ASSERT_TRUE((bool)(*info.triangles)[4].position2.hasNormal);
  mTEST_ASSERT_TRUE((bool)(*info.triangles)[4].position2.hasTextureCoord);
  mTEST_ASSERT_FLOAT_EQUALS(0.5f, (*info.triangles)[4].position2.textureCoord.x);
  mTEST_ASSERT_FLOAT_EQUALS(0.f, (*info.triangles)[4].position2.textureCoord.y);
  mTEST_ASSERT_FLOAT_EQUALS(0.f, (*info.triangles)[4].position2.textureCoord.z);
  mTEST_ASSERT_FLOAT_EQUALS(1.1f, (*info.triangles)[4].position2.normal.x);
  mTEST_ASSERT_FLOAT_EQUALS(1234567890.f, (*info.triangles)[4].position2.normal.y);
  mTEST_ASSERT_FLOAT_EQUALS(0.0f, (*info.triangles)[4].position2.normal.z);

  mTEST_ASSERT_EQUAL((*info.triangles)[5].position0.position, (*info.vertices)[0].position);
  mTEST_ASSERT_EQUAL((*info.triangles)[5].position0.hasColour, (*info.vertices)[0].hasColour);
  mTEST_ASSERT_TRUE((bool)(*info.triangles)[5].position0.hasNormal);
  mTEST_ASSERT_FALSE((bool)(*info.triangles)[5].position0.hasTextureCoord);
  mTEST_ASSERT_FLOAT_EQUALS(0.0f, (*info.triangles)[5].position0.normal.x);
  mTEST_ASSERT_FLOAT_EQUALS(0.000f, (*info.triangles)[5].position0.normal.y);
  mTEST_ASSERT_FLOAT_EQUALS(1.f, (*info.triangles)[5].position0.normal.z);

  mTEST_ASSERT_EQUAL((*info.triangles)[5].position1.position, (*info.vertices)[1].position);
  mTEST_ASSERT_EQUAL((*info.triangles)[5].position1.hasColour, (*info.vertices)[1].hasColour);
  mTEST_ASSERT_TRUE((bool)(*info.triangles)[5].position1.hasNormal);
  mTEST_ASSERT_FALSE((bool)(*info.triangles)[5].position1.hasTextureCoord);
  mTEST_ASSERT_FLOAT_EQUALS(1.1f, (*info.triangles)[5].position1.normal.x);
  mTEST_ASSERT_FLOAT_EQUALS(1234567890.f, (*info.triangles)[5].position1.normal.y);
  mTEST_ASSERT_FLOAT_EQUALS(0.0f, (*info.triangles)[5].position1.normal.z);

  mTEST_ASSERT_EQUAL((*info.triangles)[5].position2.position, (*info.vertices)[2].position);
  mTEST_ASSERT_EQUAL((*info.triangles)[5].position2.hasColour, (*info.vertices)[2].hasColour);
  mTEST_ASSERT_EQUAL((*info.triangles)[5].position2.colour, (*info.vertices)[2].colour);
  mTEST_ASSERT_TRUE((bool)(*info.triangles)[5].position2.hasNormal);
  mTEST_ASSERT_FALSE((bool)(*info.triangles)[5].position2.hasTextureCoord);
  mTEST_ASSERT_FLOAT_EQUALS(.01f, (*info.triangles)[5].position2.normal.x);
  mTEST_ASSERT_FLOAT_EQUALS(-.1f, (*info.triangles)[5].position2.normal.y);
  mTEST_ASSERT_FLOAT_EQUALS(.0f, (*info.triangles)[5].position2.normal.z);

  mTEST_ASSERT_EQUAL((*info.lines)[0].position0.position, (*info.vertices)[0].position);
  mTEST_ASSERT_EQUAL((*info.lines)[0].position0.hasColour, (*info.vertices)[0].hasColour);

  mTEST_ASSERT_EQUAL((*info.lines)[0].position1.position, (*info.vertices)[1].position);
  mTEST_ASSERT_EQUAL((*info.lines)[0].position1.hasColour, (*info.vertices)[1].hasColour);

  mTEST_ASSERT_EQUAL((*info.lines)[1].position0.position, (*info.vertices)[1].position);
  mTEST_ASSERT_EQUAL((*info.lines)[1].position0.hasColour, (*info.vertices)[1].hasColour);

  mTEST_ASSERT_EQUAL((*info.lines)[1].position1.position, (*info.vertices)[2].position);
  mTEST_ASSERT_EQUAL((*info.lines)[1].position1.hasColour, (*info.vertices)[2].hasColour);
  mTEST_ASSERT_EQUAL((*info.lines)[1].position1.colour, (*info.vertices)[2].colour);

  mTEST_ASSERT_EQUAL((*info.lines)[2].position0.position, (*info.vertices)[2].position);
  mTEST_ASSERT_EQUAL((*info.lines)[2].position0.hasColour, (*info.vertices)[2].hasColour);
  mTEST_ASSERT_EQUAL((*info.lines)[2].position0.colour, (*info.vertices)[2].colour);

  mTEST_ASSERT_EQUAL((*info.lines)[2].position1.position, (*info.vertices)[3].position);
  mTEST_ASSERT_EQUAL((*info.lines)[2].position1.hasColour, (*info.vertices)[3].hasColour);
  mTEST_ASSERT_EQUAL((*info.lines)[2].position1.colour, (*info.vertices)[3].colour);

  mTEST_ASSERT_EQUAL((*info.lines)[3].position0.position, (*info.vertices)[3].position);
  mTEST_ASSERT_EQUAL((*info.lines)[3].position0.hasColour, (*info.vertices)[3].hasColour);
  mTEST_ASSERT_EQUAL((*info.lines)[3].position0.colour, (*info.vertices)[3].colour);

  mTEST_ASSERT_EQUAL((*info.lines)[3].position1.position, (*info.vertices)[0].position);
  mTEST_ASSERT_EQUAL((*info.lines)[3].position1.hasColour, (*info.vertices)[0].hasColour);

  mTEST_ASSERT_EQUAL((*info.lines)[4].position0.position, (*info.vertices)[0].position);
  mTEST_ASSERT_EQUAL((*info.lines)[4].position0.hasColour, (*info.vertices)[0].hasColour);

  mTEST_ASSERT_EQUAL((*info.lines)[4].position1.position, (*info.vertices)[1].position);
  mTEST_ASSERT_EQUAL((*info.lines)[4].position1.hasColour, (*info.vertices)[1].hasColour);

  mTEST_ALLOCATOR_ZERO_CHECK();
}
