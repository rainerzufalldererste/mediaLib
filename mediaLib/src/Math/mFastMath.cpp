#include "mediaLib.h"
#include "mFastMath.h"

mVector::mVector() :
  x(0),
  y(0),
  z(0),
  w(1)
{ }

mVector::mVector(const mVector &a) :
  v(a.v)
{ }

mVector::mVector(const float_t s)
{
  const DirectX::XMFLOAT4 f(s, s, s, s);
  v = DirectX::XMLoadFloat4(&f);
}

mVector mVector::Scalar(const float_t s) { return mVector(DirectX::XMLoadFloat(&s)); }

mVector::mVector(const float_t x, const float_t y)
{
  const DirectX::XMFLOAT2 f(x, y);
  v = DirectX::XMLoadFloat2(&f);
}

mVector::mVector(const float_t x, const float_t y, const float_t z)
{
  const DirectX::XMFLOAT3 f(x, y, z);
  v = DirectX::XMLoadFloat3(&f);
}

mVector::mVector(const float_t x, const float_t y, const float_t z, const float_t w)
{
  const DirectX::XMFLOAT4 f(x, y, z, w);
  v = DirectX::XMLoadFloat4(&f);
}

mVector::mVector(const mVec2f &v) : mVector(v.x, v.y) { }
mVector::mVector(const mVec3f &v) : mVector(v.x, v.y, v.z) { }
mVector::mVector(const mVec4f &v) : mVector(v.x, v.y, v.z, v.w) { }

mVector::mVector(DirectX::XMVECTOR _v) : v(_v) { }

mVector::mVector(DirectX::XMFLOAT2 _v) { v = DirectX::XMLoadFloat2(&_v); }
mVector::mVector(DirectX::XMFLOAT3 _v) { v = DirectX::XMLoadFloat3(&_v); }
mVector::mVector(DirectX::XMFLOAT4 _v) { v = DirectX::XMLoadFloat4(&_v); }

mQuaternion::mQuaternion(DirectX::XMVECTOR v) : q(v) { }

mMatrix::mMatrix(DirectX::XMMATRIX _m) : m(_m) { }
