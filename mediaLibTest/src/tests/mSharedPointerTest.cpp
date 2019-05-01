#include "mTestLib.h"

mTEST(mSharedPointer, TestCastDerived)
{
  mTEST_ALLOCATOR_SETUP();

  struct Base
  {
    size_t a;
  };

  struct Derived : Base
  {
    size_t b;
  };

  mPtr<Base> a;
  mPtr<Base> b;
  mPtr<Derived> c;
  mPtr<Derived> d;

  a = c;
  b = std::move(c);

  mPtr<Base> e(d);
  mPtr<Base> f(std::move(d));

  mPtr<Derived> g(a);
  mPtr<Derived> h(std::move(b));

  mTEST_ALLOCATOR_ZERO_CHECK();
}

mTEST(mReferencePack, TestCreateCleanup)
{
  mTEST_ALLOCATOR_SETUP();

  {
    mReferencePack<size_t> refPack;
  }

  {
    mReferencePack<size_t> refPack;

    {
      mPtr<size_t> ptr = refPack.ToPtr();
      mTEST_ASSERT_TRUE(ptr == nullptr);
    }
  }

  {
    size_t value = 1;
    mReferencePack<size_t> refPack(&value);

    mTEST_ASSERT_EQUAL(*refPack, 1);
    mTEST_ASSERT_EQUAL(refPack.GetPointer(), &value);
    mTEST_ASSERT_EQUAL(refPack.GetReferenceCount(), 1);

    *refPack = 5;
    mTEST_ASSERT_EQUAL(*refPack, 5);
    mTEST_ASSERT_EQUAL(value, 5);
  }

  {
    size_t value = 1;
    mReferencePack<size_t> refPack = mReferencePack<size_t>(&value);

    mTEST_ASSERT_EQUAL(*refPack, 1);
    mTEST_ASSERT_EQUAL(refPack.GetPointer(), &value);
    mTEST_ASSERT_EQUAL(refPack.GetReferenceCount(), 1);

    *refPack = 5;
    mTEST_ASSERT_EQUAL(*refPack, 5);
    mTEST_ASSERT_EQUAL(value, 5);
  }

  {
    size_t value = 1;
    mReferencePack<size_t> refPack = std::move(mReferencePack<size_t>(&value));

    mTEST_ASSERT_EQUAL(*refPack, 1);
    mTEST_ASSERT_EQUAL(refPack.GetPointer(), &value);
    mTEST_ASSERT_EQUAL(refPack.GetReferenceCount(), 1);

    *refPack = 5;
    mTEST_ASSERT_EQUAL(*refPack, 5);
    mTEST_ASSERT_EQUAL(value, 5);
  }

  {
    size_t value = 1;
    mReferencePack<size_t> refPack(std::move(mReferencePack<size_t>(&value)));

    mTEST_ASSERT_EQUAL(*refPack, 1);
    mTEST_ASSERT_EQUAL(refPack.GetPointer(), &value);
    mTEST_ASSERT_EQUAL(refPack.GetReferenceCount(), 1);

    *refPack = 5;
    mTEST_ASSERT_EQUAL(*refPack, 5);
    mTEST_ASSERT_EQUAL(value, 5);
  }

  {
    size_t value = 1;

    {
      mReferencePack<size_t> refPack(&value, [](size_t *pData) { *pData += 10; });

      mTEST_ASSERT_EQUAL(*refPack, 1);
      mTEST_ASSERT_EQUAL(refPack.GetPointer(), &value);
      mTEST_ASSERT_EQUAL(refPack.GetReferenceCount(), 1);

      *refPack = 5;
      mTEST_ASSERT_EQUAL(*refPack, 5);
      mTEST_ASSERT_EQUAL(value, 5);

      {
        mPtr<size_t> sharedPointer = refPack.ToPtr();
        mTEST_ASSERT_EQUAL(sharedPointer.GetReferenceCount(), 2);

        *sharedPointer = 10;
        mTEST_ASSERT_EQUAL(*sharedPointer, 10);
        mTEST_ASSERT_EQUAL(*refPack, 10);
        mTEST_ASSERT_EQUAL(value, 10);
      }

      mTEST_ASSERT_EQUAL(*refPack, 10);
    }

    mTEST_ASSERT_EQUAL(value, 20);
  }

  {
    size_t value = 1;

    {
      mReferencePack<size_t> refPack(&value, std::move([](size_t *pData) { *pData += 10; }));

      mTEST_ASSERT_EQUAL(*refPack, 1);
      mTEST_ASSERT_EQUAL(refPack.GetPointer(), &value);
      mTEST_ASSERT_EQUAL(refPack.GetReferenceCount(), 1);

      *refPack = 5;
      mTEST_ASSERT_EQUAL(*refPack, 5);
      mTEST_ASSERT_EQUAL(value, 5);

      {
        mPtr<size_t> sharedPointer = refPack.ToPtr();
        mTEST_ASSERT_EQUAL(sharedPointer.GetReferenceCount(), 2);

        *sharedPointer = 10;
        mTEST_ASSERT_EQUAL(*sharedPointer, 10);
        mTEST_ASSERT_EQUAL(*refPack, 10);
        mTEST_ASSERT_EQUAL(value, 10);
      }

      mTEST_ASSERT_EQUAL(*refPack, 10);
    }

    mTEST_ASSERT_EQUAL(value, 20);
  }

  mTEST_ALLOCATOR_ZERO_CHECK();
}

mTEST(mUniqueContainer, TestCreateCleanup)
{
  mTEST_ALLOCATOR_SETUP();

  {
    mUniqueContainer<size_t> container;
  }

  {
    mUniqueContainer<size_t> container;

    {
      mPtr<size_t> ptr = container.ToPtr();
      mTEST_ASSERT_EQUAL(ptr.GetPointer(), &container.m_value);
    }
  }

  {
    mUniqueContainer<size_t> container(1);

    mTEST_ASSERT_EQUAL(*container, 1);
    mTEST_ASSERT_EQUAL(container.GetPointer(), &container.m_value);
    mTEST_ASSERT_EQUAL(container.GetReferenceCount(), 1);

    *container = 5;
    mTEST_ASSERT_EQUAL(*container, 5);
    mTEST_ASSERT_EQUAL(*(container.GetPointer()), 5);
  }

  {
    mUniqueContainer<size_t> container = mUniqueContainer<size_t>(1);

    mTEST_ASSERT_EQUAL(*container, 1);
    mTEST_ASSERT_EQUAL(container.GetPointer(), &container.m_value);
    mTEST_ASSERT_EQUAL(container.GetReferenceCount(), 1);

    *container = 5;
    mTEST_ASSERT_EQUAL(*container, 5);
    mTEST_ASSERT_EQUAL(*(container.GetPointer()), 5);
  }

  {
    mUniqueContainer<size_t> container = std::move(mUniqueContainer<size_t>(1));

    mTEST_ASSERT_EQUAL(*container, 1);
    mTEST_ASSERT_EQUAL(container.GetPointer(), &container.m_value);
    mTEST_ASSERT_EQUAL(container.GetReferenceCount(), 1);

    *container = 5;
    mTEST_ASSERT_EQUAL(*container, 5);
    mTEST_ASSERT_EQUAL(*(container.GetPointer()), 5);
  }

  {
    mUniqueContainer<size_t> container(std::move(mUniqueContainer<size_t>(1)));

    mTEST_ASSERT_EQUAL(*container, 1);
    mTEST_ASSERT_EQUAL(container.GetPointer(), &container.m_value);
    mTEST_ASSERT_EQUAL(container.GetReferenceCount(), 1);

    *container = 5;
    mTEST_ASSERT_EQUAL(*container, 5);
    mTEST_ASSERT_EQUAL(*(container.GetPointer()), 5);
  }

  {
    size_t value = 1;

    {
      mUniqueContainer<size_t *> container = mUniqueContainer<size_t *>::CreateWithCleanupFunction([](size_t **ppData) { **ppData += 10; }, &value);

      mTEST_ASSERT_EQUAL(*(*container), 1);
      mTEST_ASSERT_EQUAL(container.GetPointer(), &container.m_value);
      mTEST_ASSERT_EQUAL(*container.GetPointer(), &value);
      mTEST_ASSERT_EQUAL(container.GetReferenceCount(), 1);

      **container = 5;
      mTEST_ASSERT_EQUAL(**container, 5);
      mTEST_ASSERT_EQUAL(value, 5);

      {
        mPtr<size_t *> sharedPointer = container.ToPtr();
        mTEST_ASSERT_EQUAL(sharedPointer.GetReferenceCount(), 2);

        **sharedPointer = 10;
        mTEST_ASSERT_EQUAL(**sharedPointer, 10);
        mTEST_ASSERT_EQUAL(**container, 10);
      }

      mTEST_ASSERT_EQUAL(**container, 10);
    }

    mTEST_ASSERT_EQUAL(value, 20);
  }

  {
    size_t value = 1;

    {
      mUniqueContainer<size_t *> container = mUniqueContainer<size_t *>::CreateWithCleanupFunction(std::move([](size_t **ppData) { **ppData += 10; }), &value);

      mTEST_ASSERT_EQUAL(**container, 1);
      mTEST_ASSERT_EQUAL(*container.GetPointer(), &value);
      mTEST_ASSERT_EQUAL(container.GetReferenceCount(), 1);

      **container = 5;
      mTEST_ASSERT_EQUAL(**container, 5);
      mTEST_ASSERT_EQUAL(value, 5);

      {
        mPtr<size_t *> sharedPointer = container.ToPtr();
        mTEST_ASSERT_EQUAL(sharedPointer.GetReferenceCount(), 2);

        **sharedPointer = 10;
        mTEST_ASSERT_EQUAL(**sharedPointer, 10);
        mTEST_ASSERT_EQUAL(**container, 10);
        mTEST_ASSERT_EQUAL(value, 10);
      }

      mTEST_ASSERT_EQUAL(**container, 10);
    }

    mTEST_ASSERT_EQUAL(value, 20);
  }

  mTEST_ALLOCATOR_ZERO_CHECK();
}
