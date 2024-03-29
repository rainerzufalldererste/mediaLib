#include "mTestLib.h"

mTEST(mSharedPointer, TestCastDerived)
{
  mTEST_ALLOCATOR_SETUP();

  struct Base
  {
    size_t a;
    size_t *pData;
  };

  struct Derived : Base
  {
    size_t b;
    size_t *pDerivedData;
  };

  {
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
  }

  {
    mPtr<Derived> derived;
    size_t derived_data = 0;
    mTEST_ASSERT_SUCCESS(mSharedPointer_Allocate<Derived>(&derived, pAllocator, [](Derived *pData) { (*pData->pDerivedData)++; }, 1));

    derived->pDerivedData = &derived_data;

    mPtr<Base> base;
    size_t base_data = 10;
    mTEST_ASSERT_SUCCESS(mSharedPointer_Allocate<Base>(&base, pAllocator, [](Base *pData) { (*pData->pData)++; }, 1));

    base->pData = &base_data;

    mPtr<Base> tmp = (mPtr<Base>)derived;
    derived = (mPtr<Derived>)base;
    base = tmp;

    mTEST_ASSERT_SUCCESS(mSharedPointer_Destroy(&tmp));

    mTEST_ASSERT_SUCCESS(mSharedPointer_Destroy(&derived));
    mTEST_ASSERT_EQUAL(base_data, 11);

    mTEST_ASSERT_SUCCESS(mSharedPointer_Destroy(&base));
    mTEST_ASSERT_EQUAL(derived_data, 1);
  }

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
    mTEST_ASSERT_EQUAL(container.GetPointer(), nullptr);
  }

  {
    mUniqueContainer<size_t> container(1);

    {
      mPtr<size_t> ptr = container.ToPtr();
      mTEST_ASSERT_EQUAL(ptr.GetPointer(), reinterpret_cast<decltype(container.GetPointer())>(container.m_value));
    }
  }

  {
    mUniqueContainer<size_t> container(1);

    mTEST_ASSERT_EQUAL(*container, 1);
    mTEST_ASSERT_EQUAL(container.GetPointer(), reinterpret_cast<decltype(container.GetPointer())>(container.m_value));
    mTEST_ASSERT_EQUAL(container.GetReferenceCount(), 1);

    *container = 5;
    mTEST_ASSERT_EQUAL(*container, 5);
    mTEST_ASSERT_EQUAL(*(container.GetPointer()), 5);
  }

  {
    mUniqueContainer<size_t> container = mUniqueContainer<size_t>(1);

    mTEST_ASSERT_EQUAL(*container, 1);
    mTEST_ASSERT_EQUAL(container.GetPointer(), reinterpret_cast<decltype(container.GetPointer())>(container.m_value));
    mTEST_ASSERT_EQUAL(container.GetReferenceCount(), 1);

    *container = 5;
    mTEST_ASSERT_EQUAL(*container, 5);
    mTEST_ASSERT_EQUAL(*(container.GetPointer()), 5);
  }

  {
    mUniqueContainer<size_t> container = std::move(mUniqueContainer<size_t>(1));

    mTEST_ASSERT_EQUAL(*container, 1);
    mTEST_ASSERT_EQUAL(container.GetPointer(), reinterpret_cast<decltype(container.GetPointer())>(container.m_value));
    mTEST_ASSERT_EQUAL(container.GetReferenceCount(), 1);

    *container = 5;
    mTEST_ASSERT_EQUAL(*container, 5);
    mTEST_ASSERT_EQUAL(*(container.GetPointer()), 5);
  }

  {
    mUniqueContainer<size_t> container(std::move(mUniqueContainer<size_t>(1)));

    mTEST_ASSERT_EQUAL(*container, 1);
    mTEST_ASSERT_EQUAL(container.GetPointer(), reinterpret_cast<decltype(container.GetPointer())>(container.m_value));
    mTEST_ASSERT_EQUAL(container.GetReferenceCount(), 1);

    *container = 5;
    mTEST_ASSERT_EQUAL(*container, 5);
    mTEST_ASSERT_EQUAL(*(container.GetPointer()), 5);
  }

  {
    size_t value = 1;

    {
      mUniqueContainer<size_t *> container;
      mUniqueContainer<size_t *>::ConstructWithCleanupFunction(&container, [](size_t **ppData) { **ppData += 10; }, &value);

      mTEST_ASSERT_EQUAL(*(*container), 1);
      mTEST_ASSERT_EQUAL(container.GetPointer(), reinterpret_cast<decltype(container.GetPointer())>(container.m_value));
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
      mUniqueContainer<size_t *> container;
      mUniqueContainer<size_t *>::ConstructWithCleanupFunction(&container, std::move([](size_t **ppData) { **ppData += 10; }), &value);

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

mTEST(mSharedPointer, TestSelfAssign)
{
  mTEST_ALLOCATOR_SETUP();

  mPtr<size_t> value;
  mTEST_ASSERT_SUCCESS(mSharedPointer_Allocate<size_t>(&value, pAllocator, 1234));

  *value = 10;

  struct _internal
  {
    static void AssignToSelf(mPtr<size_t> &val, const mPtr<size_t> &b)
    {
      val = b;
    }
  };

  _internal::AssignToSelf(value, value);

  mTEST_ASSERT_TRUE(value != nullptr);

  mTEST_ALLOCATOR_ZERO_CHECK();
}
