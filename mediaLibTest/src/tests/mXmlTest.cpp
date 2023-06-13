#include "mTestLib.h"
#include "mXml.h"
#include "mFile.h"

mTEST(mXmlReader, TestRead)
{
  mTEST_ALLOCATOR_SETUP();

  const char xmlString[] = R"XML(<tag_a>test<tag_b key="x">content</tag_b><tag_b key="value" key2="value2" b="c">content2</tag_b></tag_a>
)XML";
  const char filename[] = "test.xml";

  mPtr<mXmlReader> xmlReader;
  mDEFER_CALL(&xmlReader, mXmlReader_Destroy);
  mTEST_ASSERT_SUCCESS(mXmlReader_CreateFromString(&xmlReader, pAllocator, xmlString));
  mTEST_ASSERT_SUCCESS(mXmlReader_CreateFromString(&xmlReader, pAllocator, xmlString, mARRAYSIZE(xmlString)));
  mTEST_ASSERT_SUCCESS(mFile_WriteAllText("test.xml", xmlString));
  mTEST_ASSERT_SUCCESS(mXmlReader_CreateFromFile(&xmlReader, pAllocator, filename));

  mString key, value;

  mTEST_ASSERT_SUCCESS(mXmlReader_StepInto(xmlReader, "tag_a"));
  mTEST_ASSERT_SUCCESS(mXmlReader_GetContent(xmlReader, &value));
  mTEST_ASSERT_EQUAL(value, "test");
  mTEST_ASSERT_EQUAL(mR_ResourceNotFound, mXmlReader_GetAttributeProperty(xmlReader, "missingKey", &value));
  mTEST_ASSERT_EQUAL(mR_ResourceNotFound, mXmlReader_VisitAttribute(xmlReader, &key, &value));
  mTEST_ASSERT_EQUAL(mR_ResourceNotFound, mXmlReader_VisitNextAttribute(xmlReader, &key, &value));
  mTEST_ASSERT_SUCCESS(mXmlReader_Visit(xmlReader, &value));
  mTEST_ASSERT_EQUAL(value, "tag_b");
  mTEST_ASSERT_SUCCESS(mXmlReader_VisitAttribute(xmlReader, &key, &value));
  mTEST_ASSERT_EQUAL(key, "key");
  mTEST_ASSERT_EQUAL(value, "x");
  mTEST_ASSERT_EQUAL(mR_ResourceNotFound, mXmlReader_VisitNextAttribute(xmlReader, &key, &value));
  mTEST_ASSERT_SUCCESS(mXmlReader_GetContent(xmlReader, &value));
  mTEST_ASSERT_EQUAL(value, "content");
  mTEST_ASSERT_SUCCESS(mXmlReader_StepIntoNext(xmlReader));
  mTEST_ASSERT_SUCCESS(mXmlReader_VisitAttribute(xmlReader, &key, &value));
  mTEST_ASSERT_EQUAL(key, "key");
  mTEST_ASSERT_EQUAL(value, "value");
  mTEST_ASSERT_SUCCESS(mXmlReader_VisitNextAttribute(xmlReader, &key, &value));
  mTEST_ASSERT_EQUAL(key, "key2");
  mTEST_ASSERT_EQUAL(value, "value2");
  mTEST_ASSERT_SUCCESS(mXmlReader_VisitNextAttribute(xmlReader, &key, &value));
  mTEST_ASSERT_EQUAL(key, "b");
  mTEST_ASSERT_EQUAL(value, "c");
  mTEST_ASSERT_SUCCESS(mXmlReader_GetAttributeProperty(xmlReader, "key", &value));
  mTEST_ASSERT_EQUAL(value, "value");
  mTEST_ASSERT_SUCCESS(mXmlReader_GetAttributeProperty(xmlReader, "key2", &value));
  mTEST_ASSERT_EQUAL(value, "value2");
  mTEST_ASSERT_EQUAL(mR_ResourceNotFound, mXmlReader_GetAttributeProperty(xmlReader, "key3", &value));
  mTEST_ASSERT_SUCCESS(mXmlReader_GetContent(xmlReader, &value));
  mTEST_ASSERT_EQUAL(value, "content2");
  mTEST_ASSERT_EQUAL(mR_ResourceNotFound, mXmlReader_StepIntoNext(xmlReader));
  mTEST_ASSERT_SUCCESS(mXmlReader_StepInto(xmlReader, "tag_b"));
  mTEST_ASSERT_SUCCESS(mXmlReader_VisitNextAttribute(xmlReader, &key, &value));
  mTEST_ASSERT_EQUAL(key, "key");
  mTEST_ASSERT_EQUAL(value, "x");
  mTEST_ASSERT_SUCCESS(mXmlReader_VisitNext(xmlReader, &value));
  mTEST_ASSERT_EQUAL(value, "tag_b");
  mTEST_ASSERT_SUCCESS(mXmlReader_GetContent(xmlReader, &value));
  mTEST_ASSERT_EQUAL(value, "content2");
  mTEST_ASSERT_EQUAL(mR_ResourceNotFound, mXmlReader_StepInto(xmlReader, "tag_b"));
  mTEST_ASSERT_SUCCESS(mXmlReader_ExitTag(xmlReader));
  mTEST_ASSERT_SUCCESS(mXmlReader_ExitTag(xmlReader));
  mTEST_ASSERT_EQUAL(mR_ResourceStateInvalid, mXmlReader_ExitTag(xmlReader));

  mTEST_ALLOCATOR_ZERO_CHECK();
}

mTEST(mXmlWriter, TestWriteRead)
{
  mTEST_ALLOCATOR_SETUP();

  const char filename[] = "test.xml";

  // Write.
  {
    mPtr<mXmlWriter> writer;
    mDEFER_CALL(&writer, mXmlWriter_Destroy);
    mTEST_ASSERT_SUCCESS(mXmlWriter_Create(&writer, pAllocator, "tag_a"));
    mTEST_ASSERT_SUCCESS(mXmlWriter_AddContent(writer, "🌵"));
    mTEST_ASSERT_SUCCESS(mXmlWriter_BeginElement(writer, "tag_b"));
    mTEST_ASSERT_EQUAL(mR_InvalidParameter, mXmlWriter_BeginElement(writer, "🌵"));
    mTEST_ASSERT_EQUAL(mR_InvalidParameter, mXmlWriter_BeginElement(writer, ""));
    mTEST_ASSERT_SUCCESS(mXmlWriter_AddContent(writer, "content"));
    mTEST_ASSERT_SUCCESS(mXmlWriter_AddAttribute(writer, "key", "x"));
    mTEST_ASSERT_EQUAL(mR_InvalidParameter, mXmlWriter_AddAttribute(writer, "🌵", ""));
    mTEST_ASSERT_EQUAL(mR_InvalidParameter, mXmlWriter_AddAttribute(writer, "", ""));
    mTEST_ASSERT_SUCCESS(mXmlWriter_EndElement(writer));
    mTEST_ASSERT_SUCCESS(mXmlWriter_BeginElement(writer, "tag_b"));
    mTEST_ASSERT_SUCCESS(mXmlWriter_AddAttribute(writer, "key", " 🦎  🦎  "));
    mTEST_ASSERT_SUCCESS(mXmlWriter_AddAttribute(writer, "key2", "value2"));
    mTEST_ASSERT_SUCCESS(mXmlWriter_AddAttribute(writer, "b", "c"));
    mTEST_ASSERT_SUCCESS(mXmlWriter_AddContent(writer, "  content  2 "));

    mTEST_ASSERT_SUCCESS(mXmlWriter_ToFile(writer, filename));
  }

  // Validate.
  {
    mPtr<mXmlReader> xmlReader;
    mDEFER_CALL(&xmlReader, mXmlReader_Destroy);
    mTEST_ASSERT_SUCCESS(mXmlReader_CreateFromFile(&xmlReader, pAllocator, filename));

    mString key, value;

    mTEST_ASSERT_SUCCESS(mXmlReader_StepInto(xmlReader, "tag_a"));
    mTEST_ASSERT_SUCCESS(mXmlReader_GetContent(xmlReader, &value));
    mTEST_ASSERT_EQUAL(value, "🌵");
    mTEST_ASSERT_EQUAL(mR_ResourceNotFound, mXmlReader_GetAttributeProperty(xmlReader, "missingKey", &value));
    mTEST_ASSERT_EQUAL(mR_ResourceNotFound, mXmlReader_VisitAttribute(xmlReader, &key, &value));
    mTEST_ASSERT_EQUAL(mR_ResourceNotFound, mXmlReader_VisitNextAttribute(xmlReader, &key, &value));
    mTEST_ASSERT_SUCCESS(mXmlReader_Visit(xmlReader, &value));
    mTEST_ASSERT_EQUAL(value, "tag_b");
    mTEST_ASSERT_SUCCESS(mXmlReader_VisitAttribute(xmlReader, &key, &value));
    mTEST_ASSERT_EQUAL(key, "key");
    mTEST_ASSERT_EQUAL(value, "x");
    mTEST_ASSERT_EQUAL(mR_ResourceNotFound, mXmlReader_VisitNextAttribute(xmlReader, &key, &value));
    mTEST_ASSERT_SUCCESS(mXmlReader_GetContent(xmlReader, &value));
    mTEST_ASSERT_EQUAL(value, "content");
    mTEST_ASSERT_SUCCESS(mXmlReader_StepIntoNext(xmlReader));
    mTEST_ASSERT_SUCCESS(mXmlReader_VisitAttribute(xmlReader, &key, &value));
    mTEST_ASSERT_EQUAL(key, "key");
    mTEST_ASSERT_EQUAL(value, " 🦎  🦎  ");
    mTEST_ASSERT_SUCCESS(mXmlReader_VisitNextAttribute(xmlReader, &key, &value));
    mTEST_ASSERT_EQUAL(key, "key2");
    mTEST_ASSERT_EQUAL(value, "value2");
    mTEST_ASSERT_SUCCESS(mXmlReader_VisitNextAttribute(xmlReader, &key, &value));
    mTEST_ASSERT_EQUAL(key, "b");
    mTEST_ASSERT_EQUAL(value, "c");
    mTEST_ASSERT_SUCCESS(mXmlReader_GetAttributeProperty(xmlReader, "key", &value));
    mTEST_ASSERT_EQUAL(value, " 🦎  🦎  ");
    mTEST_ASSERT_SUCCESS(mXmlReader_GetAttributeProperty(xmlReader, "key2", &value));
    mTEST_ASSERT_EQUAL(value, "value2");
    mTEST_ASSERT_EQUAL(mR_ResourceNotFound, mXmlReader_GetAttributeProperty(xmlReader, "key3", &value));
    mTEST_ASSERT_SUCCESS(mXmlReader_GetContent(xmlReader, &value));
    mTEST_ASSERT_EQUAL(value, "  content  2 ");
    mTEST_ASSERT_EQUAL(mR_ResourceNotFound, mXmlReader_StepIntoNext(xmlReader));
    mTEST_ASSERT_SUCCESS(mXmlReader_StepInto(xmlReader, "tag_b"));
    mTEST_ASSERT_SUCCESS(mXmlReader_VisitNextAttribute(xmlReader, &key, &value));
    mTEST_ASSERT_EQUAL(key, "key");
    mTEST_ASSERT_EQUAL(value, "x");
    mTEST_ASSERT_SUCCESS(mXmlReader_VisitNext(xmlReader, &value));
    mTEST_ASSERT_EQUAL(value, "tag_b");
    mTEST_ASSERT_SUCCESS(mXmlReader_GetContent(xmlReader, &value));
    mTEST_ASSERT_EQUAL(value, "  content  2 ");
    mTEST_ASSERT_EQUAL(mR_ResourceNotFound, mXmlReader_StepInto(xmlReader, "tag_b"));
    mTEST_ASSERT_SUCCESS(mXmlReader_ExitTag(xmlReader));
    mTEST_ASSERT_SUCCESS(mXmlReader_ExitTag(xmlReader));
    mTEST_ASSERT_EQUAL(mR_ResourceStateInvalid, mXmlReader_ExitTag(xmlReader));
  }

  mTEST_ALLOCATOR_ZERO_CHECK();
}
