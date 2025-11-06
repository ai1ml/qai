# safer way to inspect the schema of compile_job or compiled_model
schema_attr = None
for attr in ["shapes", "shape", "io_schema", "schema", "input_specs"]:
    if hasattr(compile_job, attr):
        schema_attr = attr
        schema = getattr(compile_job, attr)
        if callable(schema):  # sometimes it's a function
            schema = schema()
        print(f"\nFound attribute: {attr}")
        print(type(schema))
        try:
            # print cleanly even if complex objects exist
            for k, v in (schema.items() if hasattr(schema, "items") else enumerate(schema)):
                print("  ", k, ":", v)
        except Exception as e:
            print("Schema keys:", dir(schema))
        break

if schema_attr is None:
    print("⚠️ No schema-like attribute found. Try compile_job.__dict__")
    print(dir(compile_job))
