import qai_hub as hub

compile_job = hub.get_job("j9aegxovs5")
print("Status:", compile_job.get_status().code)
print("Shapes:", getattr(compile_job, "shapes", None) or getattr(compile_job, "target_shapes", None))
compiled_model = compile_job.get_target_model()
