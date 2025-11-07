print("type:", type(compile_job).__name__)
print("job_type:", getattr(compile_job, "job_type", None))
print("status:", getattr(compile_job, "get_status", lambda:None)())
