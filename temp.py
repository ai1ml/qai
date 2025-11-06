# 1) See what fields exist
print("Attributes:", [a for a in dir(compile_job) if not a.startswith("_")])

# 2) If available, get a dict view
if hasattr(compile_job, "to_dict"):
    jd = compile_job.to_dict()
    print("Keys:", list(jd.keys()))
    print("job dict sample:", {k: jd[k] for k in list(jd)[:10]})
else:
    # Fallback: try __dict__
    try:
        print("job.__dict__ keys:", list(compile_job.__dict__.keys()))
    except Exception as e:
        print("No __dict__ available:", e)
