import qai_hub as hub

hub.login(api_token=QAIHUB_API_TOKEN)

compile_job.refresh()
print("Compile status:", compile_job.status)

if compile_job.status == "COMPLETED":
    compiled_model = compile_job.get_target_model()
    device = hub.Device("Samsung Galaxy S24 (Family)")
    
    profile_job = hub.submit_profile_job(
        model=compiled_model,
        device=device,
        name="mobilenetv2_profile_test"
    )

    print("Submitted profile job:", profile_job.id)
    profile_job.wait()

    print("Profile status:", profile_job.status)
    print("Model type:", profile_job.model_type)
    print("Message:", getattr(profile_job, "status_message", None))
else:
    print("Compile not complete — cannot profile yet.")
