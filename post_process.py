POSTPROCESSORS = {
    "clean_names": clean_name_list,
    # future:
    # "clean_ip": clean_ip_list,
}

# In your task runner:
if task.postprocess:
    func = POSTPROCESSORS[task.postprocess]
    data = func(data)
