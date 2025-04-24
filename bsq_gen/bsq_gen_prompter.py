def zero_shot_gen_bsq(topic, n_bsqs, label_list, base_on_feature):
    return f"""You are an expert in the {topic} field, 
your task this time is to create {n_bsqs} Binary Subtask Questions (BSQ) (true/false questions) in natural language 
to be able to classify classes: [{ ','.join(label_list) }], 
based on {base_on_feature}"""
