# generate_response.py
def generate_response(model, tokenizer, prompt):
    inputs = tokenizer(prompt, return_tensors="pt").to('cuda')
    outputs = model.generate(inputs['input_ids'], max_length=100, num_return_sequences=1)
    response = tokenizer.decode(outputs[0].cpu(), skip_special_tokens=True)
    return response
