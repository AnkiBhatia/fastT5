from fastpegasus import export_and_get_onnx_model
from transformers import AutoTokenizer

model_name = 'google/pegasus-cnn_dailymail'
model = export_and_get_onnx_model(model_name)

tokenizer = AutoTokenizer.from_pretrained(model_name)
t_input = """PG&E stated it scheduled the blackouts in response to forecasts for high winds amid dry conditions. The aim is to reduce the risk of wildfires. Nearly 800 thousand customers were scheduled to be affected by the shutoffs which were expected to last through at least midday tomorrow."""
token = tokenizer(t_input, return_tensors='pt')

tokens = model.generate(input_ids=token['input_ids'],
               attention_mask=token['attention_mask'],
               num_beams=2)

output = tokenizer.decode(tokens.squeeze(), skip_special_tokens=True)
print(output)