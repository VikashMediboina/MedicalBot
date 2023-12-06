import torch
from transformers import LlamaForCausalLM, LlamaTokenizer

print("Model loading...")

llm_path = "/home/mediboina.v/Vikash/medicalBot/LLM_HF"
tokenizer = LlamaTokenizer.from_pretrained(llm_path)

# Check if GPU is available and use it; otherwise, use CPU
device = torch.device("cuda" if (torch.cuda.is_available()) else "cpu")
print(f"Using device: {device}")

model = LlamaForCausalLM.from_pretrained(llm_path).to(device)
print("Model loaded")

def doctorOutput(patientInput, previousInput):
    prompt_template = f''' {previousInput}[INST]{patientInput}[/INST]'''
    inputs = tokenizer(prompt_template, return_tensors="pt").to(device)
    generate_ids = model.generate(inputs.input_ids, max_length=4096)
    output = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    return output

# Example usage
# print(doctorOutput("I have been having a headache for 2 days.", ""))
