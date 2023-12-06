from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch

model_name = f"facebook/nllb-200-distilled-600M"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Check if GPU is available and use it; otherwise, use CPU
device = torch.device("cuda" if (torch.cuda.is_available()) else "cpu")
print(f"Using device: {device}")
model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)

def translate(text, src_lang, tgt_lang):
    # Load the model and tokenizer
 

    tokenizer.src_lang = src_lang
    tokenizer.tgt_lang = tgt_lang
    # Tokenize the text
    batch = tokenizer(text, return_tensors="pt").to(device)

    # Generate translation
    translated = model.generate(**batch,forced_bos_token_id=tokenizer.lang_code_to_id[tgt_lang])

    # Decode the translated text
    translated_text = tokenizer.batch_decode(translated, skip_special_tokens=True)
    return translated_text[0]

# # Example usage
# src_language = "eng_Latn"  # Source language (English in this case)
# tgt_language = "hin_Deva"  # Target language (French in this case)
# text_to_translate = "Hello, world!"

# translated_text = translate(text_to_translate, src_language, tgt_language)
# print(f"Translated text: {translated_text}")
