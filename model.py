from transformers import T5Tokenizer, T5ForConditionalGeneration
from googletrans import Translator

# Initialize the model and tokenizer
model_name = "t5-small"
model = T5ForConditionalGeneration.from_pretrained(model_name)
tokenizer = T5Tokenizer.from_pretrained(model_name)

# Initialize the translator for language translation
translator = Translator()

def summarize_with_t5(text, language='en'):
    # Translate to English if the text is not in English
    if language != 'en':
        text = translator.translate(text, dest='en').text

    # Prepare the input for summarization
    input_text = "summarize: " + text
    input_tokenized = tokenizer.encode(input_text, return_tensors="pt", max_length=512, truncation=True)

    # Generate the summary
    summary_ids = model.generate(input_tokenized, max_length=100, min_length=5, length_penalty=2.0, num_beams=4, early_stopping=True)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    
    # Translate back to the selected language if needed
    if language != 'en':
        summary = translator.translate(summary, dest=language).text

    return summary
