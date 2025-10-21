from transformers import T5Tokenizer, T5ForConditionalGeneration

def summarize_text(text, min_length=20, max_length=60):
    model_name = "t5-small"
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name)

    input_text = "summarize: " + text
    inputs = tokenizer.encode(input_text, return_tensors="pt", truncation=True)

    summary_ids = model.generate(
        inputs,
        max_length=max_length,
        min_length=min_length,
        length_penalty=2.0,
        num_beams=4,
        early_stopping=True
    )

    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

if __name__ == "__main__":
    text = input("Enter text to summarize:\n")
    summary = summarize_text(text)
    print("\nSummary:\n", summary)

    # Save the summary result
    with open("text_summarizer_result.txt", "w") as file:
        file.write("Original Text:\n" + text + "\n\nSummary:\n" + summary)
    print("\n✅ Result saved to text_summarizer_result.txt")

