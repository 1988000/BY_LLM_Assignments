from transformers import T5Tokenizer, T5ForConditionalGeneration

def summarize_paragraphs(file_path, output_path, min_length=20, max_length=60):
    model_name = "t5-small"
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name)

    with open(file_path, "r") as f:
        paragraphs = f.read().split("\n\n")

    all_summaries = []
    for i, paragraph in enumerate(paragraphs, 1):
        if not paragraph.strip():
            continue
        print(f"Summarizing paragraph {i}...")
        input_text = "summarize: " + paragraph
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
        all_summaries.append(f"Paragraph {i} Summary:\n{summary}\n")

    with open(output_path, "w") as f:
        f.write("\n\n".join(all_summaries))

    print(f"\n✅ Summaries saved to {output_path}")

if __name__ == "__main__":
    input_file = "multi_text_input.txt"
    output_file = "multi_text_summarizer_result.txt"

    print("Reading paragraphs from:", input_file)
    summarize_paragraphs(input_file, output_file)

