from transformers import T5Tokenizer, T5ForConditionalGeneration
import os

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


def summarize_from_file(file_path):
    with open(file_path, "r") as f:
        text = f.read()
    return summarize_text(text)


def main():
    print("🧠 Welcome to the Interactive AI Summarizer!")
    print("Choose an option below:")
    print("1️⃣  Type or paste text manually")
    print("2️⃣  Summarize text from a file (e.g., input.txt)\n")

    choice = input("Enter 1 or 2: ").strip()
    output_path = "interactive_summarizer_result.txt"

    if choice == "1":
        text = input("\nEnter text to summarize:\n")
        summary = summarize_text(text)
    elif choice == "2":
        file_path = input("Enter the file name (e.g., input.txt): ").strip()
        if not os.path.exists(file_path):
            print(f"❌ File '{file_path}' not found!")
            return
        summary = summarize_from_file(file_path)
        with open(file_path, "r") as f:
            text = f.read()
    else:
        print("❌ Invalid choice. Please enter 1 or 2.")
        return

    # Display and save the summary
    print("\n✅ Summary:\n", summary)
    with open(output_path, "w") as f:
        f.write("Original Text:\n" + text + "\n\nSummary:\n" + summary)

    print(f"\n💾 Result saved to {output_path}\n")


if __name__ == "__main__":
    main()

