from transformers import GPT2LMHeadModel, GPT2Tokenizer
from googlesearch import search
import torch

class ChatApp:
    def __init__(self):
        # Load pretrained GPT-2 model and tokenizer
        try:
            self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
            self.model = GPT2LMHeadModel.from_pretrained('gpt2')
            print("Model and tokenizer loaded successfully.")
        except Exception as e:
            print(f"Error loading model or tokenizer: {e}")
            exit(1)

    def send_message(self, message: str) -> None:
        if message:
            print(f"You: {message}")
            self.process_message(message)

    def process_message(self, message: str) -> None:
        try:
            if "search for" in message.lower():
                query = message.lower().split("search for")[-1].strip()
                if query:
                    print(f"ChatGPT: Performing Google search for: {query}")
                    results = self.google_search(query)
                    for result in results:
                        print(result)
                else:
                    print("ChatGPT: No query provided for search.")
            else:
                input_ids = self.tokenizer.encode(message, return_tensors='pt')
                generated_ids = self.model.generate(
                    input_ids,
                    max_length=50,
                    num_return_sequences=1,
                    no_repeat_ngram_size=2,  # Prevents repeating n-grams
                    temperature=0.7,  # Controls randomness, higher value means more diverse outputs
                    pad_token_id=self.tokenizer.eos_token_id
                )
                generated_text = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
                print(f"ChatGPT: {generated_text}")
        except Exception as e:
            print(f"Error processing message: {e}")

    def google_search(self, query: str, num_results: int = 5) -> list:
        try:
            search_results = []
            for result in search(query, num_results=num_results, stop=num_results, pause=2):
                search_results.append(result)
            return search_results
        except Exception as e:
            print(f"Error during Google search: {e}")
            return []

if __name__ == "__main__":
    app = ChatApp()
    try:
        while True:
            user_input = input("You: ")
            if user_input.lower() in ["exit", "quit"]:
                print("Exiting chat...")
                break
            app.send_message(user_input)
    except KeyboardInterrupt:
        print("\nExiting chat...")
    except Exception as e:
        print(f"Unexpected error: {e}")
