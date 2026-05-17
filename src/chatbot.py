"""
TechBot CLI - Command Line Interface for TechBot

Interactive chatbot for technology and IT questions.

Usage:
CPU
    python bot/chatbot.py --adapter models/techbot-finetuned
GPU
    python bot/chatbot.py --adapter models/techbot-finetuned --8bit
"""

import argparse
from typing import List, Dict
from model_handler import ModelHandler


class TechBotCLI:
    """Command-line interface for TechBot"""

    def __init__(self, model_name: str, adapter_path: str = None, load_in_8bit: bool = False):
        print("🚀 Initializing TechBot...")
        print("=" * 60)

        self.model_handler = ModelHandler(
            model_name=model_name,
            adapter_path=adapter_path,
            load_in_8bit=load_in_8bit
        )

        # Initial system prompt
        self.conversation_history: List[Dict[str, str]] = [
            {
                "role": "system",
                "content": (
                    "You are TechBot, an AI assistant specialized in technology and IT. "
                    "Provide accurate, concise technical answers. Include code examples if relevant."
                )
            }
        ]

        print("=" * 60)
        print("✅ TechBot is ready!")
        print("Type 'quit' or 'exit' to end conversation, 'clear' to reset history, 'help' for examples.")
        print("=" * 60)

    def clear_history(self):
        self.conversation_history = [self.conversation_history[0]]
        print("🧹 Conversation history cleared!")

    def show_help(self):
        examples = [
            "How do I implement authentication in Flask?",
            "Difference between REST and GraphQL APIs?",
            "Explain Docker containers and their benefits",
            "Optimize React component rendering",
            "Python decorators explained",
            "SQL vs NoSQL databases",
            "Handling async/await errors in JS",
            "Purpose of Python virtual environments"
        ]
        print("\n💡 Example Questions:")
        print("=" * 60)
        for i, ex in enumerate(examples, 1):
            print(f"{i}. {ex}")
        print("=" * 60)

    def chat(self, user_input: str) -> str:
        self.conversation_history.append({"role": "user", "content": user_input})
        print("\n🤖 TechBot is thinking...\n")

        response = self.model_handler.chat(
            messages=self.conversation_history,
            max_new_tokens=512,
            temperature=0.7,
            top_p=0.9
        )

        self.conversation_history.append({"role": "assistant", "content": response})
        return response

    def run(self):
        while True:
            try:
                user_input = input("\n👤 You: ").strip()
                if not user_input:
                    continue

                if user_input.lower() in ("quit", "exit", "q"):
                    print("\n👋 Goodbye!")
                    break

                if user_input.lower() == "clear":
                    self.clear_history()
                    continue

                if user_input.lower() == "help":
                    self.show_help()
                    continue

                response = self.chat(user_input)
                print(f"\n🤖 TechBot:\n{response}")
                print("-" * 60)

            except KeyboardInterrupt:
                print("\n👋 Interrupted. Goodbye!")
                break
            except Exception as e:
                print(f"\n❌ Error: {e}")


def main():
    parser = argparse.ArgumentParser(description="TechBot - Technology & IT Assistant")
    parser.add_argument("--model", type=str, default="meta-llama/Llama-3.2-1B-Instruct")
    parser.add_argument("--adapter", type=str, default=None)
    parser.add_argument("--8bit", action="store_true", help="Load model in 8-bit (GPU only)")
    args = parser.parse_args()

    bot = TechBotCLI(
        model_name=args.model,
        adapter_path=args.adapter,
        load_in_8bit=args.__dict__["8bit"]
    )
    bot.run()


if __name__ == "__main__":
    main()
