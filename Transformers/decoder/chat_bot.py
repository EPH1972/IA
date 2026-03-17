import torch
import torch.nn as nn
import pandas as pd
from transformers import AutoTokenizer
from model import TransformerDecoder, load_data, initialize_tokenizer, create_pairs, tokenize_data


class ChatBot:
    def __init__(self, model_path=None):
        """Initialize the chatbot with tokenizer and optionally load a pre-trained model."""
        self.tokenizer = initialize_tokenizer()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize model
        self.model = TransformerDecoder(
            vocab_size=self.tokenizer.vocab_size,
            block_size=512,
            embedding_dim=768,
            num_heads=12,
            num_layers=6,
            dropout=0.1
        ).to(self.device)
        
        print(f"ChatBot initialized on {self.device}")
        print(f"Vocabulary size: {self.tokenizer.vocab_size}")
    
    def predict_next_word(self, context, top_k=5):
        """
        Predict the next word given a context of 5 words.
        
        Args:
            context: String with 5 or fewer words
            top_k: Return top k predictions
        
        Returns:
            List of tuples (word, probability)
        """
        words = context.strip().split()
        
        # Check if we have enough words
        if len(words) < 5:
            print(f"Please provide 5 words. You provided {len(words)}.")
            return None
        
        # Take only last 5 words if more provided
        if len(words) > 5:
            words = words[-5:]
        
        # Tokenize the context
        context_str = ' '.join(words)
        tokens = self.tokenizer(context_str, return_tensors="pt", padding=True, truncation=True)
        input_ids = tokens['input_ids'].to(self.device)
        
        # Get model prediction
        with torch.no_grad():
            logits, _ = self.model(input_ids)
            # Take last position logits
            last_logits = logits[0, -1, :]
            probabilities = torch.softmax(last_logits, dim=-1)
        
        # Get top k predictions
        top_probs, top_indices = torch.topk(probabilities, top_k)
        
        predictions = []
        for prob, idx in zip(top_probs, top_indices):
            word = self.tokenizer.decode([idx.item()])
            predictions.append((word, prob.item()))
        
        return predictions
    
    def chat(self):
        """Interactive chatbot loop."""
        print("\n" + "="*60)
        print("🤖 Welcome to the Transformer Decoder ChatBot!")
        print("="*60)
        print("Instructions:")
        print("- Provide 5 words, and I'll predict the 6th word")
        print("- Type 'quit' to exit")
        print("="*60 + "\n")
        
        while True:
            try:
                context = input("You: ").strip()
                
                if context.lower() == 'quit':
                    print("Bot: Goodbye! 👋")
                    break
                
                if not context:
                    print("Bot: Please enter some words.\n")
                    continue
                
                predictions = self.predict_next_word(context)
                
                if predictions is None:
                    print()
                    continue
                
                print("\nBot: Top predictions for the next word:")
                for i, (word, prob) in enumerate(predictions, 1):
                    print(f"  {i}. '{word}' ({prob*100:.2f}%)")
                print()
                
            except Exception as e:
                print(f"Bot: Error - {str(e)}\n")


if __name__ == "__main__":
    bot = ChatBot()
    bot.chat()
