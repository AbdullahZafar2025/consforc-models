# model4.py
from transformers import pipeline

# ---------- Load chatbot model ONCE ----------
print("Loading chatbot model...")

# Using Flan-T5 with a simple output length limit
chatbot = pipeline(
    "text2text-generation",
    model="google/flan-t5-base",
    device=-1,  # CPU
)

print("Chatbot loaded!")

# ---------- Simple chat function ----------
def chat(message: str) -> str:
    """
    Simple chat function with a default output token limit.
    """
    try:
        # Generate response with output token limit
        response = chatbot(
            message,
            max_length=100,   # default max tokens
            do_sample=True,
            temperature=0.8,  # Higher temperature for more variety
            top_p=0.9,        # Nucleus sampling to avoid repetition
            repetition_penalty=1.5,
            no_repeat_ngram_size=3,
            early_stopping=True
        )
        
        # Extract the response
        bot_response = response[0]['generated_text'].strip()
        
        # Additional cleaning to remove repetitive patterns
        words = bot_response.split()
        if len(words) > 3:
            if len(set(words[-3:])) == 1:  # Avoid repeated last 3 words
                words = words[:-2]
                bot_response = " ".join(words)
        
        # Validate response quality
        if not bot_response or len(bot_response) < 3:
            return "I'd be happy to chat! What would you like to talk about?"
        
        return bot_response
        
    except Exception as e:
        return f"Sorry, I encountered an issue: {str(e)}"

