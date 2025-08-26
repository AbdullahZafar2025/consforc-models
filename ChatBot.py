# model4.py
from transformers import pipeline

# ---------- Load chatbot model ONCE ----------
print("Loading chatbot model...")

# Using Flan-T5 with better token control
chatbot = pipeline(
    "text2text-generation",
    model="google/flan-t5-base",
    device=-1,  # CPU
)

print("Chatbot loaded!")

# ---------- Simple chat function ----------
def chat(message: str) -> str:
    """
    Simple chat function with proper token control to prevent repetition.
    """
    try:
        # Dynamic token calculation based on input
        input_tokens = len(message.split())
        
        # Calculate appropriate response length
        if input_tokens <= 3:  # Short inputs like "hi", "hello"
            min_tokens = 5
            max_tokens = 25
        elif input_tokens <= 10:  # Medium inputs
            min_tokens = 10
            max_tokens = 50
        else:  # Long inputs
            min_tokens = 15
            max_tokens = 100
        
        # Generate response with anti-repetition settings
        response = chatbot(
            message,
            min_length=min_tokens,
            max_length=max_tokens,
            do_sample=True,
            temperature=0.8,  # Higher temperature for more variety
            top_p=0.9,  # Nucleus sampling to avoid repetition
            repetition_penalty=1.5,  # Penalize repetition heavily
            no_repeat_ngram_size=3,  # Don't repeat 3-grams
            early_stopping=True
        )
        
        # Extract the response
        bot_response = response[0]['generated_text'].strip()
        
        # Additional cleaning to remove repetitive patterns
        words = bot_response.split()
        if len(words) > 3:
            # Remove if last 3 words are identical
            if len(set(words[-3:])) == 1:
                words = words[:-2]
                bot_response = " ".join(words)
        
        # Validate response quality
        if not bot_response or len(bot_response) < 3:
            return "I'd be happy to chat! What would you like to talk about?"
        
        return bot_response
        
    except Exception as e:
        return f"Sorry, I encountered an issue: {str(e)}"