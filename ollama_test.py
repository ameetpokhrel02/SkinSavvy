# Import the actual ollama library
import ollama

# Send a prompt to the model
response = ollama.chat(
    model='llama3.2:1b',
    messages=[
        {
            'role': 'user',
            'content': 'Which product is recommeded for acne problems?',
        },
    ],
)

# Print the model's response
print(response['message']['content'])