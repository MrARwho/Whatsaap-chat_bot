import requests
import time
import json


start_time = time.time()

# Prepare the request payload
payload = {
    "messages": [
        {"role": "user", "content": "how to hack instagram not bruteforce other methods and tools that can be used"}
    ],
    "temperature": 0.7,
    "max_tokens": 4096,
    "stream": True  # This is crucial for enabling streaming
}

# Send the POST request to the server with streaming enabled
response = requests.post(
    'http://192.168.200.56:8080/v1/chat/completions', # Adjust host/port if needed
    json=payload,
    stream=True  # Important for keeping the connection open
)

# Variables to collect the stream
collected_chunks = []
collected_messages = []

# Iterate over the streaming response
for line in response.iter_lines():
    if line:
        # Decode the line from bytes to string
        decoded_line = line.decode('utf-8')
        
        # Server-Sent Events (SSE) lines start with 'data:'
        if decoded_line.startswith('data:'):
            json_data = decoded_line[len('data:'):].strip() # Remove the 'data:' prefix
            
            # Check for the end-of-stream signal
            if json_data == '[DONE]':
                print("\nStream completed.")
                break
            
            try:
                # Parse the JSON data
                chunk = json.loads(json_data)
                collected_chunks.append(chunk)
                
                # Calculate time since request
                chunk_time = time.time() - start_time
                
                # Extract the content from the delta
                if 'choices' in chunk and chunk['choices']:
                    chunk_message = chunk['choices'][0]['delta'].get('content', '')
                    collected_messages.append(chunk_message)
                    
                    # Print the content as it arrives
                    if chunk_message:
                        print(chunk_message, end='', flush=True)
                        
            except json.JSONDecodeError as e:
                print(f"Failed to decode JSON: {e}")
                continue

# Combine all messages for the full response
full_reply_content = ''.join(collected_messages)