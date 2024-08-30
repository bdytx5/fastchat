import os
from flask import Flask, request, render_template_string
import threading
import sys
from transformers import AutoTokenizer
from cerebras.cloud.sdk import Cerebras
import weave

# Initialize Weave for logging
weave.init("cerebras_llama31_performance")

# Flask app initialization
app = Flask(__name__)

# Load API key from file if it exists
API_KEY_FILE = "cerebras_api_key.txt"
CEREBRAS_API_KEY = None

def load_api_key():
    global CEREBRAS_API_KEY
    if os.path.exists(API_KEY_FILE):
        with open(API_KEY_FILE, "r") as file:
            CEREBRAS_API_KEY = file.read().strip()
            if CEREBRAS_API_KEY:
                initialize_cerebras_client()

def save_api_key(api_key):
    global CEREBRAS_API_KEY
    CEREBRAS_API_KEY = api_key
    with open(API_KEY_FILE, "w") as file:
        file.write(api_key)
    initialize_cerebras_client()

def initialize_cerebras_client():
    global client
    client = Cerebras(api_key=CEREBRAS_API_KEY)

# Load the API key at startup
load_api_key()

# Check for --test and --port in the command line arguments
is_test = '--test' in sys.argv
port_index = sys.argv.index('--port') if '--port' in sys.argv else None
port = int(sys.argv[port_index + 1]) if port_index else 5001

# Initialize tokenizer from Hugging Face model
tokenizer = AutoTokenizer.from_pretrained("akjindal53244/Llama-3.1-Storm-8B")

# Determine if token-based or character-based context length is used
use_token_context = '--token_contextlen' in sys.argv
char_context_len = int(sys.argv[sys.argv.index('--char_contextlen') + 1]) if '--char_contextlen' in sys.argv else None
token_context_len = int(sys.argv[sys.argv.index('--token_contextlen') + 1]) if '--token_contextlen' in sys.argv else 8000
# Cache to hold previous chat messages and responses
chat_cache = []

def manage_cache(prompt):
    global chat_cache
    if use_token_context:
        # Tokenize the full conversation
        all_tokens = tokenizer.encode("".join(chat_cache + [prompt]), add_special_tokens=False)
        
        # Keep only the last token_context_len tokens
        if len(all_tokens) > token_context_len:
            all_tokens = all_tokens[-token_context_len:]
        
        # Decode back to text for processing
        chat_cache = [tokenizer.decode(all_tokens, skip_special_tokens=True)]
        print(len(chat_cache))
    else:
        # Character-based context management
        chat_cache.append(prompt)
        if len("".join(chat_cache)) > char_context_len:
            # Trim chat cache to fit within the character context length
            while len("".join(chat_cache)) > char_context_len:
                chat_cache.pop(0)

@app.route('/chat')
def chat():
    return render_template_string(html_content)

@app.route('/clear_chat', methods=['POST'])
def clear_chat():
    global chat_cache
    chat_cache.clear()
    return 'Cleared'

@app.route('/save_api_key', methods=['POST'])
def save_api_key_route():
    data = request.json
    api_key = data.get("api_key")
    if api_key:
        save_api_key(api_key)
    return "API Key Saved"




def perform_inference(prompt):
    if not CEREBRAS_API_KEY:
        return "API Key not set. Please enter your Cerebras API Key."
    
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
        model="llama3.1-70b",
    )

    response_content = chat_completion.choices[0].message.content
    return response_content

@app.route('/send_message', methods=['POST'])
def send_message():
    global chat_cache
    
    data = request.json
    new_prompt = f"User: {data['prompt']}\n"
    
    # Manage context based on token or character length
    manage_cache(new_prompt)
    full_prompt = ''.join(chat_cache)

    try:
        response_content = perform_inference(full_prompt)

        if "```" in response_content:
            parts = response_content.split("```")
            for i in range(1, len(parts), 2):
                code_content = parts[i].strip().split('\n', 1)
                if len(code_content) > 1 and code_content[0].strip() in [
                    "python", "cpp", "javascript", "java", "html", "css", "bash",
                    "csharp", "go", "ruby", "php", "swift", "r", "typescript", "kotlin", "dart"
                ]:
                    parts[i] = "<pre><code>" + code_content[1].strip() + "</code></pre>"
                else:
                    parts[i] = "<pre><code>" + parts[i].strip() + "</code></pre>"
            response_content = "".join(parts)
        else:
            response_content = f'<div class="bot-message">{response_content}</div>'

        api_response = f"Bot: {response_content}\n"
        chat_cache.append(api_response)

        return api_response
    except Exception as e:
        print(f"Exception caught: {e}")
        return str(e)


html_content = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Chat with API</title>
    <style>
        body, html {
            height: 100%;
            margin: 0;
            font-family: Arial, sans-serif;
            display: flex;
            flex-direction: column;
            background-color: #f8f9fa;
        }
        #chat-container {
            flex-grow: 1;
            overflow-y: auto;
            padding: 20px;
            box-sizing: border-box;
            margin-bottom: 70px; /* Leave space for the input field */
        }
        #input-container {
            position: fixed;
            bottom: 0;
            width: 100%;
            background-color: #f8f9fa;
            padding: 10px 0;
            box-shadow: 0 -2px 5px rgba(0, 0, 0, 0.1);
        }
        #userInput {
            width: calc(100% - 40px);
            margin: 0 20px;
            padding: 10px;
            border: 1px solid #ccc;
            border-radius: 5px;
            box-sizing: border-box;
            font-size: 16px;
            height: 50px; /* Fixed height */
            background-color: #fff;
            resize: none;
        }
        #apiKeyInput {
            width: 200px;
            margin-right: 10px;
            padding: 10px;
            border: 1px solid #ccc;
            border-radius: 5px;
            box-sizing: border-box;
            font-size: 16px;
            height: 40px;
            background-color: #fff;
        }
        #loader {
            position: fixed;
            bottom: 70px;
            width: 100%;
            text-align: center;
            display: none;
        }
        #chat {
            padding: 10px;
            word-wrap: break-word;
        }
        pre {
            background-color: #f4f4f4;
            border: 1px solid #ccc;
            padding: 10px;
            border-radius: 5px;
            overflow-x: auto;
        }
        code {
            font-family: Consolas, 'Courier New', Courier, monospace;
            color: #d63384;
        }
        .bot-message {
            border: 1px solid #007BFF;
            background-color: #E9F7FF;
            border-radius: 5px;
            padding: 10px;
            margin: 10px 0;
        }
        .user-message {
            border: 1px solid #333;
            background-color: #f0f0f0;
            border-radius: 5px;
            padding: 10px;
            margin: 10px 0;
            text-align: right;
        }
        .control-panel {
            display: flex;
            align-items: center;
            padding: 0 20px;
        }
        button {
            padding: 10px 20px;
            background-color: #007BFF;
            color: #fff;
            border: none;
            border-radius: 5px;
            cursor: pointer;
        }
        button:hover {
            background-color: #0056b3;
        }
        p {
            text-align: center;
            margin: 0;
            color: #333;
        }
    </style>
</head>
<body>
    <div id="chat-container">
        <div id="chat"></div>
    </div>
    <div id="input-container">
        <div class="control-panel">
            <input type="password" id="apiKeyInput" placeholder="Enter API Key" onkeydown="handleApiKeyInput(event)">
            <button onclick="clearChat()">Clear Chat</button>
        </div>
        <textarea id="userInput" placeholder="Hit shift+enter to send..." onkeydown="handleKeyDown(event)"></textarea>
    </div>
    <div id="loader">Loading...</div>
    <p>To send your message, hit Shift+Enter.</p>

    <script>
        function handleKeyDown(event) {
            if (event.key === 'Enter' && event.shiftKey) {
                event.preventDefault();
                sendMessage();
            }
        }

        function handleApiKeyInput(event) {
            if (event.key === 'Enter') {
                event.preventDefault();
                saveApiKey();
            }
        }

        function saveApiKey() {
            const apiKeyInput = document.getElementById('apiKeyInput');
            const apiKey = apiKeyInput.value.trim();
            if (apiKey !== '') {
                fetch('/save_api_key', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({ api_key: apiKey }),
                })
                .then(response => response.text())
                .then(data => {
                    console.log('API Key saved:', data);
                    apiKeyInput.value = '';  // Clear the input field
                })
                .catch(error => {
                    console.error('Error saving API key:', error);
                });
            }
        }

        function sendMessage() {
            const inputField = document.getElementById('userInput');
            const message = inputField.value.trim();

            if (message === '') return;

            inputField.value = '';
            const chatContainer = document.getElementById('chat');
            chatContainer.innerHTML += `<div class="user-message">User: ${message}</div>`;

            const loader = document.getElementById('loader');
            loader.style.display = 'block';

            fetch('/send_message', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ prompt: message }),
            })
            .then(response => response.text())
            .then(data => {
                loader.style.display = 'none';
                chatContainer.innerHTML += `<div class="bot-message">${data}</div>`;
                autoScrollToBottom();
            })
            .catch(error => {
                loader.style.display = 'none';
                chatContainer.innerHTML += `<div class="bot-message">Error: ${error}</div>`;
                autoScrollToBottom();
            });
        }

        function clearChat() {
            const chatContainer = document.getElementById('chat');
            chatContainer.innerHTML = '';
            fetch('/clear_chat', { method: 'POST' });
        }

        function autoScrollToBottom() {
            const chatContainer = document.getElementById('chat-container');
            chatContainer.scrollTop = chatContainer.scrollHeight;
        }
    </script>
</body>
</html>
"""

if __name__ == "__main__":
    t = threading.Thread(target=app.run, kwargs={'host': '0.0.0.0', 'port': 5001})
    t.start()
    print(f"Visit http://127.0.0.1:{port}/chat to start chatting.")
