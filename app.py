"""
Flask Web Application for Agentic Workflow
------------------------------------------
This provides a simple web interface to interact with the agentic workflow.
"""
import sys
sys.stdout = open('flask_debug_output.txt', 'w', buffering=1)
sys.stderr = sys.stdout
print(">>> FLASK APP STARTED <<<", flush=True)

import os
from flask import Flask, request, jsonify, render_template
import threading
import time
import json
from typing import Dict, Any, List, Optional

# Import the agent functionality
from agent import run_agent, get_conversation

# Create the Flask app
app = Flask(__name__)

# Store ongoing conversations
conversations = {}

# Create a lock for thread safety
lock = threading.Lock()

@app.route('/')
def index():
    """Render the main page."""
    return serve_template()

@app.route('/api/query', methods=['POST'])
def handle_query():
    """Handle a new user query."""
    data = request.json
    query = data.get('query')
    
    if not query:
        return jsonify({'error': 'No query provided'}), 400
    
    # Generate a unique ID for this conversation
    conversation_id = f"conv_{int(time.time())}"
    
    # Start the agent in a background thread
    def run_agent_thread():
        try:
            with lock:
                conversations[conversation_id] = {
                    'status': 'running',
                    'query': query,
                    'tasks': [],
                    'final_answer': None,
                    'updates': [{'type': 'status', 'content': 'Started processing query', 'timestamp': time.time()}]
                }
            
            result = run_agent(query, conversation_id)
            
            with lock:
                conversations[conversation_id] = {
                    'status': 'completed',
                    'query': query,
                    'tasks': [
                        {
                            'id': task.id,
                            'description': task.description,
                            'status': task.status,
                            'executions': [
                                {
                                    'tool': exe.tool.name if exe.tool else None,
                                    'parameters': exe.tool.parameters if exe.tool else {},
                                    'result': exe.result,
                                    'timestamp': exe.timestamp
                                } for exe in task.executions
                            ],
                            'reflection': task.reflection
                        } for task in result.tasks
                    ],
                    'final_answer': result.final_answer,
                    'updates': conversations[conversation_id]['updates'] + [
                        {'type': 'completion', 'content': 'Query processing completed', 'timestamp': time.time()}
                    ],
                    'state_id': result.id
                }
        except Exception as e:
            print("EXCEPTION IN run_agent_thread:", e, flush=True)
            import traceback; traceback.print_exc()
            with lock:
                if conversation_id in conversations:
                    conversations[conversation_id]['status'] = 'error'
                    conversations[conversation_id]['error'] = str(e)
                    conversations[conversation_id]['updates'].append(
                        {'type': 'error', 'content': f'Error: {str(e)}', 'timestamp': time.time()}
                    )
    
    # Start the thread
    threading.Thread(target=run_agent_thread).start()
    
    # Return the conversation ID
    return jsonify({
        'conversation_id': conversation_id,
        'status': 'processing'
    })

@app.route('/api/status/<conversation_id>', methods=['GET'])
def get_status(conversation_id):
    """Get the status of a conversation."""
    with lock:
        conversation = conversations.get(conversation_id)
    
    if not conversation:
        # Try to load from persistent storage
        state = get_conversation(conversation_id)
        
        if state:
            return jsonify({
                'status': 'completed',
                'query': state.user_query,
                'tasks': [
                    {
                        'id': task.id,
                        'description': task.description,
                        'status': task.status,
                        'executions': [
                            {
                                'tool': exe.tool.name if exe.tool else None,
                                'parameters': exe.tool.parameters if exe.tool else {},
                                'result': exe.result,
                                'timestamp': exe.timestamp
                            } for exe in task.executions
                        ],
                        'reflection': task.reflection
                    } for task in state.tasks
                ],
                'final_answer': state.final_answer,
                'state_id': state.id
            })
        
        return jsonify({'error': 'Conversation not found'}), 404
    
    return jsonify(conversation)

@app.route('/api/conversations', methods=['GET'])
def list_conversations():
    """List all ongoing conversations."""
    with lock:
        return jsonify({
            'conversations': [
                {
                    'id': conv_id,
                    'query': data['query'],
                    'status': data['status'],
                    'timestamp': data['updates'][0]['timestamp'] if data['updates'] else time.time()
                }
                for conv_id, data in conversations.items()
            ]
        })

@app.route('/templates/index.html')
def serve_template():
    """Serve the HTML template."""
    return """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Agentic Workflow Demo</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            line-height: 1.6;
            max-width: 800px;
            margin: 0 auto;
            padding: 20px;
        }
        .container {
            border: 1px solid #ddd;
            border-radius: 5px;
            padding: 20px;
            margin-bottom: 20px;
        }
        .form-group {
            margin-bottom: 15px;
        }
        .form-group label {
            display: block;
            margin-bottom: 5px;
            font-weight: bold;
        }
        .form-group textarea {
            width: 100%;
            padding: 8px;
            border: 1px solid #ddd;
            border-radius: 4px;
            min-height: 100px;
        }
        button {
            background-color: #4CAF50;
            color: white;
            padding: 10px 15px;
            border: none;
            border-radius: 4px;
            cursor: pointer;
        }
        button:hover {
            background-color: #45a049;
        }
        #result {
            white-space: pre-wrap;
        }
        .tasks {
            margin-top: 20px;
        }
        .task {
            border: 1px solid #eee;
            border-radius: 4px;
            padding: 10px;
            margin-bottom: 10px;
        }
        .task-header {
            display: flex;
            justify-content: space-between;
            margin-bottom: 5px;
        }
        .task-status {
            font-weight: bold;
        }
        .status-pending {
            color: #f39c12;
        }
        .status-completed {
            color: #2ecc71;
        }
        .status-failed {
            color: #e74c3c;
        }
        .task-execution {
            margin-left: 20px;
            border-left: 3px solid #ddd;
            padding-left: 10px;
            margin-top: 5px;
        }
        .task-reflection {
            margin-top: 10px;
            padding: 5px;
            background-color: #f8f9fa;
            border-radius: 4px;
        }
        #status {
            margin-bottom: 10px;
            padding: 10px;
            border-radius: 4px;
        }
        .status-running {
            background-color: #f0f8ff;
        }
        .status-completed {
            background-color: #f0fff0;
        }
        .status-error {
            background-color: #fff0f0;
        }
        #conversations {
            margin-top: 30px;
        }
        .conversation-item {
            cursor: pointer;
            padding: 10px;
            border: 1px solid #eee;
            margin-bottom: 5px;
            border-radius: 4px;
        }
        .conversation-item:hover {
            background-color: #f5f5f5;
        }
    </style>
</head>
<body>
    <h1>Agentic Workflow Demo</h1>
    <div class="container">
        <div class="form-group">
            <label for="query">Enter your query:</label>
            <textarea id="query" placeholder="e.g., Research the latest developments in renewable energy and summarize the top 3 innovations."></textarea>
        </div>
        <button id="submit">Submit Query</button>
        <div id="status" style="display: none;"></div>
        <div class="tasks" id="tasks" style="display: none;"></div>
        <div id="result" style="display: none;"></div>
    </div>
    
    <div id="conversations">
        <h2>Recent Conversations</h2>
        <div id="conversation-list"></div>
    </div>

    <script>
        let currentConversationId = null;
        let statusCheckInterval = null;

        document.getElementById('submit').addEventListener('click', async () => {
            const query = document.getElementById('query').value.trim();
            if (!query) {
                alert('Please enter a query');
                return;
            }

            // Clear previous results
            document.getElementById('result').textContent = '';
            document.getElementById('result').style.display = 'none';
            document.getElementById('tasks').innerHTML = '';
            document.getElementById('tasks').style.display = 'none';
            
            // Update status
            const statusEl = document.getElementById('status');
            statusEl.textContent = 'Processing query...';
            statusEl.className = 'status-running';
            statusEl.style.display = 'block';

            try {
                const response = await fetch('/api/query', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({ query })
                });

                const data = await response.json();
                
                if (response.ok) {
                    currentConversationId = data.conversation_id;
                    startStatusChecking();
                } else {
                    statusEl.textContent = `Error: ${data.error || 'Failed to process query'}`;
                    statusEl.className = 'status-error';
                }
            } catch (error) {
                statusEl.textContent = `Error: ${error.message}`;
                statusEl.className = 'status-error';
            }
        });

        function startStatusChecking() {
            if (statusCheckInterval) {
                clearInterval(statusCheckInterval);
            }
            
            statusCheckInterval = setInterval(checkStatus, 2000);
            checkStatus(); // Also check immediately
        }

        async function checkStatus() {
            if (!currentConversationId) return;
            
            try {
                const response = await fetch(`/api/status/${currentConversationId}`);
                
                if (!response.ok) {
                    clearInterval(statusCheckInterval);
                    document.getElementById('status').textContent = 'Error retrieving status';
                    document.getElementById('status').className = 'status-error';
                    return;
                }
                
                const data = await response.json();
                updateUI(data);
                
                // If completed or error, stop checking
                if (data.status === 'completed' || data.status === 'error') {
                    clearInterval(statusCheckInterval);
                    loadConversations(); // Refresh the conversation list
                }
            } catch (error) {
                document.getElementById('status').textContent = `Error: ${error.message}`;
                document.getElementById('status').className = 'status-error';
                clearInterval(statusCheckInterval);
            }
        }

        function updateUI(data) {
            // Update status
            const statusEl = document.getElementById('status');
            if (data.status === 'error' && data.error) {
                statusEl.textContent = `Status: error - ${data.error}`;
                statusEl.className = 'status-error';
                const resultEl = document.getElementById('result');
                resultEl.textContent = `Error details: ${data.error}`;
                resultEl.style.display = 'block';
            } else {
                statusEl.textContent = `Status: ${data.status}`;
                statusEl.className = `status-${data.status}`;
            }
            
            // Update tasks
            if (data.tasks && data.tasks.length > 0) {
                const tasksEl = document.getElementById('tasks');
                tasksEl.style.display = 'block';
                tasksEl.innerHTML = '<h3>Tasks:</h3>';
                
                data.tasks.forEach(task => {
                    const taskEl = document.createElement('div');
                    taskEl.className = 'task';
                    
                    const taskHeader = document.createElement('div');
                    taskHeader.className = 'task-header';
                    
                    const taskTitle = document.createElement('div');
                    taskTitle.textContent = task.description;
                    
                    const taskStatus = document.createElement('div');
                    taskStatus.className = `task-status status-${task.status}`;
                    taskStatus.textContent = task.status;
                    
                    taskHeader.appendChild(taskTitle);
                    taskHeader.appendChild(taskStatus);
                    taskEl.appendChild(taskHeader);
                    
                    // Add executions
                    if (task.executions && task.executions.length > 0) {
                        const latestExecution = task.executions[task.executions.length - 1];
                        
                        const executionEl = document.createElement('div');
                        executionEl.className = 'task-execution';
                        
                        if (latestExecution.tool) {
                            const toolEl = document.createElement('div');
                            toolEl.innerHTML = `<strong>Tool:</strong> ${latestExecution.tool}`;
                            executionEl.appendChild(toolEl);
                            
                            const paramsEl = document.createElement('div');
                            paramsEl.innerHTML = `<strong>Parameters:</strong> ${JSON.stringify(latestExecution.parameters)}`;
                            executionEl.appendChild(paramsEl);
                        }
                        
                        const resultEl = document.createElement('div');
                        resultEl.innerHTML = `<strong>Result:</strong> <pre>${latestExecution.result}</pre>`;
                        executionEl.appendChild(resultEl);
                        
                        taskEl.appendChild(executionEl);
                    }
                    
                    // Add reflection
                    if (task.reflection) {
                        const reflectionEl = document.createElement('div');
                        reflectionEl.className = 'task-reflection';
                        reflectionEl.innerHTML = `<strong>Reflection:</strong> ${task.reflection}`;
                        taskEl.appendChild(reflectionEl);
                    }
                    
                    tasksEl.appendChild(taskEl);
                });
            }
            
            // Update final answer
            if (data.final_answer) {
                const resultEl = document.getElementById('result');
                resultEl.textContent = data.final_answer;
                resultEl.style.display = 'block';
            }
        }

        async function loadConversations() {
            try {
                const response = await fetch('/api/conversations');
                
                if (!response.ok) {
                    console.error('Failed to load conversations');
                    return;
                }
                
                const data = await response.json();
                
                const listEl = document.getElementById('conversation-list');
                listEl.innerHTML = '';
                
                if (data.conversations.length === 0) {
                    listEl.innerHTML = '<p>No conversations yet</p>';
                    return;
                }
                
                data.conversations.forEach(conv => {
                    const item = document.createElement('div');
                    item.className = 'conversation-item';
                    item.dataset.id = conv.id;
                    
                    const date = new Date(conv.timestamp * 1000);
                    
                    item.innerHTML = `
                        <div><strong>${conv.query.substring(0, 50)}${conv.query.length > 50 ? '...' : ''}</strong></div>
                        <div>Status: ${conv.status} - ${date.toLocaleString()}</div>
                    `;
                    
                    item.addEventListener('click', () => loadConversation(conv.id));
                    
                    listEl.appendChild(item);
                });
            } catch (error) {
                console.error('Error loading conversations:', error);
            }
        }

        async function loadConversation(id) {
            try {
                const response = await fetch(`/api/status/${id}`);
                
                if (!response.ok) {
                    alert('Failed to load conversation');
                    return;
                }
                
                const data = await response.json();
                
                // Set as current conversation
                currentConversationId = id;
                
                // Update the query input
                document.getElementById('query').value = data.query;
                
                // Update the UI
                updateUI(data);
                
                // Show results section
                document.getElementById('status').style.display = 'block';
                
                // Scroll to top
                window.scrollTo(0, 0);
            } catch (error) {
                console.error('Error loading conversation:', error);
                alert(`Error: ${error.message}`);
            }
        }

        // Load conversations on page load
        document.addEventListener('DOMContentLoaded', loadConversations);
    </script>
</body>
</html>
    """

if __name__ == '__main__':
    # Set the port
    port = int(os.environ.get('PORT', 5000))
    
    # Run the app
    app.run(host='0.0.0.0', port=port, debug=True)