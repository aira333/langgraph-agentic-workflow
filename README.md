# LangGraph-Agentic Workflow

## Overview
This project implements an agentic workflow system using LangGraph, LangChain, Flask, and Groq LLMs. It enables users to submit queries which are broken down into sub-tasks, solved iteratively with tool use and reflection, and then synthesized into a final answer. The architecture is inspired by the classic outer/inner loop agentic design.

---

## Technologies Used
- **Python 3.10+**
- **Flask**: Web application framework for the API and UI
- **LangGraph**: For agentic workflow orchestration
- **LangChain**: LLM and tool integration
- **Groq API**: Fast, free LLMs (Llama 3 70B, etc.)
- **dotenv**: For environment variable management
- **Pydantic**: Data validation and parsing

---

## Setup Instructions
1. **Clone the repository:**
   ```bash
   git clone <repo-url>
   cd langgraph-agent
   ```
2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
3. **Set up environment variables:**
   - Copy `.env.example` to `.env` and add your Groq API key:
     ```env
     GROQ_API_KEY=your-groq-api-key
     ```
4. **Run the Flask app:**
   ```bash
   python app.py
   ```
5. **Visit the app:**
   - Open [http://localhost:5000](http://localhost:5000) in your browser.

---

## Usage
- Submit a query via the web interface or API.
- The agent will:
  1. **Plan:** Break down the query into sub-tasks (PlanAgent)
  2. **Execute:** Solve each sub-task using available tools (ToolAgent)
  3. **Reflect:** Critically assess results and suggest refinements (Reflection)
  4. **Iterate:** Refine, modify, or add tasks if needed
  5. **Finalize:** Synthesize all results into a final answer
- All steps are logged and visible in the UI and logs.

---

## Architecture
- **Outer Loop:**
  - **PlanAgent:** Splits the user query into sub-tasks
  - **Refinement:** Iteratively modifies, deletes, or adds sub-tasks based on feedback
  - **Finalization:** Produces the end result
- **Inner Loop:**
  - **ToolAgent:** Executes each sub-task, dispatches tools (e.g., search, calculator, weather)
  - **Reflection:** Evaluates results, provides feedback for refinement

### Mapping to Implementation
- The implementation closely follows the architecture in your provided image:
  - **PlanAgent, ToolAgent, and Reflection** are implemented as separate functions/nodes in the LangGraph workflow.
  - **Outer loop** (planning, refinement, finalization) and **inner loop** (tool dispatch, reflection) are faithfully represented in code.
  - **Tool execution** is simulated or real, depending on configuration.
  - **Feedback/refinement** is handled after each sub-task.
- **Note:** The actual tool dispatch is modular; you can add more tools easily in `tools.py`.

---

## Troubleshooting
- If you see model errors, ensure you are using an exact supported model name from Groq (e.g., `llama3-70b-8192`).
- Check your `.env` for the correct `GROQ_API_KEY`.
- Logs are written to `flask_debug_output.txt` for debugging.

---

## Credits
- Architecture inspired by agentic loop research and open-source agent frameworks.
- Powered by [Groq](https://console.groq.com/), [LangGraph](https://github.com/langchain-ai/langgraph), and [LangChain](https://python.langchain.com/).
