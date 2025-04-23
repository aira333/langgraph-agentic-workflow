"""
Enhanced Agentic Workflow Implementation using Langgraph
-------------------------------------------------
This enhanced version integrates actual tools and adds state tracking.
"""

import os
from typing import Dict, List, Tuple, Any, Optional, Annotated, TypedDict, Sequence, Literal
import json
from pydantic import BaseModel, Field
import uuid
import datetime
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Ensure OpenAI API key is set
if not os.environ.get("OPENAI_API_KEY"):
    print("Warning: OPENAI_API_KEY environment variable not found.")
    print("Either set it in your environment or create a .env file.")

# Langgraph imports
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver

# LangChain imports
from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate

# Import the tools module
from tools import get_tool, execute_tool, get_available_tools_description

# For testing, if tools.py isn't available yet
def get_available_tools_description():
    return """
    - search: Search the web for information on a given query
    - calculator: Perform mathematical calculations
    - weather: Get weather information for a location
    - data_analysis: Analyze data from a provided dataset or description
    - news: Get recent news articles on a topic
    """

def execute_tool(tool_name: str, **kwargs):
    return f"Simulated execution of {tool_name} with parameters {kwargs}"

# Define enhanced state schema
class TaskTool(BaseModel):
    """Tool selected for a task."""
    name: str
    parameters: Dict[str, Any] = Field(default_factory=dict)
    
class TaskExecution(BaseModel):
    """Record of a task execution."""
    attempt: int
    tool: Optional[TaskTool] = None
    result: Optional[str] = None
    timestamp: str

class Task(BaseModel):
    """A single task to be completed."""
    id: str
    description: str
    status: str = "pending"  # pending, in_progress, completed, failed
    executions: List[TaskExecution] = Field(default_factory=list)
    reflection: Optional[str] = None
    
    def latest_result(self) -> Optional[str]:
        """Get the most recent execution result."""
        if not self.executions:
            return None
        return self.executions[-1].result

class AgentState(BaseModel):
    """The state of the agent workflow."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    user_query: str
    tasks: List[Task] = Field(default_factory=list)
    current_task_index: Optional[int] = None
    needs_refinement: bool = False
    refinement_feedback: Optional[str] = None
    final_answer: Optional[str] = None
    conversation_history: List[Dict[str, Any]] = Field(default_factory=list)
    # Add routing state
    next_step: str = "plan"  # Used for explicit routing
    thread_id: str = ""  # Used for checkpointing and persistence
    
    def current_task(self) -> Optional[Task]:
        """Get the current task being processed."""
        if self.current_task_index is not None and 0 <= self.current_task_index < len(self.tasks):
            return self.tasks[self.current_task_index]
        return None
    
    def all_tasks_completed(self) -> bool:
        """Check if all tasks are completed."""
        return all(task.status == "completed" for task in self.tasks)
    
    def get_task_summary(self) -> str:
        """Get a summary of all tasks and their status."""
        summary = []
        for i, task in enumerate(self.tasks):
            summary.append(f"Task {i+1}: {task.description} - Status: {task.status}")
            if task.executions:
                latest_exec = task.executions[-1]
                summary.append(f"  Latest execution result: {latest_exec.result}")
            if task.reflection:
                summary.append(f"  Reflection: {task.reflection}")
        return "\n".join(summary)
    
    def add_to_conversation(self, role: str, content: str):
        """Add a message to the conversation history."""
        self.conversation_history.append({
            "role": role,
            "content": content,
            "timestamp": datetime.datetime.now().isoformat()
        })

# Use Groq LLM via OpenAI-compatible API
from langchain_openai import ChatOpenAI

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
GROQ_BASE_URL = "https://api.groq.com/openai/v1"
GROQ_MODEL = "llama3-70b-8192"

planner_llm = ChatOpenAI(
    api_key=GROQ_API_KEY,
    base_url=GROQ_BASE_URL,
    model=GROQ_MODEL,
    temperature=0.1
)
tool_llm = ChatOpenAI(
    api_key=GROQ_API_KEY,
    base_url=GROQ_BASE_URL,
    model=GROQ_MODEL,
    temperature=0.2
)
reflection_llm = ChatOpenAI(
    api_key=GROQ_API_KEY,
    base_url=GROQ_BASE_URL,
    model=GROQ_MODEL,
    temperature=0.1
)


# Enhanced PlanAgent with better parsing
def plan_agent(state: AgentState) -> AgentState:
    """
    Creates an initial plan by breaking down the user query into subtasks.
    """
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a planning agent that breaks down user queries into clear, specific sub-tasks.
        Each sub-task should be a single actionable step that can be completed independently.
        Generate between 2-5 sub-tasks based on the complexity of the query.
        
        Return your response in the following JSON format:
        {{
          "explanation": "Brief explanation of your plan",
          "tasks": [
            {{
              "id": "task_1",
              "description": "Detailed description of the first task"
            }},
            {{
              "id": "task_2",
              "description": "Detailed description of the second task"
            }},
            ...
          ]
        }}"""),
        ("user", "Please break down the following query into sub-tasks: {query}")
    ])
    
    response = planner_llm.invoke(prompt.format(query=state.user_query))
    
    # Parse the JSON response
    try:
        # Extract JSON from the response
        response_text = response.content
        if "```json" in response_text:
            response_text = response_text.split("```json")[1].split("```")[0].strip()
        elif "```" in response_text:
            response_text = response_text.split("```")[1].split("```")[0].strip()
        
        plan_data = json.loads(response_text)
        
        # Create tasks from the plan
        tasks = []
        for task_data in plan_data["tasks"]:
            tasks.append(Task(
                id=task_data["id"],
                description=task_data["description"]
            ))
        
        new_state = state.model_copy(deep=True)
        new_state.tasks = tasks
        new_state.current_task_index = 0  # Start with the first task
        
        # Record the planning in conversation history
        new_state.add_to_conversation("system", f"Created plan: {plan_data['explanation']}")
        new_state.add_to_conversation("system", f"Generated {len(tasks)} tasks")
        
        # Set next step
        new_state.next_step = "execute"
        
        return new_state
        
    except Exception as e:
        print(f"Error parsing plan: {e}")
        
        # Create a simple default task if parsing fails
        new_state = state.model_copy(deep=True)
        new_state.tasks = [Task(
            id="task_1",
            description=f"Complete the query: {state.user_query}"
        )]
        new_state.current_task_index = 0
        new_state.add_to_conversation("system", f"Error creating detailed plan. Created a simple task.")
        
        # Set next step
        new_state.next_step = "execute"
        
        return new_state

# Enhanced task refinement with structured output
def refine_tasks(state: AgentState) -> AgentState:
    """
    Refines the tasks based on feedback from execution or reflection.
    Can modify, delete, or add tasks.
    """
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a task refinement agent. Based on the feedback provided, 
        you should refine the task list by modifying, deleting, or adding tasks.
        Your goal is to ensure that the tasks will successfully address the user's query.
        
        Output a JSON with the following format:
        {
            "explanation": "Brief explanation of your refinements",
            "tasks": [
                {"id": "task_1", "description": "Task description", "status": "pending"},
                {"id": "task_2", "description": "Task description", "status": "pending"},
                ...
            ]
        }"""),
        ("user", """
        Original user query: {query}
        
        Current tasks:
        {task_summary}
        
        Feedback requiring refinement:
        {feedback}
        
        Available tools:
        {tools}
        
        Please refine the tasks accordingly.
        """)
    ])
    
    response = planner_llm.invoke(
        prompt.format(
            query=state.user_query,
            task_summary=state.get_task_summary(),
            feedback=state.refinement_feedback,
            tools=get_available_tools_description()
        )
    )
    
    # Parse the JSON response
    try:
        # Extract JSON from the response
        response_text = response.content
        if "```json" in response_text:
            response_text = response_text.split("```json")[1].split("```")[0].strip()
        elif "```" in response_text:
            response_text = response_text.split("```")[1].split("```")[0].strip()
            
        refinement_data = json.loads(response_text)
        
        # Create a copy of the state to modify
        new_state = state.model_copy(deep=True)
        
        # Replace tasks with the refined tasks
        new_tasks = []
        for task_data in refinement_data["tasks"]:
            # Preserve existing results if the task ID matches
            existing_executions = []
            existing_reflection = None
            
            for old_task in state.tasks:
                if old_task.id == task_data["id"]:
                    existing_executions = old_task.executions
                    existing_reflection = old_task.reflection
                    break
            
            new_task = Task(
                id=task_data["id"],
                description=task_data["description"],
                status=task_data["status"],
                executions=existing_executions,
                reflection=existing_reflection
            )
            new_tasks.append(new_task)
        
        new_state.tasks = new_tasks
        new_state.needs_refinement = False  # Reset the refinement flag
        new_state.refinement_feedback = None
        
        # If we were in the middle of execution, update the task index
        # to either the first pending task or reset if all are completed
        if any(task.status == "pending" for task in new_state.tasks):
            # Find the first pending task
            for i, task in enumerate(new_state.tasks):
                if task.status == "pending":
                    new_state.current_task_index = i
                    break
        else:
            # All tasks are either completed or failed
            new_state.current_task_index = None
        
        # Log the refinement
        new_state.add_to_conversation("system", f"Refined tasks: {refinement_data['explanation']}")
        
        # Determine next step
        if new_state.current_task_index is not None:
            new_state.next_step = "execute"
        else:
            new_state.next_step = "finalize"
        
        return new_state
        
    except Exception as e:
        # If parsing fails, handle gracefully and return the original state
        print(f"Error parsing refinement response: {e}")
        
        # Set error as feedback and keep needs_refinement flag
        new_state = state.model_copy(deep=True)
        new_state.refinement_feedback = f"The previous refinement failed. Please try again with clearer formatting. Error: {e}"
        new_state.add_to_conversation("system", f"Error in task refinement: {e}")
        
        # Try to continue if possible
        if new_state.current_task_index is not None:
            new_state.next_step = "execute"
        else:
            new_state.next_step = "finalize"
            
        return new_state

# Enhanced tool agent with actual tool execution
def tool_agent(state: AgentState) -> AgentState:
    """
    Executes the current task using the appropriate tools.
    """
    current_task = state.current_task()
    
    if not current_task:
        # No current task to execute
        new_state = state.model_copy(deep=True)
        new_state.next_step = "finalize"
        return new_state
    
    # Provide available tools and ask for tool selection
    tool_selection_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a tool agent responsible for executing tasks.
        Select the most appropriate tool for the current task from the available tools.
        
        Return your response in the following JSON format:
        {{
          "reasoning": "Your step-by-step reasoning about which tool to use",
          "selected_tool": "tool_name",
          "parameters": {{
            "param1": "value1",
            "param2": "value2",
            ...
          }}
        }}"""),
        ("user", """
        Original user query: {query}
        
        Current task to execute: {task_description}
        
        Available tools:
        {tools}
        
        Select the best tool and parameters for this task.
        """)
    ])
    
    tool_selection_response = tool_llm.invoke(
        tool_selection_prompt.format(
            query=state.user_query,
            task_description=current_task.description,
            tools=get_available_tools_description()
        )
    )
    
    # Parse the tool selection response
    try:
        # Extract JSON from the response
        response_text = tool_selection_response.content
        if "```json" in response_text:
            response_text = response_text.split("```json")[1].split("```")[0].strip()
        elif "```" in response_text:
            response_text = response_text.split("```")[1].split("```")[0].strip()
            
        tool_data = json.loads(response_text)
        
        # Execute the selected tool
        tool_name = tool_data["selected_tool"]
        parameters = tool_data["parameters"]
        
        tool_result = execute_tool(tool_name, **parameters)
        
        # Record the execution
        execution = TaskExecution(
            attempt=len(current_task.executions) + 1,
            tool=TaskTool(name=tool_name, parameters=parameters),
            result=tool_result,
            timestamp=datetime.datetime.now().isoformat()
        )
        
        # Update the state
        new_state = state.model_copy(deep=True)
        current_task_index = new_state.current_task_index
        
        if current_task_index is not None:
            # Add the execution to the task
            new_state.tasks[current_task_index].executions.append(execution)
            new_state.tasks[current_task_index].status = "completed"
            
            # Log the execution
            new_state.add_to_conversation(
                "system", 
                f"Executed tool '{tool_name}' for task '{current_task.description}'"
            )
            
            # Move to the next pending task, if any
            next_task_index = None
            for i in range(current_task_index + 1, len(new_state.tasks)):
                if new_state.tasks[i].status == "pending":
                    next_task_index = i
                    break
            
            if next_task_index is None:
                # If no next pending task, check from the beginning (in case we refined)
                for i in range(0, current_task_index):
                    if new_state.tasks[i].status == "pending":
                        next_task_index = i
                        break
            
            new_state.current_task_index = next_task_index
        
        # Set next step
        new_state.next_step = "reflect"
        
        return new_state
        
    except Exception as e:
        print(f"Error in tool execution: {e}")
        
        # Record the error as a failed execution
        execution = TaskExecution(
            attempt=len(current_task.executions) + 1,
            tool=None,
            result=f"Error in tool selection or execution: {str(e)}",
            timestamp=datetime.datetime.now().isoformat()
        )
        
        # Update the state
        new_state = state.model_copy(deep=True)
        current_task_index = new_state.current_task_index
        
        if current_task_index is not None:
            # Add the failed execution to the task
            new_state.tasks[current_task_index].executions.append(execution)
            new_state.tasks[current_task_index].status = "failed"
            
            # Set refinement feedback
            new_state.needs_refinement = True
            new_state.refinement_feedback = f"Task execution failed: {str(e)}. Please refine the task."
            
            # Log the error
            new_state.add_to_conversation("system", f"Error in tool execution: {e}")
        
        # Set next step
        new_state.next_step = "reflect"
        
        return new_state

# Enhanced reflection agent with more detailed analysis
def reflection_agent(state: AgentState) -> AgentState:
    """
    Reflects on task execution to provide feedback for improvement.
    """
    current_task = state.current_task()
    
    if not current_task or current_task.status not in ["completed", "failed"]:
        # No task to reflect on or task not in a state for reflection
        new_state = state.model_copy(deep=True)
        
        if state.needs_refinement:
            new_state.next_step = "refine"
        elif new_state.current_task_index is not None:
            new_state.next_step = "execute"
        else:
            new_state.next_step = "finalize"
            
        return new_state
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a reflection agent that evaluates task execution results.
        Your job is to identify any issues, assess if the task was completed successfully,
        and provide feedback for improvement.
        
        For failed tasks, suggest specific improvements or alternative approaches.
        For successful tasks, validate that the result is correct and sufficient.
        
        Your reflection should be critical and specific about what went well and what could be improved.
        
        Return your reflection in the following format:
        {{
          "evaluation": "successful | partially_successful | failed",
          "reasoning": "Your detailed analysis of the task execution",
          "needs_refinement": true | false,
          "refinement_suggestion": "Specific suggestion for refinement if needed"
        }}"""),
        ("user", """
        Original user query: {query}
        
        Task: {task_description}
        
        Execution result: {task_result}
        
        Please reflect on this task execution and provide feedback.
        """)
    ])
    
    latest_result = current_task.latest_result() or "No execution result available."
    
    response = reflection_llm.invoke(
        prompt.format(
            query=state.user_query,
            task_description=current_task.description,
            task_result=latest_result
        )
    )
    
    # Parse the reflection response
    try:
        # Extract JSON from the response
        response_text = response.content
        if "```json" in response_text:
            response_text = response_text.split("```json")[1].split("```")[0].strip()
        elif "```" in response_text:
            response_text = response_text.split("```")[1].split("```")[0].strip()
            
        reflection_data = json.loads(response_text)
        
        # Update the state with the reflection
        new_state = state.model_copy(deep=True)
        current_task_index = new_state.current_task_index
        
        if current_task_index is not None:
            # Add the reflection to the task
            new_state.tasks[current_task_index].reflection = reflection_data["reasoning"]
            
            # Determine if the task needs refinement
            if reflection_data.get("needs_refinement", False):
                new_state.needs_refinement = True
                new_state.refinement_feedback = reflection_data.get("refinement_suggestion", "Task needs improvement.")
                
                # Log the need for refinement
                new_state.add_to_conversation(
                    "system", 
                    f"Task needs refinement: {reflection_data.get('refinement_suggestion')}"
                )
        
        # Determine next step
        if new_state.needs_refinement:
            new_state.next_step = "refine"
        elif new_state.current_task_index is not None:
            new_state.next_step = "execute"
        else:
            new_state.next_step = "finalize"
        
        return new_state
        
    except Exception as e:
        print(f"Error in reflection: {e}")
        
        # Handle reflection failure gracefully
        new_state = state.model_copy(deep=True)
        
        if current_task.status == "failed":
            # If the task already failed, definitely needs refinement
            new_state.needs_refinement = True
            new_state.refinement_feedback = f"Task failed and reflection failed too. Please revise the task approach."
        
        # Log the error
        new_state.add_to_conversation("system", f"Error in reflection: {e}")
        
        # Determine next step
        if new_state.needs_refinement:
            new_state.next_step = "refine"
        elif new_state.current_task_index is not None:
            new_state.next_step = "execute"
        else:
            new_state.next_step = "finalize"
        
        return new_state

# Enhanced finalization with better synthesis
def finalize_answer(state: AgentState) -> AgentState:
    """
    Combines the results of all tasks to create a final answer.
    """
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a finalization agent that synthesizes the results of multiple tasks
        into a coherent final answer for the user. Make sure to incorporate relevant information
        from all the completed tasks and present a comprehensive response to the original query.
        
        Your response should be well-structured, clear, and directly address the user's query.
        
        If there were any limitations or caveats to the information gathered, mention them
        briefly at the end of your response."""),
        ("user", """
        Original user query: {query}
        
        Task results:
        {task_summary}
        
        Please synthesize these results into a final answer for the user.
        """)
    ])
    
    response = planner_llm.invoke(
        prompt.format(
            query=state.user_query,
            task_summary=state.get_task_summary()
        )
    )
    
    # Update the state with the final answer
    new_state = state.model_copy(deep=True)
    new_state.final_answer = response.content
    
    # Log the finalization
    new_state.add_to_conversation("system", "Generated final answer")
    new_state.add_to_conversation("assistant", new_state.final_answer)
    
    # Set next step to END
    new_state.next_step = "end"
    
    return new_state

# Initialize the workflow graph
workflow = StateGraph(AgentState)

# Add nodes to the graph
workflow.add_node("plan", plan_agent)
workflow.add_node("refine", refine_tasks)
workflow.add_node("execute", tool_agent)
workflow.add_node("reflect", reflection_agent)
workflow.add_node("finalize", finalize_answer)

# Add the START edge to the initial node (plan)
workflow.set_entry_point("plan")

# Define the conditional edges based on the state's next_step field
workflow.add_conditional_edges(
    "plan",
    lambda state: state.next_step,
    {
        "execute": "execute",
        "refine": "refine",
        "finalize": "finalize",
        "reflect": "reflect",
        "end": END
    }
)

workflow.add_conditional_edges(
    "refine",
    lambda state: state.next_step,
    {
        "execute": "execute",
        "finalize": "finalize",
        "reflect": "reflect",
        "plan": "plan",
        "end": END
    }
)

workflow.add_conditional_edges(
    "execute",
    lambda state: state.next_step,
    {
        "reflect": "reflect",
        "refine": "refine",
        "finalize": "finalize",
        "plan": "plan",
        "end": END
    }
)

workflow.add_conditional_edges(
    "reflect",
    lambda state: state.next_step,
    {
        "execute": "execute",
        "refine": "refine",
        "finalize": "finalize",
        "plan": "plan",
        "end": END
    }
)

workflow.add_conditional_edges(
    "finalize",
    lambda state: state.next_step,
    {
        "execute": "execute",
        "refine": "refine",
        "reflect": "reflect",
        "plan": "plan",
        "end": END
    }
)

# Add memory/checkpoint support
# Create a memory saver to persist state
memory = MemorySaver()

# Compile the graph with memory
workflow_app = workflow.compile(checkpointer=memory)

# Function to run the agent with a query
def run_agent(query: str, conversation_id: str = None) -> Dict:
    print(">>> run_agent CALLED <<<", flush=True)
    """
    Run the agent workflow with the given query.
    Optionally provide conversation_id to use as thread_id for checkpointing.
    Args:
        query: The user's query
        conversation_id: The conversation ID to use for checkpointing (thread_id)
    Returns:
        The final state of the agent
    """
    # Use conversation_id as thread_id if provided, else generate
    thread_id = conversation_id if conversation_id else str(uuid.uuid4())
    initial_state = AgentState(
        user_query=query,
        next_step="plan",
        thread_id=thread_id
    )
    initial_state.add_to_conversation("user", query)
    # Debug: Ensure thread_id is set before invoking the workflow
    if not initial_state.thread_id:
        print("ERROR: thread_id is missing in AgentState before workflow invocation!", flush=True)
        print("AgentState:", initial_state.dict(), flush=True)
        raise ValueError("thread_id is missing in AgentState before workflow invocation!")
    else:
        print("AgentState before workflow invoke:", initial_state.dict(), flush=True)

    state_dict = initial_state.dict()
    config = {"configurable": {"thread_id": initial_state.thread_id}}
    result = workflow_app.invoke(state_dict, config)
    # Ensure result is always an AgentState object
    if not isinstance(result, AgentState):
        result = AgentState(**result)
    return result

# Function to get a conversation by ID
def get_conversation(conversation_id: str) -> Optional[AgentState]:
    """
    Get a conversation by ID from memory.
    
    Args:
        conversation_id: The ID of the conversation to retrieve
        
    Returns:
        The conversation state, or None if not found
    """
    try:
        return memory.get(conversation_id)
    except:
        return None

# Example usage with error handling
if __name__ == "__main__":
    try:
        # Example query
        user_query = "Research the latest developments in renewable energy technologies and create a summary of the top 3 innovations with the highest potential impact."
        
        print(f"Running agent with query: {user_query}")
        print("-" * 80)
        
        # Always provide a conversation_id for checkpointing
        final_state = run_agent(user_query, conversation_id="cli_test")
        
        print("\nFinal Task Summary:")
        print("-" * 80)
        print(final_state.get_task_summary())
        
        print("\nFinal Answer:")
        print("-" * 80)
        print(final_state.final_answer)
        
        print(f"\nConversation ID for reference: {final_state.id}")
        
    except Exception as e:
        print(f"Error running agent: {e}")
        
    
