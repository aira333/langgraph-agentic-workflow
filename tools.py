# Mock tools implementation for testing
# This is a simplified version of tools.py for initial testing

import json
import datetime
from typing import Dict, Any, Optional

class BaseTool:
    name: str
    description: str
    
    def run(self, **kwargs) -> str:
        """Execute the tool with the given parameters"""
        raise NotImplementedError("Tool must implement run method")

class SearchTool(BaseTool):
    name = "search"
    description = "Search the web for information on a given query"
    
    def run(self, query: str) -> str:
        """Simulated search tool"""
        current_date = datetime.datetime.now().strftime("%Y-%m-%d")
        
        return json.dumps([
            {
                "title": f"Result for: {query}",
                "snippet": f"This is a simulated search result for '{query}'.",
                "url": f"https://example.com/search?q={query.replace(' ', '+')}",
            },
            {
                "title": f"More info about {query}",
                "snippet": f"Additional information about '{query}'.",
                "url": f"https://example.com/info?topic={query.replace(' ', '+')}",
            }
        ], indent=2)

class CalculatorTool(BaseTool):
    name = "calculator"
    description = "Perform mathematical calculations"
    
    def run(self, expression: str) -> str:
        """Simple calculator"""
        try:
            # Very limited set of operations for safety
            allowed_chars = set("0123456789+-*/().% ")
            if not all(c in allowed_chars for c in expression):
                return "Error: Expression contains invalid characters"
            
            result = eval(expression)
            return f"Result: {result}"
        except Exception as e:
            return f"Error calculating result: {str(e)}"

class WeatherTool(BaseTool):
    name = "weather"
    description = "Get weather information for a location"
    
    def run(self, location: str) -> str:
        """Simulated weather tool"""
        import random
        conditions = ["Sunny", "Cloudy", "Rainy", "Partly Cloudy"]
        
        return json.dumps({
            "location": location,
            "condition": random.choice(conditions),
            "temperature": random.randint(10, 30),
            "humidity": random.randint(30, 90),
        }, indent=2)

class DataAnalysisTool(BaseTool):
    name = "data_analysis"
    description = "Analyze data from a provided dataset or description"
    
    def run(self, data_description: str) -> str:
        """Simulated data analysis"""
        return f"""
Data Analysis Results for: {data_description}

This is a simulation of a data analysis tool.
Some simulated insights:
- The data shows interesting patterns
- Key metrics indicate positive trends
- The analysis suggests correlation between main factors
"""

class NewsTool(BaseTool):
    name = "news"
    description = "Get recent news articles on a topic"
    
    def run(self, topic: str) -> str:
        """Simulated news tool"""
        current_date = datetime.datetime.now().strftime("%Y-%m-%d")
        
        return json.dumps([
            {
                "title": f"Latest news on {topic}",
                "source": "News Daily",
                "date": current_date,
                "summary": f"This is a simulated news article about {topic}."
            },
            {
                "title": f"Recent developments in {topic}",
                "source": "The Times",
                "date": current_date,
                "summary": f"More simulated content about {topic}."
            }
        ], indent=2)

# Create a dictionary of available tools
AVAILABLE_TOOLS = {
    "search": SearchTool(),
    "calculator": CalculatorTool(),
    "weather": WeatherTool(),
    "data_analysis": DataAnalysisTool(),
    "news": NewsTool()
}

def get_tool(tool_name: str) -> Optional[BaseTool]:
    """Get a tool by name"""
    return AVAILABLE_TOOLS.get(tool_name)

def execute_tool(tool_name: str, **kwargs) -> str:
    """Execute a tool with the given parameters"""
    tool = get_tool(tool_name)
    if not tool:
        return f"Error: Tool '{tool_name}' not found"
    
    try:
        return tool.run(**kwargs)
    except Exception as e:
        return f"Error executing tool '{tool_name}': {str(e)}"

def get_available_tools_description() -> str:
    """Get a description of all available tools"""
    descriptions = []
    for name, tool in AVAILABLE_TOOLS.items():
        descriptions.append(f"- {name}: {tool.description}")
    
    return "\n".join(descriptions)