"""This module provides example tools for web scraping and search functionality.

It includes a basic Tavily search function (as an example)

These tools are intended as free examples to get started. For production use,
consider implementing more robust and specialized tools tailored to your needs.
"""

import codecs
import os
import asyncio
from pathlib import Path
from datetime import datetime
from typing import Any, Callable, List, Optional, cast, Literal
from pydantic import BaseModel, Field
from langchain_tavily import TavilySearch
from langgraph.runtime import get_runtime
from langchain_core.tools import tool
from react_agent.context import Context
from react_agent.doc_processing.doc_process import process_document
import logging

logger = logging.getLogger(__name__)


class DocumentProcessInput(BaseModel):
    file_path: str = Field(
        description="String path to the document file to process (not the file contents)",
        example="/path/to/document.pdf"
    )


async def search(query: str) -> Optional[dict[str, Any]]:
    """Search for general web results.

    This function performs a search using the Tavily search engine, which is designed
    to provide comprehensive, accurate, and trusted results. It's particularly useful
    for answering questions about current events.
    """
    runtime = get_runtime(Context)
    wrapped = TavilySearch(max_results=runtime.context.max_search_results)
    return cast(dict[str, Any], await wrapped.ainvoke({"query": query}))

async def get_weather(city: Literal["nyc", "sf"]):
    """Use this to get weather information."""
    if city == "nyc":
        return "It might be cloudy in nyc"
    elif city == "sf":
        return "It's always sunny in sf"
    else:
        raise AssertionError("Unknown city")

@tool(args_schema=DocumentProcessInput)
async def process_singledocument(file_path: str) -> str:
    """Tool to process single document"""

    #Normalize filepath
    try:
        if isinstance(file_path, str):
            return f"Error: Expected string file path, got {type(file_path)}"
        
        #Normalize file path
        file_path = str(file_path).strip().strip('"').strip("'")

        #convert to Path object and resolve
        path_obj = Path(file_path).resolve()

        # Validate file exists and is accessible
        if not path_obj.exists():
            return f"Error: File does not exist: {path_obj}"
            
        if not path_obj.is_file():
            return f"Error: Path is not a file: {path_obj}"
            
        # Check file permissions
        if not os.access(path_obj, os.R_OK):
            return f"Error: Cannot read file (permission denied): {path_obj}"
    
        # Call the async process_document function
        loop = asyncio.get_event_loop()
        result = loop.run_until_complete(process_document(str(path_obj)))
            
        if isinstance(result, dict) and result.get("status") == "success":
            return f"Successfully processed document: {path_obj.name}. " \
                    f"Content length: {result.get('content_length', 0)} chars, " \
                    f"Processing time: {result.get('processing_time', 0):.2f}s"
        else:
            return f"Document processed: {path_obj.name}"
        
    except Exception as e:
        error_msg = f"Error processing document '{file_path}': {str(e)}"
        logger.error(error_msg)
        return error_msg
    
#TOOLS = List[Callable[..., Any]] = [search, get_weather]
TOOLS = [search, get_weather, process_document]
