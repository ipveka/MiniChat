# UI module
"""
UI components for MiniChat application.
"""

from miniLM.src.ui.chat import (
    render_chat_page,
    render_message,
    render_chat_input,
    handle_regenerate,
    build_chat_context,
)
from miniLM.src.ui.studio import (
    render_studio_page,
    render_upload_section,
    render_query_section,
    render_sources,
)
from miniLM.src.ui.agents import (
    render_agents_page,
    render_agent_form,
    render_agent_list,
)

__all__ = [
    # Chat UI
    "render_chat_page",
    "render_message",
    "render_chat_input",
    "handle_regenerate",
    "build_chat_context",
    # Studio UI
    "render_studio_page",
    "render_upload_section",
    "render_query_section",
    "render_sources",
    # Agents UI
    "render_agents_page",
    "render_agent_form",
    "render_agent_list",
]
