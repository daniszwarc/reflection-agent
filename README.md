# reflection-agent

A basic Reflection agent implemented with LangGraph. The agent generates a tweet, receives structured critique from a simulated reviewer, then revises based on that critique — cycling through generate → reflect → generate until a fixed iteration limit is reached.

This was built as a learning project to understand the core reflection loop pattern before tackling more complex variants like Reflexion with tool use and web search.

---

## How it works

The loop is intentionally simple — two nodes, one conditional edge:

1. A **generator** writes or improves a tweet based on the conversation history
2. A **reflector** critiques it, evaluating length, virality, style, and tone
3. The critique is injected back into the message history as a `HumanMessage`, so the generator treats it as new input on the next pass
4. The loop exits when the message history exceeds 6 messages (3 full generate → reflect cycles)

The key design choice: the reflector's output is wrapped as a `HumanMessage` rather than an `AIMessage`. This keeps the conversation structure alternating user/assistant, which is what the generator prompt expects, and means no special handling is needed to pass critique between nodes.

---

## Graph architecture

```
    START
      │
      ▼
  ┌──────────┐
  │ generate │  Writes or revises the tweet
  └─────┬────┘
        │
  should_continue?
  (len(messages) > 6)
        │
   ┌────┴─────┐
   ▼          ▼
reflect      END
   │
   ▼
  ┌──────────┐
  │ reflect  │  Critiques the tweet, returns as HumanMessage
  └─────┬────┘
        │
        └──────────────────────▶ generate
```

The compiled graph is exported as `flow.png` at startup.

---

## Project structure

```
reflection-agent/
├── main.py       LangGraph graph definition, node functions, entry point
├── chains.py     generate_chain and reflect_chain with prompt templates
├── flow.png      Auto-generated graph diagram
└── pyproject.toml  Dependencies (managed with Poetry)
```

---

## Chains

Both chains share the same `ChatOpenAI` model instance and use `MessagesPlaceholder` to receive the full conversation history, giving each node full context of all prior drafts and critiques.

```python
# Generator — writes the tweet, revises on critique
generation_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a twitter techie influencer assistant tasked with writing excellent twitter posts. "
               "If the user provides critique, respond with a revised version of your previous attempts."),
    MessagesPlaceholder(variable_name="messages"),
])

# Reflector — critiques the tweet
reflection_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a viral twitter influencer grading a tweet. "
               "Generate critique and recommendations for the user's tweet. "
               "Always provide detailed recommendations, including requests for length, virality, style, etc."),
    MessagesPlaceholder(variable_name="messages"),
])
```

---

## Iteration control

Iteration count is derived from the length of the message state rather than a separate counter. Each generate → reflect cycle adds 2 messages, so `len(messages) > 6` corresponds to 3 complete cycles.

```python
def should_continue(state: MessageGraph):
    if len(state["messages"]) > 6:
        return END
    return REFLECT
```

To run more refinement cycles, increase the threshold. Each additional cycle adds one LLM call for generation and one for reflection.

---

## State

The graph uses a simple typed dict with `add_messages` as the reducer, which appends new messages rather than replacing the list on each node call.

```python
class MessageGraph(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
```

---

## Setup

**Requirements:** Python 3.13+, [Poetry](https://python-poetry.org)

```bash
git clone https://github.com/daniszwarc/reflection-agent.git
cd reflection-agent
poetry install
```

Create a `.env` file:

```
OPENAI_API_KEY=your_openai_key
```

Run the agent:

```bash
poetry run python main.py
```

The input tweet is hardcoded in `main.py`. Edit the `HumanMessage` content to use a different prompt.

---

## Relation to reflexion-agent

This repo implements the simpler of the two reflection patterns:

| | reflection-agent | reflexion-agent |
|---|---|---|
| Critique | Internal (LLM self-review) | Internal (structured Pydantic schema) |
| External knowledge | None | Tavily web search |
| Citations | No | Yes (numbered references) |
| Output schema | Unstructured text | Structured via tool binding |
| Use case | Content refinement | Research and factual Q&A |

reflection-agent is the conceptual foundation. reflexion-agent extends it with tool use, structured schemas, and web-grounded revision.

---

## Dependencies

Managed with [Poetry](https://python-poetry.org). Key packages:

- `langgraph` — graph orchestration and state management
- `langchain-openai` — LLM calls
- `python-dotenv` — environment variable loading
