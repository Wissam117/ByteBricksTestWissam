# app/services/concierge.py

from typing import Dict, List, Any, Optional
import uuid
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import create_openai_tools_agent
from langchain_core.agents import AgentFinish
from langgraph.graph import StateGraph
from langgraph.checkpoint.memory import MemorySaver

from app.services.tools.retrieve_docs import RetrieveDocsTool
from app.services.tools.manage_tasks import ManageTasksTool
from app.services.self_grading import SelfGradingService
from app.services.self_reflection import SelfReflectionService
from app.services.memory import MemoryService
from app.utils.vector_store import VectorStore
from app.core.config import settings

class ConciergeService:
    def __init__(self, llm=None):
        # Create LLM - can be configured to use different models
        if llm is None:
            # Default to OpenAI if API key is available, otherwise use a free alternative
            if settings.OPENAI_API_KEY:
                # Note: Requires API key and payment
                from langchain_openai import ChatOpenAI
                self.llm = ChatOpenAI(temperature=0, model="gpt-4")
            else:
                # Try to use a lightweight CPU-based model first (no CUDA required)
                try:
                    # CTransformers is a lightweight CPU inference engine
                    from langchain_community.llms import CTransformers
                    
                    # Use a smaller model that works well on CPU
                    self.llm = CTransformers(
                        model="TheBloke/Llama-2-7B-Chat-GGUF",
                        model_file="llama-2-7b-chat.Q4_K_M.gguf",  # Quantized for CPU
                        model_type="llama",
                        config={
                            'max_new_tokens': 1024,
                            'temperature': 0.2,
                            'context_length': 2048,
                        }
                    )
                    print("Using CTransformers with Llama-2 (CPU-only)")
                except Exception as e:
                    print(f"Could not load CTransformers model: {e}")
                    
                    # Fallback to Hugging Face Inference API
                    try:
                        from langchain_community.llms import HuggingFaceEndpoint
                        # Try a smaller model that's publicly accessible
                        self.llm = HuggingFaceEndpoint(
                            repo_id="google/flan-t5-base",  # Small model, no token needed
                            max_length=1024
                        )
                        print("Using Hugging Face Inference API with flan-t5")
                    except Exception as e:
                        print(f"Could not load HuggingFace model: {e}")
                        
                        # Final fallback - a very minimal model
                        print("Falling back to minimal text generation")
                        from langchain.llms import FakeListLLM
                        self.llm = FakeListLLM(
                            responses=["I can provide general assistance, but I'm currently running in a limited mode without access to a full language model."]
                        )
        else:
            self.llm = llm
        
        # Set up vector store and tools
        self.vector_store = VectorStore(settings.KNOWLEDGE_BASE_PATH)
        self.retrieve_docs_tool = RetrieveDocsTool(self.vector_store)
        self.manage_tasks_tool = ManageTasksTool()
        self.tools = [self.retrieve_docs_tool, self.manage_tasks_tool]
        
        # Set up services
        self.self_grading = SelfGradingService(self.llm)
        self.self_reflection = SelfReflectionService()
        self.memory = MemoryService()
        
        # Set up agent
        self.agent = self._create_agent()
        
        # Set up workflow
        self.workflow = self._create_workflow()
        
        # Session trackers
        self.sessions: Dict[str, Any] = {}
    
    def _create_agent(self):
        """Create the agent with tools."""
        # Base system prompt
        system_prompt = """You are an AI Concierge for small businesses looking to adopt AI technologies.
        You must answer questions about AI technologies, schedule demo calls, and keep a running to-do list.
        Be helpful, informative, and cite your sources when providing information.
        
        When a user wants to schedule something, make sure to use the manage_tasks tool.
        When a user asks about AI technologies, use the retrieve_docs tool to find relevant information.
        
        {custom_instructions}
        """
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
        ])
        
        return create_openai_tools_agent(self.llm, self.tools, prompt)
    
    def _create_workflow(self):
        """Create the workflow graph that handles the conversation flow."""
        
        def initial_state(inputs):
            """Initialize the state with inputs."""
            return {
                "input": inputs["question"],
                "chat_history": self.memory.get_messages(inputs["session_id"]),
                "custom_instructions": "",
                "session_id": inputs["session_id"]
            }
        
        def check_feedback(state):
            """Check if the message contains feedback commands."""
            user_input = state["input"].strip().lower()
            if user_input.startswith("/good_answer"):
                return "handle_feedback"
            elif user_input.startswith("/bad_answer"):
                return "handle_feedback"
            else:
                return "check_retrieval_need"
        
        def handle_feedback(state):
            """Process feedback commands."""
            user_input = state["input"].strip().lower()
            session_id = state["session_id"]
            
            if user_input.startswith("/good_answer"):
                self.self_reflection.apply_feedback(session_id, "good_answer")
                return {"response": "✓ Thank you for the positive feedback! I'll maintain this style."}
            elif user_input.startswith("/bad_answer"):
                self.self_reflection.apply_feedback(session_id, "bad_answer")
                return {"response": "✓ I appreciate your feedback. I'll be more concise and cite sources explicitly."}
        
        def check_retrieval_need(state):
            """Determine if retrieval is needed."""
            # Simple heuristic - if the message mentions AI, ML, or technology terms, use retrieval
            user_input = state["input"].lower()
            ai_terms = ["ai", "artificial intelligence", "ml", "machine learning", "neural", "algorithm", 
                      "transformer", "llm", "large language model", "deep learning", "training", 
                      "model", "rag", "retrieval", "vector"]
            
            if any(term in user_input for term in ai_terms):
                return "retrieve_docs"
            
            # Check for task management keywords
            task_terms = ["schedule", "appointment", "demo", "meeting", "call", "todo", "to-do", 
                        "to do", "task", "list"]
            
            if any(term in user_input for term in task_terms):
                return "run_agent"
            
            # Default to running the agent directly
            return "run_agent"
        
        def retrieve_docs(state):
            """Retrieve relevant documents and grade them."""
            query = state["input"]
            session_id = state["session_id"]
            
            # Call the retrieve_docs tool
            result = self.retrieve_docs_tool._run(query)
            docs = result["docs"]
            
            # Grade the retrieved documents
            scores = self.self_grading.grade_retrieval(query, docs)
            print(f"Retrieval scores: {scores}")
            
            # Check if the scores are above threshold
            if self.self_grading.is_above_threshold(scores):
                # Add documents to the state
                state["retrieved_docs"] = docs
                return "run_agent"
            else:
                # Try once more with a refined query
                refined_query = f"{query} ai technology business"
                refined_result = self.retrieve_docs_tool._run(refined_query)
                refined_docs = refined_result["docs"]
                
                # Grade the refined retrieval
                refined_scores = self.self_grading.grade_retrieval(query, refined_docs)
                print(f"Refined retrieval scores: {refined_scores}")
                
                if self.self_grading.is_above_threshold(refined_scores):
                    state["retrieved_docs"] = refined_docs
                    return "run_agent"
                else:
                    # Return a response indicating knowledge gap
                    return {
                        "response": "I'm sorry, my knowledge base doesn't cover that topic. Is there something else about AI technologies I can help you with?"
                    }
        
        def run_agent(state):
            """Run the agent to generate a response."""
            session_id = state["session_id"]
            
            # Apply custom instructions based on feedback
            custom_instruction = self.self_reflection.get_modified_prompt(session_id) or ""
            state["custom_instructions"] = custom_instruction
            
            # Prepare the retrieved documents if available
            docs_context = ""
            if "retrieved_docs" in state and state["retrieved_docs"]:
                docs = state["retrieved_docs"]
                docs_context = "Based on my knowledge:\n\n"
                for i, doc in enumerate(docs):
                    docs_context += f"[Source: {doc['source']}]\n{doc['content']}\n\n"
                
                # Add the context to the input
                state["input"] = f"{state['input']}\n\nContext: {docs_context}"
            
            # Run the agent
            response = self.agent.invoke({
                "input": state["input"],
                "chat_history": state["chat_history"],
                "custom_instructions": state["custom_instructions"]
            })
            
            # Extract the response content
            if isinstance(response, AgentFinish):
                agent_response = response.return_values["output"]
            else:
                agent_response = str(response)
            
            # Add messages to memory
            self.memory.add_message(session_id, "human", state["input"])
            self.memory.add_message(session_id, "ai", agent_response)
            
            # Apply decay to feedback score
            self.self_reflection.apply_decay(session_id)
            
            return {"response": agent_response}
        
        # Define the graph
        workflow = StateGraph(inputs=["question", "session_id"])
        
        # Add nodes
        workflow.add_node("initial", initial_state)
        workflow.add_node("check_feedback", check_feedback)
        workflow.add_node("handle_feedback", handle_feedback)
        workflow.add_node("check_retrieval_need", check_retrieval_need)
        workflow.add_node("retrieve_docs", retrieve_docs)
        workflow.add_node("run_agent", run_agent)
        
        # Add edges
        workflow.set_entry_point("initial")
        workflow.add_edge("initial", "check_feedback")
        workflow.add_edge("check_feedback", "handle_feedback")
        workflow.add_edge("check_feedback", "check_retrieval_need")
        workflow.add_edge("check_retrieval_need", "retrieve_docs")
        workflow.add_edge("check_retrieval_need", "run_agent")
        workflow.add_edge("retrieve_docs", "run_agent")
        
        # Compile the graph
        return workflow.compile()
    
    async def chat(self, user_id: str, question: str) -> Dict[str, Any]:
        """
        Process a user message and generate a response.
        
        Args:
            user_id: The user's ID
            question: The user's message
            
        Returns:
            Dict containing the AI's response
        """
        # Get or create session ID
        session_key = f"user_{user_id}"
        if session_key not in self.sessions:
            self.sessions[session_key] = str(uuid.uuid4())
        
        session_id = self.sessions[session_key]
        
        # Run the workflow
        result = await self.workflow.ainvoke({
            "question": question,
            "session_id": session_id
        })
        
        return {"response": result["response"]}