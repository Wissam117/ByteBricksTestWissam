# app/services/concierge.py

from typing import Dict, List, Any, Optional
import uuid
import os
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
                print("Using OpenAI GPT-4")
            else:
                # Try to use local model with GPU if available
                try:
                    # Check if CUDA is available
                    try:
                        import torch
                        cuda_available = torch.cuda.is_available()
                        print(f"CUDA available: {cuda_available}")
                        if cuda_available:
                            print(f"CUDA device count: {torch.cuda.device_count()}")
                            print(f"Current CUDA device: {torch.cuda.current_device()}")
                    except ImportError:
                        cuda_available = False
                        print("PyTorch not installed, CUDA unavailable")
                    
                    # Path to your local model
                    local_model_path = os.getenv('LOCAL_MODEL_PATH', 'app/models/Llama-3.2-3B-Instruct-IQ4_XS.gguf')
                    
                    if os.path.exists(local_model_path):
                        # Try LlamaCpp first (preferred for GGUF files)
                        try:
                            from langchain_community.llms import LlamaCpp
                            
                            if cuda_available:
                                self.llm = LlamaCpp(
                                    model_path=local_model_path,
                                    temperature=0.2,
                                    max_tokens=1024,
                                    n_ctx=8192,  # Increased context length
                                    n_gpu_layers=35,  # Use GPU layers (adjust based on your model)
                                    n_batch=512,  # Batch size for processing
                                    verbose=False,
                                )
                                print(f"Using LlamaCpp with CUDA acceleration: {local_model_path}")
                            else:
                                self.llm = LlamaCpp(
                                    model_path=local_model_path,
                                    temperature=0.2,
                                    max_tokens=1024,
                                    n_ctx=8192,  # Increased context length
                                    n_gpu_layers=0,  # CPU only
                                    n_batch=8,  # Smaller batch for CPU
                                    verbose=False,
                                )
                                print(f"Using LlamaCpp with CPU only: {local_model_path}")
                        except ImportError:
                            print("LlamaCpp not available, trying CTransformers...")
                            # Fallback to CTransformers
                            from langchain_community.llms import CTransformers
                            
                            gpu_layers = 35 if cuda_available else 0
                            self.llm = CTransformers(
                                model=local_model_path,
                                model_type="llama",
                                config={
                                    'max_new_tokens': 1024,
                                    'temperature': 0.2,
                                    'context_length': 4096,
                                    'gpu_layers': gpu_layers,
                                    'threads': 4,
                                }
                            )
                            print(f"Using CTransformers with {'GPU' if cuda_available else 'CPU'}: {local_model_path}")
                    else:
                        # Model file not found, try downloading a model
                        print(f"Local model not found at: {local_model_path}")
                        print("Attempting to download a model...")
                        
                        # Try CTransformers with auto-download
                        from langchain_community.llms import CTransformers
                        
                        gpu_layers = 35 if cuda_available else 0
                        self.llm = CTransformers(
                            model="TheBloke/Llama-2-7B-Chat-GGUF",
                            model_file="llama-2-7b-chat.Q4_K_M.gguf",
                            model_type="llama",
                            config={
                                'max_new_tokens': 1024,
                                'temperature': 0.2,
                                'context_length': 2048,
                                'gpu_layers': gpu_layers,
                                'threads': 4,
                            }
                        )
                        print(f"Using CTransformers with downloaded model ({'GPU' if cuda_available else 'CPU'})")
                        
                except Exception as e:
                    print(f"Could not load local model: {e}")
                    
                    # Fallback to Hugging Face Inference API
                    try:
                        from langchain_community.llms import HuggingFaceEndpoint
                        # Try a smaller model that's publicly accessible
                        self.llm = HuggingFaceEndpoint(
                            repo_id="microsoft/DialoGPT-medium",
                            max_length=1024,
                            temperature=0.7,
                        )
                        print("Using Hugging Face Inference API with DialoGPT")
                    except Exception as e:
                        print(f"Could not load HuggingFace model: {e}")
                        
                        # Final fallback - a very minimal model
                        print("Falling back to minimal text generation")
                        from langchain.llms import FakeListLLM
                        self.llm = FakeListLLM(
                            responses=[
                                "I can provide general assistance, but I'm currently running in a limited mode without access to a full language model.",
                                "For full functionality, please configure an OpenAI API key or ensure a local model is properly installed.",
                                "I can still help with task management and accessing the knowledge base."
                            ]
                        )
        else:
            self.llm = llm
        
        # Set up vector store and tools
        try:
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
            print("✅ ConciergeService initialized successfully")
            
        except Exception as e:
            print(f"⚠️  Failed to initialize full ConciergeService: {e}")
            # Continue with minimal functionality
            self.vector_store = None
            self.retrieve_docs_tool = None
            self.manage_tasks_tool = ManageTasksTool()
            self.tools = [self.manage_tasks_tool] if self.manage_tasks_tool else []
            self.self_grading = None
            self.self_reflection = SelfReflectionService()
            self.memory = MemoryService()
            self.agent = None
            self.workflow = None
            self.sessions: Dict[str, Any] = {}
            print("⚠️  Running with limited functionality")
    
    def _create_agent(self):
        """Create the agent with tools."""
        if not self.tools:
            return None
            
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
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])
        
        return create_openai_tools_agent(self.llm, self.tools, prompt)
    
    def _create_workflow(self):
        """Create the workflow graph that handles the conversation flow."""
        if not self.agent:
            return None
            
        def initial_state(state):
            """Initialize the state with inputs."""
            return {
                "input": state["question"],
                "chat_history": self.memory.get_messages(state["session_id"]),
                "custom_instructions": "",
                "session_id": state["session_id"]
            }
        
        def check_feedback(state):
            """Check if the message contains feedback commands."""
            user_input = state["input"].strip().lower()
            if user_input.startswith("/good_answer"):
                return {"next": "handle_feedback"}
            elif user_input.startswith("/bad_answer"):
                return {"next": "handle_feedback"}
            else:
                return {"next": "check_retrieval_need"}
        
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
            return state
        
        def check_retrieval_need(state):
            """Determine if retrieval is needed."""
            # Simple heuristic - if the message mentions AI, ML, or technology terms, use retrieval
            user_input = state["input"].lower()
            ai_terms = ["ai", "artificial intelligence", "ml", "machine learning", "neural", "algorithm", 
                      "transformer", "llm", "large language model", "deep learning", "training", 
                      "model", "rag", "retrieval", "vector"]
            
            if self.retrieve_docs_tool and any(term in user_input for term in ai_terms):
                return {"next": "retrieve_docs"}
            
            # Check for task management keywords
            task_terms = ["schedule", "appointment", "demo", "meeting", "call", "todo", "to-do", 
                        "to do", "task", "list"]
            
            if any(term in user_input for term in task_terms):
                return {"next": "run_agent"}
            
            # Default to running the agent directly
            return {"next": "run_agent"}
        
        def retrieve_docs(state):
            """Retrieve relevant documents and grade them."""
            if not self.retrieve_docs_tool:
                return {"next": "run_agent"}
                
            query = state["input"]
            session_id = state["session_id"]
            
            try:
                # Call the retrieve_docs tool
                result = self.retrieve_docs_tool._run(query)
                docs = result["docs"]
                
                # Skip grading for now and just use the docs
                # TODO: Fix self-grading compatibility with local LLM
                print(f"Retrieved {len(docs)} documents")
                return {
                    "retrieved_docs": docs,
                    "next": "run_agent"
                }
            except Exception as e:
                print(f"Error in retrieve_docs: {e}")
                return {"next": "run_agent"}
        
        def run_agent(state):
            """Run the agent to generate a response."""
            session_id = state["session_id"]
            
            try:
                # Prepare context from retrieved documents
                context = ""
                if "retrieved_docs" in state and state["retrieved_docs"]:
                    docs = state["retrieved_docs"]
                    context = "Based on my knowledge:\n\n"
                    for i, doc in enumerate(docs):
                        context += f"[Source: {doc['source']}]\n{doc['content']}\n\n"
                
                # Format chat history
                chat_history = ""
                if state["chat_history"]:
                    for msg in state["chat_history"][-5:]:  # Last 5 messages for context
                        if hasattr(msg, 'content'):
                            content = msg.content
                            msg_type = msg.type if hasattr(msg, 'type') else 'unknown'
                        else:
                            content = str(msg)
                            msg_type = 'message'
                        chat_history += f"{msg_type}: {content}\n"
                
                # Run the simple agent (LLM chain)
                response = self.agent.invoke({
                    "context": context,
                    "chat_history": chat_history,
                    "input": state["input"]
                })
                
                # For LLMChain, response is a dict with 'text' key
                if isinstance(response, dict) and 'text' in response:
                    agent_response = response['text']
                else:
                    agent_response = str(response)
                
                # Add messages to memory
                self.memory.add_message(session_id, "human", state["input"])
                self.memory.add_message(session_id, "ai", agent_response)
                
                # Apply decay to feedback score
                self.self_reflection.apply_decay(session_id)
                
                return {"response": agent_response}
                
            except Exception as e:
                print(f"Error in run_agent: {e}")
                import traceback
                traceback.print_exc()
                
                # Fallback to direct LLM response
                try:
                    context = ""
                    if "retrieved_docs" in state and state["retrieved_docs"]:
                        docs = state["retrieved_docs"]
                        context = "\n\nContext: "
                        for doc in docs[:2]:  # Use first 2 docs
                            context += f"{doc['content'][:200]}... "
                    
                    prompt = f"Question: {state['input']}{context}\n\nAnswer:"
                    response = self.llm.invoke(prompt)
                    return {"response": response}
                except Exception as fallback_error:
                    print(f"Fallback also failed: {fallback_error}")
                    return {"response": "I apologize, but I encountered an error processing your request. Please try again."}
        
        # Define the state structure
        from typing import TypedDict, Optional, List, Any
        
        class WorkflowState(TypedDict, total=False):
            question: str
            session_id: str
            input: str
            chat_history: List[Any]
            custom_instructions: str
            retrieved_docs: Optional[List[Any]]
            response: Optional[str]
            next: Optional[str]
        
        # Create the workflow
        from langgraph.graph import StateGraph, END
        
        workflow = StateGraph(WorkflowState)
        
        # Add nodes
        workflow.add_node("initial", initial_state)
        workflow.add_node("check_feedback", check_feedback)
        workflow.add_node("handle_feedback", handle_feedback)
        workflow.add_node("check_retrieval_need", check_retrieval_need)
        workflow.add_node("retrieve_docs", retrieve_docs)
        workflow.add_node("run_agent", run_agent)
        
        # Add conditional edges
        def decide_next_step(state):
            """Route to the next step based on the 'next' field."""
            return state.get("next", END)
        
        # Set entry point
        workflow.set_entry_point("initial")
        
        # Add edges
        workflow.add_edge("initial", "check_feedback")
        
        workflow.add_conditional_edges(
            "check_feedback",
            decide_next_step,
            {
                "handle_feedback": "handle_feedback",
                "check_retrieval_need": "check_retrieval_need"
            }
        )
        
        workflow.add_edge("handle_feedback", END)
        
        workflow.add_conditional_edges(
            "check_retrieval_need", 
            decide_next_step,
            {
                "retrieve_docs": "retrieve_docs",
                "run_agent": "run_agent"
            }
        )
        
        workflow.add_conditional_edges(
            "retrieve_docs",
            decide_next_step,
            {
                "run_agent": "run_agent",
                END: END
            }
        )
        
        workflow.add_edge("run_agent", END)
        
        # Compile the workflow
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
        
        # If workflow is not available, provide a simple response
        if not self.workflow:
            # Simple fallback response
            if self.retrieve_docs_tool and any(term in question.lower() for term in ["ai", "artificial intelligence", "ml", "machine learning"]):
                try:
                    result = self.retrieve_docs_tool._run(question)
                    docs = result["docs"]
                    if docs:
                        response = f"Based on my knowledge base:\n\n{docs[0]['content'][:500]}..."
                    else:
                        response = "I couldn't find specific information about that in my knowledge base."
                except:
                    response = "I'm currently running in limited mode. Please check the system configuration."
            else:
                response = "I'm currently running in limited mode. I can help with basic questions, but full functionality requires proper system configuration."
            
            return {"response": response}
        
        # Run the workflow
        try:
            result = await self.workflow.ainvoke({
                "question": question,
                "session_id": session_id
            })
            return {"response": result["response"]}
        except Exception as e:
            print(f"Error in workflow: {e}")
            return {"response": "I apologize, but I encountered an error processing your request. Please try again."}

# Simple fallback service for compatibility
class SimpleConciergeService:
    """Simplified version of ConciergeService for fallback scenarios."""
    
    def __init__(self):
        self.sessions = {}
    
    async def chat(self, user_id: str, question: str) -> Dict[str, Any]:
        """Simple chat implementation."""
        # Basic responses for common queries
        question_lower = question.lower()
        
        if any(word in question_lower for word in ["hello", "hi", "hey"]):
            response = "Hello! I'm your AI Concierge. How can I help you today?"
        elif any(word in question_lower for word in ["help", "what can you do"]):
            response = "I can help with AI technology questions and task management. However, I'm currently running in simplified mode."
        else:
            response = "I'm currently running in simplified mode. For full functionality, please ensure all dependencies are properly configured."
        
        return {"response": response}