
# app/services/self_grading.py

from typing import Dict, List, Any, Tuple
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

class SelfGradingService:
    def __init__(self, llm=None):
        self.llm = llm or ChatOpenAI(temperature=0, model="gpt-4")
        self.threshold = 0.6
        self.grading_prompt = ChatPromptTemplate.from_template("""
        Rate the following on a scale of 0-1:
        
        Question: {question}
        Retrieved Documents: {docs}
        
        Factual Relevance: [0-1] - How relevant are the documents to the question?
        Answer Coverage: [0-1] - How completely can the question be answered with these documents?
        
        Return your evaluation as a JSON object with the following format:
        {{
            "factual_relevance": <score between 0 and 1>,
            "answer_coverage": <score between 0 and 1>
        }}
        
        Do not include any explanations, just the JSON object.
        """)
    
    def grade_retrieval(self, question: str, docs: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        Grade the relevance and coverage of retrieved documents.
        
        Args:
            question: The user's question
            docs: The retrieved documents
            
        Returns:
            Dict with relevance and coverage scores
        """
        # Format documents for evaluation
        docs_str = "\n".join([
            f"Document {i+1}:\nContent: {doc['content']}\nSource: {doc['source']}"
            for i, doc in enumerate(docs)
        ])
        
        # Create the grading prompt
        chain = self.grading_prompt | self.llm
        
        # Call the LLM to evaluate the retrieval
        response = chain.invoke({"question": question, "docs": docs_str})
        
        try:
            # Parse the evaluation result
            result = response.content
            if isinstance(result, str):
                # Extract JSON part if there's extra text
                import json
                import re
                json_match = re.search(r'\{.*\}', result, re.DOTALL)
                if json_match:
                    result = json.loads(json_match.group(0))
                else:
                    result = json.loads(result)
            
            return {
                "factual_relevance": float(result.get("factual_relevance", 0)),
                "answer_coverage": float(result.get("answer_coverage", 0))
            }
        except Exception as e:
            print(f"Error parsing evaluation result: {e}")
            return {"factual_relevance": 0, "answer_coverage": 0}
    
    def is_above_threshold(self, scores: Dict[str, float]) -> bool:
        """Check if the scores are above the defined threshold."""
        return (scores["factual_relevance"] >= self.threshold and 
                scores["answer_coverage"] >= self.threshold)