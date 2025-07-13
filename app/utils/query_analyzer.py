"""Query analysis utilities.

Allows to break down complex queries into easier subqueries,
hence enhancing and extending search space by number of related requests.
"""

from typing import Dict, Any, List
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser


class QueryAnalyzer:
    """Analyze query complexity and generate sub-questions"""

    def __init__(self, llm):
        self.llm = llm

    async def analyze_query(self, question: str) -> Dict[str, Any]:
        """Analyze query complexity and determine retrieval strategy.

        Query complexity score is a heuristic algorithm to be fair.
        """

        # Some random query complexity score
        word_count = len(question.split())
        has_multiple_concepts = any(word in question.lower() for word in ['and', 'or', 'but', 'however', 'also'])
        complexity_score = word_count / 10 + (2 if has_multiple_concepts else 0)

        search_queries = [question]
        if complexity_score > 1.5:
            try:
                sub_questions = await self._generate_sub_questions(question)
                search_queries.extend(sub_questions)
            except Exception:
                pass  # Fall back to original question

        return {
            "search_queries": search_queries,
            "complexity_score": complexity_score
        }

    async def _generate_sub_questions(self, question: str) -> List[str]:
        """Generate sub-questions for complex queries"""
        prompt = ChatPromptTemplate.from_template("""
        Break down this complex question into 2-3 simpler sub-questions that would help 
        answer the original question comprehensively.

        Original question: {question}

        Sub-questions (one per line):
        """)

        chain = prompt | self.llm | StrOutputParser()
        response = await chain.ainvoke({"question": question})

        sub_questions = [q.strip() for q in response.split('\n') if q.strip()]
        return sub_questions[:3]
