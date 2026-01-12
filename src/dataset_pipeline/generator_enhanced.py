"""
Enhanced Question Generator with Self-Instruct Diversity Filtering

기존 generator.py를 래핑하여 diversity filtering 추가.
"""

from __future__ import annotations

from typing import List, Dict, Any

from .generator import QuestionGenerator as BaseGenerator
from .diversity import is_diverse, calculate_diversity_score
from .llm_connector import LLMConnector


class EnhancedQuestionGenerator(BaseGenerator):
    """
    Self-Instruct diversity filtering이 적용된 질문 생성기.
    
    Wang et al. (2023, ACL)의 ROUGE-L 기반 중복 제거 적용.
    """
    
    def __init__(self, llm_connector: LLMConnector, config: Dict[str, Any]):
        super().__init__(llm_connector, config)
        
        # Diversity threshold (기본 0.7)
        self.diversity_threshold = config.get('question_diversity_threshold', 0.7)
        
        # 통계 수집
        self.stats = {
            'total_generated': 0,
            'filtered_by_diversity': 0,
            'filtered_by_exact_match': 0,
        }
    
    def generate_dataset(self, contexts: List[str], seed_questions: List[str] = None) -> List[Dict[str, Any]]:
        """
        Self-Instruct diversity filtering이 적용된 데이터셋 생성.
        
        원본 generate_dataset에 ROUGE-L 기반 중복 제거 추가.
        """
        if seed_questions is None:
            seed_questions = self.config.seed_questions.copy()
        
        all_questions = seed_questions.copy()
        qa_pairs = []
        
        import random
        
        # 반복적 생성
        for iteration in range(self.config.max_iterations):
            print(f"반복 {iteration + 1}/{self.config.max_iterations}")
            
            # 각 컨텍스트에서 질문 생성
            for context in contexts:
                if len(all_questions) >= self.config.num_questions:
                    break
                
                new_questions = self.generate_from_context(context, all_questions)
                
                for q_data in new_questions:
                    question = q_data["question"]
                    self.stats['total_generated'] += 1
                    
                    # Self-Instruct diversity filtering (Wang et al., 2023)
                    is_div, similarity = is_diverse(
                        question, 
                        all_questions, 
                        threshold=self.diversity_threshold
                    )
                    
                    if not is_div:
                        # ROUGE-L 유사도가 threshold 이상 -> 중복 제거
                        self.stats['filtered_by_diversity'] += 1
                        continue
                    
                    # Exact match 체크 (보조)
                    if question in all_questions:
                        self.stats['filtered_by_exact_match'] += 1
                        continue
                    
                    all_questions.append(question)
                    
                    # 진화 적용 (일부 질문만)
                    if random.random() < 0.3:  # 30% 확률로 진화
                        evolved = self.evolve_question(question)
                        question = evolved["evolved_question"]
                    
                    qa_pairs.append({
                        "question": question,
                        "answer": "",  # 나중에 생성
                        "difficulty": q_data.get("difficulty", "medium"),
                        "context": context,
                        "iteration": iteration + 1,
                        "diversity_score": 1.0 - similarity,  # 다양성 점수 기록
                    })
                    
                    if len(qa_pairs) >= self.config.num_questions:
                        break
                
                if len(qa_pairs) >= self.config.num_questions:
                    break
        
        # 최종 통계 출력
        print(f"\n[Diversity Filtering 통계]")
        print(f"  총 생성: {self.stats['total_generated']}개")
        print(f"  Diversity filter: {self.stats['filtered_by_diversity']}개 제거")
        print(f"  Exact match filter: {self.stats['filtered_by_exact_match']}개 제거")
        print(f"  최종 선택: {len(qa_pairs)}개")
        
        # 최종 다양성 점수 계산
        if qa_pairs:
            final_diversity = calculate_diversity_score([q['question'] for q in qa_pairs])
            print(f"  최종 다양성 점수: {final_diversity:.3f}")
        
        return qa_pairs[:self.config.num_questions]
