"""
Self-Instruct Diversity Filtering 모듈

Wang et al. (2023, ACL) "Self-Instruct: Aligning Language Models 
with Self-Generated Instructions" 의 diversity filtering 구현.

핵심: ROUGE-L 유사도가 threshold 이상이면 중복으로 간주.
"""

from __future__ import annotations

from typing import List, Tuple
import re


def calculate_rouge_l(text1: str, text2: str) -> float:
    """
    ROUGE-L (Longest Common Subsequence) F1 score 계산.
    
    rouge-score 라이브러리 없이도 작동하도록 순수 Python 구현.
    
    Args:
        text1: 첫 번째 텍스트
        text2: 두 번째 텍스트
    
    Returns:
        ROUGE-L F1 score (0.0 ~ 1.0)
    
    Reference:
        Lin, C. Y. (2004). "ROUGE: A Package for Automatic Evaluation 
        of Summaries"
    """
    # Tokenize (간단한 공백 분리)
    tokens1 = text1.lower().split()
    tokens2 = text2.lower().split()
    
    if not tokens1 or not tokens2:
        return 0.0
    
    # LCS (Longest Common Subsequence) 계산
    lcs_length = _lcs_length(tokens1, tokens2)
    
    # ROUGE-L = LCS-based F1
    if lcs_length == 0:
        return 0.0
    
    precision = lcs_length / len(tokens2)
    recall = lcs_length / len(tokens1)
    
    if precision + recall == 0:
        return 0.0
    
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def _lcs_length(seq1: List[str], seq2: List[str]) -> int:
    """
    Longest Common Subsequence 길이 계산 (동적 프로그래밍).
    
    Args:
        seq1: 첫 번째 시퀀스
        seq2: 두 번째 시퀀스
    
    Returns:
        LCS 길이
    """
    m, n = len(seq1), len(seq2)
    
    # DP 테이블
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if seq1[i - 1] == seq2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    
    return dp[m][n]


def is_diverse(
    new_question: str,
    existing_questions: List[str],
    threshold: float = 0.7,
    window_size: int = 50
) -> Tuple[bool, float]:
    """
    새 질문이 기존 질문들과 충분히 다른지 판단 (Self-Instruct 기준).
    
    Args:
        new_question: 새로 생성된 질문
        existing_questions: 이미 생성된 질문 리스트
        threshold: ROUGE-L 임계값 (기본 0.7, Wang et al. 2023)
        window_size: 비교할 최근 질문 수 (성능 최적화)
    
    Returns:
        (is_diverse, max_similarity)
        - is_diverse: True면 다양성 충족
        - max_similarity: 최대 유사도 값
    
    Example:
        >>> existing = ["와사비의 최적 온도는?", "와사비 재배 온도 범위는?"]
        >>> new = "와사비에 적합한 온도는 몇 도인가요?"
        >>> is_diverse(new, existing, threshold=0.7)
        (False, 0.85)  # 유사도 높아서 중복으로 판정
    """
    if not existing_questions:
        return True, 0.0
    
    # 최근 window_size개만 비교 (성능 최적화)
    recent_questions = existing_questions[-window_size:]
    
    max_similarity = 0.0
    for existing_q in recent_questions:
        similarity = calculate_rouge_l(new_question, existing_q)
        max_similarity = max(max_similarity, similarity)
        
        if similarity >= threshold:
            # 임계값 초과 → 중복으로 간주
            return False, similarity
    
    return True, max_similarity


def filter_diverse_questions(
    questions: List[str],
    threshold: float = 0.7
) -> List[str]:
    """
    질문 리스트에서 중복을 제거하여 다양한 질문만 반환.
    
    Args:
        questions: 질문 리스트
        threshold: ROUGE-L 임계값
    
    Returns:
        다양성이 확보된 질문 리스트
    
    Example:
        >>> questions = [
        ...     "와사비의 최적 온도는?",
        ...     "와사비 재배 온도는?",  # 중복
        ...     "와사비 병해충 대처법은?",
        ... ]
        >>> filter_diverse_questions(questions, threshold=0.7)
        ["와사비의 최적 온도는?", "와사비 병해충 대처법은?"]
    """
    filtered = []
    
    for q in questions:
        is_div, sim = is_diverse(q, filtered, threshold)
        if is_div:
            filtered.append(q)
    
    return filtered


def calculate_diversity_score(questions: List[str]) -> float:
    """
    질문 집합의 전체 다양성 점수 계산.
    
    점수가 높을수록 다양성이 높음 (0.0 ~ 1.0).
    
    Args:
        questions: 질문 리스트
    
    Returns:
        Diversity score (1 - 평균 pairwise 유사도)
    
    Example:
        >>> questions = ["A", "B", "C"]
        >>> calculate_diversity_score(questions)
        0.85  # 높은 다양성
    """
    if len(questions) < 2:
        return 1.0
    
    total_similarity = 0.0
    comparisons = 0
    
    for i in range(len(questions)):
        for j in range(i + 1, len(questions)):
            similarity = calculate_rouge_l(questions[i], questions[j])
            total_similarity += similarity
            comparisons += 1
    
    avg_similarity = total_similarity / comparisons
    diversity_score = 1.0 - avg_similarity
    
    return diversity_score
