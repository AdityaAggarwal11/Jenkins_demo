import os
import pytest
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Your existing code
student_files = [doc for doc in os.listdir() if doc.endswith('.txt')]
student_notes = [open(_file, encoding='utf-8').read() for _file in student_files]

def vectorize(Text): 
    return TfidfVectorizer().fit_transform(Text).toarray()

def similarity(doc1, doc2): 
    return cosine_similarity([doc1, doc2])

vectors = vectorize(student_notes)
s_vectors = list(zip(student_files, vectors))
plagiarism_results = set()

def check_plagiarism():
    global s_vectors
    for student_a, text_vector_a in s_vectors:
        new_vectors = s_vectors.copy()
        current_index = new_vectors.index((student_a, text_vector_a))
        del new_vectors[current_index]
        for student_b, text_vector_b in new_vectors:
            sim_score = similarity(text_vector_a, text_vector_b)[0][1]
            student_pair = sorted((student_a, student_b))
            score = (student_pair[0], student_pair[1], sim_score)
            plagiarism_results.add(score)
    return plagiarism_results

# Test Functions
def test_vectorize():
    # Test with a simple, predictable text input
    test_text = ["This is a test", "This is a test."]
    vectorized_result = vectorize(test_text)
    
    # Check if the result is a 2D array (since TfidfVectorizer returns an array)
    assert isinstance(vectorized_result, list)
    assert len(vectorized_result) == 2  # Two sentences, so two vectors
    assert len(vectorized_result[0]) > 0  # Vectors should have some length

def test_similarity():
    # Test similarity function with two identical texts
    text1 = ["This is a test sentence."]
    text2 = ["This is a test sentence."]
    
    vec1 = vectorize(text1)
    vec2 = vectorize(text2)
    
    similarity_score = similarity(vec1[0], vec2[0])[0][1]
    assert similarity_score == 1.0  # Identical texts should have a similarity score of 1
    
    # Test similarity with completely different texts
    text1 = ["This is a test sentence."]
    text2 = ["Another completely different sentence."]
    
    vec1 = vectorize(text1)
    vec2 = vectorize(text2)
    
    similarity_score = similarity(vec1[0], vec2[0])[0][1]
    assert similarity_score < 0.1  # Different texts should have a very low similarity score

def test_check_plagiarism():
    # Mock data for testing
    mock_notes = [
        "This is a test file.",
        "This is a test file.",
        "Completely different content."
    ]
    
    mock_files = ["student1.txt", "student2.txt", "student3.txt"]
    
    # Simulate the vectors and plagiarism results
    vectors = vectorize(mock_notes)
    s_vectors = list(zip(mock_files, vectors))
    
    global plagiarism_results
    plagiarism_results.clear()
    
    # Run the plagiarism check
    result = check_plagiarism()
    
    # Test that plagiarism results contain expected pair
    assert len(result) == 2  # Should only be two pairs: (student1, student2) and (student2, student3)
    
    # Test a specific plagiarism result
    student_pair = sorted(("student1.txt", "student2.txt"))
    assert any(
        student_pair == sorted((r[0], r[1])) for r in result
    )  # Verify the pair exists in the results
    
    # Ensure no plagiarism result for student3 (different content)
    assert not any(
        "student3.txt" in (r[0], r[1]) for r in result
    )  # student3 should not have plagiarism with anyone

# To run the tests directly within this script
if __name__ == "__main__":
    pytest.main()
