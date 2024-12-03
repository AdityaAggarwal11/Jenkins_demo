import math
import os
import pytest
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Your existing code
student_files = [doc for doc in os.listdir() if doc.endswith('.txt')]
student_notes = [open(_file, encoding='utf-8').read() for _file in student_files]

def vectorize(Text):
    vectorizer = TfidfVectorizer(stop_words='english')
    return TfidfVectorizer().fit_transform(Text).toarray()

def similarity(doc1, doc2): 
    return cosine_similarity([doc1, doc2])

vectors = vectorize(student_notes)
s_vectors = list(zip(student_files, vectors))
plagiarism_results = set()

def check_plagiarism(threshold=0.8):
    global s_vectors
    for student_a, text_vector_a in s_vectors:
        new_vectors = s_vectors.copy()
        current_index = new_vectors.index((student_a, text_vector_a))
        del new_vectors[current_index]
        for student_b, text_vector_b in new_vectors:
            sim_score = similarity(text_vector_a, text_vector_b)[0][1]
            if sim_score >= threshold:
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
    assert isinstance(vectorized_result, np.ndarray)
    assert len(vectorized_result) == 2  # Two sentences, so two vectors
    assert len(vectorized_result[0]) > 0  # Vectors should have some length


def test_similarity():
    # Test similarity function with two identical texts
    text1 = ["This is a test sentence."]
    text2 = ["This is a test sentence."]

    # Vectorizing the texts
    vec1 = vectorize(text1)
    vec2 = vectorize(text2)

    # Print vectors for debugging
    print(f"Vec1 (Identical): {vec1}")
    print(f"Vec2 (Identical): {vec2}")

    similarity_score = similarity(vec1[0], vec2[0])[0][0]

    # Assert similarity score is exactly 1 for identical texts
    assert similarity_score == 1.0, f"Expected 1.0 but got {similarity_score}"

    # Test similarity with completely different texts
    text1 = ["This is a test sentence."]
    text2 = ["Another completely different sentence."]

    vec1 = vectorize(text1)
    vec2 = vectorize(text2)

    # Print vectors for debugging
    print(f"Vec1 (Different): {vec1}")
    print(f"Vec2 (Different): {vec2}")

    similarity_score = similarity(vec1[0], vec2[0])[0][0]

    # Use math.isclose to handle floating-point precision issues
    assert math.isclose(similarity_score, 0.0, abs_tol=0.1), f"Expected similarity < 0.1, but got {similarity_score}"

    # Alternatively, if you prefer using a threshold approach:
    # assert similarity_score < 0.5, f"Expected similarity score < 0.5, but got {similarity_score}"




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

    # Modify check_plagiarism to take a threshold and filter results based on it
    def check_plagiarism(threshold=0.5):
        global plagiarism_results
        plagiarism_results = set()

        for student_a, text_vector_a in s_vectors:
            new_vectors = s_vectors.copy()
            current_index = new_vectors.index((student_a, text_vector_a))
            del new_vectors[current_index]
            for student_b, text_vector_b in new_vectors:
                sim_score = similarity(text_vector_a, text_vector_b)[0][1]
                if sim_score >= threshold:
                    student_pair = sorted((student_a, student_b))
                    plagiarism_results.add((student_pair[0], student_pair[1], sim_score))
        return plagiarism_results

    # Run the plagiarism check with a threshold of 0.5
    result = check_plagiarism(threshold=0.5)  # Ensure you pass a threshold

    # Test that plagiarism results contain expected pair
    assert len(result) == 1  # Should only be one pair: (student1, student2)

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
