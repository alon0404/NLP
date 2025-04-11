import numpy as np


def softmax(x):
    """Compute the softmax function for each row of the input x.
    It is crucial that this function is optimized for speed because
    it will be used frequently in later code.

    Arguments:
    x -- A D dimensional vector or N x D dimensional numpy matrix.
    Return:
    x -- You are allowed to modify x in-place
    """
    orig_shape = x.shape

    if len(x.shape) > 1:
        # Matrix
        ### YOUR CODE HERE
        x = x - np.max(x, axis=1, keepdims=True)
        exp_x = np.exp(x)
        sum_exp_x = np.sum(exp_x, axis=1, keepdims=True)
        x = np.divide(exp_x, sum_exp_x)
        ### END YOUR CODE
    else:
        # Vector
        ### YOUR CODE HERE
        x = x - np.max(x)
        exp_x = np.exp(x)
        sum_exp_x = np.sum(exp_x)
        x = np.divide(exp_x, sum_exp_x)
        ### END YOUR CODE

    assert x.shape == orig_shape
    return x


def test_softmax_basic():
    """
    Some simple tests to get you started.
    Warning: these are not exhaustive.
    """
    print("Running basic tests...")
    test1 = softmax(np.array([1, 2]))
    print(test1)
    ans1 = np.array([0.26894142,  0.73105858])
    assert np.allclose(test1, ans1, rtol=1e-05, atol=1e-06)

    test2 = softmax(np.array([[1001, 1002], [3, 4]]))
    print(test2)
    ans2 = np.array([
        [0.26894142, 0.73105858],
        [0.26894142, 0.73105858]])
    assert np.allclose(test2, ans2, rtol=1e-05, atol=1e-06)

    test3 = softmax(np.array([[-1001, -1002]]))
    print(test3)
    ans3 = np.array([0.73105858, 0.26894142])
    assert np.allclose(test3, ans3, rtol=1e-05, atol=1e-06)

    print("You should be able to verify these results by hand!\n")


def your_softmax_test():
    """
    Use this space to test your softmax implementation by running:
        python q1_softmax.py
    This function will not be called by the autograder, nor will
    your tests be graded.
    """
    print("Running your tests...")
    ### YOUR OPTIONAL CODE HERE
    import scipy.special
    
    # Test that softmax outputs sum to 1 for vectors
    print("Testing vector sum to one...")
    test_vectors = [
        np.array([1, 2, 3, 4]),
        np.array([-10, -5, 0, 5, 10]),
        np.array([0, 0, 0, 0]),
        np.array([1000, 1000, 1000])
    ]
    for vec in test_vectors:
        result = softmax(vec)
        assert np.isclose(np.sum(result), 1.0)
        assert result.shape == vec.shape
    
    # Test that softmax outputs sum to 1 for each row in matrices
    print("Testing matrix sum to one...")
    test_matrices = [
        np.array([[1, 2, 3], [4, 5, 6]]),
        np.array([[-10, -5, 0], [5, 10, 15]]),
        np.array([[0, 0, 0], [1000, 1000, 1000]])
    ]
    for matrix in test_matrices:
        result = softmax(matrix)
        assert np.allclose(np.sum(result, axis=1), np.ones(matrix.shape[0]))
        assert result.shape == matrix.shape
    
    # Test softmax with very large values to check numerical stability
    print("Testing large values...")
    large_vec = np.array([1000, 1001, 1002])
    result = softmax(large_vec)
    # The largest value should dominate but not cause overflow
    expected = scipy.special.softmax(large_vec)
    assert np.allclose(result, expected, rtol=1e-5, atol=1e-6)
    
    # Test softmax with very small values
    print("Testing small values...")
    small_vec = np.array([-1000, -1001, -1002])
    result = softmax(small_vec)
    expected = scipy.special.softmax(small_vec)
    assert np.allclose(result, expected, rtol=1e-5, atol=1e-6)
    
    # Test that softmax preserves relative ordering of inputs
    print("Testing ordering preservation...")
    test_vec = np.array([1, 3, 2])
    result = softmax(test_vec)
    # Verify ordering: second element should be largest, third second largest
    assert np.argmax(result) == 1
    assert np.argsort(result)[-2] == 2
    
    # Test edge cases like single value arrays, etc.
    print("Testing edge cases...")
    # Single value array should give 1.0
    single_val = np.array([5])
    assert np.isclose(softmax(single_val)[0], 1.0)
    
    # Equal values should give equal probabilities
    equal_vals = np.array([7, 7, 7, 7])
    expected = scipy.special.softmax(equal_vals)
    assert np.allclose(softmax(equal_vals), expected)
    
    # Test that adding a constant to all values doesn't change the output
    print("Testing shift invariance...")
    orig = np.array([1, 2, 3])
    shifted = orig + 100  # Add a large constant
    
    result_orig = softmax(orig)
    result_shifted = softmax(shifted)
    
    assert np.allclose(result_orig, result_shifted, rtol=1e-5, atol=1e-6)
    
    print("All tests passed!")
    pass
    ### END YOUR CODE


if __name__ == "__main__":
    test_softmax_basic()
    your_softmax_test()
