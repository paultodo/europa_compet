import numpy as np
import sys
sys.path.append('./')

from src.target_data_io import get_necessary_steps

def test_two_candidates_last_point():
    # Test two candidates for last point
    t = np.array([300, 600, 700])
    indices_kept, timesteps_integer_keps = get_necessary_steps(t, history_length= 10)
    np.testing.assert_array_equal(indices_kept, np.array([0, 2], dtype = int))
    np.testing.assert_array_equal(timesteps_integer_keps, np.array([8, 9], dtype = int))

def test_three_perfect_candidates():
    # Test three perfect candidates
    t = np.array([300, 600, 900])
    indices_kept, timesteps_integer_keps = get_necessary_steps(t, history_length= 10)
    np.testing.assert_array_equal(indices_kept, np.array([0, 1, 2], dtype = int))
    np.testing.assert_array_equal(timesteps_integer_keps, np.array([7, 8, 9], dtype = int))

    
def test_missing_candidate_and_perfect_candidates():
    # Test a missing candidate for the middle timestep, but perfect candidates
    t = np.array([300, 900])
    indices_kept, timesteps_integer_keps = get_necessary_steps(t, history_length= 10)
    np.testing.assert_array_equal(indices_kept, np.array([0, 1], dtype = int))
    np.testing.assert_array_equal(timesteps_integer_keps, np.array([7, 9], dtype = int))

def test_missing_candidate_and_two_candidates():
    # Test a missing candidate for the middle timestep, and two candidates for the first one
    t = np.array([270, 320, 900])
    indices_kept, timesteps_integer_keps = get_necessary_steps(t, history_length= 10)
    np.testing.assert_array_equal(indices_kept, np.array([1, 2], dtype = int))
    np.testing.assert_array_equal(timesteps_integer_keps, np.array([7, 9], dtype = int))

def test_missing_candidate_and_unperfect_candidate():
    # Test a missing candidate for the middle timestep, and one unperfect candidate for the first one
    t = np.array([270, 900])
    indices_kept, timesteps_integer_keps = get_necessary_steps(t, history_length= 10)
    np.testing.assert_array_equal(indices_kept, np.array([0, 1], dtype = int))
    np.testing.assert_array_equal(timesteps_integer_keps, np.array([7, 9], dtype = int))
    
def test_prune_when_too_much_history():
    # Test that the output is pruned when there is too much history
    t = np.array([270, 610, 900])
    indices_kept, timesteps_integer_keps = get_necessary_steps(t, history_length= 2)
    np.testing.assert_array_equal(indices_kept, np.array([1, 2], dtype = int))
    np.testing.assert_array_equal(timesteps_integer_keps, np.array([0, 1], dtype = int))

if __name__ == '__main__':
    test_two_candidates_last_point()
    test_three_perfect_candidates()
    test_missing_candidate_and_perfect_candidates()
    test_missing_candidate_and_two_candidates()
    test_missing_candidate_and_unperfect_candidate()
    test_prune_when_too_much_history()
