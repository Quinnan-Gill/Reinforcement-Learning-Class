from checkpoints import load_checkpoint
import numpy as np

# Load phase 1 checkpoint (use YOUR actual timestamp)
phase1 = load_checkpoint("results/training/q-learning_curriculum_20251111_010525/phase1_vsrandom", action_size=4)

# Check Q-table statistics
red_nonzero = sum(1 for state in phase1['red'] if np.any(phase1['red'][state] != 0))
black_nonzero = sum(1 for state in phase1['black'] if np.any(phase1['black'][state] != 0))

print(f"Phase 1 Red states with learning: {red_nonzero}")
print(f"Phase 1 Black states with learning: {black_nonzero}")

# Do the same for phase 2 to see if black got training
print("\n--- Phase 2 (after self-play) ---")
phase2 = load_checkpoint("results/training/q-learning_curriculum_20251111_010525/phase2_selfplay", action_size=4)

red_nonzero_p2 = sum(1 for state in phase2['red'] if np.any(phase2['red'][state] != 0))
black_nonzero_p2 = sum(1 for state in phase2['black'] if np.any(phase2['black'][state] != 0))

print(f"Phase 2 Red states with learning: {red_nonzero_p2}")
print(f"Phase 2 Black states with learning: {black_nonzero_p2}")

# And final
print("\n--- Final (iteration3) ---")
final = load_checkpoint("results/training/q-learning_curriculum_20251111_010525/iteration3", action_size=4)

red_nonzero_final = sum(1 for state in final['red'] if np.any(final['red'][state] != 0))
black_nonzero_final = sum(1 for state in final['black'] if np.any(final['black'][state] != 0))

print(f"Final Red states with learning: {red_nonzero_final}")
print(f"Final Black states with learning: {black_nonzero_final}")


print("\n--- Phase 3 ---")
phase3 = load_checkpoint("results/training/q-learning_curriculum_20251111_010525/phase3_vscheckpoint", action_size=4)
red_p3 = sum(1 for state in phase3['red'] if np.any(phase3['red'][state] != 0))
black_p3 = sum(1 for state in phase3['black'] if np.any(phase3['black'][state] != 0))
print(f"Phase 3 Red: {red_p3}, Black: {black_p3}")

print("\n--- Iteration 1 ---")
iter1 = load_checkpoint("results/training/q-learning_curriculum_20251111_010525/iteration1", action_size=4)
red_i1 = sum(1 for state in iter1['red'] if np.any(iter1['red'][state] != 0))
black_i1 = sum(1 for state in iter1['black'] if np.any(iter1['black'][state] != 0))
print(f"Iter1 Red: {red_i1}, Black: {black_i1}")
