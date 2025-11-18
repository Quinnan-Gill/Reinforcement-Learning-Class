"""
Training methods for RL agents in Connect Four.

These functions work with any agent class (Q-Learning, SARSA, Monte Carlo, etc.)
as long as the agent has:
- select_action(state, player) method
- train_step(episode) method for self-play
- q['red'] and q['black'] dictionaries
"""

import numpy as np
from pathlib import Path
from typing import Dict, Any
from random_agent import RandomAgent
from frozen_agent import FrozenAgent


def train_vs_random(
    agent,
    env,
    episodes: int,
    bidirectional: bool = True,
    workspace: Path = None
) -> Dict[str, Any]:
    """
    Train agent against random opponent.
    
    Args:
        agent: RL agent to train
        env: ConnectFour environment
        episodes: Number of episodes per color (total = episodes * 2 if bidirectional)
        bidirectional: If True, train both colors equally
        workspace: Directory to save learning curves (optional)
    
    Returns:
        Dictionary with training stats
    """
    random_opponent = RandomAgent(env)
    learning_curve_red = []
    learning_curve_black = []
    
    # Phase 1: Agent as RED
    print(f"\nPhase 1: Agent as RED vs Random")
    for i in range(episodes):
        print(f"\rRed Episode {i+1}/{episodes}", end="")
        
        # Start episode tracking for Monte Carlo
        if hasattr(agent, 'start_episode'):
            agent.start_episode()
        
        env.reset()
        done = False
        episode_reward_red = 0.0
        
        while not done:
            current_player = env.current_player
            state = env.get_state_key()
            
            # Select action
            if current_player == 1:  # Agent (Red)
                action = agent.select_action(state, current_player)
            else:  # Random opponent (Black)
                action = random_opponent.make_move()
            
            # Execute move
            _, reward, done, _ = env.make_move(action)
            next_state = env.get_state_key()
            
            # Record step for Monte Carlo
            if current_player == 1 and hasattr(agent, 'record_step'):
                agent.record_step(state, current_player, action, reward)
            
            # Update agent (if it has update method for TD learning)
            if current_player == 1 and hasattr(agent, 'update_q_values') and not hasattr(agent, 'record_step'):
                agent.update_q_values(current_player, state, action, reward, next_state, done)
                episode_reward_red += reward
        
        # For Monte Carlo agents - update at end of episode
        if hasattr(agent, 'update_after_episode'):
            agent.update_after_episode()
        
        learning_curve_red.append(episode_reward_red)
    
    if bidirectional:
        # Phase 2: Agent as BLACK
        print(f"\nPhase 2: Agent as BLACK vs Random")
        for i in range(episodes):
            print(f"\rBlack Episode {i+1}/{episodes}", end="")
            
            # Start episode tracking for Monte Carlo
            if hasattr(agent, 'start_episode'):
                agent.start_episode()
            
            env.reset()
            done = False
            episode_reward_black = 0.0
            
            while not done:
                current_player = env.current_player
                state = env.get_state_key()
                
                # Select action
                if current_player == 1:  # Random opponent (Red)
                    action = random_opponent.make_move()
                else:  # Agent (Black)
                    action = agent.select_action(state, current_player)
                
                # Execute move
                _, reward, done, _ = env.make_move(action)
                next_state = env.get_state_key()
                
                # Record step for Monte Carlo
                if current_player == -1 and hasattr(agent, 'record_step'):
                    agent.record_step(state, current_player, action, reward)
                
                # Update agent
                if current_player == -1 and hasattr(agent, 'update_q_values') and not hasattr(agent, 'record_step'):
                    agent.update_q_values(current_player, state, action, reward, next_state, done)
                    episode_reward_black += reward
            
            # For Monte Carlo agents
            if hasattr(agent, 'update_after_episode'):
                agent.update_after_episode()
            
            learning_curve_black.append(episode_reward_black)
    
    print(f"\n✓ Training complete")
    
    # Save learning curves if workspace provided
    if workspace:
        from checkpoints import save_learning_curve
        if learning_curve_red:
            save_learning_curve(str(workspace), player='red', 
                              data=np.array(learning_curve_red).reshape(-1, 1))
        if learning_curve_black:
            save_learning_curve(str(workspace), player='black',
                              data=np.array(learning_curve_black).reshape(-1, 1))
    
    return {
        'red_states': len(agent.q['red']),
        'black_states': len(agent.q['black']),
        'total_episodes': episodes * 2 if bidirectional else episodes
    }


def train_self_play(
    agent,
    env,
    episodes: int,
    workspace: Path = None
) -> Dict[str, Any]:
    """
    Train agent via self-play.
    
    Args:
        agent: RL agent to train
        env: ConnectFour environment
        episodes: Number of self-play episodes
        workspace: Directory to save learning curves (optional)
    
    Returns:
        Dictionary with training stats
    """
    learning_curve_red = []
    learning_curve_black = []
    
    for i in range(episodes):
        print(f"\rEpisode {i+1}/{episodes}", end="")
        
        # Use agent's train_step if available (handles self-play internally)
        if hasattr(agent, 'train_step'):
            total_reward = agent.train_step(i)
            if isinstance(total_reward, dict):
                learning_curve_red.append(total_reward.get('red', 0))
                learning_curve_black.append(total_reward.get('black', 0))
        else:
            # Manual self-play implementation
            env.reset()
            done = False
            rewards = {'red': 0, 'black': 0}
            
            while not done:
                current_player = env.current_player
                state = env.get_state_key()
                action = agent.select_action(state, current_player)
                
                _, reward, done, _ = env.make_move(action)
                next_state = env.get_state_key()
                
                # Update agent
                if hasattr(agent, 'update_q_values'):
                    agent.update_q_values(current_player, state, action, reward, next_state, done)
                
                player_key = 'red' if current_player == 1 else 'black'
                rewards[player_key] += reward
            
            if hasattr(agent, 'update_after_episode'):
                agent.update_after_episode()
            
            learning_curve_red.append(rewards['red'])
            learning_curve_black.append(rewards['black'])
    
    print(f"\n✓ Training complete")
    
    # Save learning curves if workspace provided
    if workspace and learning_curve_red:
        from checkpoints import save_learning_curve
        save_learning_curve(str(workspace), player='red',
                          data=np.array(learning_curve_red).reshape(-1, 1))
        save_learning_curve(str(workspace), player='black',
                          data=np.array(learning_curve_black).reshape(-1, 1))
    
    return {
        'red_states': len(agent.q['red']),
        'black_states': len(agent.q['black']),
        'total_episodes': episodes
    }


def train_vs_checkpoint(
    agent,
    env,
    checkpoint_path: str,
    episodes: int,
    bidirectional: bool = True,
    workspace: Path = None
) -> Dict[str, Any]:
    """
    Train agent against a frozen checkpoint opponent.
    
    Args:
        agent: RL agent to train (continues learning)
        env: ConnectFour environment
        checkpoint_path: Path to checkpoint directory
        episodes: Number of episodes per color
        bidirectional: If True, train both colors equally
        workspace: Directory to save learning curves (optional)
    
    Returns:
        Dictionary with training stats
    """
    # Load frozen checkpoint opponent
    frozen_opponent = FrozenAgent(env, checkpoint_path, player=-1)
    
    learning_curve_red = []
    learning_curve_black = []
    
    # Phase 1: Agent as RED vs Frozen as BLACK
    print(f"\nAgent as RED vs Checkpoint")
    for i in range(episodes):
        print(f"\rRed Episode {i+1}/{episodes}", end="")
        
        # Start episode tracking for Monte Carlo
        if hasattr(agent, 'start_episode'):
            agent.start_episode()
        
        env.reset()
        done = False
        episode_reward_red = 0.0
        
        while not done:
            current_player = env.current_player
            state = env.get_state_key()
            
            if current_player == 1:  # Agent (Red)
                action = agent.select_action(state, current_player)
            else:  # Frozen opponent (Black)
                action = frozen_opponent.make_move()
            
            _, reward, done, _ = env.make_move(action)
            next_state = env.get_state_key()
            
            # Record step for Monte Carlo
            if current_player == 1 and hasattr(agent, 'record_step'):
                agent.record_step(state, current_player, action, reward)
            
            # TD learning update
            if current_player == 1 and hasattr(agent, 'update_q_values') and not hasattr(agent, 'record_step'):
                agent.update_q_values(current_player, state, action, reward, next_state, done)
                episode_reward_red += reward
        
        if hasattr(agent, 'update_after_episode'):
            agent.update_after_episode()
        
        learning_curve_red.append(episode_reward_red)
    
    if bidirectional:
        # Phase 2: Agent as BLACK vs Frozen as RED
        frozen_opponent.player = 1  # Switch frozen to play red
        
        print(f"\nAgent as BLACK vs Checkpoint")
        for i in range(episodes):
            print(f"\rBlack Episode {i+1}/{episodes}", end="")
            
            # Start episode tracking for Monte Carlo
            if hasattr(agent, 'start_episode'):
                agent.start_episode()
            
            env.reset()
            done = False
            episode_reward_black = 0.0
            
            while not done:
                current_player = env.current_player
                state = env.get_state_key()
                
                if current_player == 1:  # Frozen opponent (Red)
                    action = frozen_opponent.make_move()
                else:  # Agent (Black)
                    action = agent.select_action(state, current_player)
                
                _, reward, done, _ = env.make_move(action)
                next_state = env.get_state_key()
                
                # Record step for Monte Carlo
                if current_player == -1 and hasattr(agent, 'record_step'):
                    agent.record_step(state, current_player, action, reward)
                
                # TD learning update
                if current_player == -1 and hasattr(agent, 'update_q_values') and not hasattr(agent, 'record_step'):
                    agent.update_q_values(current_player, state, action, reward, next_state, done)
                    episode_reward_black += reward
            
            if hasattr(agent, 'update_after_episode'):
                agent.update_after_episode()
            
            learning_curve_black.append(episode_reward_black)
    
    print(f"\n✓ Training complete")
    
    # Save learning curves if workspace provided
    if workspace:
        from checkpoints import save_learning_curve
        if learning_curve_red:
            save_learning_curve(str(workspace), player='red',
                              data=np.array(learning_curve_red).reshape(-1, 1))
        if learning_curve_black:
            save_learning_curve(str(workspace), player='black',
                              data=np.array(learning_curve_black).reshape(-1, 1))
    
    return {
        'red_states': len(agent.q['red']),
        'black_states': len(agent.q['black']),
        'total_episodes': episodes * 2 if bidirectional else episodes
    }


def train_curriculum(
    agent,
    env,
    config: Dict[str, Any],
    base_workspace: Path
) -> Dict[str, Any]:
    """
    Train agent using curriculum learning.
    
    Curriculum phases:
    1. vs Random opponent (foundation)
    2. Self-play (strategy development)
    3. vs Phase 1 checkpoint (test improvement)
    4-N. vs previous best checkpoints (iterative refinement)
    
    Args:
        agent: RL agent to train
        env: ConnectFour environment
        config: Dictionary with:
            - 'vs_random_episodes': Episodes for phase 1
            - 'selfplay_episodes': Episodes for phase 2
            - 'vs_checkpoint_episodes': Episodes for phases 3+
            - 'iterations': Number of iterative checkpoint phases
        base_workspace: Base directory for saving checkpoints
    
    Returns:
        Dictionary with final checkpoint path and stats
    """
    print(f"\n{'='*70}")
    print("CURRICULUM LEARNING")
    print(f"{'='*70}")
    print(f"Phase 1 (vs random): {config.get('vs_random_episodes', 0)} episodes")
    print(f"Phase 2 (self-play): {config.get('selfplay_episodes', 0)} episodes")
    print(f"Phase 3+ (vs checkpoint): {config.get('vs_checkpoint_episodes', 0)} episodes each")
    print(f"Iterations: {config.get('iterations', 0)}")
    
    base_workspace.mkdir(parents=True, exist_ok=True)
    
    # Phase 1: vs Random
    if config.get('vs_random_episodes', 0) > 0:
        print(f"\n--- Phase 1: Training vs Random Opponent ---")
        phase1_workspace = base_workspace / "phase1_vsrandom"
        phase1_workspace.mkdir(exist_ok=True)
        
        train_vs_random(agent, env, config['vs_random_episodes'], 
                       bidirectional=True, workspace=phase1_workspace)
        
        # Save checkpoint
        np.save(phase1_workspace / "best_red_agent", dict(agent.q['red']))
        np.save(phase1_workspace / "best_black_agent", dict(agent.q['black']))
        print(f"✓ Phase 1 complete - Checkpoint saved")
    
    # Phase 2: Self-play
    if config.get('selfplay_episodes', 0) > 0:
        print(f"\n--- Phase 2: Training via Self-Play ---")
        phase2_workspace = base_workspace / "phase2_selfplay"
        phase2_workspace.mkdir(exist_ok=True)
        
        train_self_play(agent, env, config['selfplay_episodes'], 
                       workspace=phase2_workspace)
        
        # Save checkpoint
        np.save(phase2_workspace / "best_red_agent", dict(agent.q['red']))
        np.save(phase2_workspace / "best_black_agent", dict(agent.q['black']))
        print(f"✓ Phase 2 complete - Checkpoint saved")
    
    # Phase 3: vs Phase 1 checkpoint
    if config.get('vs_checkpoint_episodes', 0) > 0:
        print(f"\n--- Phase 3: Training vs Phase 1 Checkpoint ---")
        phase3_workspace = base_workspace / "phase3_vscheckpoint"
        phase3_workspace.mkdir(exist_ok=True)
        
        train_vs_checkpoint(agent, env, str(base_workspace / "phase1_vsrandom"),
                           config['vs_checkpoint_episodes'],
                           bidirectional=True, workspace=phase3_workspace)
        
        # Save checkpoint
        np.save(phase3_workspace / "best_red_agent", dict(agent.q['red']))
        np.save(phase3_workspace / "best_black_agent", dict(agent.q['black']))
        print(f"✓ Phase 3 complete - Checkpoint saved")
        
        previous_checkpoint = str(phase3_workspace)
    else:
        previous_checkpoint = str(base_workspace / "phase2_selfplay")
    
    # Iterations: vs previous best
    iterations = config.get('iterations', 0)
    for i in range(iterations):
        print(f"\n--- Iteration {i+1}: Training vs Previous Best ---")
        iter_workspace = base_workspace / f"iteration{i+1}"
        iter_workspace.mkdir(exist_ok=True)
        
        train_vs_checkpoint(agent, env, previous_checkpoint,
                           config['vs_checkpoint_episodes'],
                           bidirectional=True, workspace=iter_workspace)
        
        # Save checkpoint
        np.save(iter_workspace / "best_red_agent", dict(agent.q['red']))
        np.save(iter_workspace / "best_black_agent", dict(agent.q['black']))
        print(f"✓ Iteration {i+1} complete - Checkpoint saved")
        
        previous_checkpoint = str(iter_workspace)
    
    total_phases = 3 + iterations  # phase1, phase2, phase3, + iterations
    print(f"\n✓ Full curriculum training complete ({total_phases} phases total)")
    print(f"✓ Final checkpoint: {previous_checkpoint}")
    
    return {
        'final_checkpoint': previous_checkpoint,
        'total_phases': total_phases,
        'red_states': len(agent.q['red']),
        'black_states': len(agent.q['black'])
    }
