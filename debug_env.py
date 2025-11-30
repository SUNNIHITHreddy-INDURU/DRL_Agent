"""
Debug script to identify environment issues
"""

import traceback
import sys

print("="*60)
print("DEBUGGING ENVIRONMENT SETUP")
print("="*60)

# Test 1: Check imports
print("\n1. Testing imports...")
try:
    import gymnasium as gym
    print("✓ gymnasium imported")
except Exception as e:
    print(f"✗ gymnasium import failed: {e}")
    sys.exit(1)

try:
    import highway_env
    print("✓ highway_env imported")
except Exception as e:
    print(f"✗ highway_env import failed: {e}")
    sys.exit(1)

try:
    import numpy as np
    print("✓ numpy imported")
except Exception as e:
    print(f"✗ numpy import failed: {e}")
    sys.exit(1)

# Test 2: Check if multi_stage_env.py exists
print("\n2. Checking multi_stage_env.py...")
try:
    import multi_stage_env
    print("✓ multi_stage_env.py found and imported")
except Exception as e:
    print(f"✗ multi_stage_env.py import failed: {e}")
    print("\nFull traceback:")
    traceback.print_exc()
    sys.exit(1)

# Test 3: Check if environment is registered
print("\n3. Checking environment registration...")
try:
    from gymnasium.envs.registration import registry
    if 'multi-stage-highway-v0' in registry:
        print("✓ multi-stage-highway-v0 is registered")
    else:
        print("✗ multi-stage-highway-v0 NOT registered")
        print("\nRegistered environments containing 'stage':")
        for env_id in registry:
            if 'stage' in env_id.lower():
                print(f"  - {env_id}")
except Exception as e:
    print(f"✗ Registry check failed: {e}")
    traceback.print_exc()

# Test 4: Try to create environment
print("\n4. Attempting to create environment...")
try:
    env = gym.make('multi-stage-highway-v0')
    print("✓ Environment created successfully!")
    env.close()
except Exception as e:
    print(f"✗ Environment creation failed: {e}")
    print("\nFull traceback:")
    traceback.print_exc()
    
    print("\n" + "="*60)
    print("ATTEMPTING MANUAL ENVIRONMENT CREATION")
    print("="*60)
    
    # Try creating environment directly
    try:
        from multi_stage_env import HighwayConstructionEnv
        print("\nTrying direct instantiation...")
        env = HighwayConstructionEnv()
        print("✓ Direct instantiation successful!")
        env.close()
    except Exception as e2:
        print(f"✗ Direct instantiation also failed: {e2}")
        print("\nFull traceback:")
        traceback.print_exc()
    
    sys.exit(1)

# Test 5: Try reset
print("\n5. Testing environment reset...")
try:
    env = gym.make('multi-stage-highway-v0')
    obs, info = env.reset()
    print("✓ Environment reset successful!")
    print(f"  - Observation shape: {np.array(obs).shape if hasattr(obs, '__len__') else 'scalar'}")
    print(f"  - Info keys: {list(info.keys())}")
    env.close()
except Exception as e:
    print(f"✗ Environment reset failed: {e}")
    traceback.print_exc()
    sys.exit(1)

print("\n" + "="*60)
print("✓ ALL TESTS PASSED!")
print("="*60)
print("\nYour environment is working correctly.")
print("You can now run: python test_environment.py")