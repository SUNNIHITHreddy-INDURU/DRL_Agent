# DRL_Agent

🛣️ Custom Environment: Highway Construction Zone

Content:

4-lane straight highway (length = 1000m)

Construction cones placed in lane 0 at randomized offsets

Traffic vehicles: 18 cars, constant speed (~30 m/s)

Ego vehicle starts in random lane [0,1,2]

Ego speed = traffic speed + 1 m/s

LidarObservation (128 cells) used as state input

DiscreteMetaAction (accelerate, brake, left, right, keep lane)

Key Challenges:

Avoiding cones in lane 0

Maintaining safe distance

Lane selection under traffic

Navigating with limited Lidar vision
Slide: State, Action, Reward Definitions
State (Observation)

LidarObservation (128 cells)

Detects distances to:

vehicles ahead

traffic behind

lane borders

construction cones

Normalized values ∈ [0,1]

Action Space (DiscreteMetaAction)
Action ID    Meaning
0    Keep lane
1    Accelerate
2    Brake
3    Lane Left
4    Lane Right
Reward Function Components

Positive rewards:

+1.0 = ideal speed (60–70 mph)

+1.0 = safe lane during construction zone

+0.3 = safe distance (>20m)

+0.07 × (speed/30) = forward progress

+0.2 = preferred middle lanes

Penalties:

-50 = crash

-2.5 = staying in construction lane inside zone

-0.5 = unsafe tailgating

-0.02 × |lateral drift| = lane-centering penalty

-0.05 = unnecessary lane-change

Reward Clipping:

Final r ∈ [-10, +10] for DRL stability
