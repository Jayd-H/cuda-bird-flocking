# cuda bird flocking
 Bird flocking simulation in CUDA

![Visualisation of Birds](visualisation.png)

When running the visual simulation, bird colour indicates the strongest force:

- Red = Separation
- Blue = Alignment
- Green = Cohesion

## Basic Commands

For regular visualization with FPS tracking:
```sh
BirdSim.exe
```
For raw computational benchmark of steps without graphics:
```sh
BirdSim.exe benchmark                  # Default: 200 birds, 1000 steps
BirdSim.exe benchmark 500              # 500 birds, 1000 steps
BirdSim.exe benchmark 500 2000         # 500 birds, 2000 steps
```

For scaling analysis across different flock sizes:
```sh
BirdSim.exe scaling
```

**NOTE: MAX BIRD COUNT IS 5000**