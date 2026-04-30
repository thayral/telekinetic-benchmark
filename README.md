
# Benchmarking framework

Interactive environement / dataset generator for spatial reasoning experiments, using mujoco.
(This project is under development. It shifted from training environement to a benchmarking philosophy, inspired by the DeepMind paper / Kaggle competition on cognitive benchmarks)


# Telekinetics

Designing cognitive benchmarks for **physical AI**: spatial understanding, object-centric representations, action-conditioned...
It is set in a "telekinesis" framework: agents control objects in an abstract way (motion without contact). The low-level complexity of physical interaction and dynamics is partly removed, to focus on higher-level physical constraints: object collision, 3D structure, permanent identity of objects, ... . Telekinesis corresponds to the **object-centric control**, without embodied interaction, a good compromise for the cognitive evaluation of arbitrary embodiements in physical tasks with scene understanding, spatial reasoning and action planning. The interface and evaluation protocols remain a big challenge (visual QA, structured text output, interactive evaluation...). In an interactive setup, the benchmark can consist in **visual interactive puzzles**, but less abstract than ARC-AGI-3, and testing more basic cognitive factulties that are still difficult.


This complements the vision of my **PhD work** on [low-level reflex control](https://thayral.github.io/reactive-slip-control/) for manipulation. The fast-feedback stabilization layers can handle the physical contacts, and provide abstraction over interaction. But higher-level planning and reasoning still have to be grounded in objects and causal interaction for the 3D physical world. 

TLDR: Can AI understand the structured 3D physical scene in an image from video alone ? Does grounding require contact or is it about **interaction** ?


  <em>Task: "Move the yellow object to the left"</em>

<p align="center">
  <img src="media/scene-209652396_9d1e69bb5206669c_correct_000001_move_1777516124330688300.png" width="300"/><br>
  <img src="media/yellow cube.png" width="300"/> <img src="media/purple collision.png" width="300"/>
</p>


## Why this framework?


- **Object transformations**  
  States evolve through explicit transformations, structuring interaction as causal* transformations of objects.
    (*causality as interpretative mecanism, grounded in the affordances at the level of the agents)

- **Transformation graph as a dataset**  
  Each scene becomes a graph of states and transitions, allowing:
  - trajectory-based learning  
  - counterfactual reasoning  
  - structured supervision beyond flat samples  

- **QA triplets generation**  
  Built-in support for `(state, question, answer)` data, making it easy to generate supervised reasoning datasets.

- **Occlusion-aware reasoning**  
    - Partial observability, object permanence...

- **Explicit collision modeling**  
    - 3D structure and visual puzzles based on physical constraints.


- **Clear separation of concerns**  
  - Scene specification (reproducible structure)  
  - Factory (random generation)  
  - Environment (physics engine)  
  - Interaction dataset (-> train & eval)



## Roadmap

- [x] Simulation environement 
- [x] Scene factory
- [x] Dataset generation VQA
- [x] Step callbacks 
- [x] Composition of interactions

- [ ] Canonical replay path for loading scenes, with stronger seeding and history
- [ ] Add collision summaries to transition edges
- [ ] Add occlusion summaries to transition edges
- [ ] Multi-POV, agent motion

- [ ] Benchmark difficulty axes:
  - visual ambiguity (clip)
  - scene difficulty
  - interaction vocabulary
  - prompt variants
  - camera/viewpoint variants
  - object textures / floor grid
  - 2D / 3D
