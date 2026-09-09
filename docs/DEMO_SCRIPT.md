# Failsafe — demo reel narration

Voiceover for `docs/site/video/failsafe-reel.mp4` (~81 s, silent). Plain, calm, ~2.5 words/second,
a short beat at each card change. Numbers match the compiled policy and the report; keep them exact.

| # | On screen (dwell) | Say |
|---|---|---|
| 1 | **Failsafe** title (3.5 s) | "This is Failsafe. Don't ask an AI what to do while a machine is failing." |
| 2 | The problem (5 s) | "Cameras, robots, and drones fail in messy ways. The real danger isn't being less accurate. It's quietly stopping the one job they were given." |
| 3 | How it's done today (5 s) | "Today the backup plan is hand-written and never tested. And letting an AI improvise mid-failure, on something that can hurt people, is worse." |
| 4 | Find it. Prove it. Lock it in. (5 s) | "So Failsafe tries many backups, breaks the network on purpose, and measures each one. A fixed checker, not the AI, decides what passes." |
| 5 | Built by AI. Run without it. (6.5 s) | "The AI proposes plans and the checker proves them, before anything ships. On the device there's no AI, just a lookup. Every green mode shows how many people it catches out of a hundred and how fast it alerts. All measured." |
| 6 | The live demo (2.5 s) | "Here it is, running on real video." |
| — | **Warehouse split-screen** (~13 s) | "Same footage, two systems. Then the network is cut. On the left, an ordinary cloud system goes blind. On the right, Failsafe stops waiting on the cloud, decides on the device, and keeps watching the zone." |
| — | Self-driving dashcam (~5 s) | "Same system on a self-driving dashcam. Link lost, still watching the road." |
| — | Drone (~5 s) | "On a drone. Comms down, still tracking the ground below." |
| — | Robot safety (~5 s) | "On a robot's safety zone. The planner drops out, the robot keeps its eyes on people." |
| — | Robot arm (~5 s) | "And on a robot arm, where no backup was ever proven safe, it stops and asks a person instead of guessing." |
| 7 | Slow ≠ dead (5.5 s) | "The main finding: a slow cloud is more dangerous than a dead one. When the cloud is dead, every alert arrives on time. When it's slow but alive, every one arrives too late." |
| 8 | One system, five real jobs (5.5 s) | "One compiler, five real jobs: a warehouse, a car, a drone, and two robots. Four find a verified backup. The fifth is where it correctly refuses." |
| 9 | Trust + sponsors (5.5 s) | "The AI never decides what a failing system does. NVIDIA Nemotron proposes; a fixed checker proves. In its one completed search it found the best plan first, and two others were rejected before any test ran." |
| 10 | Close (4 s) | "Prove what works. Use only what's proven. It's open source, and every result can be reproduced." |

**Delivery notes**
- Total script is ~200 words for ~81 s: unhurried, with a clear pause at every card cut.
- The demo section (cards 6 through the five clips) is where the voice does the most work; let the "goes blind / keeps watching" line land on the warehouse cut.
- If you run long, the trimmable lines are the drone and robot-safety one-liners (the point is already made by the warehouse and dashcam).
- Tone: matter-of-fact, not salesy. The system's restraint is the pitch.
