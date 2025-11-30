# Reference Pick Policy

This module mirrors the structure of the reference team's motion-planning code
(`/yyc_prj_ref/policy`). The goal is to keep the same control phases—approach,
descend, grip, lift—while adapting them to the lightweight Sapien scene used in
`eai2025`.

Key pieces:

- `ik_pick_policy.py`: builds per-episode joint-space waypoints via Sapien's
  inverse-kinematics solver. The stages are recomputed using the live cube pose,
  so the arm does not rely on hard-coded joint presets.
- `run_reference_pick_policy.py`: driver script analogous to
  `solve_lift_cube_so101.py` from the reference repo. It logs planner stages and
  records front-camera frames under `outputs/reference_pick/`.

Usage:

```bash
python -m policies.reference_pick.run_reference_pick_policy --episodes 1 --task lift
```

Generate a video from the recorded front-camera frames with:

```bash
ffmpeg -y -framerate 30 -pattern_type glob -i 'outputs/reference_pick/episode_000/front/*.png' outputs/reference_pick_episode_000_front.mp4
```

You can tailor approach/descend heights via the `IKPickPolicy` constructor or by
adding new command-line flags (e.g., to sweep over lift heights for stack/sort
variants). The implementation intentionally stays close to the reference code's
style—planning happens via a dedicated class, stages are named for debugging,
and the runner keeps per-step telemetry for easier comparisons.
