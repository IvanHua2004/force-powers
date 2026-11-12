# Force Powers

Move real physics objects with your bare hands through a webcam. Grab things from a
distance, fling them, pull them in with a fist, and blast them away with an open palm.

Hand tracking uses MediaPipe, physics uses pymunk, and drawing uses pygame.

## Play

You need Windows, a webcam, and [Python 3.12](https://www.python.org/downloads/).

Double-click `run.bat`. The first run installs the libraries and downloads the hand
tracking model, which takes a few minutes. After that it starts straight away.

If the wrong camera opens, run `run.bat --camera 1` from a terminal.

On other systems:

```bash
python -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python game.py
```

## Controls

| Gesture | Power |
| --- | --- |
| Pinch (thumb and index together, other fingers up) | Grab the nearest object, carry it, and throw it by letting go mid-swing |
| Fist | Pull nearby objects in and charge up |
| Open palm after a fist | Blast everything away; a longer charge hits harder |
| Pinch one object with both hands | Stretch it or shrink it |

Smash the red-and-white targets with thrown or blasted objects before the 90 seconds
run out. Faster hits score more. Carrying an object into a target does not count.

Keys: `R` restarts, `C` hides or shows the camera background, `Esc` quits. When a round
ends, holding two fists for a second also restarts.

## Files

- `game.py`: the game, physics, and drawing.
- `hands.py`: webcam capture, hand tracking, and gesture detection.
- `run.bat`: Windows launcher that sets everything up on first run.

## Check without a camera

```bash
.venv\Scripts\python.exe game.py --selftest
```

This drives the game with scripted fake hands and prints `selftest passed`.
