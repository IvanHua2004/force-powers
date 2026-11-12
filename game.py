"""Force Powers: move real physics objects with your hands through a webcam.

  Pinch (thumb + index, other fingers up)  grab an object from a distance, move it, throw it
  Fist                                     pull nearby objects in and charge up
  Open palm after a fist                   blast everything away
  Pinch the same object with both hands    stretch or shrink it

Smash the glowing targets with thrown or blasted objects before the time runs out.
Keys: R restart, C toggle camera background, Esc quit.
"""
import argparse
import math
import os
import random
import sys

W, H = 1280, 720
ROUND_SECONDS = 90
GRAVITY = 900
MAX_SPEED = 2600
GRAB_REACH = 140
PULL_RADIUS = 380
MAX_CHARGE = 1.5
HIT_SPEED = 350

GESTURE_COLOURS = {
    "none": (225, 230, 240),
    "pinch": (70, 225, 255),
    "fist": (255, 150, 50),
    "open": (190, 130, 255),
}
THING_COLOURS = [(255, 107, 107), (255, 200, 87), (78, 205, 196), (120, 160, 255), (200, 130, 255)]


def parse_args():
    parser = argparse.ArgumentParser(description="Force Powers hand-tracking game")
    parser.add_argument("--camera", type=int, default=0, help="camera index (default 0)")
    parser.add_argument("--selftest", action="store_true", help="run a camera-free check and exit")
    return parser.parse_args()


ARGS = parse_args() if __name__ == "__main__" else argparse.Namespace(camera=0, selftest=False)
if ARGS.selftest:
    os.environ["SDL_VIDEODRIVER"] = "dummy"
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

import pygame  # noqa: E402
import pymunk  # noqa: E402

from hands import BONES, FIST, OPEN, PINCH, CameraSource, HandSet, fake_hand  # noqa: E402


def dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


class Thing:
    """A throwable box or ball."""

    def __init__(self, space, pos, kind, size, colour):
        self.space = space
        self.kind = kind
        self.base = size
        self.scale = 1.0
        self.colour = colour
        self.body = pymunk.Body(1, 1)
        self.body.position = pos
        self.shape = None
        space.add(self.body)
        self.build()

    @property
    def radius(self):
        return self.base * self.scale * (1.0 if self.kind == "ball" else 1.2)

    def build(self):
        """(Re)create the collision shape for the current scale."""
        s = self.base * self.scale
        if self.shape is not None:
            self.space.remove(self.shape)
        if self.kind == "ball":
            mass = 0.002 * math.pi * s * s
            self.body.mass = mass
            self.body.moment = pymunk.moment_for_circle(mass, 0, s)
            self.shape = pymunk.Circle(self.body, s)
        else:
            mass = 0.002 * 4 * s * s
            self.body.mass = mass
            self.body.moment = pymunk.moment_for_box(mass, (2 * s, 2 * s))
            self.shape = pymunk.Poly.create_box(self.body, (2 * s, 2 * s), radius=2)
        self.shape.elasticity = 0.55
        self.shape.friction = 0.7
        self.space.add(self.shape)


class Game:
    def __init__(self, seed=None):
        self.rng = random.Random(seed)
        self.space = pymunk.Space()
        self.space.gravity = (0, GRAVITY)
        edge = 60
        for a, b in (((-edge, H + edge), (W + edge, H + edge)), ((-edge, -edge), (W + edge, -edge)),
                     ((-edge, -edge), (-edge, H + edge)), ((W + edge, -edge), (W + edge, H + edge))):
            wall = pymunk.Segment(self.space.static_body, a, b, edge)
            wall.elasticity = 0.6
            wall.friction = 0.8
            self.space.add(wall)

        self.things = []
        for i in range(7):
            kind = "box" if i % 2 == 0 else "ball"
            pos = (self.rng.uniform(150, W - 150), self.rng.uniform(H - 260, H - 80))
            self.things.append(Thing(self.space, pos, kind, self.rng.uniform(26, 40),
                                     THING_COLOURS[i % len(THING_COLOURS)]))

        self.handset = HandSet()
        self.stretch = None  # (thing, start hand distance, start scale)
        self.particles = []
        self.shockwaves = []
        self.popups = []
        self.blasts = 0
        self.best = 0
        self.show_camera = True
        self.seen_hands = False
        self.restart_hold = 0.0
        self.font = pygame.font.SysFont("segoeui,arial", 30, bold=True)
        self.big = pygame.font.SysFont("segoeui,arial", 72, bold=True)
        self.small = pygame.font.SysFont("segoeui,arial", 20)
        self.restart()

    def restart(self):
        self.score = 0
        self.time_left = float(ROUND_SECONDS)
        self.state = "play"
        self.restart_hold = 0.0
        self.targets = []
        for _ in range(3):
            self.spawn_target()

    def spawn_target(self):
        for _ in range(40):
            pos = (self.rng.uniform(120, W - 120), self.rng.uniform(90, H - 280))
            if all(dist(pos, t["pos"]) > 180 for t in self.targets):
                break
        self.targets.append({"pos": pos, "r": 38, "age": 0.0})

    # ---- hands -------------------------------------------------------------

    def add_hand(self, hand):
        hand.anchor = pymunk.Body(body_type=pymunk.Body.KINEMATIC)
        hand.anchor.position = tuple(hand.pinch_point)
        self.space.add(hand.anchor)
        hand.grab = None
        hand.spring = None
        hand.charge = 0.0
        hand.charge_hold = 0.0

    def remove_hand(self, hand):
        self.release(hand)
        self.space.remove(hand.anchor)

    def try_grab(self, hand):
        point = hand.pinch_point
        best, best_gap = None, GRAB_REACH
        for thing in self.things:
            gap = dist(thing.body.position, point) - thing.radius
            if gap < best_gap:
                best, best_gap = thing, gap
        if best is None:
            return
        mass = best.body.mass
        hand.spring = pymunk.DampedSpring(hand.anchor, best.body, (0, 0), (0, 0), 0, 120 * mass, 14 * mass)
        self.space.add(hand.spring)
        hand.grab = best

    def release(self, hand):
        if hand.spring is not None:
            self.space.remove(hand.spring)
        hand.spring = None
        hand.grab = None

    def blast(self, hand):
        centre = hand.palm
        power = hand.charge / MAX_CHARGE
        radius = 300 + 160 * power
        for thing in self.things:
            offset = thing.body.position - tuple(centre)
            d = offset.length
            if d < radius:
                direction = offset / d if d > 1 else pymunk.Vec2d(0, -1)
                strength = thing.body.mass * (700 + 1400 * power) * math.sqrt(1 - d / radius)
                thing.body.apply_impulse_at_world_point(direction * strength, thing.body.position)
        self.shockwaves.append({"pos": tuple(centre), "r": 20.0, "max": radius, "life": 1.0})
        self.blasts += 1

    def update_hand(self, hand, dt):
        target = pymunk.Vec2d(*hand.pinch_point)
        velocity = (target - hand.anchor.position) / dt
        if velocity.length > 6000:
            velocity = velocity.scale_to_length(6000)
        hand.anchor.velocity = velocity

        if hand.gesture == PINCH:
            if hand.grab is None:
                self.try_grab(hand)
        elif hand.grab is not None:
            self.release(hand)

        if hand.gesture == FIST:
            hand.charge = min(MAX_CHARGE, hand.charge + dt)
            hand.charge_hold = 0.3
        elif hand.gesture == OPEN and hand.charge > 0.2:
            self.blast(hand)
            hand.charge = 0.0
        else:
            hand.charge_hold -= dt
            if hand.charge_hold <= 0:
                hand.charge = 0.0

    def update_stretch(self):
        hands = self.handset.hands
        both = len(hands) == 2 and hands[0].grab is not None and hands[0].grab is hands[1].grab
        if not both:
            self.stretch = None
            return
        thing = hands[0].grab
        span = max(dist(hands[0].pinch_point, hands[1].pinch_point), 1.0)
        if self.stretch is None or self.stretch[0] is not thing:
            self.stretch = (thing, span, thing.scale)
            return
        _, span0, scale0 = self.stretch
        scale = max(0.5, min(2.5, scale0 * span / span0))
        if abs(scale - thing.scale) > 0.03:
            thing.scale = scale
            thing.build()
            for hand in hands:
                hand.spring.stiffness = 120 * thing.body.mass
                hand.spring.damping = 14 * thing.body.mass

    def apply_pull(self):
        """Fists drag nearby objects in and hold them up against gravity."""
        for hand in self.handset.hands:
            if hand.gesture != FIST:
                continue
            centre = pymunk.Vec2d(*hand.palm)
            for thing in self.things:
                offset = centre - thing.body.position
                d = offset.length
                if 1 < d < PULL_RADIUS:
                    falloff = 1 - d / PULL_RADIUS
                    force = offset / d * (2600 * falloff) + pymunk.Vec2d(0, -GRAVITY * falloff)
                    thing.body.apply_force_at_world_point(force * thing.body.mass, thing.body.position)
                    thing.body.velocity = thing.body.velocity * 0.985

    # ---- simulation --------------------------------------------------------

    def step(self, dt, now, detections):
        added, removed = self.handset.update(detections, now)
        for hand in removed:
            self.remove_hand(hand)
        for hand in added:
            self.add_hand(hand)
        if self.handset.hands:
            self.seen_hands = True
        for hand in self.handset.hands:
            self.update_hand(hand, dt)
        self.update_stretch()

        for _ in range(2):
            self.apply_pull()
            self.space.step(dt / 2)
            for thing in self.things:
                speed = thing.body.velocity.length
                if speed > MAX_SPEED:
                    thing.body.velocity = thing.body.velocity * (MAX_SPEED / speed)

        if self.state == "play":
            self.check_targets()
            self.time_left -= dt
            if self.time_left <= 0:
                self.time_left = 0.0
                self.state = "over"
                self.best = max(self.best, self.score)
        else:
            hands = self.handset.hands
            if len(hands) == 2 and all(h.gesture == FIST for h in hands):
                self.restart_hold += dt
                if self.restart_hold > 1.0:
                    self.restart()
            else:
                self.restart_hold = 0.0

        for target in self.targets:
            target["age"] += dt
        for p in self.particles:
            p["pos"] = (p["pos"][0] + p["vel"][0] * dt, p["pos"][1] + p["vel"][1] * dt)
            p["vel"] = (p["vel"][0] * 0.96, p["vel"][1] * 0.96 + 600 * dt)
            p["life"] -= dt * 1.6
        self.particles = [p for p in self.particles if p["life"] > 0]
        for wave in self.shockwaves:
            wave["r"] += (wave["max"] - wave["r"]) * min(1.0, dt * 9)
            wave["life"] -= dt * 2.4
        self.shockwaves = [w for w in self.shockwaves if w["life"] > 0]
        for popup in self.popups:
            popup["pos"] = (popup["pos"][0], popup["pos"][1] - 50 * dt)
            popup["life"] -= dt
        self.popups = [p for p in self.popups if p["life"] > 0]

    def check_targets(self):
        held = {h.grab for h in self.handset.hands if h.grab is not None}
        for target in list(self.targets):
            for thing in self.things:
                if thing in held:
                    continue  # carrying an object into a target does not count
                speed = thing.body.velocity.length
                if speed > HIT_SPEED and dist(thing.body.position, target["pos"]) < target["r"] + thing.radius:
                    points = 10 + int(speed / 100)
                    self.score += points
                    self.targets.remove(target)
                    self.spawn_target()
                    self.popups.append({"pos": target["pos"], "text": f"+{points}", "life": 1.0})
                    for _ in range(28):
                        angle = self.rng.uniform(0, math.tau)
                        v = self.rng.uniform(120, 520)
                        self.particles.append({"pos": target["pos"], "life": 1.0, "colour": thing.colour,
                                               "vel": (math.cos(angle) * v, math.sin(angle) * v)})
                    break

    # ---- drawing -----------------------------------------------------------

    def draw(self, screen, background, now):
        if background is not None and self.show_camera:
            screen.blit(background, (0, 0))
        else:
            screen.fill((14, 16, 26))

        for target in self.targets:
            pulse = 1 + 0.08 * math.sin(target["age"] * 5)
            grow = min(1.0, target["age"] * 4)
            x, y = int(target["pos"][0]), int(target["pos"][1])
            for frac, colour in ((1.0, (255, 80, 110)), (0.66, (255, 255, 255)), (0.33, (255, 80, 110))):
                pygame.draw.circle(screen, colour, (x, y), max(2, int(target["r"] * frac * pulse * grow)))

        held = {h.grab for h in self.handset.hands if h.grab is not None}
        for thing in self.things:
            pos = thing.body.position
            outline = (255, 255, 255) if thing in held else (20, 22, 30)
            width = 4 if thing in held else 3
            if thing.kind == "ball":
                r = int(thing.base * thing.scale)
                pygame.draw.circle(screen, thing.colour, (int(pos.x), int(pos.y)), r)
                pygame.draw.circle(screen, outline, (int(pos.x), int(pos.y)), r, width)
                tip = thing.body.local_to_world((r * 0.8, 0))
                pygame.draw.line(screen, outline, (pos.x, pos.y), (tip.x, tip.y), 3)
            else:
                corners = [thing.body.local_to_world(v) for v in thing.shape.get_vertices()]
                corners = [(c.x, c.y) for c in corners]
                pygame.draw.polygon(screen, thing.colour, corners)
                pygame.draw.polygon(screen, outline, corners, width)

        for wave in self.shockwaves:
            shade = int(255 * max(0.0, wave["life"]))
            pygame.draw.circle(screen, (shade, int(shade * 0.7), 255),
                               (int(wave["pos"][0]), int(wave["pos"][1])), int(wave["r"]), 6)
        for p in self.particles:
            pygame.draw.circle(screen, p["colour"], (int(p["pos"][0]), int(p["pos"][1])),
                               max(1, int(6 * p["life"])))

        for hand in self.handset.hands:
            self.draw_hand(screen, hand, now)

        for popup in self.popups:
            text = self.font.render(popup["text"], True, (255, 240, 150))
            screen.blit(text, text.get_rect(center=popup["pos"]))

        self.draw_hud(screen)

    def draw_hand(self, screen, hand, now):
        colour = GESTURE_COLOURS[hand.gesture]
        pts = [(int(x), int(y)) for x, y in hand.pts]
        if hand.grab is not None:
            pinch = hand.pinch_point
            target = hand.grab.body.position
            pygame.draw.line(screen, colour, (pinch[0], pinch[1]), (target.x, target.y), 3)
        for a, b in BONES:
            pygame.draw.line(screen, (10, 12, 20), pts[a], pts[b], 9)
        for a, b in BONES:
            pygame.draw.line(screen, colour, pts[a], pts[b], 4)
        for p in pts:
            pygame.draw.circle(screen, (255, 255, 255), p, 4)
        palm = (int(hand.palm[0]), int(hand.palm[1]))
        if hand.gesture == FIST:
            power = hand.charge / MAX_CHARGE
            wobble = 4 * math.sin(now * 20)
            pygame.draw.circle(screen, colour, palm, int(30 + 60 * power + wobble), 4)
            pygame.draw.circle(screen, (90, 60, 30), palm, PULL_RADIUS, 1)
        elif hand.gesture == PINCH and hand.grab is None:
            pinch = (int(hand.pinch_point[0]), int(hand.pinch_point[1]))
            pygame.draw.circle(screen, colour, pinch, 12, 3)

    def draw_hud(self, screen):
        def label(text, font, colour, **where):
            shadow = font.render(text, True, (0, 0, 0))
            image = font.render(text, True, colour)
            rect = image.get_rect(**where)
            screen.blit(shadow, rect.move(2, 2))
            screen.blit(image, rect)

        label(f"Score {self.score}", self.font, (255, 255, 255), topleft=(24, 16))
        label(f"Best {self.best}", self.small, (200, 205, 220), topleft=(26, 56))
        colour = (255, 110, 110) if self.time_left < 10 else (255, 255, 255)
        label(f"{math.ceil(self.time_left)}s", self.font, colour, topright=(W - 24, 16))
        label("Pinch: grab and throw     Fist: pull and charge     Open palm: blast     "
              "Two pinches on one object: stretch", self.small, (220, 225, 235), midbottom=(W // 2, H - 12))
        if not self.seen_hands:
            label("Show your hands to the camera", self.font, (255, 240, 150), center=(W // 2, H // 2))
        if self.state == "over":
            label("Time!", self.big, (255, 255, 255), center=(W // 2, H // 2 - 70))
            label(f"Score {self.score}", self.font, (255, 240, 150), center=(W // 2, H // 2))
            label("Hold two fists for a second, or press R, to play again", self.small,
                  (220, 225, 235), center=(W // 2, H // 2 + 48))


def run(camera):
    pygame.init()
    screen = pygame.display.set_mode((W, H))
    pygame.display.set_caption("Force Powers")
    clock = pygame.time.Clock()
    game = Game()
    source = CameraSource((W, H), camera)
    source.start()
    last_seq, background, now = 0, None, 0.0
    waiting = game.font.render("Starting camera...", True, (255, 255, 255))
    try:
        while True:
            dt = min(clock.tick(60) / 1000, 1 / 30)
            now += dt
            for event in pygame.event.get():
                if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                    return 0
                if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                    game.restart()
                if event.type == pygame.KEYDOWN and event.key == pygame.K_c:
                    game.show_camera = not game.show_camera
            if source.error is not None:
                print(f"Camera or hand tracking failed: {source.error}", file=sys.stderr)
                return 1
            seq, frame, detections = source.latest()
            fresh = seq != last_seq
            if fresh:
                last_seq = seq
                background = pygame.image.frombuffer(frame.tobytes(), (W, H), "RGB")
            if background is None:
                screen.fill((14, 16, 26))
                screen.blit(waiting, waiting.get_rect(center=(W // 2, H // 2)))
            else:
                game.step(dt, now, detections if fresh else None)
                game.draw(screen, background, now)
            pygame.display.flip()
    finally:
        source.stop()
        pygame.quit()


def selftest():
    """Drive the game with scripted fake hands; no camera or window needed."""
    pygame.init()
    screen = pygame.display.set_mode((W, H))
    game = Game(seed=1)
    game.space.gravity = (0, GRAVITY)
    dt = 1 / 60
    clock = {"t": 0.0, "frame": 0}

    def advance(frames, hands_at):
        for i in range(frames):
            clock["t"] += dt
            clock["frame"] += 1
            game.step(dt, clock["t"], hands_at(i / max(frames - 1, 1)))
            if clock["frame"] % 10 == 0:
                game.draw(screen, None, clock["t"])

    def lerp(a, b, t):
        return a + (b - a) * t

    thing = game.things[0]
    for other in game.things[1:]:
        other.body.position = (1150, 650)  # out of the way

    # 1. Pinch-grab an object and carry it across the screen.
    thing.body.position, thing.body.velocity = (400, 500), (0, 0)
    advance(20, lambda t: [fake_hand(400, 500, PINCH)])
    advance(50, lambda t: [fake_hand(lerp(400, 800, t), lerp(500, 300, t), PINCH)])
    advance(20, lambda t: [fake_hand(800, 300, PINCH)])
    hand = game.handset.hands[0]
    assert hand.gesture == PINCH and hand.grab is thing, "pinch did not grab the object"
    assert dist(thing.body.position, (800, 300)) < 90, f"object did not follow: {thing.body.position}"

    # 2. Opening the hand lets go, and with no charge there is no blast.
    advance(10, lambda t: [fake_hand(800, 300, OPEN)])
    assert hand.grab is None and game.blasts == 0, "release failed"

    # 3. A fist charges and pulls; opening it afterwards blasts.
    thing.body.position, thing.body.velocity = (500, 560), (0, 0)
    advance(70, lambda t: [fake_hand(640, 400, FIST)])
    assert hand.charge > 0.5, f"fist did not charge: {hand.charge}"
    assert dist(thing.body.position, (640, 400)) < 150, f"fist did not pull: {thing.body.position}"
    advance(10, lambda t: [fake_hand(640, 400, OPEN)])
    assert game.blasts == 1, "open palm did not blast"
    assert thing.body.velocity.length > 200, "blast did not push the object"

    # 4. Two pinches on the same object stretch it.
    advance(30, lambda t: [])  # hands leave
    assert not game.handset.hands, "hands were not removed"
    thing.body.position, thing.body.velocity = (640, 400), (0, 0)
    advance(20, lambda t: [fake_hand(600, 400, PINCH), fake_hand(680, 400, PINCH)])
    advance(40, lambda t: [fake_hand(lerp(600, 440, t), 400, PINCH), fake_hand(lerp(680, 840, t), 400, PINCH)])
    assert thing.scale > 1.5, f"stretch failed: scale {thing.scale}"

    # 5. A fast free object pops a target and scores.
    advance(30, lambda t: [])
    target = game.targets[0]
    thing.body.position, thing.body.velocity = target["pos"], (900, 0)
    advance(2, lambda t: [])
    assert game.score > 0, "target hit did not score"

    # 6. The round ends when the clock runs out.
    game.time_left = 0.01
    advance(3, lambda t: [])
    assert game.state == "over", "round did not end"
    game.draw(screen, None, clock["t"])
    pygame.quit()
    print("selftest passed")
    return 0


if __name__ == "__main__":
    sys.exit(selftest() if ARGS.selftest else run(ARGS.camera))
