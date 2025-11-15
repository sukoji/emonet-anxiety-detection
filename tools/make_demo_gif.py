"""
Generate a synthetic demo animation for the README.

The real webcam demo (src/cap.py) shows actual team members' faces, so this
script re-creates the *same information* the demo surfaces -- face detection
box, Anxiety label + Score, valence/arousal readouts and the 2D valence-arousal
chart -- as a polished monitoring dashboard driven by an illustrated avatar
instead of a real person.

Scenario: normal ATM use -> voice-phishing call received -> anxiety detected
-> counselor alert. Output: docs/demo/demo_scenario.gif (+ poster .png files).

Rendering is 3x super-sampled and downscaled with Lanczos for crisp, anti-
aliased edges. Requires only Pillow + numpy (already used by the project).

    python tools/make_demo_gif.py
"""

import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "docs" / "demo"
OUT_DIR.mkdir(parents=True, exist_ok=True)
GIF_PATH = OUT_DIR / "demo_scenario.gif"
POSTER_PATH = OUT_DIR / "demo_scenario.png"

W, H = 960, 540
SS = 3  # super-sampling factor

FONT_DIR = Path("C:/Windows/Fonts")
_font_cache = {}


def font(size, bold=True):
    key = (size, bold)
    if key not in _font_cache:
        name = "malgunbd.ttf" if bold else "malgun.ttf"
        _font_cache[key] = ImageFont.truetype(str(FONT_DIR / name), int(size * SS))
    return _font_cache[key]


# ---- palette --------------------------------------------------------------
BG = (14, 17, 25)
PANEL = (23, 27, 38)
PANEL2 = (30, 35, 48)
STROKE = (49, 57, 75)
STROKE2 = (68, 78, 98)
TEXT = (228, 232, 240)
SUB = (139, 149, 168)
GREEN = (46, 204, 113)
GREEN_D = (30, 130, 74)
RED = (231, 76, 60)
RED_D = (150, 40, 34)
AMBER = (241, 196, 15)
CAMBG = (17, 21, 31)
SKIN = (238, 203, 173)
SKIN_SH = (208, 168, 140)
HAIR = (58, 45, 40)
SHIRT = (66, 86, 118)
SHIRT_SH = (52, 68, 96)


def lerp(a, b, t):
    return a + (b - a) * t


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def smooth(t):
    return t * t * (3 - 2 * t)


def mix(c1, c2, t):
    return tuple(int(lerp(c1[i], c2[i], t)) for i in range(3))


# ---- super-sampled drawing wrapper ----------------------------------------
class Canvas:
    def __init__(self):
        self.img = Image.new("RGB", (W * SS, H * SS), BG)
        self.d = ImageDraw.Draw(self.img, "RGBA")

    def _b(self, box):
        return [round(c * SS) for c in box]

    def _pts(self, pts):
        return [(round(x * SS), round(y * SS)) for x, y in pts]

    def rect(self, box, fill=None, outline=None, width=1):
        self.d.rectangle(self._b(box), fill=fill, outline=outline, width=round(width * SS))

    def rrect(self, box, radius, fill=None, outline=None, width=1):
        self.d.rounded_rectangle(self._b(box), radius=round(radius * SS), fill=fill,
                                 outline=outline, width=round(width * SS))

    def ellipse(self, box, fill=None, outline=None, width=1):
        self.d.ellipse(self._b(box), fill=fill, outline=outline, width=round(width * SS))

    def line(self, pts, fill, width=1):
        self.d.line(self._pts(pts), fill=fill, width=round(width * SS))

    def arc(self, box, start, end, fill, width=1):
        self.d.arc(self._b(box), start, end, fill=fill, width=round(width * SS))

    def polygon(self, pts, fill=None, outline=None, width=1):
        self.d.polygon(self._pts(pts), fill=fill, outline=outline, width=round(width * SS))

    def pieslice(self, box, start, end, fill=None):
        self.d.pieslice(self._b(box), start, end, fill=fill)

    def chord(self, box, start, end, fill=None):
        self.d.chord(self._b(box), start, end, fill=fill)

    def text(self, xy, s, size, fill, bold=True, anchor="la"):
        self.d.text((xy[0] * SS, xy[1] * SS), s, font=font(size, bold), fill=fill, anchor=anchor)

    def textlen(self, s, size, bold=True):
        return self.d.textlength(s, font=font(size, bold)) / SS

    def result(self):
        return self.img.resize((W, H), Image.LANCZOS)


# ---- scenario keyframes: (frame, valence, arousal, worry, phase) -----------
# phase: 0 normal, 1 call incoming, 2 analyzing, 3 anxiety alert, 4 connecting
KEYS = [
    (0, 0.62, -0.24, 0.07, 0),
    (20, 0.60, -0.20, 0.10, 0),
    (26, 0.55, -0.08, 0.24, 1),
    (40, 0.08, 0.28, 0.56, 2),
    (52, -0.56, 0.63, 0.90, 2),
    (60, -0.63, 0.67, 0.93, 3),
    (84, -0.60, 0.64, 0.90, 3),
    (92, -0.58, 0.60, 0.88, 4),
    (104, -0.58, 0.60, 0.88, 4),
]
N_FRAMES = KEYS[-1][0] + 1


def state_at(f):
    for i in range(len(KEYS) - 1):
        f0 = KEYS[i][0]
        f1 = KEYS[i + 1][0]
        if f0 <= f <= f1:
            t = smooth((f - f0) / max(1, (f1 - f0)))
            _, v0, a0, w0, p0 = KEYS[i]
            _, v1, a1, w1, p1 = KEYS[i + 1]
            phase = p1 if (f - f0) / max(1, (f1 - f0)) > 0.45 else p0
            return lerp(v0, v1, t), lerp(a0, a1, t), lerp(w0, w1, t), phase
    _, v, a, w, p = KEYS[-1]
    return v, a, w, p


# ---- icon helpers ---------------------------------------------------------
def warning_icon(c, cx, cy, r, color, ink=(150, 40, 34)):
    c.polygon([(cx, cy - r), (cx - r, cy + r * 0.85), (cx + r, cy + r * 0.85)], fill=color)
    c.line([(cx, cy - r * 0.3), (cx, cy + r * 0.25)], fill=ink, width=max(1, r * 0.22))
    c.ellipse([cx - r * 0.12, cy + r * 0.45, cx + r * 0.12, cy + r * 0.68], fill=ink)


def phone_icon(c, cx, cy, r, color):
    c.arc([cx - r, cy - r, cx + r, cy + r], 130, 320, fill=color, width=max(1, r * 0.42))
    c.ellipse([cx - r * 1.05, cy - r * 1.05, cx - r * 0.35, cy - r * 0.35], fill=color)
    c.ellipse([cx + r * 0.35, cy + r * 0.35, cx + r * 1.05, cy + r * 1.05], fill=color)


# ---- avatar ---------------------------------------------------------------
def draw_avatar(c, cx, cy, worry, accent, blink):
    worry = clamp(worry, 0.0, 1.0)
    hw, hh = 78, 94

    # shoulders / shirt
    c.chord([cx - 138, cy + 104, cx + 138, cy + 330], 180, 360, fill=SHIRT)
    c.chord([cx - 138, cy + 104, cx + 16, cy + 330], 196, 318, fill=SHIRT_SH)
    # neck + collar
    c.rrect([cx - 20, cy + 62, cx + 20, cy + 124], 12, fill=SKIN_SH)
    c.rrect([cx - 18, cy + 58, cx + 18, cy + 112], 12, fill=SKIN)
    c.polygon([(cx - 24, cy + 116), (cx, cy + 146), (cx + 24, cy + 116),
               (cx + 13, cy + 104), (cx - 13, cy + 104)], fill=(236, 240, 245))

    # ears
    for sx in (-1, 1):
        c.ellipse([cx + sx * hw - 9, cy + 2, cx + sx * hw + 9, cy + 28], fill=SKIN)
        c.ellipse([cx + sx * hw - 3, cy + 8, cx + sx * hw + 5, cy + 22], fill=SKIN_SH)

    # head (single clean ellipse)
    c.ellipse([cx - hw, cy - hh, cx + hw, cy + hh - 4], fill=SKIN)

    # hair: clean rounded crown, no floating strands
    c.chord([cx - hw - 2, cy - hh - 12, cx + hw + 2, cy + 4], 180, 360, fill=HAIR)

    ex, eye_y = 30, cy - 2

    # eyebrows (restrained: slight raise + inner-up when worried)
    brow_y = eye_y - 22 - worry * 3
    tilt = worry * 7
    for side, sx in ((-1, -ex), (1, ex)):
        inner = cx + sx - side * 12
        outer = cx + sx + side * 12
        c.line([(inner, brow_y - tilt), (outer, brow_y + tilt * 0.3)], fill=HAIR, width=5)

    # eyes
    eye_open = 8 + worry * 3
    for sx in (-ex, ex):
        if blink:
            c.arc([cx + sx - 12, eye_y - 6, cx + sx + 12, eye_y + 8], 20, 160,
                  fill=(120, 95, 85), width=3)
            continue
        c.ellipse([cx + sx - 12, eye_y - eye_open, cx + sx + 12, eye_y + eye_open], fill=(250, 250, 252))
        c.ellipse([cx + sx - 6, eye_y - 6, cx + sx + 6, eye_y + 6], fill=(76, 60, 54))
        c.ellipse([cx + sx - 3, eye_y - 3, cx + sx + 3, eye_y + 3], fill=(26, 20, 18))
        c.ellipse([cx + sx + 1, eye_y - 5, cx + sx + 4, eye_y - 2], fill=(244, 244, 248))

    # nose
    c.line([(cx - 1, eye_y + 8), (cx - 6, cy + 26)], fill=SKIN_SH, width=3)
    c.arc([cx - 8, cy + 18, cx + 6, cy + 32], 25, 155, fill=SKIN_SH, width=2)

    # mouth (restrained: gentle smile -> modest open)
    my = cy + 50
    if worry < 0.45:
        t = worry / 0.45
        c.arc([cx - 22, my - 14, cx + 22, my + 10], lerp(18, 6, t), lerp(162, 174, t),
              fill=(178, 100, 96), width=5)
    else:
        t = (worry - 0.45) / 0.55
        mo = lerp(4, 13, t)
        c.ellipse([cx - 12, my - mo, cx + 12, my + mo], fill=(126, 68, 68), outline=(172, 104, 100), width=2)
        if mo > 8:
            c.chord([cx - 12, my - mo, cx + 12, my + mo * 0.35], 0, 180, fill=(206, 132, 128))

    # calm blush / anxious sweat
    if worry < 0.35:
        for sx in (-1, 1):
            c.ellipse([cx + sx * 42 - 10, cy + 20, cx + sx * 42 + 10, cy + 33], fill=(238, 172, 158, 80))
    elif worry > 0.72:
        t = (worry - 0.72) / 0.28
        sx = cx + hw - 16
        sy = cy - 30 + t * 42
        c.ellipse([sx - 6, sy, sx + 6, sy + 15], fill=(120, 205, 255))
        c.polygon([(sx, sy - 8), (sx - 6, sy + 4), (sx + 6, sy + 4)], fill=(120, 205, 255))
        c.ellipse([sx - 3, sy + 3, sx - 1, sy + 7], fill=(205, 236, 255))


# ---- panels ---------------------------------------------------------------
def panel(c, box, title=None):
    c.rrect(box, 12, fill=PANEL, outline=STROKE, width=1.4)
    if title:
        c.text((box[0] + 16, box[1] + 12), title, 13, SUB, bold=True)


def draw_chart(c, box, valence, arousal, accent, trail):
    panel(c, box, "VALENCE – AROUSAL")
    x0, y0, x1, y1 = box
    m = 22
    px0, py0, px1, py1 = x0 + m, y0 + 40, x1 - m, y1 - 30
    cxx, cyy = (px0 + px1) / 2, (py0 + py1) / 2
    # quadrant tints
    c.rect([px0, py0, cxx, cyy], fill=(60, 30, 30, 70))     # high arousal, low valence -> 불안
    c.rect([cxx, cyy, px1, py1], fill=(30, 60, 40, 70))     # low arousal, high valence -> 안정
    # grid
    for gx in range(1, 4):
        xx = lerp(px0, px1, gx / 4)
        c.line([(xx, py0), (xx, py1)], fill=(255, 255, 255, 16), width=1)
    for gy in range(1, 4):
        yy = lerp(py0, py1, gy / 4)
        c.line([(px0, yy), (px1, yy)], fill=(255, 255, 255, 16), width=1)
    c.rect([px0, py0, px1, py1], outline=STROKE2, width=1.2)
    # axes
    c.line([(px0, cyy), (px1, cyy)], fill=STROKE2, width=1.4)
    c.line([(cxx, py0), (cxx, py1)], fill=STROKE2, width=1.4)
    # labels
    c.text((px0 + 6, py0 + 4), "불안", 12, (214, 130, 122), bold=True)
    c.text((px1 - 6, py1 - 18), "안정", 12, (128, 200, 150), bold=True, anchor="la")
    c.text(((px0 + px1) / 2, py1 + 6), "Valence →", 11, SUB, anchor="ma")
    c.text((px0 - 6, (py0 + py1) / 2), "Arousal ↑", 11, SUB, anchor="mm")

    def to_px(v, a):
        return (cxx + v * (px1 - px0) / 2 * 0.9, cyy - a * (py1 - py0) / 2 * 0.9)

    # trail
    n = len(trail)
    for i, (tv, ta) in enumerate(trail):
        tx, ty = to_px(tv, ta)
        rr = 2 + 4 * (i / max(1, n))
        al = int(30 + 90 * (i / max(1, n)))
        c.ellipse([tx - rr, ty - rr, tx + rr, ty + rr], fill=(accent[0], accent[1], accent[2], al))
    # current dot with glow
    dx, dy = to_px(valence, arousal)
    for gr, ga in ((13, 40), (9, 80)):
        c.ellipse([dx - gr, dy - gr, dx + gr, dy + gr], fill=(accent[0], accent[1], accent[2], ga))
    c.ellipse([dx - 6, dy - 6, dx + 6, dy + 6], fill=accent, outline=(250, 250, 250), width=1.6)


def draw_camera(c, box, cx, cy, worry, accent, phase, f, blink):
    x0, y0, x1, y1 = box
    c.rrect(box, 12, fill=CAMBG, outline=STROKE, width=1.4)
    # faint feed grid
    for gx in range(1, 6):
        xx = lerp(x0, x1, gx / 6)
        c.line([(xx, y0 + 6), (xx, y1 - 6)], fill=(255, 255, 255, 8), width=1)
    for gy in range(1, 7):
        yy = lerp(y0, y1, gy / 7)
        c.line([(x0 + 6, yy), (x1 - 6, yy)], fill=(255, 255, 255, 8), width=1)

    draw_avatar(c, cx, cy, worry, accent, blink)

    # scan line sweep (subtle)
    sweep = (math.sin(f * 0.18) * 0.5 + 0.5)
    sy = lerp(y0 + 20, y1 - 20, sweep)
    c.line([(x0 + 8, sy), (x1 - 8, sy)], fill=(accent[0], accent[1], accent[2], 45), width=2)

    # detection bounding box with corner brackets + glow
    bw, bh = 112, 144
    bx0, by0, bx1, by1 = cx - bw, cy - bh + 8, cx + bw, cy + bh - 6
    glow = 40 + (30 if phase >= 3 else 0)
    c.rrect([bx0 - 3, by0 - 3, bx1 + 3, by1 + 3], 8, outline=(accent[0], accent[1], accent[2], glow), width=3)
    L = 26
    for (ox, oy, dx, dy) in [(bx0, by0, 1, 1), (bx1, by0, -1, 1), (bx0, by1, 1, -1), (bx1, by1, -1, -1)]:
        c.line([(ox, oy), (ox + dx * L, oy)], fill=accent, width=3)
        c.line([(ox, oy), (ox, oy + dy * L)], fill=accent, width=3)
    # label chip on box (bottom-left corner, clear of the CAM/REC row)
    conf = 0.9 + 0.09 * (math.sin(f * 0.5) * 0.5 + 0.5)
    chip = f"FACE  {conf*100:.0f}%"
    cw = c.textlen(chip, 12) + 20
    c.rrect([bx0, by1 + 4, bx0 + cw, by1 + 24], 6, fill=accent)
    c.text((bx0 + 10, by1 + 6), chip, 12, (12, 16, 22), bold=True)

    # CAM chip + REC
    c.text((x0 + 14, y0 + 12), "CAM 01 · ATM", 12, SUB, bold=True)
    rec_on = (f // 6) % 2 == 0
    c.ellipse([x1 - 66, y0 + 14, x1 - 56, y0 + 24], fill=RED if rec_on else (90, 50, 50))
    c.text((x1 - 50, y0 + 12), "REC", 12, SUB, bold=True)

    # phishing call toast (phase 1/2)
    if phase in (1, 2):
        tw = 176
        tx, ty = x1 - tw - 14, y1 - 52
        c.rrect([tx, ty, tx + tw, ty + 38], 10, fill=(38, 36, 22, 235), outline=AMBER, width=1.4)
        phone_icon(c, tx + 20, ty + 19, 9, AMBER)
        c.text((tx + 38, ty + 6), "수신 전화", 12, AMBER, bold=True)
        c.text((tx + 38, ty + 21), "발신번호 표시제한", 10, (210, 195, 130), bold=False)


def draw_status(c, box, anxiety, score, valence, arousal, accent, phase):
    panel(c, box, "분석 결과 · ANALYSIS")
    x0, y0, x1, y1 = box
    cw = x1 - x0

    # anxiety chip
    chip_col = accent
    c.rrect([x0 + 16, y0 + 36, x1 - 16, y0 + 74], 10, fill=(chip_col[0], chip_col[1], chip_col[2], 40),
            outline=chip_col, width=1.6)
    c.ellipse([x0 + 28, y0 + 49, x0 + 40, y0 + 61], fill=chip_col)
    label = "불안 감지  ·  ANXIETY: YES" if anxiety else "안정  ·  ANXIETY: NO"
    c.text((x0 + 50, y0 + 47), label, 13, TEXT, bold=True)

    # score gauge (circular)
    gcx, gcy, gr = (x0 + x1) / 2, y0 + 150, 52
    c.arc([gcx - gr, gcy - gr, gcx + gr, gcy + gr], 130, 410, fill=STROKE2, width=9)
    frac = clamp(score / 100, 0, 1)
    c.arc([gcx - gr, gcy - gr, gcx + gr, gcy + gr], 130, 130 + 280 * frac, fill=accent, width=9)
    c.text((gcx, gcy - 12), f"{score:.0f}", 34, TEXT, anchor="mm")
    c.text((gcx, gcy + 20), "ANXIETY SCORE", 11, SUB, anchor="mm")

    # valence / arousal bars
    def bar(yy, name, val, lo=-1, hi=1):
        c.text((x0 + 16, yy), name, 12, SUB, bold=True)
        bx0, bx1 = x0 + 16, x1 - 16
        by = yy + 18
        c.rrect([bx0, by, bx1, by + 12], 6, fill=PANEL2)
        # center marker
        cxm = lerp(bx0, bx1, 0.5)
        c.line([(cxm, by - 2), (cxm, by + 14)], fill=STROKE2, width=1)
        frac2 = clamp((val - lo) / (hi - lo), 0, 1)
        fx = lerp(bx0, bx1, frac2)
        if fx >= cxm:
            c.rrect([cxm, by, fx, by + 12], 6, fill=accent)
        else:
            c.rrect([fx, by, cxm, by + 12], 6, fill=accent)
        c.text((bx1, yy), f"{val:+.2f}", 12, TEXT, bold=True, anchor="ra")

    bar(y0 + 224, "Valence", valence)
    bar(y0 + 268, "Arousal", arousal)

    # phase steps
    steps = ["감지 대기", "전화 수신", "표정 분석", "불안 감지", "상담 연결"]
    active = phase
    sy = y0 + 322
    c.text((x0 + 16, sy), "PROCESS", 11, SUB, bold=True)
    for i, name in enumerate(steps):
        yy = sy + 22 + i * 20
        done = i <= active
        col = accent if i == active else (GREEN if done else STROKE2)
        c.ellipse([x0 + 18, yy + 1, x0 + 30, yy + 13], fill=col if done else PANEL2,
                  outline=col, width=1.4)
        if done:
            c.text((x0 + 24, yy + 6), "", 8, TEXT, anchor="mm")
        c.text((x0 + 40, yy + 1), name, 12, TEXT if i == active else (SUB if done else STROKE2), bold=(i == active))


# ---- frame ----------------------------------------------------------------
_trail = []


def render_frame(f, for_trail=True):
    v, a, worry, phase = state_at(f)
    anxiety = 1 if worry >= 0.5 else 0
    accent = RED if anxiety else (AMBER if phase in (1, 2) else GREEN)
    score = clamp(10 + worry * 84 + math.sin(f * 0.6) * 1.2, 0, 100)
    blink = (f % 30) in (15, 16)

    if for_trail:
        _trail.append((v, a))
        if len(_trail) > 14:
            _trail.pop(0)

    c = Canvas()
    # header
    c.rect([0, 0, W, 46], fill=(10, 12, 18))
    c.line([(0, 46), (W, 46)], fill=STROKE, width=1)
    c.text((22, 13), "보이스피싱 방지 · ATM 감정 인식 모니터", 17, TEXT, bold=True)
    live = (f // 8) % 2 == 0
    c.ellipse([W - 118, 18, W - 108, 28], fill=GREEN if live else (40, 90, 60))
    c.text((W - 100, 13), "LIVE", 14, SUB, bold=True)
    c.text((W - 56, 13), "AI", 14, GREEN, bold=True)

    # panels
    chart_box = (22, 60, 300, 500)
    cam_box = (312, 60, 648, 500)
    stat_box = (660, 60, 938, 500)
    draw_chart(c, chart_box, v, a, accent, list(_trail))
    ccx = (cam_box[0] + cam_box[2]) / 2
    ccy = cam_box[1] + 168 + math.sin(f * 0.28) * 2.2
    draw_camera(c, cam_box, ccx, ccy, worry, accent, phase, f, blink)
    draw_status(c, stat_box, anxiety, score, v, a, accent, phase)

    # bottom caption
    c.rect([0, 506, W, H], fill=(10, 12, 18))
    c.line([(0, 506), (W, 506)], fill=STROKE, width=1)
    cap_map = {
        0: ("정상 거래 진행 중", (130, 200, 150)),
        1: ("발신번호 표시제한 전화 수신 — 통화 시작", AMBER),
        2: ("표정에서 arousal 상승 · valence 하강 감지 중", AMBER),
        3: ("불안 상태 감지 — 보이스피싱 의심, 상담원 호출", RED),
        4: ("상담원 연결 중 — 거래를 잠시 멈추고 안내드립니다", RED),
    }
    text, col = cap_map[phase]
    c.text((W / 2, 523), text, 15, col, bold=True, anchor="mm")

    # alert banner overlay on detection
    if phase in (3, 4):
        pulse = 150 + int(70 * (math.sin(f * 0.7) * 0.5 + 0.5))
        c.rrect([cam_box[0] + 8, cam_box[1] + 8, cam_box[2] - 8, cam_box[1] + 46], 8,
                fill=(180, 40, 34, pulse))
        if phase == 3:
            warning_icon(c, cam_box[0] + 32, cam_box[1] + 27, 12, (255, 255, 255), ink=(180, 40, 34))
            msg = "보이스피싱 의심 — 상담원 호출"
        else:
            phone_icon(c, cam_box[0] + 32, cam_box[1] + 27, 10, (255, 255, 255))
            msg = "상담원 연결 중…"
        c.text((cam_box[0] + 52, cam_box[1] + 16), msg, 15, (255, 255, 255), bold=True)

    return c.result()


def main():
    # warm the trail so frame 0 already has a little history (loop-friendly)
    frames = [render_frame(f) for f in range(N_FRAMES)]
    durations = [70] * N_FRAMES
    durations[20] = 500   # linger on calm baseline
    durations[84] = 300
    durations[-1] = 900   # hold the connecting state before loop

    frames[0].save(POSTER_PATH)
    render_frame(66, for_trail=False).save(OUT_DIR / "demo_scenario_detected.png")

    # Build one shared adaptive palette from frames sampled across the whole
    # timeline (calm-green + alert-red states), so every frame quantizes to the
    # same 200 colors -> far smaller GIF with no per-frame palette flicker.
    sample_idx = [0, 24, 40, 52, 66, 84, 100]
    strip = Image.new("RGB", (W, H * len(sample_idx)))
    for i, idx in enumerate(sample_idx):
        strip.paste(frames[idx], (0, H * i))
    pal = strip.quantize(colors=200, method=Image.FASTOCTREE, dither=Image.NONE)
    q = [fr.quantize(palette=pal, dither=Image.NONE) for fr in frames]

    q[0].save(
        GIF_PATH,
        save_all=True,
        append_images=q[1:],
        duration=durations,
        loop=0,
        optimize=True,
        disposal=1,
    )
    kb = GIF_PATH.stat().st_size / 1024
    print(f"wrote {GIF_PATH} ({N_FRAMES} frames, {kb:.0f} KB)")
    print(f"wrote {POSTER_PATH}")


if __name__ == "__main__":
    main()
