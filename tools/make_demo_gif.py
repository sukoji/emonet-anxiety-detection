"""
Generate a synthetic demo animation for the README.

The real webcam demo (src/cap.py) shows actual team members' faces, so this
script re-creates the *same on-screen UI* that cap.py renders -- face bounding
box, Anxiety label + Score, valence/arousal bars, and the 2D valence-arousal
chart -- but drives it with an illustrated avatar instead of a real person.

Scenario: normal ATM use -> voice-phishing call received -> anxiety detected
-> counselor alert. Output: docs/demo/demo_scenario.gif (+ a poster .png).

Requires only Pillow + numpy (already used by the project).
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

# colors (RGB)
GREEN = (60, 220, 90)
RED = (235, 60, 60)
WHITE = (240, 240, 240)
YELLOW = (255, 210, 70)
BG_TOP = (26, 30, 40)
BG_BOT = (12, 14, 20)
SKIN = (233, 196, 162)
HAIR = (60, 48, 44)

FONT_DIR = Path("C:/Windows/Fonts")


def font(name, size):
    return ImageFont.truetype(str(FONT_DIR / name), size)


F_KR = lambda s: font("malgun.ttf", s)
F_KRB = lambda s: font("malgunbd.ttf", s)
F_EN = lambda s: font("arialbd.ttf", s)


def lerp(a, b, t):
    return a + (b - a) * t


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


# ---- scenario keyframes: (frame, valence, arousal, worry, phase) -----------
# phase: 0 normal, 1 call incoming, 2 analyzing, 3 anxiety alert, 4 connecting
KEYS = [
    (0, 0.62, -0.22, 0.08, 0),
    (14, 0.60, -0.20, 0.10, 0),
    (18, 0.55, -0.10, 0.22, 1),
    (26, 0.10, 0.25, 0.55, 2),
    (34, -0.55, 0.62, 0.90, 2),
    (40, -0.62, 0.66, 0.92, 3),
    (52, -0.60, 0.64, 0.90, 3),
    (58, -0.58, 0.60, 0.88, 4),
    (66, -0.58, 0.60, 0.88, 4),
]
N_FRAMES = KEYS[-1][0] + 1


def state_at(f):
    for i in range(len(KEYS) - 1):
        f0, *_ = KEYS[i]
        f1 = KEYS[i + 1][0]
        if f0 <= f <= f1:
            t = (f - f0) / max(1, (f1 - f0))
            _, v0, a0, w0, p0 = KEYS[i]
            _, v1, a1, w1, p1 = KEYS[i + 1]
            v = lerp(v0, v1, t)
            a = lerp(a0, a1, t)
            w = lerp(w0, w1, t)
            phase = p1 if t > 0.5 else p0
            return v, a, w, phase
    _, v, a, w, p = KEYS[-1]
    return v, a, w, p


def gradient_bg():
    img = Image.new("RGB", (W, H), BG_BOT)
    px = img.load()
    for y in range(H):
        t = y / H
        c = tuple(int(lerp(BG_TOP[i], BG_BOT[i], t)) for i in range(3))
        for x in range(W):
            px[x, y] = c
    return img


def draw_avatar(d, cx, cy, worry, color):
    """Simple illustrated face (clearly not a real person)."""
    # head
    hw, hh = 78, 96
    d.ellipse([cx - hw, cy - hh, cx + hw, cy + hh], fill=SKIN, outline=(150, 120, 95), width=2)
    # hair cap
    d.pieslice([cx - hw, cy - hh - 6, cx + hw, cy + 30], 180, 360, fill=HAIR)
    d.rectangle([cx - hw, cy - hh - 6, cx + hw, cy - 40], fill=HAIR)
    # ears
    d.ellipse([cx - hw - 12, cy - 6, cx - hw + 8, cy + 26], fill=SKIN)
    d.ellipse([cx + hw - 8, cy - 6, cx + hw + 12, cy + 26], fill=SKIN)

    eye_y = cy - 6
    ex = 30
    eye_open = 6 + int(worry * 5)  # eyes widen with worry
    for sx in (-ex, ex):
        d.ellipse([cx + sx - 13, eye_y - eye_open, cx + sx + 13, eye_y + eye_open], fill=WHITE)
        d.ellipse([cx + sx - 6, eye_y - 6, cx + sx + 6, eye_y + 6], fill=(40, 40, 55))
    # eyebrows: flat when calm, angled/raised (inner up) when worried
    brow_y = eye_y - 20 - int(worry * 4)
    tilt = int(worry * 12)
    for side, sx in ((-1, -ex), (1, ex)):
        inner_x = cx + sx + side * -14
        outer_x = cx + sx + side * 14
        d.line([(inner_x, brow_y - tilt), (outer_x, brow_y + tilt // 2)], fill=HAIR, width=5)
    # nose
    d.line([(cx, eye_y + 6), (cx - 5, cy + 26)], fill=(180, 140, 110), width=3)
    d.line([(cx - 5, cy + 26), (cx + 6, cy + 26)], fill=(180, 140, 110), width=3)
    # mouth: smile -> open worried oval
    my = cy + 52
    if worry < 0.4:
        # gentle smile
        d.arc([cx - 26, my - 18, cx + 26, my + 14], 20, 160, fill=(150, 70, 70), width=5)
    else:
        mo = int(lerp(4, 20, (worry - 0.4) / 0.6))
        d.ellipse([cx - 16, my - mo, cx + 16, my + mo], fill=(120, 55, 60), outline=(150, 70, 70), width=3)
    # sweat drop when anxious
    if worry > 0.6:
        sx = cx + hw - 10
        sy = cy - 30 + int((worry - 0.6) * 40)
        d.ellipse([sx - 7, sy, sx + 7, sy + 16], fill=(120, 200, 255))
        d.polygon([(sx, sy - 8), (sx - 7, sy + 4), (sx + 7, sy + 4)], fill=(120, 200, 255))


def draw_warning_icon(d, cx, cy, color):
    """Filled warning triangle with an exclamation mark."""
    r = 14
    d.polygon([(cx, cy - r), (cx - r, cy + r), (cx + r, cy + r)], fill=color)
    d.line([(cx, cy - 4), (cx, cy + 5)], fill=(180, 40, 40), width=3)
    d.ellipse([cx - 2, cy + 8, cx + 2, cy + 12], fill=(180, 40, 40))


def draw_phone_icon(d, cx, cy, color):
    """Small telephone handset."""
    d.arc([cx - 12, cy - 12, cx + 12, cy + 12], 135, 315, fill=color, width=5)
    d.ellipse([cx - 13, cy - 13, cx - 5, cy - 5], fill=color)
    d.ellipse([cx + 5, cy + 5, cx + 13, cy + 13], fill=color)


def draw_va_chart(d, valence, arousal, color):
    """Bottom-left 2D valence/arousal chart, mirrors cap.py (250x220)."""
    gw, gh = 250, 220
    gx, gy = 12, H - gh - 12
    d.rectangle([gx, gy, gx + gw, gy + gh], fill=(20, 24, 34), outline=(80, 90, 110), width=2)
    cxx, cyy = gx + gw // 2, gy + gh // 2
    # axes
    d.line([(gx + 12, cyy), (gx + gw - 12, cyy)], fill=(90, 100, 120), width=1)
    d.line([(cxx, gy + 12), (cxx, gy + gh - 12)], fill=(90, 100, 120), width=1)
    # quadrant tints
    d.text((gx + 8, gy + 6), "Arousal +", font=F_EN(12), fill=(140, 150, 170))
    d.text((gx + gw - 78, gy + gh - 20), "Valence +", font=F_EN(12), fill=(140, 150, 170))
    d.text((gx + 8, gy + gh // 2 - 34), "불안", font=F_KR(13), fill=(210, 120, 120))
    d.text((gx + gw - 52, gy + gh // 2 + 16), "안정", font=F_KR(13), fill=(120, 190, 140))
    # dot
    px = int(cxx + valence * (gw / 2 - 16))
    py = int(cyy - arousal * (gh / 2 - 16))
    d.ellipse([px - 7, py - 7, px + 7, py + 7], fill=color, outline=WHITE, width=2)
    d.text((gx + 8, gy - 22), "Valence–Arousal", font=F_KRB(14), fill=WHITE)


def render_frame(f):
    v, a, worry, phase = state_at(f)
    breathe = math.sin(f * 0.5) * 1.5
    anxiety = 1 if worry >= 0.5 else 0
    color = RED if anxiety else GREEN
    # score: calm ~15, anxious ~90
    score = clamp(12 + worry * 82 + math.sin(f * 0.7) * 1.5, 0, 100)

    img = gradient_bg()
    d = ImageDraw.Draw(img, "RGBA")

    # header bar
    d.rectangle([0, 0, W, 44], fill=(0, 0, 0, 140))
    d.text((16, 12), "Face Detection and Emotion Recognition", font=F_EN(18), fill=WHITE)
    d.text((W - 150, 12), "ATM · CAM 01", font=F_EN(16), fill=(150, 160, 180))

    # face box geometry (avatar centered a bit right of chart)
    fx, fy, fw, fh = 560, 150, 200, 250
    cx, cy = fx + fw // 2, int(fy + fh // 2 + breathe)
    draw_avatar(d, cx, cy, worry, color)

    # bounding box
    d.rectangle([fx, fy, fx + fw, fy + fh], outline=color, width=3)
    # label above box
    d.text((fx, fy - 30), f"Anxiety: {'Yes' if anxiety else 'No'}, Score: {score:.2f}",
           font=F_EN(20), fill=color)

    # valence vertical bar (left of box)
    vn = (v + 1) / 2
    bar_h = int(vn * fh)
    d.rectangle([fx - 26, fy + fh - bar_h, fx - 10, fy + fh], fill=color)
    # arousal horizontal bar (below box)
    an = (a + 1) / 2
    d.rectangle([fx, fy + fh + 12, fx + int(fw * an), fy + fh + 26], fill=color)
    d.text((fx, fy + fh + 34), f"Valence: {v:+.2f}", font=F_EN(16), fill=WHITE)
    d.text((fx, fy + fh + 56), f"Arousal: {a:+.2f}", font=F_EN(16), fill=WHITE)

    # 2D chart
    draw_va_chart(d, v, a, color)

    # scenario caption / phone bubble / banner
    cap_map = {
        0: ("정상 거래 진행 중", (120, 190, 140)),
        1: ("발신번호 표시제한 전화 수신", YELLOW),
        2: ("표정 분석 중  ·  arousal ↑  valence ↓", YELLOW),
        3: ("불안 상태 감지 — 보이스피싱 의심", RED),
        4: ("상담원 연결 중 — 잠시만 기다려 주세요", RED),
    }
    text, ccol = cap_map[phase]
    # caption strip bottom
    d.rectangle([0, H - 40, W, H], fill=(0, 0, 0, 160))
    d.text((300, H - 32), text, font=F_KRB(20), fill=ccol)

    # phone bubble in phase 1 (with a drawn handset icon)
    if phase == 1:
        bx, by = 690, 90
        d.rounded_rectangle([bx, by, bx + 224, by + 46], radius=12, fill=(40, 40, 55, 230),
                            outline=YELLOW, width=2)
        draw_phone_icon(d, bx + 20, by + 23, YELLOW)
        d.text((bx + 40, by + 13), "수신 전화 · 사기 의심", font=F_KR(15), fill=YELLOW)

    # alert banner top in phase 3/4
    if phase in (3, 4):
        pulse = 120 + int(80 * (0.5 + 0.5 * math.sin(f * 0.9)))
        d.rectangle([0, 48, W, 96], fill=(180, 40, 40, pulse))
        if phase == 3:
            draw_warning_icon(d, 34, 72, WHITE)
            msg = "보이스피싱 의심 — 직원/상담원에게 알림 전송"
        else:
            draw_phone_icon(d, 34, 72, WHITE)
            msg = "상담원 연결 중 — 거래를 잠시 멈춰 주세요"
        d.text((58, 60), msg, font=F_KRB(22), fill=WHITE)

    return img


def main():
    frames = [render_frame(f) for f in range(N_FRAMES)]
    # hold last frame a bit longer, then loop
    durations = [90] * N_FRAMES
    for i in range(len(KEYS[-1:])):
        pass
    durations[-1] = 900
    durations[14] = 500  # linger on calm
    frames[0].save(POSTER_PATH)
    # also save a "detected" poster
    render_frame(44).save(OUT_DIR / "demo_scenario_detected.png")
    frames[0].save(
        GIF_PATH,
        save_all=True,
        append_images=frames[1:],
        duration=durations,
        loop=0,
        optimize=True,
        disposal=2,
    )
    print(f"wrote {GIF_PATH} ({N_FRAMES} frames)")
    print(f"wrote {POSTER_PATH}")


if __name__ == "__main__":
    main()
