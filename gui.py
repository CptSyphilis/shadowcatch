import math
import queue
import random
import re
import subprocess
import threading
from typing import Any
import tkinter as tk
from tkinter import font as tkfont

import customtkinter as ctk
import numpy as np
import matplotlib
from matplotlib.backends._backend_tk import NavigationToolbar2Tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from PIL import Image, ImageDraw, ImageFilter, ImageTk

import core

VOID = "#0E0B1F"
PANEL = "#18142E"
RAISED = "#241D42"
LINE = "#33295A"
FG = "#ECE8F7"
MUTED = "#9A92B8"
VIOLET = "#8B6CF0"
VIOLET_HOVER = "#A28BF6"
GOLD = "#FFC46B"
CYAN = "#6CD4E0"
SLATE = "#6E6790"
NEBULA = "#4A2A7A"
ROSE = "#7A2D6B"
EXAMPLES = ["Pi Mensae", "TOI-700", "LHS 3844", "WASP-18", "Proxima Centauri", "TIC 261136679"]
HEADINGS = {"How this works", "The three plots", "The words", "The options", "Status colors"}
KEY = re.compile(r"^(ORBIT|DIP|SIZE|LENGTH|COUNT|SDE|SNR|EXOFOP|CATALOGS|NEIGHBORS|GAIA|RECHECK|PIXELS|VERDICT|STAR)\b")
HELP = """How this works

When a planet passes in front of its star, it blocks a tiny bit of light.
The star looks slightly dimmer for a few hours. If the dip repeats on a
regular schedule, something is orbiting the star.

NASA's TESS telescope measures the brightness of stars every 2 minutes.
Shadowcatch downloads that data and looks for repeating dips.


The three plots

1. Raw light curve (top)
   The star's brightness over time. Each dot is one measurement; the
   color just shows time passing (cyan = early, pink = late).
   Gaps are when TESS was pointed elsewhere or sending data home.
   Upward spikes are flares - explosions on the star's surface.

2. BLS periodogram (middle)
   The search tries thousands of possible orbit lengths and scores
   each one. A tall, narrow spike = a repeating dip at that orbit.
   The gold line marks the best one. A forest of equal spikes = noise.

3. Folded light curve (bottom)
   All the data cut into pieces one orbit long and stacked on top of
   each other, lined up on the dip. A real planet shows a clean dip in
   the middle. Dot colors show WHEN each dip happened - if the colors
   all overlap, every dip looks the same, which is what you want.
   The gold line is the average.


The words

Transit      A planet crossing in front of its star.
Period       How long one orbit takes (the planet's year).
Depth (ppm)  How much dimmer the star gets, in parts per million.
             100 ppm = 0.01%. Earth crossing the Sun: ~84 ppm.
SDE          Signal strength: how far the best score stands above the
             rest. Under 7 is ignored here; over ~12 is strong.
TIC          TESS Input Catalog - every star TESS watches has a number.
TOI          TESS Object of Interest - a signal NASA already flagged.
CTOI         Community TOI - a candidate someone outside NASA posted on
             ExoFOP, the TESS follow-up website. New candidates are
             checked against all of them, plus the star's ExoFOP page
             (notes, close companions, extra light from other stars,
             telescope follow-up already done).
Sector       TESS watches one patch of sky for ~27 days. One sector =
             one of those watches. More sectors = more dips to catch.
SNR          Signal-to-noise: how many times bigger the dip is than the
             star's natural flickering. A planet needs about 7 or more.
Blend        A fainter star next to yours that is really two stars
             eclipsing each other. TESS pixels are 21 arcseconds wide, so
             its light mixes in and its eclipses look like a shallow dip on
             your star. For new candidates, Shadowcatch checks the Gaia star
             catalog for neighbors bright enough to do this.
Pixels       New candidates get two extra pictures at the bottom: the
             star as TESS sees it, and a "difference image" of where light
             went missing during the dips. The gold + should sit inside the
             cyan circle (this star). Rings mark other nearby stars.
Flare        A sudden burst of light on the star. Removed before
             searching because it can fake or hide dips.


The options

Sectors   How many 27-day chunks to download. More = finds longer
          orbits, but slower. 3-6 is a good range.
Signals   How many separate signals to look for in one star.
Min SDE   How strong a signal must be to be reported.


Status colors

Cyan    Known - NASA already has it, or someone posted it on ExoFOP as a
        community candidate ("known CTOI"). Good test that the search works.
Gold    New candidate - not in NASA's list. Check the plots carefully.
Grey    Not a planet:
        likely noise  - fewer than 3 dips with real data in them (dips
                        that fall in data gaps don't count), or a dip
                        too small to stand out from the star's own
                        flickering (SNR under 7)
        likely binary - failed a check that catches two stars eclipsing
                        each other (odd/even depths, V shape, a deep second
                        dip halfway round the orbit, dips far too long, or
                        an object wider than ~25 Earths - bigger than any
                        planet)
        repeat        - same rhythm as an earlier signal in this star
        likely variable star - the folded data is a smooth wave, not a
                        sharp dip: the star's own spots or pulsations.
                        "unsteady" means the dip is only there in part
                        of the data - a planet never switches off
        likely blend  - TESS's pictures show the dip coming from a spot
                        next to the star, not the star itself
        not confirmed - TESS watched the star in other sectors too and
                        the dip isn't there - a planet never switches off -
                        or the dip is no deeper than what random noise
                        makes in that data ("chance-sized dip")
        catalogued binary - the VSX or SIMBAD star catalog already lists
                        an eclipsing binary here with this period
"""


def blend(a, b, t):
    a = [int(a[i : i + 2], 16) for i in (1, 3, 5)]
    b = [int(b[i : i + 2], 16) for i in (1, 3, 5)]
    return "#" + "".join(f"{round(x + (y - x) * t):02x}" for x, y in zip(a, b))


def rgb(color):
    return np.array([int(color[i : i + 2], 16) for i in (1, 3, 5)], np.float32) / 255


def rgba(color, alpha):
    return tuple(int(color[i : i + 2], 16) for i in (1, 3, 5)) + (alpha,)


def covered(d, big, small):
    if d >= big + small:
        return 0.0
    if d <= big - small:
        return (small / big) ** 2
    a = small**2 * math.acos((d * d + small * small - big * big) / (2 * d * small))
    b = big**2 * math.acos((d * d + big * big - small * small) / (2 * d * big))
    c = 0.5 * math.sqrt((-d + small + big) * (d + small - big) * (d - small + big) * (d + small + big))
    return (a + b - c) / (math.pi * big * big)


def status_color(label):
    if label == "NEW CANDIDATE":
        return GOLD
    if label.startswith(("known", "alias")):
        return CYAN
    return SLATE


def status_text(label):
    if label == "NEW CANDIDATE":
        return "New candidate"
    return label[0].upper() + label[1:]


def pick_family(*names):
    available = set(tkfont.families())
    return next((n for n in names if n in available), "TkDefaultFont")


class Transit(tk.Canvas):
    W, H = 440, 80
    STAR_X, STAR_Y, STAR_R, PLANET_R = 50, 40, 26, 8.5
    CURVE_X0, CURVE_X1, CURVE_Y = 108, 428, 30
    SS = 4

    def __init__(self, master):
        self.k = max(1.0, ctk.ScalingTracker.get_widget_scaling(master))
        self.w, self.h = round(self.W * self.k), round(self.H * self.k)
        super().__init__(master, width=self.w, height=self.h, bg=VOID, highlightthickness=0)
        self.shown = 0.0
        self.goal = 0.0
        self.ticking = False
        self.sky = self.render_sky()
        self.photo = ImageTk.PhotoImage(self.sky)
        self.create_image(0, 0, anchor="nw", image=self.photo)
        self.draw_frame()

    def render_sky(self):
        s = self.k * self.SS
        h, w = self.h * self.SS, self.w * self.SS
        y, x = np.mgrid[0:h, 0:w].astype(np.float32) / s
        img = np.empty((h, w, 3), np.float32)
        img[:] = rgb(VOID)

        def glow(cx, cy, rx, ry, color, strength, tilt=0.0):
            dx, dy = x - cx, y - cy - tilt * (x - cx)
            return strength * np.exp(-((dx / rx) ** 2 + (dy / ry) ** 2))[..., None] * rgb(color)

        img += glow(270, 40, 170, 30, NEBULA, 0.75, -0.06)
        img += glow(330, 34, 80, 17, ROSE, 0.55, -0.06)
        img += glow(225, 46, 70, 13, "#3B5BA8", 0.30, -0.06)
        img += glow(300, 37, 26, 9, "#E8D4F5", 0.22, -0.06)
        lane = np.exp(-(((y - 41 + 0.06 * (x - 290)) / 3.2) ** 2)) * np.exp(-(((x - 290) / 110) ** 2))
        img *= (1 - 0.45 * lane)[..., None]

        rng = random.Random(7)
        for _ in range(140):
            cx, cy = rng.uniform(0, self.W), rng.uniform(0, self.H)
            if math.hypot(cx - self.STAR_X, cy - self.STAR_Y) < self.STAR_R + 14:
                continue
            bright = rng.random() ** 3
            sigma = 0.35 + 0.55 * bright
            color = rng.choice(("#FFFFFF", "#D8E2FF", "#FFE9D0", "#E9D8FF"))
            x0, x1 = int((cx - 5 * sigma) * s), int((cx + 5 * sigma) * s) + 1
            y0, y1 = int((cy - 5 * sigma) * s), int((cy + 5 * sigma) * s) + 1
            x0, y0, x1, y1 = max(x0, 0), max(y0, 0), min(x1, w), min(y1, h)
            px, py = x[y0:y1, x0:x1], y[y0:y1, x0:x1]
            d2 = (px - cx) ** 2 + (py - cy) ** 2
            light = (0.25 + 0.9 * bright) * np.exp(-d2 / (2 * sigma**2)) + 0.12 * bright * np.exp(-np.sqrt(d2) / 2.2)
            img[y0:y1, x0:x1] += light[..., None] * rgb(color)

        d = np.hypot(x - self.STAR_X, y - self.STAR_Y) / self.STAR_R
        outside = np.clip(d - 1, 0, None) * self.STAR_R
        img += (0.55 * np.exp(-outside / 4) + 0.28 * np.exp(-outside / 14))[..., None] * rgb(GOLD) * (d > 1)[..., None]
        edge = np.minimum(np.minimum(x / 30, (self.W - x) / 110), np.minimum(y, self.H - y) / 14)
        fade = np.clip(edge, 0, 1)
        fade = fade * fade * (3 - 2 * fade)
        img = rgb(VOID) + (img - rgb(VOID)) * fade[..., None]
        mu = np.sqrt(np.clip(1 - d**2, 0, 1))
        limb = 1 - 0.45 * (1 - mu) - 0.25 * (1 - mu) ** 2
        tint = mu[..., None] ** 1.6
        surface = (rgb("#F07A22") * (1 - tint) + rgb("#FFE3A3") * tint) * limb[..., None]
        disc = (d <= 1)[..., None]
        img = np.where(disc, surface, img)

        out = Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8))
        return out.reduce(self.SS)

    def planet_x(self, f):
        reach = self.STAR_R + self.PLANET_R + 5
        return self.STAR_X - reach + 2 * reach * f

    def draw_frame(self):
        s = self.k * self.SS
        size = (self.w * self.SS, self.h * self.SS)
        sharp = Image.new("RGBA", size, (0, 0, 0, 0))
        haze = Image.new("RGBA", size, (0, 0, 0, 0))
        pen, soft = ImageDraw.Draw(sharp), ImageDraw.Draw(haze)

        dash = self.CURVE_X0
        while dash < self.CURVE_X1:
            pen.line([(dash * s, self.CURVE_Y * s), ((dash + 2) * s, self.CURVE_Y * s)], fill=rgba(MUTED, 70), width=round(s))
            dash += 6

        if self.shown > 0:
            steps = max(2, int(160 * self.shown))
            points = []
            for i in range(steps + 1):
                f = self.shown * i / steps
                dip = covered(abs(self.planet_x(f) - self.STAR_X), self.STAR_R, self.PLANET_R) * 180
                points.append(((self.CURVE_X0 + (self.CURVE_X1 - self.CURVE_X0) * f) * s, (self.CURVE_Y + dip) * s))
            soft.line(points, fill=rgba(GOLD, 150), width=round(5 * s), joint="curve")
            pen.line(points, fill=rgba(blend(GOLD, "#FFFFFF", 0.35), 255), width=round(1.7 * s), joint="curve")
            hx, hy = points[-1]
            soft.ellipse([hx - 7 * s, hy - 7 * s, hx + 7 * s, hy + 7 * s], fill=rgba(GOLD, 170))
            pen.ellipse([hx - 2.6 * s, hy - 2.6 * s, hx + 2.6 * s, hy + 2.6 * s], fill=rgba("#FFF4DC", 255))

        px, py, r = self.planet_x(self.shown) * s, self.STAR_Y * s, self.PLANET_R * s
        pen.ellipse([px - r, py - r, px + r, py + r], fill=rgba("#07050F", 255), outline=rgba("#B9A8E8", 90), width=round(0.9 * s))

        haze = haze.reduce(self.SS).filter(ImageFilter.GaussianBlur(2.2 * self.k))
        frame = self.sky.convert("RGBA")
        frame.alpha_composite(haze)
        frame.alpha_composite(sharp.reduce(self.SS))
        self.photo.paste(frame)

    def set(self, fraction, instant=False):
        self.goal = max(0.0, min(1.0, fraction))
        if instant:
            self.shown = self.goal
            self.draw_frame()
        elif not self.ticking:
            self.ticking = True
            self.tick()

    def tick(self):
        gap = self.goal - self.shown
        if abs(gap) < 0.0015:
            self.shown = self.goal
            self.ticking = False
        else:
            self.shown += gap * 0.12
            self.after(16, self.tick)
        self.draw_frame()


class Toolbar(NavigationToolbar2Tk):
    toolitems = tuple(t for t in NavigationToolbar2Tk.toolitems if t[0] in ("Home", "Pan", "Zoom", "Save"))


class SignalRow(ctk.CTkFrame):
    def __init__(self, master, signal, fonts, on_pick):
        super().__init__(master, fg_color=PANEL, corner_radius=12, border_width=1, border_color=PANEL)
        self.signal = signal
        self.selected = False
        color = status_color(signal["label"])
        self.grid_columnconfigure(1, weight=1)
        dot = ctk.CTkFrame(self, width=10, height=10, corner_radius=5, fg_color=color)
        dot.grid(row=0, column=0, rowspan=2, padx=(14, 12), pady=14)
        period = signal["period"]
        when = f"{period * 24:.1f} hours" if period < 1 else f"{period:.3f} days"
        title = ctk.CTkLabel(self, text=when, font=fonts["number"], text_color=FG, anchor="w")
        title.grid(row=0, column=1, sticky="w", pady=(10, 0))
        detail = (
            f"{signal['depth'] * 1e6:.0f} ppm dip, {signal['duration'] * 24:.1f} h long, "
            f"{signal['transits']} transits, SDE {signal['sde']:.1f}"
        )
        sub = ctk.CTkLabel(self, text=detail, font=fonts["small"], text_color=MUTED, anchor="w")
        sub.grid(row=1, column=1, sticky="w", pady=(0, 10))
        tag = ctk.CTkLabel(self, text=status_text(signal["label"]), font=fonts["body_bold"], text_color=color, anchor="e")
        tag.grid(row=0, column=2, rowspan=2, padx=(8, 14))
        for widget in (self, dot, title, sub, tag):
            widget.bind("<Button-1>", lambda _: on_pick(self))
            widget.bind("<Enter>", lambda _: self.hover(True))
            widget.bind("<Leave>", lambda _: self.hover(False))
        self.configure(cursor="hand2")

    def hover(self, on):
        if not self.selected:
            self.configure(fg_color=RAISED if on else PANEL)

    def select(self, on):
        self.selected = on
        self.configure(fg_color=RAISED if on else PANEL, border_color=VIOLET if on else PANEL)


class App(ctk.CTk):
    def __init__(self):
        ctk.set_appearance_mode("dark")
        matplotlib.rcParams["font.family"] = ["Fira Sans", "Ubuntu Sans", "DejaVu Sans"]
        super().__init__(fg_color=VOID)
        self.title("Shadowcatch")
        self.geometry("1440x920")
        self.minsize(1100, 700)
        self.events = queue.Queue()
        self.signals = []
        self.rows = []
        self.canvas = None
        self.toolbar = None
        self.star_text = ""
        self.running = False
        self.resume = threading.Event()
        self.resume.set()
        self.log_open = False
        sans = pick_family("Fira Sans", "Ubuntu Sans", "Cantarell")
        narrow = pick_family("Fira Sans Condensed", "Ubuntu Sans", sans)
        mono = pick_family("Fira Mono", "Ubuntu Sans Mono", "DejaVu Sans Mono")
        self.fonts = {
            "brand": ctk.CTkFont(narrow, 26, "bold"),
            "number": ctk.CTkFont(narrow, 19, "bold"),
            "body": ctk.CTkFont(sans, 14),
            "body_bold": ctk.CTkFont(sans, 13, "bold"),
            "small": ctk.CTkFont(sans, 12),
            "section": ctk.CTkFont(sans, 13, "bold"),
            "status": ctk.CTkFont(sans, 15),
            "mono": ctk.CTkFont(mono, 12),
        }
        self.build()
        self.after(50, self.poll)

    def build(self):
        f = self.fonts
        header = ctk.CTkFrame(self, fg_color=VOID, corner_radius=0)
        header.pack(fill="x", padx=24, pady=(18, 6))
        ctk.CTkLabel(header, text="Shadowcatch", font=f["brand"], text_color=FG).pack(side="left", padx=(0, 28))

        self.target = tk.StringVar(value="Pi Mensae")
        self.sectors = tk.StringVar(value="3")
        self.max_signals = tk.StringVar(value="3")
        self.min_sde = tk.StringVar(value="7")

        field: dict[str, Any] = dict(fg_color=PANEL, border_color=LINE, border_width=1, corner_radius=10, height=36, text_color=FG, font=f["body"])
        ctk.CTkComboBox(
            header, variable=self.target, values=EXAMPLES, width=230,
            button_color=LINE, button_hover_color=VIOLET, dropdown_fg_color=PANEL,
            dropdown_hover_color=RAISED, dropdown_text_color=FG, dropdown_font=f["body"], **field,
        ).pack(side="left")
        for text, var in (("Sectors", self.sectors), ("Signals", self.max_signals), ("Min SDE", self.min_sde)):
            ctk.CTkLabel(header, text=text, font=f["small"], text_color=MUTED).pack(side="left", padx=(18, 6))
            ctk.CTkEntry(header, textvariable=var, width=52, justify="center", **field).pack(side="left")

        ghost: dict[str, Any] = dict(fg_color="transparent", hover_color=RAISED, text_color=MUTED, font=f["body"], corner_radius=10, height=36, width=0)
        ctk.CTkButton(header, text="Help", command=self.show_help, **ghost).pack(side="right")
        ctk.CTkButton(header, text="Open output folder", command=self.open_output, **ghost).pack(side="right", padx=(0, 4))
        self.button = ctk.CTkButton(
            header, text="Search", command=self.press, fg_color=VIOLET, hover_color=VIOLET_HOVER,
            text_color="#FFFFFF", font=f["body_bold"], corner_radius=10, height=36, width=120,
        )
        self.button.pack(side="left", padx=(22, 0))
        self.bind("<Return>", lambda _: self.start())

        strip = ctk.CTkFrame(self, fg_color=VOID, corner_radius=0)
        strip.pack(fill="x", padx=24, pady=(4, 10))
        self.transit = Transit(strip)
        self.transit.pack(side="left")
        self.status = ctk.CTkLabel(strip, text="Ready when you are", font=f["status"], text_color=MUTED, anchor="w")
        self.status.pack(side="left", padx=(18, 0))
        self.percent = ctk.CTkLabel(strip, text="", font=f["number"], text_color=FG)
        self.percent.pack(side="right")

        body = ctk.CTkFrame(self, fg_color=VOID, corner_radius=0)
        body.pack(fill="both", expand=True, padx=24, pady=(0, 20))
        body.grid_columnconfigure(0, weight=0, minsize=470)
        body.grid_columnconfigure(1, weight=1)
        body.grid_rowconfigure(0, weight=1)

        left = ctk.CTkFrame(body, fg_color=VOID, corner_radius=0, width=470)
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 18))
        left.pack_propagate(False)

        ctk.CTkLabel(left, text="Signals", font=f["section"], text_color=MUTED, anchor="w").pack(fill="x", pady=(0, 6))
        self.list = ctk.CTkScrollableFrame(
            left, fg_color=VOID, height=250, corner_radius=0,
            scrollbar_button_color=RAISED, scrollbar_button_hover_color=LINE,
        )
        self.list.pack(fill="x")
        self.empty = ctk.CTkLabel(
            self.list, text="Signals the search finds will be listed here.", font=f["body"], text_color=SLATE, anchor="w",
        )
        self.empty.pack(fill="x", pady=8)

        heading = ctk.CTkFrame(left, fg_color=VOID, corner_radius=0)
        heading.pack(fill="x", pady=(16, 6))
        ctk.CTkLabel(heading, text="What this means", font=f["section"], text_color=MUTED, anchor="w").pack(side="left")
        self.ctoi_button = ctk.CTkButton(
            heading, text="Copy CTOI values", command=self.copy_ctoi, fg_color="transparent", hover_color=RAISED,
            text_color=GOLD, text_color_disabled=PANEL, font=f["small"], height=26, width=0, corner_radius=8, state="disabled",
        )
        self.ctoi_button.pack(side="right")
        self.chosen = None
        self.info_box = ctk.CTkTextbox(
            left, fg_color=PANEL, text_color=FG, font=f["body"], corner_radius=14, wrap="word",
            border_spacing=14, scrollbar_button_color=RAISED, scrollbar_button_hover_color=LINE,
        )
        self.info_box.pack(fill="both", expand=True)
        self.info_box.tag_config("key", foreground=GOLD)
        self.set_info("Pick a star and press Search. Then click a signal to read what it means.\n\nNew to this? Press Help.")

        self.log_toggle = ctk.CTkButton(
            left, text="Show log", command=self.toggle_log, anchor="w", fg_color="transparent",
            hover_color=PANEL, text_color=MUTED, font=f["small"], height=28, corner_radius=8,
        )
        self.log_toggle.pack(fill="x", pady=(10, 0))
        self.log_box = ctk.CTkTextbox(
            left, fg_color=PANEL, text_color=MUTED, font=f["mono"], corner_radius=12, height=140,
            border_spacing=10, wrap="word", scrollbar_button_color=RAISED,
        )
        self.log_box.configure(state="disabled")

        self.plot_frame = ctk.CTkFrame(body, fg_color=core.PLOT_BG, corner_radius=16)
        self.plot_frame.grid(row=0, column=1, sticky="nsew")
        self.placeholder = ctk.CTkLabel(
            self.plot_frame, text="Each signal gets three plots here:\nthe raw light curve, the period search, and the folded dip.",
            font=f["body"], text_color=SLATE, justify="center",
        )
        self.placeholder.place(relx=0.5, rely=0.5, anchor="center")

    def toggle_log(self):
        self.log_open = not self.log_open
        if self.log_open:
            self.log_box.pack(fill="x", pady=(4, 0))
        else:
            self.log_box.pack_forget()
        self.log_toggle.configure(text="Hide log" if self.log_open else "Show log")

    def log(self, message):
        self.resume.wait()
        self.events.put(("log", message))

    def set_info(self, text):
        box = self.info_box
        box.configure(state="normal")
        box.delete("1.0", "end")
        for line in text.split("\n"):
            match = KEY.match(line)
            if match:
                box.insert("end", match.group(0), "key")
                box.insert("end", line[match.end() :] + "\n")
            else:
                box.insert("end", line + "\n")
        box.configure(state="disabled")

    def show_help(self):
        window = ctk.CTkToplevel(self, fg_color=VOID)
        window.title("How to read the results")
        window.geometry("760x860")
        box = ctk.CTkTextbox(window, fg_color=PANEL, text_color=FG, font=self.fonts["mono"], corner_radius=16, border_spacing=18, wrap="word", scrollbar_button_color=RAISED, scrollbar_button_hover_color=LINE)
        box.pack(fill="both", expand=True, padx=16, pady=16)
        box.tag_config("heading", foreground=GOLD)
        for line in HELP.split("\n"):
            box.insert("end", line + "\n", "heading" if line in HEADINGS else ())
        box.configure(state="disabled")
        window.after(100, window.lift)

    def report(self, fraction, text=""):
        self.resume.wait()
        self.events.put(("progress", (fraction, text)))

    def set_progress(self, fraction, text):
        self.transit.set(fraction)
        self.percent.configure(text=f"{fraction * 100:.0f}%")
        if text:
            self.status.configure(text=text, text_color=FG)

    def write_log(self, message):
        self.log_box.configure(state="normal")
        self.log_box.insert("end", message + "\n")
        self.log_box.see("end")
        self.log_box.configure(state="disabled")

    def read_number(self, var, name, kind, low, high):
        try:
            value = kind(var.get())
        except ValueError:
            value = None
        if value is None or not low <= value <= high:
            word = "a whole number" if kind is int else "a number"
            self.status.configure(text=f"{name} must be {word} from {low} to {high}", text_color=GOLD)
            return None
        return value

    def press(self):
        if not self.running:
            self.start()
        elif self.resume.is_set():
            self.resume.clear()
            self.button.configure(text="Resume", fg_color=GOLD, hover_color=GOLD, text_color=VOID)
            self.status.configure(text="Paused at the next step, press Resume to go on", text_color=GOLD)
        else:
            self.resume.set()
            self.button.configure(text="Pause", fg_color=LINE, hover_color=RAISED, text_color="#FFFFFF")
            self.status.configure(text="Resuming", text_color=FG)

    def start(self):
        if self.running:
            return
        target = self.target.get().strip()
        if not target:
            self.status.configure(text="Type a star name or TIC number first", text_color=GOLD)
            return
        sectors = self.read_number(self.sectors, "Sectors", int, 1, 60)
        signals = self.read_number(self.max_signals, "Signals", int, 1, 10)
        sde = self.read_number(self.min_sde, "Min SDE", float, 3, 30)
        if None in (sectors, signals, sde):
            return
        self.running = True
        self.resume.set()
        self.button.configure(text="Pause", fg_color=LINE, hover_color=RAISED, text_color="#FFFFFF")
        self.transit.set(0, instant=True)
        self.set_progress(0, f"Starting {target}")
        self.clear_rows()
        self.write_log(f"--- {target} ---")
        threading.Thread(target=self.work, args=(target, sectors, signals, sde), daemon=True).start()

    def work(self, target, sectors, max_signals, min_sde):
        try:
            result = core.run(target, sectors, max_signals, min_sde, log=self.log, progress=self.report)
            self.events.put(("done", result))
        except Exception as e:
            self.events.put(("error", f"{type(e).__name__}: {e}"))

    def poll(self):
        while not self.events.empty():
            kind, payload = self.events.get()
            if kind == "log":
                self.write_log(payload)
            elif kind == "progress":
                self.set_progress(*payload)
            elif kind == "error":
                self.write_log(f"ERROR {payload}")
                self.status.configure(text=f"Search failed: {payload}", text_color=GOLD)
                self.finish()
            elif kind == "done":
                self.show_results(payload)
                self.finish()
        self.after(50, self.poll)

    def finish(self):
        self.running = False
        self.resume.set()
        self.button.configure(text="Search", fg_color=VIOLET, hover_color=VIOLET_HOVER, text_color="#FFFFFF")

    def clear_rows(self):
        self.chosen = None
        self.ctoi_button.configure(state="disabled")
        for row in self.rows:
            row.destroy()
        self.rows = []
        self.signals = []
        self.empty.pack(fill="x", pady=8)

    def show_results(self, result):
        self.signals = result["signals"]
        self.star_text = core.describe_star(result["star"])
        new = sum(s["label"] == "NEW CANDIDATE" for s in self.signals)
        if not self.signals:
            self.status.configure(text="No signals strong enough to report", text_color=FG)
            self.set_info(
                self.star_text + "\n\nNo signals strong enough to report. That's the normal result - "
                "most stars have no planet that passes in front of them from our point of view."
            )
            return
        found = f"{len(self.signals)} signal{'s' if len(self.signals) != 1 else ''} found"
        self.status.configure(text=found + (f", {new} new" if new else ""), text_color=GOLD if new else FG)
        self.empty.pack_forget()
        for signal in self.signals:
            row = SignalRow(self.list, signal, self.fonts, self.pick)
            row.pack(fill="x", pady=(0, 8))
            self.rows.append(row)
        self.pick(self.rows[0])

    def copy_ctoi(self):
        signal = self.chosen.signal if self.chosen else None
        if not signal or "ctoi" not in signal:
            return
        self.clipboard_clear()
        self.clipboard_append(signal["ctoi"].read_text())
        self.status.configure(text=f"CTOI values copied - also saved to {signal['ctoi'].relative_to(core.OUTPUT.parent)}", text_color=GOLD)

    def pick(self, chosen):
        self.chosen = chosen
        self.ctoi_button.configure(state="normal" if "ctoi" in chosen.signal else "disabled")
        for row in self.rows:
            row.select(row is chosen)
        signal = chosen.signal
        self.set_info(signal["summary"] + "\n\n" + self.star_text)
        self.show_figure(signal["figure"])

    def show_figure(self, figure):
        self.placeholder.place_forget()
        if self.canvas is None:
            holder = tk.Frame(self.plot_frame, bg=core.PLOT_BG)
            holder.pack(fill="both", expand=True, padx=14, pady=14)
            self.canvas = FigureCanvasTkAgg(figure, master=holder)
            widget = self.canvas.get_tk_widget()
            widget.configure(bg=core.PLOT_BG, highlightthickness=0)
            self.toolbar = Toolbar(self.canvas, holder, pack_toolbar=False)
            self.style_toolbar(self.toolbar)
            self.toolbar.pack(side="bottom", fill="x")
            widget.pack(fill="both", expand=True)
        else:
            widget = self.canvas.get_tk_widget()
            figure.set_canvas(self.canvas)
            self.canvas.figure = figure
            figure.set_size_inches(widget.winfo_width() / figure.dpi, widget.winfo_height() / figure.dpi, forward=False)
            if self.toolbar is not None:
                self.toolbar.update()
        self.canvas.draw_idle()

    def style_toolbar(self, toolbar):
        toolbar.configure(bg=core.PLOT_BG)
        for child in toolbar.winfo_children():
            options = child.keys()
            colors = {k: v for k, v in (
                ("bg", core.PLOT_BG), ("activebackground", RAISED), ("highlightthickness", 0), ("bd", 0),
                ("fg", MUTED), ("activeforeground", FG), ("selectcolor", RAISED),
            ) if k in options or {"bg": "background", "fg": "foreground", "bd": "borderwidth"}.get(k) in options}
            child.configure(**colors)
            if getattr(child, "_image_file", None):
                toolbar._set_image_for_button(child)

    def open_output(self):
        core.OUTPUT.mkdir(exist_ok=True)
        subprocess.Popen(["xdg-open", str(core.OUTPUT)])


if __name__ == "__main__":
    App().mainloop()
