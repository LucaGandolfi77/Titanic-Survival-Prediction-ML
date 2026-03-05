import pygame
import random
import math
import sys

pygame.init()

WIDTH, HEIGHT = 950, 680
FPS = 60

BROWN_DARK  = (70, 40, 10)
BROWN       = (101, 67, 33)
GREEN_DARK  = (34, 85, 34)
GREEN_LIGHT = (76, 153, 0)
GOLD        = (255, 215, 0)
RED         = (200, 20, 40)
BLACK       = (0, 0, 0)
WHITE       = (255, 255, 255)
ORANGE      = (255, 140, 0)
YELLOW      = (255, 255, 0)
BLUE        = (30, 144, 255)
CYAN        = (0, 230, 230)
PURPLE      = (160, 32, 240)
PINK        = (255, 105, 180)
GRAY        = (80, 80, 80)

screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("ACORN CASINO  -  Gerald the Oak Tree, Proprietor")
clock = pygame.time.Clock()

font_big   = pygame.font.SysFont("Arial", 38, bold=True)
font_med   = pygame.font.SysFont("Arial", 24, bold=True)
font_small = pygame.font.SysFont("Arial", 18)
font_tiny  = pygame.font.SysFont("Arial", 14)
font_xs    = pygame.font.SysFont("Arial", 12)

SYMBOL_NAMES   = ["ACORN", "LEAF",   "CLOVER", "SEVEN", "DIAMOND", "SHROOM"]
SYMBOL_COLORS  = [BROWN,   ORANGE,   GREEN_LIGHT, RED,  CYAN,      PURPLE   ]
SYMBOL_WEIGHTS = [22,       20,        18,         8,     4,         12       ]
PAYOUTS = {"ACORN": 8, "LEAF": 4, "CLOVER": 6, "SEVEN": 20, "DIAMOND": 50, "SHROOM": 3}

CUSTOMER_TYPES = [
    {"name": "Squirrel",  "col": (160, 100, 40), "r": 18},
    {"name": "Fox",       "col": (210, 90,  20), "r": 22},
    {"name": "Hedgehog",  "col": (110, 85,  60), "r": 16},
    {"name": "Raccoon",   "col": (95,  95,  95), "r": 20},
    {"name": "Owl",       "col": (130, 100, 70), "r": 19},
    {"name": "Badger",    "col": (70,  70,  80), "r": 21},
]

CUSTOMER_QUOTES = [
    "I WILL BE RICH!",  "Just one more spin", "Free drinks??",
    "The oak rigged it","Lady Luck loves me", "YOLO on acorns",
    "I feel a jackpot", "All acorns GONE...", "This machine hates me",
    "GERALD IS A CHEAT","My wife left me...", "I AM INVINCIBLE",
    "One more and I'm out","I have a system!",  "There's no system",
]

GERALD_QUOTES = [
    "WELCOME TO MY CASINO",  "I am a TREE of culture",
    "The house always wins!", "I use my acorns as chips",
    "My roots are very deep", "Bark twice for drinks",
    "I have 400 branches",    "No squirrels after 10pm",
    "I photosynthesise cash", "Gerald Gerald Gerald!",
]


class SlotMachine:
    def __init__(self, x, y, idx):
        self.x, self.y, self.idx = x, y, idx
        self.reels = [random.randint(0, 5) for _ in range(3)]
        self.spin_frames  = [0, 0, 0]
        self.spin_target  = [40, 55, 70]
        self.reel_locked  = [False, False, False]
        self.spinning     = False
        self.result_text  = "-- READY --"
        self.result_color = GOLD
        self.hue          = 0
        self.BET          = 10
        self.last_win     = 0

    def start_spin(self):
        if self.spinning: return False
        self.spinning    = True
        self.spin_frames = [0, 0, 0]
        self.reel_locked = [False, False, False]
        self.result_text  = "SPINNING..."
        self.result_color = WHITE
        return True

    def update(self):
        if not self.spinning: return None
        self.hue = (self.hue + 4) % 360
        done = True
        for i in range(3):
            if not self.reel_locked[i]:
                self.spin_frames[i] += 1
                self.reels[i] = random.choices(range(6), weights=SYMBOL_WEIGHTS)[0]
                if self.spin_frames[i] >= self.spin_target[i]:
                    self.reel_locked[i] = True
            if not self.reel_locked[i]:
                done = False
        if done:
            self.spinning = False
            return self._evaluate()
        return None

    def _evaluate(self):
        s = self.reels
        if s[0] == s[1] == s[2]:
            name = SYMBOL_NAMES[s[0]]
            win = self.BET * PAYOUTS[name]
            self.last_win = win
            self.result_text  = f"JACKPOT! +{win}"
            self.result_color = GOLD
            return ("win", win, name)
        elif s[0]==s[1] or s[1]==s[2] or s[0]==s[2]:
            win = self.BET * 2
            self.last_win = win
            self.result_text  = f"PAIR! +{win}"
            self.result_color = GREEN_LIGHT
            return ("win", win, "pair")
        else:
            self.last_win = 0
            self.result_text  = f"LOST  -{self.BET}"
            self.result_color = RED
            return ("lose", self.BET, "none")

    def draw(self, surf):
        mx, my, mw, mh = self.x, self.y, 155, 215

        if self.spinning:
            r = int(127 + 127*math.sin(math.radians(self.hue)))
            g = int(127 + 127*math.sin(math.radians(self.hue+120)))
            b = int(127 + 127*math.sin(math.radians(self.hue+240)))
            pygame.draw.rect(surf, (r,g,b), (mx-4, my-4, mw+8, mh+8), border_radius=14)

        pygame.draw.rect(surf, (160, 20, 20), (mx, my, mw, mh), border_radius=10)
        pygame.draw.rect(surf, (210, 45, 45), (mx+4, my+4, mw-8, mh-8), border_radius=8)

        t = font_xs.render(f"-- MACHINE {self.idx+1} --", True, GOLD)
        surf.blit(t, (mx+mw//2-t.get_width()//2, my+8))

        pygame.draw.rect(surf, BLACK, (mx+8, my+30, mw-16, 70), border_radius=5)

        for i, si in enumerate(self.reels):
            rx = mx + 12 + i*44
            pygame.draw.rect(surf, (40,40,40), (rx, my+34, 40, 58), border_radius=4)
            nm  = SYMBOL_NAMES[si][:3]
            col = SYMBOL_COLORS[si]
            st = font_small.render(nm, True, col)
            surf.blit(st, (rx+20-st.get_width()//2, my+56))
            pygame.draw.circle(surf, col, (rx+20, my+46), 10, 2)

        bet_t = font_xs.render(f"BET: {self.BET} acorns", True, GOLD)
        surf.blit(bet_t, (mx+mw//2-bet_t.get_width()//2, my+108))

        rt = font_xs.render(self.result_text, True, self.result_color)
        surf.blit(rt, (mx+mw//2-rt.get_width()//2, my+128))

        can_spin = not self.spinning
        btn_col = GOLD if can_spin else GRAY
        pygame.draw.rect(surf, btn_col, (mx+18, my+155, mw-36, 34), border_radius=8)
        bt = font_med.render("SPIN!" if can_spin else "...", True, BLACK)
        surf.blit(bt, (mx+mw//2-bt.get_width()//2, my+161))

        pygame.draw.rect(surf, GOLD, (mx, my, mw, mh), 2, border_radius=10)

    def spin_button_rect(self):
        return pygame.Rect(self.x+18, self.y+155, self.x+self.x+155-36, 34)


class Customer:
    def __init__(self):
        ct = random.choice(CUSTOMER_TYPES)
        self.name  = ct["name"]
        self.col   = ct["col"]
        self.r     = ct["r"]
        self.x     = float(random.randint(650, 880))
        self.y     = float(random.randint(380, 570))
        self.vx    = random.uniform(-0.6, 0.6)
        self.vy    = random.uniform(-0.4, 0.4)
        self.bob   = random.uniform(0, math.pi*2)
        self.quote = random.choice(CUSTOMER_QUOTES)
        self.qtimer = random.randint(60, 200)
        self.anger  = 0

    def update(self):
        self.x += self.vx; self.y += self.vy
        if self.x < 640 or self.x > 900: self.vx *= -1
        if self.y < 370 or self.y > 580: self.vy *= -1
        self.bob += 0.08
        self.qtimer -= 1
        if self.qtimer <= 0:
            self.quote  = random.choice(CUSTOMER_QUOTES)
            self.qtimer = random.randint(120, 300)

    def draw(self, surf):
        cx = int(self.x)
        cy = int(self.y + math.sin(self.bob)*4)
        pygame.draw.circle(surf, self.col,  (cx, cy), self.r)
        pygame.draw.circle(surf, BLACK,     (cx, cy), self.r, 2)
        pygame.draw.circle(surf, WHITE, (cx-6, cy-5), 5)
        pygame.draw.circle(surf, WHITE, (cx+6, cy-5), 5)
        pygame.draw.circle(surf, BLACK, (cx-5, cy-5), 2)
        pygame.draw.circle(surf, BLACK, (cx+5, cy-5), 2)
        pygame.draw.arc(surf, BLACK, (cx-7, cy+3, 14, 10), math.pi, 2*math.pi, 2)
        nt = font_xs.render(self.name, True, WHITE)
        surf.blit(nt, (cx-nt.get_width()//2, cy+self.r+2))
        if self.qtimer > 170:
            bw = max(90, len(self.quote)*6+10)
            bx, by = cx-bw//2, cy-self.r-30
            pygame.draw.rect(surf, WHITE, (bx, by, bw, 22), border_radius=6)
            pygame.draw.polygon(surf, WHITE, [(cx-5,by+22),(cx+5,by+22),(cx,by+30)])
            qt = font_xs.render(self.quote[:18], True, BLACK)
            surf.blit(qt, (bx+bw//2-qt.get_width()//2, by+4))


def draw_gerald(surf, frame, gquote, gq_timer):
    gx, gy = 25, 80

    pygame.draw.rect(surf, BROWN_DARK, (gx+35, gy+165, 50, 130), border_radius=6)
    pygame.draw.rect(surf, BROWN,      (gx+40, gy+170, 40, 125), border_radius=5)

    for ox, oy, rw, rh, col in [
        (-15, 100, 130, 110, GREEN_DARK),
        (-5,  55,  110,  95, GREEN_DARK),
        (10,  10,   90,  85, GREEN_DARK),
        (-8,  110,  95,  65, GREEN_LIGHT),
        (2,   60,   80,  58, GREEN_LIGHT),
        (15,  14,   65,  52, GREEN_LIGHT),
    ]:
        pygame.draw.ellipse(surf, col, (gx+ox, gy+oy, rw, rh))

    hx, hy = gx+17, gy-28
    pygame.draw.rect(surf, BLACK,   (hx-6, hy+22, 75, 9),  border_radius=3)
    pygame.draw.rect(surf, BLACK,   (hx+8, hy-4,  46, 28), border_radius=3)
    pygame.draw.rect(surf, RED,     (hx+8, hy+9,  46, 7))
    pygame.draw.circle(surf, GOLD,  (hx+30, hy+5), 5)

    fx, fy = gx+55, gy+65
    blink = (frame % 110) < 5
    for ex in [fx-11, fx+11]:
        if blink:
            pygame.draw.line(surf, BLACK, (ex-6, fy), (ex+6, fy), 3)
        else:
            pygame.draw.circle(surf, WHITE, (ex, fy), 9)
            pygame.draw.circle(surf, BLACK, (ex, fy), 5)
            pygame.draw.circle(surf, WHITE, (ex+2, fy-3), 2)
    pygame.draw.circle(surf, GOLD, (fx+11, fy), 12, 2)
    pygame.draw.arc(surf, BLACK, (fx-13, fy+6, 26, 14), math.pi, 2*math.pi, 3)
    pygame.draw.arc(surf, BROWN_DARK, (fx-18, fy+1, 14, 9), 0, math.pi, 3)
    pygame.draw.arc(surf, BROWN_DARK, (fx+4,  fy+1, 14, 9), 0, math.pi, 3)

    for i,(ax,ay) in enumerate([(gx+10,gy+110),(gx+95,gy+105),(gx+45,gy+145),(gx+80,gy+72)]):
        bob = int(math.sin(frame*0.05+i)*3)
        pygame.draw.circle(surf, BROWN,      (ax, ay+bob), 9)
        pygame.draw.rect(surf,   BROWN_DARK, (ax-5, ay-13+bob, 10, 8), border_radius=2)

    label = font_xs.render("GERALD  -  OWNER & OAK", True, GOLD)
    pygame.draw.rect(surf, BLACK, (gx-5, gy+300, label.get_width()+14, 22), border_radius=5)
    surf.blit(label, (gx+2, gy+303))

    if gq_timer > 0:
        bw = max(130, len(gquote)*7+14)
        bx, by = gx+60, gy-50
        pygame.draw.rect(surf, (255,255,220), (bx, by, bw, 24), border_radius=7)
        pygame.draw.polygon(surf, (255,255,220), [(bx+10,by+24),(bx+20,by+24),(bx+8,by+34)])
        pygame.draw.rect(surf, GOLD, (bx, by, bw, 24), 2, border_radius=7)
        gqt = font_xs.render(gquote[:22], True, BLACK)
        surf.blit(gqt, (bx+bw//2-gqt.get_width()//2, by+5))


def draw_bg(surf):
    surf.fill((14, 48, 14))
    for i in range(0, WIDTH, 60):
        for j in range(0, HEIGHT, 60):
            if (i//60+j//60) % 2 == 0:
                pygame.draw.rect(surf, (17,54,17), (i, j, 60, 60))
    pygame.draw.rect(surf, GOLD, (0,0,WIDTH,HEIGHT), 4)
    pygame.draw.rect(surf, RED,  (5,5,WIDTH-10,HEIGHT-10), 2)
    pygame.draw.line(surf, GOLD, (210, 58), (210, HEIGHT-58), 2)
    pygame.draw.line(surf, GOLD, (635, 58), (635, HEIGHT-58), 1)
    banner = pygame.Rect(220, 6, 400, 46)
    pygame.draw.rect(surf, BLACK, banner, border_radius=10)
    pygame.draw.rect(surf, GOLD,  banner, 3, border_radius=10)
    t = font_big.render("ACORN  CASINO", True, GOLD)
    surf.blit(t, (WIDTH//2-t.get_width()//2-100, 13))
    s1 = font_xs.render("SLOT MACHINES", True, GOLD)
    surf.blit(s1, (335, 62))
    s2 = font_xs.render("THE FLOOR", True, GOLD)
    surf.blit(s2, (700, 62))


def draw_hud(surf, acorns, day, won, lost, hi):
    pygame.draw.rect(surf, BLACK, (210, HEIGHT-54, WIDTH-220, 48), border_radius=8)
    pygame.draw.rect(surf, GOLD,  (210, HEIGHT-54, WIDTH-220, 48), 2, border_radius=8)
    surf.blit(font_med.render(f"Acorns: {acorns}", True, GOLD),        (225, HEIGHT-48))
    surf.blit(font_med.render(f"Day: {day}",        True, WHITE),       (430, HEIGHT-48))
    surf.blit(font_small.render(f"+{won} won",       True, GREEN_LIGHT),(610, HEIGHT-50))
    surf.blit(font_small.render(f"-{lost} lost",     True, RED),        (610, HEIGHT-30))
    surf.blit(font_small.render(f"Best: {hi}",       True, CYAN),       (730, HEIGHT-50))
    hint = font_xs.render("[R] New Day  [Q] Quit", True, (160,160,160))
    surf.blit(hint, (WIDTH-hint.get_width()-10, HEIGHT-18))


def draw_msg(surf, msg, timer):
    if timer <= 0: return
    ms = font_med.render(msg, True, YELLOW)
    bw, bh = ms.get_width()+24, ms.get_height()+14
    bx, by = WIDTH//2-bw//2, HEIGHT//2-80
    s = pygame.Surface((bw,bh), pygame.SRCALPHA)
    s.fill((0,0,0,210))
    surf.blit(s, (bx, by))
    pygame.draw.rect(surf, GOLD, (bx,by,bw,bh), 2, border_radius=6)
    surf.blit(ms, (bx+12, by+7))


def draw_event_flash(surf, text, alpha):
    if alpha <= 0: return
    ft = font_big.render(text, True, YELLOW)
    ft.set_alpha(alpha)
    surf.blit(ft, (WIDTH//2-ft.get_width()//2, HEIGHT//2-160))


# ------- MAIN -------
machines      = [SlotMachine(220+i*160, 85, i) for i in range(3)]
customers     = []
acorns        = 150
day           = 1
total_won     = 0
total_lost    = 0
high_score    = 150
msg           = "Welcome! You are GERALD the OAK TREE. Run this casino!"
msg_timer     = 240
frame         = 0
cust_timer    = 0
gerald_quote  = "WELCOME, SUCKERS"
gq_timer      = 200
event_text    = ""
event_alpha   = 0

running = True
while running:
    clock.tick(FPS)
    frame += 1

    for event in pygame.event.get():
        if event.type == pygame.QUIT: running = False
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_q: running = False
            if event.key == pygame.K_r:
                day += 1
                bonus = random.randint(25, 70)
                acorns += bonus
                msg = f"New day {day}! Gerald sweeps up {bonus} acorns from the floor!"
                msg_timer = 220
                gerald_quote = random.choice(GERALD_QUOTES)
                gq_timer = 260
        if event.type == pygame.MOUSEBUTTONDOWN:
            mx, my = event.pos
            for machine in machines:
                if pygame.Rect(machine.x+18, machine.y+155, 155-36, 34).collidepoint(mx, my):
                    if acorns >= machine.BET:
                        if machine.start_spin():
                            acorns -= machine.BET

    for machine in machines:
        res = machine.update()
        if res:
            outcome, amount, symbol = res
            if outcome == "win":
                acorns    += amount
                total_won += amount
                if acorns > high_score: high_score = acorns
                if symbol in ("DIAMOND","SEVEN"):
                    msg = f"JACKPOT!!! {symbol}!!! Gerald does a victory photosynthesis! +{amount}"
                    event_text  = f"JACKPOT!!!  +{amount} ACORNS!!!"
                    event_alpha = 255
                    gerald_quote = "I LOVE JACKPOTS"
                    gq_timer = 220
                else:
                    msgs = [
                        f"WIN! +{amount} acorns! Gerald shakes his branches in joy!",
                        f"Pair! Gerald: 'The rigged machine works!' +{amount}",
                        f"Two the same! Gerald photosynthesises with excitement! +{amount}",
                    ]
                    msg = random.choice(msgs)
                msg_timer = 180
            else:
                total_lost += amount
                msgs = [
                    f"Lost {amount} acorns! Gerald weeps SAP down his trunk!",
                    f"-{amount}! Gerald: 'The house wins... wait... I AM the house?!'",
                    f"BUST! Gerald blames a nearby squirrel for the loss. -{amount}",
                ]
                msg       = random.choice(msgs)
                msg_timer = 180

    cust_timer += 1
    if cust_timer > 90 and len(customers) < 9:
        customers.append(Customer())
        cust_timer = 0
    if len(customers) > 4 and random.random() < 0.003:
        c = random.choice(customers)
        customers.remove(c)
        msg = random.choice([
            f"{c.name} left the casino broke. Gerald waves goodbye with a branch.",
            f"{c.name} rage-quit! 'YOU RIGGED IT, TREE!' Gerald: 'Correct.'",
            f"{c.name} left happy! Gerald frowns. A happy customer is a rare customer.",
        ])
        msg_timer = 160
    for c in customers: c.update()

    if event_alpha > 0: event_alpha = max(0, event_alpha-3)

    if acorns <= 0:
        msg = "BANKRUPT! Gerald is selling leaves. Press R for a new day!"
        msg_timer = 8

    draw_bg(screen)
    draw_gerald(screen, frame, gerald_quote, gq_timer)
    gq_timer = max(0, gq_timer-1)
    for m in machines: m.draw(screen)
    for c in customers: c.draw(screen)
    draw_hud(screen, acorns, day, total_won, total_lost, high_score)
    draw_msg(screen, msg, msg_timer)
    draw_event_flash(screen, event_text, event_alpha)
    msg_timer = max(0, msg_timer-1)

    pygame.display.flip()

pygame.quit()
sys.exit()
