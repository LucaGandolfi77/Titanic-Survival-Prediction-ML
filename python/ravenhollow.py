import pygame, sys, math, random

pygame.init()
W, H = 960, 640
screen = pygame.display.set_mode((W, H))
pygame.display.set_caption("Ravenhollow - Mystery RPG")
clock = pygame.time.Clock()

# ── fonts ──────────────────────────────────────────────────────────────────
F_BIG   = pygame.font.SysFont("Georgia",    28, bold=True)
F_MED   = pygame.font.SysFont("Georgia",    19, bold=True)
F_SML   = pygame.font.SysFont("Consolas",   15)
F_TINY  = pygame.font.SysFont("Consolas",   12)

# ── palette ────────────────────────────────────────────────────────────────
C = dict(
    bg=(18,22,32), fog=(28,35,52), ground=(38,48,62), path=(55,65,80),
    wall=(30,28,42), gold=(220,185,80), amber=(200,130,40),
    red=(190,50,50), green=(60,160,80), blue=(60,110,200),
    white=(230,230,230), gray=(110,110,130), dark=(12,14,20),
    panel=(18,20,34), panel2=(24,28,44), border=(80,90,120),
    suspect=(160,60,60), clue=(60,140,160), deduction=(140,100,180),
    npc_body=(100,120,160), npc_head=(200,170,140),
    player_body=(60,140,200), player_head=(220,185,140),
)

# ══════════════════════════════════════════════════════════════════════════
# GAME STATE
# ══════════════════════════════════════════════════════════════════════════
class GameState:
    def __init__(self):
        self.day          = 1
        self.hour         = 8          # 0-23
        self.clues        = set()
        self.deductions   = set()
        self.flags        = set()      # story flags
        self.suspicion    = {n:0 for n in
            ["Sheriff","Innkeeper","Fisherman","Widow","Doctor","Stranger"]}
        self.act          = 1          # 1=Discovery 2=Investigation 3=Confrontation
        self.accused      = None
        self.notifications= []         # (text, timer)
        self.ended        = False
        self.ending_text  = ""

    def add_clue(self, clue_id, desc):
        if clue_id not in self.clues:
            self.clues.add(clue_id)
            self.notifications.append([f"CLUE: {desc}", 240])
            self._check_deductions()

    def _check_deductions(self):
        c = self.clues
        d = self.deductions
        if "symbol_door" in c and "symbol_book" in c and "DED_CULT" not in d:
            d.add("DED_CULT")
            self.notifications.append(["DEDUCTION: The symbol is from an old cult!", 300])
        if "stranger_hood" in c and "stranger_boat" in c and "DED_STRANGER_SAILOR" not in d:
            d.add("DED_STRANGER_SAILOR")
            self.notifications.append(["DEDUCTION: The Stranger is a sailor in disguise!", 300])
        if "keeper_debt" in c and "sheriff_debt" in c and "DED_MOTIVE_SHERIFF" not in d:
            d.add("DED_MOTIVE_SHERIFF")
            self.notifications.append(["DEDUCTION: Sheriff had motive – debt dispute!", 300])
        if len(c) >= 6 and self.act < 2:
            self.act = 2
            self.notifications.append(["ACT II: Investigation deepens...", 360])
        if len(d) >= 2 and self.act < 3:
            self.act = 3
            self.flags.add("can_accuse")
            self.notifications.append(["ACT III: You can now make your accusation!", 400])

    def add_suspicion(self, name, amt):
        self.suspicion[name] = min(100, self.suspicion.get(name,0)+amt)

    def tick(self, minutes=10):
        self.hour = (self.hour + minutes//60) % 24
        if minutes >= 60*14 and self.hour == 0:
            self.day += 1

    @property
    def is_night(self):
        return self.hour < 6 or self.hour >= 21

GS = GameState()

# ══════════════════════════════════════════════════════════════════════════
# NPC
# ══════════════════════════════════════════════════════════════════════════
class NPC:
    def __init__(self, name, role, color, pos, location, tree):
        self.name     = name
        self.role     = role
        self.color    = color   # body color
        self.pos      = list(pos)
        self.location = location
        self.tree     = tree    # dict: node_id -> {text, opts:[{label,next,action}]}
        self.node     = "root"
        self.trust    = 0
        self.met      = False
        self.bob      = random.uniform(0, math.pi*2)

    def update(self):
        self.bob += 0.04

    def draw(self, surf, cx, cy):
        x = self.pos[0] - cx
        y = self.pos[1] - cy
        # body
        pygame.draw.rect(surf, self.color, (x-12, y-28, 24, 30), border_radius=4)
        # head
        pygame.draw.circle(surf, C["npc_head"], (x, int(y-32+math.sin(self.bob)*2)), 12)
        # eyes
        pygame.draw.circle(surf, C["dark"], (x-4, int(y-33+math.sin(self.bob)*2)), 2)
        pygame.draw.circle(surf, C["dark"], (x+4, int(y-33+math.sin(self.bob)*2)), 2)
        # exclamation if new info available
        if self.has_new_info():
            t = F_SML.render("!", True, C["gold"])
            surf.blit(t, (x-4, y-58))
        # name tag
        nt = F_TINY.render(self.name, True, C["white"])
        surf.blit(nt, (x - nt.get_width()//2, y+4))

    def has_new_info(self):
        node = self.tree.get(self.node, {})
        return bool(node.get("opts"))

    def in_range(self, px, py):
        return math.hypot(self.pos[0]-px, self.pos[1]-py) < 48

    def interact(self):
        if not self.met:
            self.met = True
            self.trust += 5
        self.node = self.tree.get(self.node, {}).get("entry", "root")

    def choose(self, idx):
        node = self.tree.get(self.node)
        if not node: return
        opts = [o for o in node["opts"] if self._opt_visible(o)]
        if idx >= len(opts): return
        opt = opts[idx]
        # execute action
        action = opt.get("action")
        if action:
            action(GS, self)
        # advance node
        nxt = opt.get("next", "root")
        self.node = nxt if nxt in self.tree else "root"

    def _opt_visible(self, opt):
        req_clue = opt.get("req_clue")
        req_flag = opt.get("req_flag")
        if req_clue and req_clue not in GS.clues: return False
        if req_flag and req_flag not in GS.flags: return False
        return True

    def current_opts(self):
        node = self.tree.get(self.node, {})
        return [o for o in node.get("opts", []) if self._opt_visible(o)]

    def current_text(self):
        return self.tree.get(self.node, {}).get("text", "...")

# ── dialogue trees ──────────────────────────────────────────────────────
def mk_sheriff():
    def give_clue_debt(gs, npc):
        gs.add_clue("sheriff_debt", "Sheriff owed money to the keeper")
        gs.add_suspicion("Sheriff", 20)
        npc.trust += 5
    def give_symbol(gs, npc):
        gs.add_clue("symbol_door","A strange symbol was burned onto the keeper's door")
        npc.trust += 3
    return {
        "root": {"text": "I'm busy. What do you want, detective?",
                 "opts": [
                     {"label": "Tell me about the missing keeper.",  "next":"keeper"},
                     {"label": "What's that symbol on the door?",    "next":"symbol"},
                     {"label": "Goodbye.", "next":"bye"},
                 ]},
        "keeper": {"text": "Elias... he was a strange one. Kept to himself. I... hadn't seen him in weeks.",
                   "opts": [
                       {"label": "Were you on good terms with him?", "next":"debt"},
                       {"label": "Any enemies?",                     "next":"enemies"},
                       {"label": "Back.",                            "next":"root"},
                   ]},
        "debt":   {"text": "*fidgets* We had a small financial disagreement. It's irrelevant.",
                   "opts": [
                       {"label": "Sounds relevant to me.", "next":"debt2", "action": give_clue_debt},
                       {"label": "I see. Back.",           "next":"root"},
                   ]},
        "debt2":  {"text": "Fine! He lent me money. I hadn't paid it back. But I didn't HURT him!",
                   "opts": [{"label":"I'll note that.", "next":"root"}]},
        "symbol": {"text": "It looked like... a circle with three lines. Probably kids.",
                   "opts": [
                       {"label":"Can I see the door?", "next":"symbol2", "action": give_symbol},
                       {"label":"Back.", "next":"root"},
                   ]},
        "symbol2":{"text": "It's sealed off. Official business. Don't go snooping.",
                   "opts": [{"label":"Understood.", "next":"root"}]},
        "enemies":{"text": "People didn't like how he ran the lighthouse. But murder? No.",
                   "opts": [{"label":"Back.", "next":"root"}]},
        "bye":    {"text": "Good. Stay out of trouble.", "opts":[]},
    }

def mk_innkeeper():
    def give_stranger(gs, npc):
        gs.add_clue("stranger_hood","The stranger always keeps his hood up, even indoors")
        gs.add_suspicion("Stranger", 15)
    def give_widow_letter(gs, npc):
        gs.add_clue("widow_grief","The widow was seen arguing with the keeper last month")
        gs.add_suspicion("Widow", 10)
    return {
        "root": {"text": "Welcome! Can I get you something? Oh, you're here about Elias...",
                 "opts": [
                     {"label":"Tell me about the stranger at the inn.", "next":"stranger"},
                     {"label":"Did you know the keeper well?",          "next":"keeper"},
                     {"label":"Heard any gossip lately?",               "next":"gossip"},
                     {"label":"Goodbye.",                               "next":"bye"},
                 ]},
        "stranger":{"text":"Arrived two nights before Elias vanished. Pays in gold. Never removes his hood!",
                    "opts":[
                        {"label":"Suspicious indeed.", "next":"stranger2", "action":give_stranger},
                        {"label":"Back.", "next":"root"},
                    ]},
        "stranger2":{"text":"He asked me about the lighthouse schedule. Why would a traveller care about that?",
                     "opts":[
                         {"label":"Very interesting.", "next":"root"},
                     ]},
        "keeper":  {"text":"Elias was... intense. Always reading old books. Nice man though.",
                    "opts":[
                        {"label":"What kind of books?", "next":"books"},
                        {"label":"Back.", "next":"root"},
                    ]},
        "books":   {"text":"Old ones. History, symbols, cults. The doctor would know more – he lent them.",
                    "opts":[{"label":"Thank you.", "next":"root"}]},
        "gossip":  {"text":"The widow and Elias had a big row last month. Something about a letter.",
                    "opts":[
                        {"label":"Tell me more.", "next":"gossip2", "action":give_widow_letter},
                        {"label":"Back.", "next":"root"},
                    ]},
        "gossip2": {"text":"She stormed out crying. He looked shaken. Never saw them speak again.",
                    "opts":[{"label":"I'll speak with her.", "next":"root"}]},
        "bye":     {"text":"Come back if you need a room!", "opts":[]},
    }

def mk_fisherman():
    def give_boat(gs, npc):
        gs.add_clue("stranger_boat","A boat was seen leaving the docks the night of the disappearance")
        gs.add_suspicion("Stranger", 25)
    def give_symbol_book(gs, npc):
        gs.add_clue("symbol_book","The symbol matches markings in an old cult book")
        gs.add_suspicion("Doctor", 10)
    return {
        "root":{"text":"*stares at the sea* The fog speaks, if you know how to listen.",
                "opts":[
                    {"label":"Did you see anything unusual that night?","next":"night"},
                    {"label":"What do you know about the symbol?",      "next":"symbol",
                     "req_clue":"symbol_door"},
                    {"label":"Goodbye.",                                "next":"bye"},
                ]},
        "night":{"text":"A boat. No lantern. Slipping out past midnight. Fast. Too fast for a fisherman.",
                 "opts":[
                     {"label":"Which direction?","next":"night2","action":give_boat},
                     {"label":"Back.", "next":"root"},
                 ]},
        "night2":{"text":"North. Toward the old ruins on the cape. The sea remembers all sins.",
                  "opts":[{"label":"Thank you, old man.","next":"root"}]},
        "symbol":{"text":"*eyes widen* You found it. That mark... I've seen it in the doctor's old book.",
                  "opts":[
                      {"label":"What book?","next":"symbol2","action":give_symbol_book},
                      {"label":"Back.","next":"root"},
                  ]},
        "symbol2":{"text":"A tome on the Hollow Covenant. A cult that vanished centuries ago. Or... did they?",
                   "opts":[{"label":"Chilling.","next":"root"}]},
        "bye":{"text":"The tide waits for no one.","opts":[]},
    }

def mk_widow():
    def give_letter(gs, npc):
        gs.add_clue("secret_letter","The keeper's letter warned of 'those who watch from the cape'")
        npc.trust += 10
    def give_cape(gs, npc):
        gs.add_clue("cape_ruins","The keeper mentioned old ruins on the cape in his private notes")
        gs.flags.add("knows_cape")
    return {
        "root":{"text":"*weeping softly* Elias... he was the only one who understood me.",
                "opts":[
                    {"label":"I'm sorry for your loss. Can you help me find him?","next":"help"},
                    {"label":"I heard you argued with him.",                       "next":"argue",
                     "req_clue":"widow_grief"},
                    {"label":"Goodbye.",                                           "next":"bye"},
                ]},
        "help":{"text":"He... he gave me a letter. Said to open it only if something happened to him.",
                "opts":[
                    {"label":"May I read it?","next":"letter","action":give_letter},
                    {"label":"Back.","next":"root"},
                ]},
        "letter":{"text":"*hands you a folded note* 'They watch from the cape. The symbol is their seal.'",
                  "opts":[
                      {"label":"Did he say who 'they' are?","next":"they"},
                      {"label":"Back.","next":"root"},
                  ]},
        "they":{"text":"He mentioned ruins. Old ruins on the north cape. He went there once and came back terrified.",
                "opts":[
                    {"label":"I'll investigate the cape.","next":"root","action":give_cape},
                ]},
        "argue":{"text":"*startled* He... he wanted to leave Ravenhollow. I begged him not to. I was afraid.",
                 "opts":[
                     {"label":"Why afraid?","next":"afraid"},
                     {"label":"Back.","next":"root"},
                 ]},
        "afraid":{"text":"Because he'd been threatened. I don't know by whom. He wouldn't tell me.",
                  "opts":[{"label":"Thank you for telling me.","next":"root"}]},
        "bye":{"text":"Please find him.","opts":[]},
    }

def mk_doctor():
    def give_book(gs, npc):
        gs.add_clue("doctor_book","The doctor lent the keeper a book about the Hollow Covenant cult")
        gs.add_suspicion("Doctor", 15)
    def give_medical(gs, npc):
        gs.add_clue("keeper_health","The keeper was in perfect health – this was no accident")
    return {
        "root":{"text":"Remarkable case. Disappearance without physical evidence. Medically speaking.",
                "opts":[
                    {"label":"Did you examine the scene?",  "next":"scene"},
                    {"label":"I heard you lent him books.", "next":"books","req_clue":"symbol_book"},
                    {"label":"Was he in good health?",      "next":"health"},
                    {"label":"Goodbye.",                    "next":"bye"},
                ]},
        "scene":{"text":"I found no blood, no signs of struggle. Just that symbol. Burns at 400 degrees minimum.",
                 "opts":[
                     {"label":"Deliberate, then.", "next":"deliberate","action":give_medical},
                     {"label":"Back.","next":"root"},
                 ]},
        "deliberate":{"text":"Precisely. Someone with knowledge and equipment. This was planned.",
                      "opts":[{"label":"Back.","next":"root"}]},
        "books":{"text":"*pauses* Yes. A historical tome on a local cult. Academic interest only.",
                 "opts":[
                     {"label":"What cult?","next":"cult","action":give_book},
                     {"label":"Back.","next":"root"},
                 ]},
        "cult":{"text":"The Hollow Covenant. 17th century. Used that symbol as a ward. Supposedly disbanded.",
                "opts":[
                    {"label":"Supposedly?","next":"cult2"},
                    {"label":"Back.","next":"root"},
                ]},
        "cult2":{"text":"I... may have shared too much. Please don't tell anyone I said this.",
                 "opts":[{"label":"Your secret is safe.","next":"root"}]},
        "health":{"text":"Excellent health. Mid-50s, strong as an ox. Whatever happened, he didn't fall ill.",
                  "opts":[{"label":"Good to know.","next":"root"}]},
        "bye":{"text":"Good day, detective.","opts":[]},
    }

def mk_stranger():
    def give_keeper(gs, npc):
        gs.add_clue("keeper_debt","The keeper owed a debt to a powerful organisation at the cape")
        gs.add_suspicion("Stranger", 30)
        gs.add_suspicion("Sheriff", 10)
    return {
        "root":{"text":"*eyes narrow under hood* I have nothing to say to you.",
                "opts":[
                    {"label":"Who are you, really?",           "next":"who"},
                    {"label":"Why were you asking about the lighthouse?","next":"light",
                     "req_clue":"stranger_hood"},
                    {"label":"I know about the boat.",         "next":"boat","req_clue":"stranger_boat"},
                    {"label":"Goodbye.",                       "next":"bye"},
                ]},
        "who":{"text":"A traveller. Nothing more. This town has nothing I want.",
               "opts":[
                   {"label":"Then why stay?","next":"stay"},
                   {"label":"Back.","next":"root"},
               ]},
        "stay":{"text":"*long pause* ...Business. Old business. None of yours.",
                "opts":[{"label":"We'll see about that.","next":"root"}]},
        "light":{"text":"*surprised* I... I was curious. Lighthouses fascinate me.",
                 "opts":[
                     {"label":"Try again.","next":"light2"},
                     {"label":"Back.","next":"root"},
                 ]},
        "light2":{"text":"*sighs* Fine. The keeper owed a debt. I was sent to collect. That's all.",
                  "opts":[
                      {"label":"Owed a debt to whom?","next":"debt","action":give_keeper},
                      {"label":"Back.","next":"root"},
                  ]},
        "debt":{"text":"*stands up* I've said too much. Ask the ruins at the cape if you want more.",
                "opts":[{"label":"I will.","next":"root"}]},
        "boat":{"text":"*goes pale* You can't prove anything. I was collecting what was owed. He was alive when I left!",
                "opts":[
                    {"label":"So you WERE there!","next":"boat2"},
                    {"label":"Back.","next":"root"},
                ]},
        "boat2":{"text":"He... he opened the door, paid half, then the symbol appeared on the door by itself. I fled.",
                 "opts":[{"label":"This changes things.","next":"root"}]},
        "bye":{"text":"*turns away*","opts":[]},
    }

# ══════════════════════════════════════════════════════════════════════════
# PLAYER
# ══════════════════════════════════════════════════════════════════════════
class Player:
    def __init__(self):
        self.x, self.y = 400, 300
        self.speed = 3
        self.energy = 20
        self.name  = "Detective"

    def move(self, keys, bounds):
        dx = dy = 0
        if keys[pygame.K_LEFT]  or keys[pygame.K_a]: dx -= self.speed
        if keys[pygame.K_RIGHT] or keys[pygame.K_d]: dx += self.speed
        if keys[pygame.K_UP]    or keys[pygame.K_w]: dy -= self.speed
        if keys[pygame.K_DOWN]  or keys[pygame.K_s]: dy += self.speed
        nx = max(bounds[0], min(bounds[2], self.x + dx))
        ny = max(bounds[1], min(bounds[3], self.y + dy))
        self.x, self.y = nx, ny

    def draw(self, surf, cx, cy):
        x = self.x - cx + W//2
        y = self.y - cy + H//2
        pygame.draw.rect(surf, C["player_body"], (x-11, y-26, 22, 28), border_radius=4)
        pygame.draw.circle(surf, C["player_head"], (x, y-30), 11)
        pygame.draw.circle(surf, C["dark"], (x-3, y-31), 2)
        pygame.draw.circle(surf, C["dark"], (x+3, y-31), 2)
        # hat
        pygame.draw.rect(surf, C["dark"], (x-9, y-43, 18, 5))
        pygame.draw.rect(surf, C["dark"], (x-6, y-54, 12, 13))
        # range indicator
        pygame.draw.circle(surf, (255,255,255,40), (x,y-14), 48, 1)

# ══════════════════════════════════════════════════════════════════════════
# DIALOGUE BOX
# ══════════════════════════════════════════════════════════════════════════
class DialogueBox:
    def __init__(self):
        self.active  = False
        self.npc     = None
        self.text    = ""
        self.shown   = 0
        self.timer   = 0
        self.opts    = []

    def open(self, npc):
        self.active  = True
        self.npc     = npc
        npc.interact()
        self._refresh()

    def _refresh(self):
        self.text  = self.npc.current_text()
        self.shown = 0
        self.opts  = self.npc.current_opts()

    def update(self):
        if not self.active: return
        self.timer += 1
        if self.timer % 2 == 0 and self.shown < len(self.text):
            self.shown += 1

    def choose(self, idx):
        self.npc.choose(idx)
        GS.tick(15)
        if not self.opts:
            self.active = False
            return
        self._refresh()

    def draw(self, surf):
        if not self.active: return
        bx, by, bw, bh = 20, H-200, W-40, 185
        pygame.draw.rect(surf, C["panel"],  (bx, by, bw, bh), border_radius=8)
        pygame.draw.rect(surf, C["border"], (bx, by, bw, bh), 2, border_radius=8)

        # portrait area
        pygame.draw.rect(surf, C["panel2"], (bx+8, by+8, 70, 70), border_radius=6)
        pc = self.npc.color
        pygame.draw.rect(surf, pc, (bx+20, by+30, 46, 40), border_radius=4)
        pygame.draw.circle(surf, C["npc_head"], (bx+43, by+25), 14)

        # name
        nt = F_MED.render(f"{self.npc.name}  [{self.npc.role}]", True, C["gold"])
        surf.blit(nt, (bx+88, by+10))
        # trust bar
        tb_w = min(self.npc.trust*2, 120)
        pygame.draw.rect(surf, C["gray"], (bx+88, by+36, 120, 8), border_radius=3)
        pygame.draw.rect(surf, C["green"],(bx+88, by+36, tb_w, 8), border_radius=3)
        surf.blit(F_TINY.render("Trust", True, C["gray"]), (bx+215, by+32))

        # dialogue text
        shown = self.text[:self.shown]
        words = shown.split()
        line, lines = "", []
        for w in words:
            test = line + w + " "
            if F_SML.size(test)[0] > bw-200: lines.append(line); line = w+" "
            else: line = test
        lines.append(line)
        for i, ln in enumerate(lines[:3]):
            surf.blit(F_SML.render(ln, True, C["white"]), (bx+88, by+52+i*18))

        # options
        opts = self.opts
        if self.shown >= len(self.text):
            for i, opt in enumerate(opts[:4]):
                col = C["gold"] if i == 0 else C["white"]
                ot = F_TINY.render(f"[{i+1}] {opt['label']}", True, col)
                surf.blit(ot, (bx+88, by+108+i*18))
        else:
            surf.blit(F_TINY.render("(press SPACE to skip)", True, C["gray"]), (bx+88, by+110))

# ══════════════════════════════════════════════════════════════════════════
# JOURNAL
# ══════════════════════════════════════════════════════════════════════════
CLUE_DESCS = {
    "symbol_door":    "Strange symbol burned onto the keeper's door",
    "stranger_hood":  "Stranger keeps hood up even indoors",
    "widow_grief":    "Widow argued with the keeper last month",
    "stranger_boat":  "Unlit boat left docks night of disappearance",
    "symbol_book":    "Symbol matches markings in an old cult tome",
    "keeper_debt":    "Keeper owed debt to organisation at the cape",
    "doctor_book":    "Doctor lent keeper a book on Hollow Covenant",
    "secret_letter":  "Keeper's letter warns of 'those at the cape'",
    "cape_ruins":     "Keeper mentioned old ruins on north cape",
    "keeper_health":  "Keeper was in perfect health – no accident",
    "sheriff_debt":   "Sheriff owed money to the keeper",
}
DED_DESCS = {
    "DED_CULT":             "The symbol is from the Hollow Covenant cult",
    "DED_STRANGER_SAILOR":  "The Stranger is a sailor / debt collector",
    "DED_MOTIVE_SHERIFF":   "Sheriff had motive: unresolved debt dispute",
}

class Journal:
    def __init__(self):
        self.open  = False
        self.tab   = 0   # 0=Clues 1=Suspects 2=Deductions

    def toggle(self): self.open = not self.open

    def draw(self, surf):
        if not self.open: return
        jx, jy, jw, jh = 60, 40, W-120, H-80
        pygame.draw.rect(surf, C["panel"],  (jx, jy, jw, jh), border_radius=10)
        pygame.draw.rect(surf, C["gold"],   (jx, jy, jw, jh), 2, border_radius=10)

        # tabs
        tabs = ["CLUES","SUSPECTS","DEDUCTIONS"]
        for i, tb in enumerate(tabs):
            col = C["gold"] if i == self.tab else C["gray"]
            tx = jx + 20 + i*130
            pygame.draw.rect(surf, C["panel2"], (tx, jy-26, 120, 28), border_radius=5)
            if i == self.tab: pygame.draw.rect(surf, C["gold"], (tx, jy-26, 120, 28), 2, border_radius=5)
            surf.blit(F_TINY.render(tb, True, col), (tx+10, jy-20))

        surf.blit(F_MED.render("DETECTIVE'S JOURNAL", True, C["gold"]), (jx+jw//2-110, jy+10))
        surf.blit(F_TINY.render("[I/TAB] close  [←→] switch tab", True, C["gray"]), (jx+jw-240, jy+14))

        if self.tab == 0:
            surf.blit(F_SML.render(f"Clues collected: {len(GS.clues)}", True, C["clue"]), (jx+20, jy+42))
            for i, cid in enumerate(sorted(GS.clues)):
                desc = CLUE_DESCS.get(cid, cid)
                surf.blit(F_TINY.render(f"◆ {desc}", True, C["white"]), (jx+30, jy+68+i*20))
        elif self.tab == 1:
            for i, (name, val) in enumerate(GS.suspicion.items()):
                col = C["red"] if val > 40 else C["white"]
                surf.blit(F_SML.render(f"{name}", True, col), (jx+30, jy+50+i*48))
                pygame.draw.rect(surf, C["gray"],    (jx+160, jy+54+i*48, 200, 14), border_radius=4)
                pygame.draw.rect(surf, C["suspect"], (jx+160, jy+54+i*48, val*2, 14), border_radius=4)
                surf.blit(F_TINY.render(f"{val}%", True, C["gold"]), (jx+368, jy+52+i*48))
        elif self.tab == 2:
            for i, did in enumerate(sorted(GS.deductions)):
                desc = DED_DESCS.get(did, did)
                surf.blit(F_SML.render(f"★ {desc}", True, C["deduction"]), (jx+30, jy+60+i*28))
            if not GS.deductions:
                surf.blit(F_SML.render("No deductions yet. Keep investigating.", True, C["gray"]), (jx+30, jy+60))
        # act indicator
        acts = ["ACT I: Discovery","ACT II: Investigation","ACT III: Confrontation"]
        surf.blit(F_TINY.render(acts[GS.act-1], True, C["amber"]), (jx+jw-200, jy+jh-30))

# ══════════════════════════════════════════════════════════════════════════
# ACCUSATION SCREEN
# ══════════════════════════════════════════════════════════════════════════
class AccusationScreen:
    def __init__(self):
        self.active   = False
        self.selected = 0
        self.suspects = list(GS.suspicion.keys())
        self.done     = False

    def draw(self, surf):
        if not self.active: return
        surf.fill(C["dark"])
        t = F_BIG.render("── THE ACCUSATION ──", True, C["gold"])
        surf.blit(t, (W//2-t.get_width()//2, 40))
        surf.blit(F_SML.render("Who kidnapped Elias, the lighthouse keeper?", True, C["white"]), (W//2-220, 90))
        for i, name in enumerate(self.suspects):
            col = C["gold"] if i == self.selected else C["gray"]
            bg  = C["panel2"] if i == self.selected else C["panel"]
            pygame.draw.rect(surf, bg, (W//2-180, 130+i*60, 360, 46), border_radius=8)
            if i == self.selected:
                pygame.draw.rect(surf, C["gold"], (W//2-180, 130+i*60, 360, 46), 2, border_radius=8)
            st = F_MED.render(name, True, col)
            surf.blit(st, (W//2-st.get_width()//2, 148+i*60))
        surf.blit(F_TINY.render("[↑↓] select  [ENTER] accuse  [ESC] cancel", True, C["gray"]), (W//2-190, H-50))

    def handle(self, event):
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_UP:   self.selected = (self.selected-1) % len(self.suspects)
            if event.key == pygame.K_DOWN: self.selected = (self.selected+1) % len(self.suspects)
            if event.key == pygame.K_RETURN:
                self._resolve(self.suspects[self.selected])
            if event.key == pygame.K_ESCAPE: self.active = False

    def _resolve(self, name):
        GS.accused = name
        self.done  = True
        self.active= False
        top = max(GS.suspicion, key=GS.suspicion.get)
        # true culprit is the Stranger (or Sheriff if player found debt)
        true_culprit = "Stranger"
        if name == true_culprit and "DED_STRANGER_SAILOR" in GS.deductions:
            GS.ending_text = (
                "CORRECT! The Stranger was a cult enforcer sent to silence Elias.\n"
                "With your evidence, justice is served. Ravenhollow sleeps safely."
            )
        elif name == "Sheriff" and "DED_MOTIVE_SHERIFF" in GS.deductions:
            GS.ending_text = (
                "PARTIAL TRUTH! The Sheriff had motive but acted only as accomplice.\n"
                "The true enforcer escapes into the fog. The case remains open..."
            )
        elif name == true_culprit:
            GS.ending_text = (
                "You named the right person but lacked solid evidence.\n"
                "The Stranger walks free on a technicality. Better luck next time."
            )
        else:
            GS.ending_text = (
                f"WRONG! {name} was innocent. The real culprit vanishes into the sea.\n"
                "Ravenhollow never fully recovers. The lighthouse stands dark forever."
            )
        GS.ended = True

# ══════════════════════════════════════════════════════════════════════════
# LOCATION
# ══════════════════════════════════════════════════════════════════════════
class Location:
    def __init__(self, name, col, wx, wy, ww, wh, npcs):
        self.name  = name
        self.col   = col
        self.wx, self.wy, self.ww, self.wh = wx, wy, ww, wh
        self.npcs  = npcs

    def draw_floor(self, surf, cx, cy):
        # ground
        pygame.draw.rect(surf, self.col,
            (self.wx-cx+W//2, self.wy-cy+H//2, self.ww, self.wh))
        # label
        lx = self.wx - cx + W//2 + self.ww//2
        ly = self.wy - cy + H//2 + 8
        lt = F_TINY.render(self.name, True, (200,200,200,120))
        surf.blit(lt, (lx-lt.get_width()//2, ly))

# ══════════════════════════════════════════════════════════════════════════
# BUILD WORLD
# ══════════════════════════════════════════════════════════════════════════
sheriff_npc   = NPC("Sheriff",    "Law",       (90,80,100),   (340, 160), "Sheriff Office",  mk_sheriff())
innkeeper_npc = NPC("Innkeeper",  "Gossip",    (120,85,60),   (200, 320), "Inn",             mk_innkeeper())
fisherman_npc = NPC("Fisherman",  "Old Salt",  (70,100,120),  (620, 520), "Docks",           mk_fisherman())
widow_npc     = NPC("Widow",      "Grieving",  (140,80,120),  (480, 290), "Old Mansion",     mk_widow())
doctor_npc    = NPC("Doctor",     "Physician", (60,120,100),  (460, 160), "Town Square",     mk_doctor())
stranger_npc  = NPC("Stranger",   "Unknown",   (60,60,80),    (180, 430), "Inn",             mk_stranger())

ALL_NPCS = [sheriff_npc, innkeeper_npc, fisherman_npc, widow_npc, doctor_npc, stranger_npc]

WORLD_W, WORLD_H = 900, 750
LOCATIONS = [
    Location("Town Square",    C["ground"],  150, 80,  350, 200, [doctor_npc]),
    Location("Sheriff Office", C["wall"],    80,  60,  160, 160, [sheriff_npc]),
    Location("Inn",            (55,42,35),   80,  280, 200, 220, [innkeeper_npc, stranger_npc]),
    Location("Old Mansion",    (40,38,55),   350, 200, 250, 220, [widow_npc]),
    Location("Docks",          (30,45,60),   440, 420, 350, 200, [fisherman_npc]),
]

player = Player()
player.x, player.y = 330, 230

dlg    = DialogueBox()
jrn    = Journal()
accuse = AccusationScreen()

WORLD_BOUNDS = (80, 60, WORLD_W-40, WORLD_H-40)

# rain particles
rain = [[random.randint(0,W), random.randint(0,H), random.randint(2,6)] for _ in range(120)]

def draw_rain(surf, night):
    alpha = 80 if night else 30
    for r in rain:
        r[1] += r[2]*2; r[0] -= 1
        if r[1] > H: r[1]=0; r[0]=random.randint(0,W)
        pygame.draw.line(surf, (180,200,220), (r[0],r[1]), (r[0]-2, r[1]+r[2]*2), 1)

def draw_hud(surf):
    # top-left panel
    pygame.draw.rect(surf, C["panel"], (8,8,200,68), border_radius=6)
    pygame.draw.rect(surf, C["border"],(8,8,200,68), 1, border_radius=6)
    surf.blit(F_SML.render(f"{player.name}", True, C["gold"]),  (16,12))
    surf.blit(F_TINY.render(f"Day {GS.day}  |  {GS.hour:02d}:00", True, C["white"]), (16,34))
    surf.blit(F_TINY.render(f"Clues: {len(GS.clues)}/11  Act {GS.act}", True, C["clue"]),  (16,52))

    # minimap (top-right)
    mx, my, ms = W-115, 8, 100
    pygame.draw.rect(surf, C["panel"],  (mx-5, my-5, ms+10, ms+10), border_radius=5)
    pygame.draw.rect(surf, C["border"], (mx-5, my-5, ms+10, ms+10), 1, border_radius=5)
    scale_x = ms / WORLD_W; scale_y = ms / WORLD_H
    for loc in LOCATIONS:
        lx = mx + int(loc.wx * scale_x)
        ly = my + int(loc.wy * scale_y)
        lw = max(6, int(loc.ww * scale_x))
        lh = max(4, int(loc.wh * scale_y))
        pygame.draw.rect(surf, loc.col, (lx,ly,lw,lh))
    px = mx + int(player.x * scale_x)
    py = my + int(player.y * scale_y)
    pygame.draw.circle(surf, C["player_body"], (px,py), 4)

    # notifications
    for i, notif in enumerate(GS.notifications[:3]):
        text, t = notif
        alpha = min(255, t*4)
        ns = F_TINY.render(text[:60], True, C["gold"])
        surf.blit(ns, (W//2-ns.get_width()//2, 14+i*20))

    # accuse hint
    if "can_accuse" in GS.flags and not GS.ended:
        ht = F_TINY.render("[A] Make your accusation!", True, C["red"])
        surf.blit(ht, (W//2-ht.get_width()//2, H-22))

    # controls hint (bottom left)
    surf.blit(F_TINY.render("[WASD/↑↓←→] Move  [E] Talk  [I/Tab] Journal  [ESC] Close", True, C["gray"]),
              (8, H-18))

def draw_ending(surf):
    surf.fill(C["dark"])
    t = F_BIG.render("─── CASE CLOSED ───", True, C["gold"])
    surf.blit(t, (W//2-t.get_width()//2, 80))
    lines = GS.ending_text.split("\n")
    for i, ln in enumerate(lines):
        lt = F_MED.render(ln, True, C["white"])
        surf.blit(lt, (W//2-lt.get_width()//2, 180+i*40))
    at = F_SML.render(f"You accused: {GS.accused}", True, C["amber"])
    surf.blit(at, (W//2-at.get_width()//2, 300))
    rt = F_MED.render("[Q] Quit  [R] Restart", True, C["gray"])
    surf.blit(rt, (W//2-rt.get_width()//2, H-60))

def _current_location(p):
    for loc in LOCATIONS:
        if loc.wx <= p.x <= loc.wx+loc.ww and loc.wy <= p.y <= loc.wy+loc.wh:
            return loc.name
    return "Unknown"

# ══════════════════════════════════════════════════════════════════════════
# MAIN LOOP
# ══════════════════════════════════════════════════════════════════════════
frame = 0
running = True
while running:
    clock.tick(60)
    frame += 1

    # camera (use player world coords as camera center)
    cam_x = player.x
    cam_y = player.y

    for event in pygame.event.get():
        if event.type == pygame.QUIT: running = False
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_q: running = False
            if GS.ended:
                if event.key == pygame.K_r:
                    GS.__init__(); player.__init__()
                continue
            if accuse.active:
                accuse.handle(event); continue
            if event.key in (pygame.K_i, pygame.K_TAB):
                if dlg.active: dlg.active = False
                else: jrn.toggle()
            if event.key == pygame.K_ESCAPE:
                dlg.active = False; jrn.open = False; accuse.active = False
            if jrn.open:
                if event.key == pygame.K_RIGHT: jrn.tab = (jrn.tab+1)%3
                if event.key == pygame.K_LEFT:  jrn.tab = (jrn.tab-1)%3
                continue
            if dlg.active:
                if event.key == pygame.K_SPACE: dlg.shown = len(dlg.text)
                for k, kk in enumerate([pygame.K_1,pygame.K_2,pygame.K_3,pygame.K_4]):
                    if event.key == kk:
                        dlg.choose(k)
                        if not dlg.active: break
                continue
            if event.key == pygame.K_e:
                for npc in ALL_NPCS:
                    # allow interaction by proximity (world distance) — don't require exact location string match
                    if npc.in_range(player.x, player.y):
                        dlg.open(npc); jrn.open = False; break
            if event.key == pygame.K_a and "can_accuse" in GS.flags:
                accuse.active = True

    # tick notifications
    GS.notifications = [[t, tm-1] for t,tm in GS.notifications if tm > 1]

    # update
    if not dlg.active and not jrn.open and not accuse.active and not GS.ended:
        keys = pygame.key.get_pressed()
        player.move(keys, WORLD_BOUNDS)
    for npc in ALL_NPCS: npc.update()
    dlg.update()

    # ── draw ──────────────────────────────────────────────────────────────
    if GS.ended:
        draw_ending(screen); pygame.display.flip(); continue

    night = GS.is_night
    sky = (10,14,26) if night else (22,32,52)
    screen.fill(sky)

    # fog overlay
    fog_surf = pygame.Surface((W,H), pygame.SRCALPHA)
    fog_alpha = 60 if night else 25
    fog_surf.fill((180,200,220, fog_alpha))
    screen.blit(fog_surf, (0,0))

    # floor of locations
    for loc in LOCATIONS:
        loc.draw_floor(screen, cam_x, cam_y)

    # world boundary lines
    for loc in LOCATIONS:
        rx = loc.wx - cam_x + W//2
        ry = loc.wy - cam_y + H//2
        pygame.draw.rect(screen, C["border"], (rx, ry, loc.ww, loc.wh), 1)

    # npcs (only current location or nearby)
    for npc in ALL_NPCS:
        nx = npc.pos[0] - cam_x + W//2
        ny = npc.pos[1] - cam_y + H//2
        if -60 < nx < W+60 and -60 < ny < H+60:
            npc.draw(screen, cam_x - W//2, cam_y - H//2)

    player.draw(screen, cam_x, cam_y)
    draw_rain(screen, night)
    draw_hud(screen)
    dlg.draw(screen)
    jrn.draw(screen)
    accuse.draw(screen)

    pygame.display.flip()

pygame.quit()
sys.exit()


