// js/outfits.js — outfit catalog, style tags, color palettes

export const OUTFITS = [
  // ── Tops ──────────────────────────────────────────────────────────
  { id:'silk-blouse',     name:'Silk Blouse',       style:'ROMANTIC',    slot:'top',
    colors:{primary:'#fce4ec',secondary:'#f48fb1'},  moodEffect:{mood:5},  compatTags:['romantic','soft'] },
  { id:'floral-crop',     name:'Floral Crop',        style:'ROMANTIC',    slot:'top',
    colors:{primary:'#e8f5e9',secondary:'#a5d6a7'},  moodEffect:{mood:8},  compatTags:['romantic','fresh'] },
  { id:'pearl-turtleneck',name:'Pearl Turtleneck',   style:'ELEGANT',     slot:'top',
    colors:{primary:'#fafafa',secondary:'#e0e0e0'},  moodEffect:{prestige:10}, compatTags:['elegant','refined'] },
  { id:'power-blazer',    name:'Power Blazer',       style:'ELEGANT',     slot:'top',
    colors:{primary:'#263238',secondary:'#78909c'},  moodEffect:{confidence:10}, compatTags:['elegant','power'] },
  { id:'graphic-tee',     name:'Graphic Tee',        style:'STREETWEAR',  slot:'top',
    colors:{primary:'#212121',secondary:'#ff1744'},  moodEffect:{energy:8},  compatTags:['street','bold'] },
  { id:'hoodie-bold',     name:'Hoodie Bold',        style:'STREETWEAR',  slot:'top',
    colors:{primary:'#1a237e',secondary:'#e8eaf6'},  moodEffect:{energy:5},  compatTags:['street','comfy'] },
  { id:'leather-jacket',  name:'Leather Jacket',     style:'PUNK',        slot:'top',
    colors:{primary:'#1b1b1b',secondary:'#9e9e9e'},  moodEffect:{attitude:12}, compatTags:['punk','edgy'] },
  { id:'athletic-top',    name:'Athletic Top',       style:'SPORTY',      slot:'top',
    colors:{primary:'#00bcd4',secondary:'#ffffff'},  moodEffect:{energy:10}, compatTags:['sporty','fresh'] },
  { id:'linen-shirt',     name:'Linen Shirt',        style:'CASUAL',      slot:'top',
    colors:{primary:'#bcaaa4',secondary:'#efebe9'},  moodEffect:{comfort:8}, compatTags:['casual','earth'] },

  // ── Bottoms ──────────────────────────────────────────────────────
  { id:'midi-skirt',      name:'Midi Skirt',         style:'ROMANTIC',    slot:'bottom',
    colors:{primary:'#f8bbd0',secondary:'#fce4ec'},  moodEffect:{mood:5},  compatTags:['romantic','soft'] },
  { id:'tailored-trousers',name:'Tailored Trousers', style:'ELEGANT',     slot:'bottom',
    colors:{primary:'#37474f',secondary:'#b0bec5'},  moodEffect:{prestige:8}, compatTags:['elegant','refined'] },
  { id:'ripped-jeans',    name:'Ripped Jeans',       style:'PUNK',        slot:'bottom',
    colors:{primary:'#212121',secondary:'#9e9e9e'},  moodEffect:{attitude:8}, compatTags:['punk','edgy'] },
  { id:'track-pants',     name:'Track Pants',        style:'SPORTY',      slot:'bottom',
    colors:{primary:'#00e5ff',secondary:'#1a237e'},  moodEffect:{energy:8},  compatTags:['sporty'] },
  { id:'mom-jeans',       name:'Mom Jeans',          style:'CASUAL',      slot:'bottom',
    colors:{primary:'#8d6e63',secondary:'#efebe9'},  moodEffect:{comfort:7}, compatTags:['casual','nostalgic'] },
  { id:'mini-skirt',      name:'Mini Skirt',         style:'STREETWEAR',  slot:'bottom',
    colors:{primary:'#f50057',secondary:'#1a1a1a'},  moodEffect:{energy:7},  compatTags:['street','bold'] },
  { id:'cargo-pants',     name:'Cargo Pants',        style:'GRUNGE',      slot:'bottom',
    colors:{primary:'#558b2f',secondary:'#33691e'},  moodEffect:{attitude:6}, compatTags:['grunge','utility'] },

  // ── Shoes ─────────────────────────────────────────────────────────
  { id:'heeled-mules',    name:'Heeled Mules',       style:'ELEGANT',     slot:'shoes',
    colors:{primary:'#d7ccc8',secondary:'#bcaaa4'},  moodEffect:{prestige:8}, compatTags:['elegant'] },
  { id:'white-sneakers',  name:'White Sneakers',     style:'STREETWEAR',  slot:'shoes',
    colors:{primary:'#ffffff',secondary:'#e0e0e0'},  moodEffect:{energy:6},  compatTags:['street','clean'] },
  { id:'platform-boots',  name:'Platform Boots',     style:'PUNK',        slot:'shoes',
    colors:{primary:'#212121',secondary:'#f44336'},  moodEffect:{attitude:10}, compatTags:['punk'] },
  { id:'running-shoes',   name:'Running Shoes',      style:'SPORTY',      slot:'shoes',
    colors:{primary:'#e3f2fd',secondary:'#1565c0'},  moodEffect:{energy:8},  compatTags:['sporty'] },
  { id:'loafers',         name:'Loafers',            style:'CASUAL',      slot:'shoes',
    colors:{primary:'#795548',secondary:'#a1887f'},  moodEffect:{comfort:6}, compatTags:['casual'] },
  { id:'ankle-boots',     name:'Ankle Boots',        style:'ROMANTIC',    slot:'shoes',
    colors:{primary:'#bf360c',secondary:'#ffccbc'},  moodEffect:{mood:5},  compatTags:['romantic'] },

  // ── Accessories ───────────────────────────────────────────────────
  { id:'pearl-necklace',  name:'Pearl Necklace',     style:'ELEGANT',     slot:'accessory',
    colors:{primary:'#f5f5f5',secondary:'#e0e0e0'},  moodEffect:{prestige:8, jealousy:5}, compatTags:['elegant'] },
  { id:'chunky-chain',    name:'Chunky Chain',       style:'STREETWEAR',  slot:'accessory',
    colors:{primary:'#bdbdbd',secondary:'#9e9e9e'},  moodEffect:{attitude:6}, compatTags:['street','bold'] },
  { id:'spike-choker',    name:'Spike Choker',       style:'PUNK',        slot:'accessory',
    colors:{primary:'#212121',secondary:'#f44336'},  moodEffect:{attitude:10, intimacy:-3}, compatTags:['punk'] },
  { id:'heart-pendant',   name:'Heart Pendant',      style:'ROMANTIC',    slot:'accessory',
    colors:{primary:'#e91e63',secondary:'#fce4ec'},  moodEffect:{love:10, compat:5}, compatTags:['romantic','love'] },
  { id:'beaded-bracelet', name:'Beaded Bracelet',    style:'CASUAL',      slot:'accessory',
    colors:{primary:'#ff9800',secondary:'#fff3e0'},  moodEffect:{friendship:8}, compatTags:['casual','friendly'] },

  // ── Outerwear ─────────────────────────────────────────────────────
  { id:'trench-coat',     name:'Trench Coat',        style:'ELEGANT',     slot:'outerwear',
    colors:{primary:'#795548',secondary:'#a1887f'},  moodEffect:{prestige:12}, compatTags:['elegant','classic'] },
  { id:'denim-jacket',    name:'Denim Jacket',       style:'GRUNGE',      slot:'outerwear',
    colors:{primary:'#1565c0',secondary:'#90caf9'},  moodEffect:{comfort:6, nostalgia:true}, compatTags:['grunge','casual'] },
  { id:'bomber-jacket',   name:'Bomber Jacket',      style:'STREETWEAR',  slot:'outerwear',
    colors:{primary:'#1b5e20',secondary:'#a5d6a7'},  moodEffect:{energy:8},  compatTags:['street','cool'] },
];

// Additional wardrobe additions
OUTFITS.push(
  // Tops
  { id:'velvet-dress',   name:'Velvet Slip Dress',  style:'ROMANTIC', slot:'top',
    colors:{primary:'#6a1b9a',secondary:'#f3e5f5'}, moodEffect:{mood:12, prestige:4}, compatTags:['romantic','luxury'] },
  { id:'peasant-blouse', name:'Peasant Blouse',     style:'BOHEMIAN', slot:'top',
    colors:{primary:'#fff3e0',secondary:'#ffe0b2'}, moodEffect:{comfort:6}, compatTags:['bohemian','soft'] },
  { id:'cropped-hoodie', name:'Cropped Hoodie',     style:'STREETWEAR', slot:'top',
    colors:{primary:'#263238',secondary:'#ffccbc'}, moodEffect:{energy:9}, compatTags:['street','comfy'] },
  { id:'satin-blouse',   name:'Satin Blouse',       style:'ELEGANT', slot:'top',
    colors:{primary:'#ffd180',secondary:'#fff3e0'}, moodEffect:{prestige:9}, compatTags:['elegant','refined'] },

  // Bottoms
  { id:'bell-bottoms',   name:'Bell Bottoms',       style:'VINTAGE', slot:'bottom',
    colors:{primary:'#5d4037',secondary:'#d7ccc8'}, moodEffect:{nostalgia:6}, compatTags:['vintage','retro'] },
  { id:'pleated-trousers',name:'Pleated Trousers',  style:'ELEGANT', slot:'bottom',
    colors:{primary:'#37474f',secondary:'#cfd8dc'}, moodEffect:{prestige:7}, compatTags:['elegant'] },

  // Shoes
  { id:'chelsea-boots',  name:'Chelsea Boots',      style:'GRUNGE', slot:'shoes',
    colors:{primary:'#3e2723',secondary:'#6d4c41'}, moodEffect:{attitude:6}, compatTags:['grunge','classic'] },
  { id:'espadrilles',    name:'Espadrilles',        style:'CASUAL', slot:'shoes',
    colors:{primary:'#ffcc80',secondary:'#fff8e1'}, moodEffect:{comfort:5}, compatTags:['casual','summer'] },

  // Accessories
  { id:'statement-belt', name:'Statement Belt',     style:'STREETWEAR', slot:'accessory',
    colors:{primary:'#000000',secondary:'#ffd54f'}, moodEffect:{confidence:6}, compatTags:['street','bold'] },
  { id:'beret',          name:'Wool Beret',         style:'ELEGANT', slot:'accessory',
    colors:{primary:'#b71c1c',secondary:'#ffebee'}, moodEffect:{prestige:4}, compatTags:['elegant','french'] },
  { id:'beanie',         name:'Cozy Beanie',        style:'CASUAL', slot:'accessory',
    colors:{primary:'#37474f',secondary:'#cfd8dc'}, moodEffect:{comfort:4}, compatTags:['casual','cozy'] },

  // Outerwear
  { id:'faux-fur-coat',  name:'Faux Fur Coat',      style:'ELEGANT', slot:'outerwear',
    colors:{primary:'#ffebee',secondary:'#f8bbd0'}, moodEffect:{prestige:14}, compatTags:['elegant','glam'] },
  { id:'parka',          name:'Parka',              style:'SPORTY', slot:'outerwear',
    colors:{primary:'#004d40',secondary:'#b2dfdb'}, moodEffect:{comfort:8}, compatTags:['sporty','utility'] },
  { id:'sequin-jacket',  name:'Sequin Jacket',      style:'STREETWEAR', slot:'outerwear',
    colors:{primary:'#ffeb3b',secondary:'#fffde7'}, moodEffect:{energy:12, prestige:5}, compatTags:['street','glam'] }
);

export const OUTFIT_MAP = Object.fromEntries(OUTFITS.map(o=>[o.id,o]));
export const bySlot = slot=>OUTFITS.filter(o=>o.slot===slot);

// Mood effect summary string
export function moodSummary(o){
  return Object.entries(o.moodEffect)
    .filter(([k])=>k!=='nostalgia'&&k!=='jealousy')
    .map(([k,v])=>`${k} ${v>0?'+'+v:v}`).join('  ');
}

