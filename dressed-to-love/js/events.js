// js/events.js — 20 random events with trigger weights + resolver
import { state } from './state.js';
import { CHARACTERS, CHAR_MAP } from './characters.js';
import { boostAffinity, damageAffinity, getPair, setPairStatus, getAllPairs } from './relationships.js';
import { rand, randInt, pairKey, pick } from './utils.js';
import { showModal, showToast, renderRelMap } from './ui.js';
import { spawnHeartParticles, setCharAnimation } from './scene3d.js';

// Each event: id, label, statusFilter[], weight, trigger(pair)->bool, resolve(pair,choice)
const EVENTS = [
  {
    id:'fashion_show', label:'Fashion Showdown',
    statusFilter:['friends','best_friends','crush','dating'],
    weight:8,
    trigger: pair => state.day % 5 === 0,
    choices:[
      {text:'👏 Cheer them on', fn:(pair)=>boostAffinity(pair.a,pair.b,12)},
      {text:'👀 Compete fiercely', fn:(pair)=>{ boostAffinity(pair.a,pair.b,4); state.score+=20; }}
    ]
  },
  {
    id:'style_clash', label:'Style Clash!',
    statusFilter:['acquaintances','friends','rivals'],
    weight:6,
    trigger: pair => pair.tension > 30,
    choices:[
      {text:'🤝 Compromise',   fn:(pair)=>{ damageAffinity(pair.a,pair.b,4); boostAffinity(pair.a,pair.b,8); }},
      {text:'🔥 Stand your ground', fn:(pair)=>{ damageAffinity(pair.a,pair.b,10); setPairStatus(pair.a,pair.b,'rivals'); }}
    ]
  },
  {
    id:'secret_admirer', label:'Secret Admirer Note',
    statusFilter:['acquaintances','friends'],
    weight:5,
    trigger: pair => pair.affinity > 40 && pair.status === 'friends',
    choices:[
      {text:'❤️ Respond warmly', fn:(pair)=>{ boostAffinity(pair.a,pair.b,15); setPairStatus(pair.a,pair.b,'crush'); }},
      {text:'🙈 Play it cool',   fn:(pair)=>boostAffinity(pair.a,pair.b,5)}
    ]
  },
  {
    id:'coffee_date', label:'Coffee Invitation',
    statusFilter:['crush','dating'],
    weight:7,
    trigger:()=>true,
    choices:[
      {text:'☕ Accept',    fn:(pair)=>boostAffinity(pair.a,pair.b,10)},
      {text:'🗓️ Raincheck', fn:(pair)=>damageAffinity(pair.a,pair.b,3)}
    ]
  },
  {
    id:'gift_outfit', label:'Surprise Outfit Gift',
    statusFilter:['dating','engaged','married'],
    weight:6,
    trigger:()=>true,
    choices:[
      {text:'🎁 Accept gratefully', fn:(pair)=>{ boostAffinity(pair.a,pair.b,14); state.score+=15; }},
      {text:'☹️ Awkwardly decline', fn:(pair)=>damageAffinity(pair.a,pair.b,6)}
    ]
  },
  {
    id:'runway_collab', label:'Runway Collaboration',
    statusFilter:['friends','best_friends','dating'],
    weight:5,
    trigger: pair => state.day > 5,
    choices:[
      {text:'💃 Join the collab', fn:(pair)=>{ boostAffinity(pair.a,pair.b,12); state.score+=25; }},
      {text:'🙅 Decline',        fn:(pair)=>damageAffinity(pair.a,pair.b,4)}
    ]
  },
  {
    id:'jealousy', label:'Green-Eyed Moment',
    statusFilter:['dating','engaged','married'],
    weight:5,
    trigger: pair => pair.tension > 25,
    choices:[
      {text:'🗣️ Talk it out', fn:(pair)=>{ boostAffinity(pair.a,pair.b,6); pair.tension=Math.max(0,pair.tension-15); }},
      {text:'😡 React badly', fn:(pair)=>{ damageAffinity(pair.a,pair.b,12); pair.tension+=20; }}
    ]
  },
  {
    id:'proposal', label:'Surprise Proposal!',
    statusFilter:['dating'],
    weight:4,
    trigger: pair => pair.affinity >= 75 && pair.daysKnown >= 18,
    choices:[
      {text:'💍 Yes!',                fn:(pair)=>{ setPairStatus(pair.a,pair.b,'engaged'); boostAffinity(pair.a,pair.b,20); state.score+=50; }},
      {text:'💔 Not ready yet', fn:(pair)=>damageAffinity(pair.a,pair.b,8)}
    ]
  },
  {
    id:'argument', label:'Heated Argument',
    statusFilter:['best_friends','dating','engaged','married'],
    weight:5,
    trigger: pair => pair.tension > 40,
    choices:[
      {text:'🤝 Apologize first',  fn:(pair)=>{ boostAffinity(pair.a,pair.b,8); pair.tension=Math.max(0,pair.tension-20); }},
      {text:'🎒 Walk away',       fn:(pair)=>{ damageAffinity(pair.a,pair.b,10); pair.tension+=10; }}
    ]
  },
  {
    id:'twin_outfits', label:'Accidental Twin Outfits',
    statusFilter:['strangers','acquaintances','friends'],
    weight:4,
    trigger:()=>true,
    choices:[
      {text:'😂 Laugh about it', fn:(pair)=>boostAffinity(pair.a,pair.b,9)},
      {text:'🙄 Ignore it',       fn:(pair)=>boostAffinity(pair.a,pair.b,2)}
    ]
  },
  {
    id:'gossip', label:'Runway Gossip',
    statusFilter:['friends','best_friends','rivals','enemies'],
    weight:6,
    trigger:()=>true,
    choices:[
      {text:'🤫 Spread it', fn:(pair)=>{ damageAffinity(pair.a,pair.b,8); state.score+=5; }},
      {text:'🤐 Stay quiet', fn:(pair)=>boostAffinity(pair.a,pair.b,4)}
    ]
  },
  {
    id:'fashion_week', label:'Fashion Week Invite',
    statusFilter:['strangers','acquaintances','friends','dating','engaged','married'],
    weight:3,
    trigger: ()=> state.day % 10 === 0,
    choices:[
      {text:'👗 Attend together', fn:(pair)=>{ boostAffinity(pair.a,pair.b,15); state.score+=30; }},
      {text:'🚪 Go solo',         fn:(pair)=>{ state.score+=15; }}
    ]
  },
  {
    id:'breakup_scare', label:'Almost Breakup',
    statusFilter:['dating','engaged'],
    weight:4,
    trigger: pair => pair.trust < 35 && pair.tension > 50,
    choices:[
      {text:'💗 Fight for it',   fn:(pair)=>{ boostAffinity(pair.a,pair.b,10); pair.tension=Math.max(0,pair.tension-25); pair.trust=Math.min(100,pair.trust+15); }},
      {text:'💔 Let it go',      fn:(pair)=>{ setPairStatus(pair.a,pair.b,'divorced'); damageAffinity(pair.a,pair.b,20); }}
    ]
  },
  {
    id:'reconcile', label:'Chance to Reconcile',
    statusFilter:['rivals','enemies'],
    weight:5,
    trigger: pair => pair.daysKnown > 10,
    choices:[
      {text:'🤝 Make peace',   fn:(pair)=>{ setPairStatus(pair.a,pair.b,'acquaintances'); boostAffinity(pair.a,pair.b,12); }},
      {text:'💪 Hold your ground', fn:(pair)=>damageAffinity(pair.a,pair.b,5)}
    ]
  },
  {
    id:'shared_memory', label:'A Fond Shared Memory',
    statusFilter:['best_friends','dating','engaged','married'],
    weight:7,
    trigger:()=>true,
    choices:[
      {text:'📸 Relive it',    fn:(pair)=>boostAffinity(pair.a,pair.b,11)},
      {text:'🤷 Shrug it off', fn:(pair)=>boostAffinity(pair.a,pair.b,2)}
    ]
  },
  {
    id:'style_compliment', label:'Total Style Compliment',
    statusFilter:['strangers','acquaintances','friends','rivals'],
    weight:8,
    trigger:()=>true,
    choices:[
      {text:'😊 Say thank you', fn:(pair)=>boostAffinity(pair.a,pair.b,7)},
      {text:'🙄 Act suspicious', fn:(pair)=>boostAffinity(pair.a,pair.b,1)}
    ]
  },
  {
    id:'wardrobe_malfunction', label:'Wardrobe Malfunction!',
    statusFilter:['acquaintances','friends','rivals'],
    weight:4,
    trigger:()=>Math.random()<0.3,
    choices:[
      {text:'💕 Help them out', fn:(pair)=>boostAffinity(pair.a,pair.b,14)},
      {text:'😂 Can’t stop laughing', fn:(pair)=>{ damageAffinity(pair.a,pair.b,5); state.score+=8; }}
    ]
  },
  {
    id:'late_night_runway', label:'Late-Night Runway Walk',
    statusFilter:['crush','dating','engaged'],
    weight:5,
    trigger:()=>true,
    choices:[
      {text:'🌟 Join them',  fn:(pair)=>boostAffinity(pair.a,pair.b,13)},
      {text:'🛌 Already in bed', fn:(pair)=>damageAffinity(pair.a,pair.b,3)}
    ]
  },
  {
    id:'anniversary', label:'One Month Anniversary!',
    statusFilter:['dating','engaged','married'],
    weight:4,
    trigger: pair => pair.daysKnown === 30,
    choices:[
      {text:'🎉 Celebrate!', fn:(pair)=>{ boostAffinity(pair.a,pair.b,18); state.score+=40; }},
      {text:'😐 Forgot...',  fn:(pair)=>damageAffinity(pair.a,pair.b,10)}
    ]
  },
  {
    id:'wedding_planning', label:'Wedding Planning Drama',
    statusFilter:['engaged'],
    weight:5,
    trigger:()=>true,
    choices:[
      {text:'💍 Agree on everything', fn:(pair)=>{ boostAffinity(pair.a,pair.b,10); state.score+=20; }},
      {text:'😤 Disagree on venue',   fn:(pair)=>{ damageAffinity(pair.a,pair.b,6); pair.tension+=10; }}
    ]
  }
];

// Pick a random event for a pair based on their status
export function rollEvent(){
  const pairs = getAllPairs();
  if(!pairs.length) return null;
  const pair = pick(pairs.filter(p=>p.daysKnown>0) || pairs);
  if(!pair) return null;
  const eligible = EVENTS.filter(e=>{
    if(!e.statusFilter.includes(pair.status)) return false;
    if(!e.trigger(pair)) return false;
    return true;
  });
  if(!eligible.length) return null;
  // Weighted pick
  const total  = eligible.reduce((s,e)=>s+e.weight, 0);
  let r = Math.random()*total;
  for(const ev of eligible){ r-=ev.weight; if(r<=0) return {event:ev, pair}; }
  return {event:eligible[eligible.length-1], pair};
}

export function resolveEvent(ev, pair, choiceIndex){
  const choice = ev.choices[choiceIndex];
  if(choice && choice.fn) choice.fn(pair);
  pair.sharedEvents++;
  const cA = CHAR_MAP[pair.a], cB = CHAR_MAP[pair.b];
  // Trigger animations
  if(choiceIndex===0){
    setCharAnimation(pair.a,'happy'); setCharAnimation(pair.b,'happy');
    spawnHeartParticles(pair.a);
  } else {
    setCharAnimation(pair.a,'sad'); setCharAnimation(pair.b,'sad');
  }
  setTimeout(()=>{ setCharAnimation(pair.a,'idle'); setCharAnimation(pair.b,'idle'); }, 2500);
}

