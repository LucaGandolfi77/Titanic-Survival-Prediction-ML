/* ===== ALL CREW CHARACTERS ===== */

export const CHARACTERS = {
  carmen: {
    id: 'carmen',
    name: 'Carmen Ruiz',
    role: 'Housekeeping Supervisor',
    age: 32,
    nationality: 'Colombian',
    flag: '🇨🇴',
    emoji: '🧹',
    avatarClass: 'carmen',
    personality: 'Warm, hardworking, secretly loves dancing',
    startingLove: 10,
    unlockDay: 1,
    tier: 2,
    interviewFame: 350,
    preferredTopics: ['passion', 'family', 'port'],
    dislikedTopics: ['secret'],
    dateLocations: ['Crew Deck', 'Laundry Room', "Ship's Theater"],
    dateBg: 'linear-gradient(135deg, #92400e, #b45309)',
    catchphrase: '"Life on this ship? It\'s like a dance — you just have to find the rhythm."',
    meetEvent: 'She knocks on your cabin door with a warm smile and fresh towels.',
    giftDescription: 'VIP cabin access key — unlocks hidden shooting location +400 fame',
    giftFameBonus: 400,
    schedule: { morning: 'Cabins', afternoon: 'Crew Deck', evening: "Ship's Theater" },
    dialogues: {
      first_meet: [
        { text: "Buenos días! I'm Carmen, head of housekeeping. Welcome aboard the Aurora Infinita!", choices: [
          { text: "Nice to meet you! The cabin looks amazing.", effect: 'best', love: 15 },
          { text: "Thanks! I'll try not to make too much of a mess.", effect: 'good', love: 8 },
          { text: "Can I get extra pillows?", effect: 'bad', love: -5 },
        ]},
      ],
      date_low: [
        { text: "You know, most people never notice how much work goes into making everything perfect. But you seem different.", choices: [
          { text: "I've always appreciated the people behind the scenes.", effect: 'best', love: 15 },
          { text: "It must be exhausting work.", effect: 'good', love: 8 },
          { text: "I guess someone has to do it.", effect: 'bad', love: -5 },
        ]},
      ],
      date_mid: [
        { text: "I used to dance salsa competitively back in Bogotá. Sometimes... I miss it.", choices: [
          { text: "Would you teach me? I'd love to learn.", effect: 'best', love: 15 },
          { text: "That sounds like it was wonderful.", effect: 'good', love: 8 },
          { text: "Why did you stop?", effect: 'bad', love: -5 },
        ]},
      ],
      date_high: [
        { text: "*takes your hand* Come to the crew deck tonight. I want to show you something special.", choices: [
          { text: "I'll be there. I wouldn't miss it for anything.", effect: 'best', love: 15 },
          { text: "Sounds mysterious... I'm intrigued.", effect: 'good', love: 8 },
          { text: "Is it safe?", effect: 'bad', love: -5 },
        ]},
      ],
      interview: [
        { text: "An interview? Me? *laughs* I'm just the cleaning lady... but okay, let me tell you about the real heart of this ship.", choices: [
          { text: "What is your favorite memory on this ship?", effect: 'best', love: 15 },
          { text: "How do you handle the busy days?", effect: 'good', love: 8 },
          { text: "Let's just take some quick photos.", effect: 'bad', love: -5 }
        ] },
      ],
      ending: "Under the Mediterranean stars, Carmen takes your hand. 'I've cleaned a thousand cabins,' she whispers, 'but you're the first person who made this ship feel like home.' She pulls you into a slow salsa, the sound of waves keeping time."
    }
  },

  baptiste: {
    id: 'baptiste',
    name: 'Chef Baptiste Lefebvre',
    role: 'Executive Chef',
    age: 45,
    nationality: 'French',
    flag: '🇫🇷',
    emoji: '🍳',
    avatarClass: 'baptiste',
    personality: 'Dramatic, perfectionist, cries when food is wasted',
    startingLove: 5,
    unlockDay: 1,
    tier: 2,
    interviewFame: 350,
    preferredTopics: ['passion', 'proud', 'port'],
    dislikedTopics: ['funny'],
    dateLocations: ['Ship Kitchen', 'Wine Cellar'],
    dateBg: 'linear-gradient(135deg, #1c1917, #292524)',
    catchphrase: '"Food is not sustenance. It is ART. And I am the artist."',
    meetEvent: 'You mention to the waiter that lunch was good. Chef Baptiste bursts through the kitchen doors.',
    giftDescription: 'Gourmet picnic for the team → team happiness +25',
    giftHappinessBonus: 25,
    schedule: { morning: 'Kitchen', afternoon: 'Market Deck', evening: 'Restaurant' },
    dialogues: {
      first_meet: [
        { text: "You said 'good'?! GOOD?! My Bouillabaisse is not 'good'! It is a MASTERPIECE!", choices: [
          { text: "You're right — it was extraordinary. I've never tasted anything like it.", effect: 'best', love: 15 },
          { text: "I'm sorry, I meant it was excellent!", effect: 'good', love: 8 },
          { text: "Calm down, it's just soup.", effect: 'bad', love: -5 },
        ]},
      ],
      date_low: [
        { text: "Tell me... do you cook? Do you understand the sacred bond between chef and flame?", choices: [
          { text: "I'd love to learn from the best. Would you teach me?", effect: 'best', love: 15 },
          { text: "I can make a decent pasta at home.", effect: 'good', love: 8 },
          { text: "I mostly just microwave things.", effect: 'bad', love: -5 },
        ]},
      ],
      date_mid: [
        { text: "*arranges a private tasting in the wine cellar* This Bordeaux... 1998. I've been saving it for someone special.", choices: [
          { text: "I'm honored, Baptiste. This moment is perfect.", effect: 'best', love: 15 },
          { text: "It's delicious. You have incredible taste.", effect: 'good', love: 8 },
          { text: "I usually drink beer.", effect: 'bad', love: -5 },
        ]},
      ],
      date_high: [
        { text: "*tears in his eyes* No one has ever appreciated my art the way you do. You are my muse.", choices: [
          { text: "And you've awakened something in me I didn't know existed.", effect: 'best', love: 15 },
          { text: "Your passion is truly inspiring, Baptiste.", effect: 'good', love: 8 },
          { text: "Don't cry, you'll salt the soup.", effect: 'bad', love: -5 },
        ]},
      ],
      interview: [
        { text: "Finally! Someone who wants to capture the TRUTH of cuisine! Let me show you my kitchen — my kingdom!", choices: [
          { text: "What is your favorite memory on this ship?", effect: 'best', love: 15 },
          { text: "How do you handle the busy days?", effect: 'good', love: 8 },
          { text: "Let's just take some quick photos.", effect: 'bad', love: -5 }
        ] },
      ],
      ending: "Baptiste presents his final dish of the cruise — a dessert shaped like a heart, with your initials in spun sugar. 'In French, we say \"l'amour est dans l'assiette\" — love is on the plate. And you, mon cher, are my greatest creation.'"
    }
  },

  marina: {
    id: 'marina',
    name: 'Marina Costa',
    role: 'Diving Instructor',
    age: 29,
    nationality: 'Brazilian',
    flag: '🇧🇷',
    emoji: '🌊',
    avatarClass: 'marina',
    personality: 'Adventurous, laid-back, speaks in surfing metaphors',
    startingLove: 0,
    unlockDay: 2,
    tier: 2,
    interviewFame: 350,
    preferredTopics: ['ocean', 'passion', 'advice'],
    dislikedTopics: ['family'],
    dateLocations: ['Pool Deck', 'Diving Platform', 'Stargazing Deck'],
    dateBg: 'linear-gradient(135deg, #0ea5e9, #06b6d4)',
    catchphrase: '"Life is like the ocean, bro — you can\'t stop the waves, but you can learn to surf."',
    meetEvent: 'You book a diving session for a content shoot. Marina surfaces with a grin.',
    giftDescription: 'Exclusive underwater photo session → +600 fame',
    giftFameBonus: 600,
    schedule: { morning: 'Pool Deck', afternoon: 'Diving Platform', evening: 'Stargazing Deck' },
    dialogues: {
      first_meet: [
        { text: "Hey! You're the social media person, right? Dude, the underwater world is SO Instagrammable!", choices: [
          { text: "Totally! Would you help me capture some underwater content?", effect: 'best', love: 15 },
          { text: "I'd love to see what's down there.", effect: 'good', love: 8 },
          { text: "I'm actually afraid of deep water.", effect: 'bad', love: -5 },
        ]},
      ],
      date_low: [
        { text: "The best sunsets are from the stargazing deck. Wanna catch one tonight? No pressure, just vibes.", choices: [
          { text: "I'd love nothing more. Lead the way!", effect: 'best', love: 15 },
          { text: "Sounds relaxing. Sure!", effect: 'good', love: 8 },
          { text: "I have work to do...", effect: 'bad', love: -5 },
        ]},
      ],
      date_mid: [
        { text: "*sits beside you on the deck* You know what I love about the ocean? It's honest. No pretending, no games.", choices: [
          { text: "I feel the same way about you. You're the most genuine person I've met.", effect: 'best', love: 15 },
          { text: "That's a beautiful way to see the world.", effect: 'good', love: 8 },
          { text: "Have you ever been stung by a jellyfish?", effect: 'bad', love: -5 },
        ]},
      ],
      date_high: [
        { text: "*watching the sunset* I've surfed every coast in Brazil. But this... this moment right here? This is the best wave I've ever caught.", choices: [
          { text: "*takes her hand* Then let's ride it together.", effect: 'best', love: 15 },
          { text: "I'm so happy you shared this with me.", effect: 'good', love: 8 },
          { text: "Are there sharks around here?", effect: 'bad', love: -5 },
        ]},
      ],
      interview: [
        { text: "An interview? Radical! Let me take you to the reef — that's where the real stories are, under the waves.", choices: [
          { text: "What is your favorite memory on this ship?", effect: 'best', love: 15 },
          { text: "How do you handle the busy days?", effect: 'good', love: 8 },
          { text: "Let's just take some quick photos.", effect: 'bad', love: -5 }
        ] },
      ],
      ending: "At the ship's bow, with the Mediterranean sunset painting the sky in gold and pink, Marina wraps her arms around you. 'You know what they say about the ocean,' she whispers. 'Once it captures your heart, there's no going back.' She kisses you as the waves applaud."
    }
  },

  theo: {
    id: 'theo',
    name: 'Theo Wells',
    role: 'Entertainment Director',
    age: 38,
    nationality: 'British',
    flag: '🇬🇧',
    emoji: '🎭',
    avatarClass: 'theo',
    personality: 'Theatrical, witty, owns 47 bow ties',
    startingLove: 20,
    unlockDay: 1,
    tier: 2,
    interviewFame: 350,
    preferredTopics: ['proud', 'funny', 'passion'],
    dislikedTopics: ['ocean'],
    dateLocations: ['Theater Backstage', 'Karaoke Bar', 'Upper Deck'],
    dateBg: 'linear-gradient(135deg, #1a0a2e, #2d1b69)',
    catchphrase: '"All the world\'s a stage, darling — and this ship is the finest theater of all."',
    meetEvent: 'You attend the evening show. Theo spots you from the stage and winks.',
    giftDescription: 'Stages a viral live show post → +800 fame',
    giftFameBonus: 800,
    schedule: { morning: 'Theater', afternoon: 'Pool Deck Stage', evening: 'Theater' },
    dialogues: {
      first_meet: [
        { text: "Ah, the new social media wizard! Finally, someone who can appreciate the ART of performance! *adjusts bow tie #23*", choices: [
          { text: "Your show was incredible! You're incredibly talented.", effect: 'best', love: 15 },
          { text: "Nice bow tie! Is that one of the famous 47?", effect: 'good', love: 8 },
          { text: "Do people actually watch these shows?", effect: 'bad', love: -5 },
        ]},
      ],
      date_low: [
        { text: "Backstage is where the real magic happens. Want a private tour? I'll even let you try on a bow tie.", choices: [
          { text: "I'd be honored! Show me everything!", effect: 'best', love: 15 },
          { text: "Only if I get to pick which bow tie.", effect: 'good', love: 8 },
          { text: "Is it dusty back there?", effect: 'bad', love: -5 },
        ]},
      ],
      date_mid: [
        { text: "*hands you a script* I wrote a scene for us. Two strangers on a ship, discovering they're not so different after all.", choices: [
          { text: "This is beautiful, Theo. You wrote this for us?", effect: 'best', love: 15 },
          { text: "Let's read it together!", effect: 'good', love: 8 },
          { text: "I'm not a good actor.", effect: 'bad', love: -5 },
        ]},
      ],
      date_high: [
        { text: "*drops the stage persona* Every night I perform for hundreds. But right now... I just want to be Theo. With you.", choices: [
          { text: "I like Theo even more than the performer.", effect: 'best', love: 15 },
          { text: "You don't need a stage to be incredible.", effect: 'good', love: 8 },
          { text: "So which bow tie is your favorite?", effect: 'bad', love: -5 },
        ]},
      ],
      interview: [
        { text: "An interview? DARLING! *strikes a pose* I was BORN for the camera. Let's make this legendary!", choices: [
          { text: "What is your favorite memory on this ship?", effect: 'best', love: 15 },
          { text: "How do you handle the busy days?", effect: 'good', love: 8 },
          { text: "Let's just take some quick photos.", effect: 'bad', love: -5 }
        ] },
      ],
      ending: "In the empty theater, under a single spotlight, Theo takes your hand. 'In my career, I've played a thousand roles,' he says softly. 'But falling for you? That's the one performance I never rehearsed.' He kisses you as the curtain falls."
    }
  },

  isabel: {
    id: 'isabel',
    name: 'Dr. Isabel Santos',
    role: "Ship's Doctor",
    age: 36,
    nationality: 'Portuguese',
    flag: '🇵🇹',
    emoji: '🏥',
    avatarClass: 'isabel',
    personality: 'Calm, intelligent, secretly reads romance novels',
    startingLove: 0,
    unlockDay: 1,
    tier: 3,
    interviewFame: 550,
    preferredTopics: ['passion', 'family', 'advice'],
    dislikedTopics: ['funny'],
    dateLocations: ['Medical Bay', "Captain's Lounge"],
    dateBg: 'linear-gradient(135deg, #ecfdf5, #d1fae5)',
    catchphrase: '"I heal bodies for a living. But some wounds... those need a different kind of medicine."',
    meetEvent: 'During a photo shoot, someone bumps a light stand and you get a small cut. Dr. Santos treats you.',
    giftDescription: 'Certifies team for exclusive health/wellness content → +300 fame per wellness post',
    giftFameBonus: 300,
    schedule: { morning: 'Medical Bay', afternoon: 'Upper Deck (break)', evening: "Captain's Lounge" },
    dialogues: {
      first_meet: [
        { text: "Hold still, let me clean that properly. *pause* You know, for a social media person, you're surprisingly... brave about needles.", choices: [
          { text: "Only because I have the best doctor on the ship.", effect: 'best', love: 15 },
          { text: "Ouch! But thank you for fixing me up.", effect: 'good', love: 8 },
          { text: "Is this going to leave a scar? That's bad for content.", effect: 'bad', love: -5 },
        ]},
      ],
      date_low: [
        { text: "*caught reading a romance novel in the medical bay* Oh! You didn't see that. I was... studying. Medical literature.", choices: [
          { text: "*smiles* Your secret is safe with me. What's it about?", effect: 'best', love: 15 },
          { text: "No judgment! Everyone needs an escape.", effect: 'good', love: 8 },
          { text: "Shouldn't you be working?", effect: 'bad', love: -5 },
        ]},
      ],
      date_mid: [
        { text: "I became a doctor because I wanted to save people. But sometimes... I forget to save time for myself.", choices: [
          { text: "Then let me be the one who reminds you. You deserve it.", effect: 'best', love: 15 },
          { text: "Balance is important. Even for heroes.", effect: 'good', love: 8 },
          { text: "That's pretty workaholic of you.", effect: 'bad', love: -5 },
        ]},
      ],
      date_high: [
        { text: "*looks at stars from the lounge* In medicine, we look for vital signs. But lately... my heart rate elevates around a certain someone.", choices: [
          { text: "I think I might be having the same symptoms, Doctor.", effect: 'best', love: 15 },
          { text: "Isabel... that's the most beautiful diagnosis I've ever heard.", effect: 'good', love: 8 },
          { text: "You should probably get that checked.", effect: 'bad', love: -5 },
        ]},
      ],
      interview: [
        { text: "An interview? I'm not sure... I'm just a doctor. But if you promise to make me look professional, not like some reality TV show...", choices: [
          { text: "What is your favorite memory on this ship?", effect: 'best', love: 15 },
          { text: "How do you handle the busy days?", effect: 'good', love: 8 },
          { text: "Let's just take some quick photos.", effect: 'bad', love: -5 }
        ] },
      ],
      ending: "In the quiet of the medical bay, surrounded by the soft hum of equipment, Isabel takes off her white coat. 'I've read hundreds of romance novels,' she confesses, 'and none of them compare to this.' She places her stethoscope on your chest. 'Just as I suspected — your heart is racing.'"
    }
  },

  james: {
    id: 'james',
    name: 'Lieutenant James Park',
    role: 'Navigation Officer',
    age: 31,
    nationality: 'Korean-American',
    flag: '🇰🇷',
    emoji: '⚓',
    avatarClass: 'james',
    personality: 'Serious on duty, surprisingly funny off duty',
    startingLove: 0,
    unlockDay: 1,
    tier: 3,
    interviewFame: 550,
    preferredTopics: ['ocean', 'proud', 'advice'],
    dislikedTopics: ['funny'],
    dateLocations: ['Navigation Room', 'Ship Bow at Night'],
    dateBg: 'linear-gradient(135deg, #1e3a5f, #0369a1)',
    catchphrase: '"On duty: \'Sir, course correction 2 degrees starboard.\' Off duty: \'What do you call a nervous ship? A WRECK.\'"',
    meetEvent: 'You coordinate with him for a safety course video. His professionalism is intimidating.',
    giftDescription: 'Shares navigation data → allows "open sea" photo shoot → +500 fame',
    giftFameBonus: 500,
    schedule: { morning: 'Bridge', afternoon: 'Navigation Room', evening: 'Ship Bow' },
    dialogues: {
      first_meet: [
        { text: "You need to film a safety course? Fine. But everything by the book. This is a SHIP, not a movie set.", choices: [
          { text: "Absolutely, Lieutenant. Safety first, great content second.", effect: 'best', love: 15 },
          { text: "Don't worry, we'll be professional!", effect: 'good', love: 8 },
          { text: "Can we make it more dramatic? Like, with explosions?", effect: 'bad', love: -5 },
        ]},
      ],
      date_low: [
        { text: "*off duty, wearing a Hawaiian shirt* Hey! Don't look at me like that. Even navigators need time off. Want to hear a joke?", choices: [
          { text: "I LOVE this side of you! Hit me with your best joke!", effect: 'best', love: 15 },
          { text: "Sure, let's hear it!", effect: 'good', love: 8 },
          { text: "Is the ship going to crash while you joke around?", effect: 'bad', love: -5 },
        ]},
      ],
      date_mid: [
        { text: "*at the ship's bow at night* See those stars? I use them to navigate. But lately... I keep looking for one particular star.", choices: [
          { text: "Let me guess... is it the one right next to you?", effect: 'best', love: 15 },
          { text: "Which star? I want to see what you see.", effect: 'good', love: 8 },
          { text: "Don't you have GPS for that?", effect: 'bad', love: -5 },
        ]},
      ],
      date_high: [
        { text: "*serious voice* I have a confession. I don't actually own 47 Hawaiian shirts. It's 48. *laughs, then softens* And I've never met anyone who makes me laugh AND takes me seriously.", choices: [
          { text: "You're incredible, James. Both versions of you.", effect: 'best', love: 15 },
          { text: "I'd surf the open sea with you any day.", effect: 'good', love: 8 },
          { text: "49 would be cooler.", effect: 'bad', love: -5 },
        ]},
      ],
      interview: [
        { text: "An interview? *adjusts collar* Alright, but only about navigation. And maybe ONE joke. Maximum.", choices: [
          { text: "What is your favorite memory on this ship?", effect: 'best', love: 15 },
          { text: "How do you handle the busy days?", effect: 'good', love: 8 },
          { text: "Let's just take some quick photos.", effect: 'bad', love: -5 }
        ] },
      ],
      ending: "On the bridge at midnight, with only the compass lights glowing, James turns to you. 'I've charted courses across every ocean,' he says, his humor gone, replaced by raw honesty. 'But you're the only destination I never want to leave.' He points to a star. 'That one. That's ours now.'"
    }
  },

  luna: {
    id: 'luna',
    name: 'Luna Greco',
    role: 'Live Music Performer',
    age: 26,
    nationality: 'Italian',
    flag: '🇮🇹',
    emoji: '🎵',
    avatarClass: 'luna',
    personality: 'Free-spirited, poetic, always has guitar',
    startingLove: 15,
    unlockDay: 1,
    tier: 2,
    interviewFame: 350,
    preferredTopics: ['passion', 'ocean', 'port'],
    dislikedTopics: ['secret'],
    dateLocations: ['Music Lounge', 'Upper Deck Jam Session'],
    dateBg: 'linear-gradient(135deg, #f59e0b, #fbbf24)',
    catchphrase: '"Every person on this ship is a song waiting to be sung."',
    meetEvent: 'You hear guitar music drifting from the music lounge. Luna plays under soft lights.',
    giftDescription: 'Writes a song about the cruise → +1,200 fame (viral post)',
    giftFameBonus: 1200,
    schedule: { morning: 'Upper Deck', afternoon: 'Music Lounge', evening: 'Music Lounge' },
    dialogues: {
      first_meet: [
        { text: "*strumming softly* Oh! I didn't see you there. I was just writing a song about the moonlight on the water. Want to hear it?", choices: [
          { text: "Please! I'd love to hear it. Music like this deserves an audience.", effect: 'best', love: 15 },
          { text: "Sure, that sounds nice!", effect: 'good', love: 8 },
          { text: "Is there WiFi in here?", effect: 'bad', love: -5 },
        ]},
      ],
      date_low: [
        { text: "Do you ever feel like the universe puts people in the right place at the right time? Like waves meeting the shore?", choices: [
          { text: "I've been feeling that ever since I heard you play.", effect: 'best', love: 15 },
          { text: "That's a beautiful way to see things.", effect: 'good', love: 8 },
          { text: "I think it's just coincidence.", effect: 'bad', love: -5 },
        ]},
      ],
      date_mid: [
        { text: "*hands you a small paper* I wrote lyrics last night. They're about someone new. Someone who makes me feel like a melody I haven't finished yet.", choices: [
          { text: "*reads them* Luna... these are about me, aren't they?", effect: 'best', love: 15 },
          { text: "These are beautiful. You're so talented.", effect: 'good', love: 8 },
          { text: "Do these lyrics rhyme?", effect: 'bad', love: -5 },
        ]},
      ],
      date_high: [
        { text: "*plays a new song, looking directly at you* 'Between the sea and stars, I found your name / A melody that plays and never sounds the same...'", choices: [
          { text: "*moved to tears* That's the most beautiful thing anyone's ever done for me.", effect: 'best', love: 15 },
          { text: "Luna... you're extraordinary.", effect: 'good', love: 8 },
          { text: "Will this be on Spotify?", effect: 'bad', love: -5 },
        ]},
      ],
      interview: [
        { text: "An interview? How about I answer with a song? *laughs* Just kidding. Mostly. Okay, partly a song.", choices: [
          { text: "What is your favorite memory on this ship?", effect: 'best', love: 15 },
          { text: "How do you handle the busy days?", effect: 'good', love: 8 },
          { text: "Let's just take some quick photos.", effect: 'bad', love: -5 }
        ] },
      ],
      ending: "In the music lounge, Luna plays one final song — the one she's been writing all week. It's called 'Infinita,' and every note carries the memory of your time together. As the last chord fades, she sets down her guitar and whispers, 'That song is yours. It always was.'"
    }
  },

  captain: {
    id: 'captain',
    name: 'Captain Dino',
    role: 'Ship Captain',
    age: 50,
    nationality: 'Neapolitan',
    flag: '🇮🇹',
    emoji: '⚓',
    avatarClass: 'captain',
    personality: 'Authoritative, fair, deeply passionate about the sea',
    startingLove: 0,
    unlockDay: 1,
    tier: 4,
    interviewFame: 1500,
    preferredTopics: ['ocean', 'proud', 'advice'],
    dislikedTopics: ['funny', 'secret'],
    dateLocations: ["Captain's Private Deck", 'Bridge (off hours)', 'Formal Dinner'],
    dateBg: 'linear-gradient(135deg, #92400e, #fbbf24)',
    catchphrase: '"The sea does not care about your plans. He demands respect — and rewards it."',
    meetEvent: 'The Captain addresses the crew at the morning briefing. His presence is commanding.',
    giftDescription: 'Full ship access — ALL locations unlocked + exclusive interview = +2,500 fame',
    giftFameBonus: 2500,
    schedule: { morning: 'Bridge', afternoon: "Captain's Quarters", evening: 'Formal Dinner (Day 5+)' },
    dialogues: {
      first_meet: [
        { text: "Ah, the new social media manager. I've heard about you. I expect professionalism, discretion, and content that honors this vessel.", choices: [
          { text: "You have my word, Captain. The Aurora Infinita deserves nothing less.", effect: 'best', love: 15 },
          { text: "I'll do my best to represent the ship well.", effect: 'good', love: 8 },
          { text: "Don't worry, I'll make this ship go viral!", effect: 'bad', love: -5 },
        ]},
      ],
      date_low: [
        { text: "I've captained this ship for 12 years. Every wave, every storm, every sunrise — I remember them all.", choices: [
          { text: "That dedication is inspiring. You must love the sea deeply.", effect: 'best', love: 15 },
          { text: "12 years! That's impressive, Captain.", effect: 'good', love: 8 },
          { text: "Don't you get bored?", effect: 'bad', love: -5 },
        ]},
      ],
      date_mid: [
        { text: "*removes her captain's hat* When we're alone... please, call me Dino. The title gets heavy sometimes.", choices: [
          { text: "Dino. I'm honored that you trust me with that.", effect: 'best', love: 15 },
          { text: "Dino it is. You deserve moments of peace.", effect: 'good', love: 8 },
          { text: "Sure thing, Dino!", effect: 'bad', love: -5 },
        ]},
      ],
      date_high: [
        { text: "*on the private deck at sunset* In 30 years of sailing, I've never anchored my heart. Until this cruise.", choices: [
          { text: "Then let me be your harbor, Dino.", effect: 'best', love: 15 },
          { text: "You deserve love as vast as the sea you command.", effect: 'good', love: 8 },
          { text: "Is this appropriate for a captain?", effect: 'bad', love: -5 },
        ]},
      ],
      interview: [
        { text: "*sits with perfect posture* Very well. If you've earned this interview, then you've earned my trust. Ask your questions.", choices: [
          { text: "What is your favorite memory on this ship?", effect: 'best', love: 15 },
          { text: "How do you handle the busy days?", effect: 'good', love: 8 },
          { text: "Let's just take some quick photos.", effect: 'bad', love: -5 }
        ] },
      ],
      ending: "On the final evening, Captain Dino invites you to the bridge. The Mediterranean stretches endlessly before you. 'I've commanded this ship through storms and calm,' he says, his voice soft for the first time. 'But you — you're the one who showed me that the greatest voyage is the one where you let someone in.' He takes your hand, and together you watch the last sunset of the cruise."
    }
  },

  koen: {
    id: 'koen',
    name: 'Koen van Dijk',
    role: 'Ship Architect',
    age: 34,
    nationality: 'Dutch',
    flag: '🇳🇱',
    emoji: '🪢',
    avatarClass: 'koen',
    personality: 'Funny, flirtatious, enjoys a drink — often falls asleep after one too many.',
    startingLove: 5,
    unlockDay: 1,
    tier: 2,
    interviewFame: 350,
    preferredTopics: ['design', 'funny', 'passion'],
    dislikedTopics: ['serious'],
    dateLocations: ["Promenade Swing", "Ship's Mast", 'Rooftop Bar'],
    dateBg: 'linear-gradient(135deg, #0ea5e9, #f97316)',
    catchphrase: "\"I built that swing. Please don't tell Safety... or maybe bring another drink.\"",
    meetEvent: "You discover him asleep in the new swing, a half-empty bottle tucked at his side.",
    giftDescription: 'Handcrafted swing design — unlocks swing photoshoot +300 fame',
    giftFameBonus: 300,
    schedule: { morning: 'Design Studio', afternoon: 'Promenade', evening: 'Rooftop Bar' },
    dialogues: {
      first_meet: [
        { text: "*murmurs, half-asleep* Huh? Oh—hello. Did you try the swing? It's my best work. Redheads love it.", choices: [
          { text: "You designed this? It's brilliant.", effect: 'best', love: 15 },
          { text: "It's a nice swing... if you stay awake.", effect: 'good', love: 8 },
          { text: "Why does it smell like gin?", effect: 'bad', love: -5 },
        ]},
      ],
      date_low: [
        { text: "Want to test the swing at sunset? I'll push you — responsibly.", choices: [
          { text: "Push away!", effect: 'best', love: 15 },
          { text: "Sure, let's try it.", effect: 'good', love: 8 },
          { text: "I prefer solid ground.", effect: 'bad', love: -5 },
        ]},
      ],
      date_mid: [
        { text: "*laughs* I may drink, but I pour my soul into wood and rope. Also—do you know any redheads?", choices: [
          { text: "Maybe I do... is that a requirement?", effect: 'best', love: 15 },
          { text: "What's wrong with brown hair?", effect: 'good', love: 8 },
          { text: "That's oddly specific.", effect: 'bad', love: -5 },
        ]},
      ],
      date_high: [
        { text: "*quietly* I fell asleep once on this very swing after a few drinks. Woke up to applause—and a redhead on my shoulder.", choices: [
          { text: "Then make more memories with me.", effect: 'best', love: 15 },
          { text: "That's a charming disaster.", effect: 'good', love: 8 },
          { text: "You should drink less.", effect: 'bad', love: -5 },
        ]},
      ],
      interview: [
        { text: "An interview? Sure — ask about load-bearing calculations, the aesthetic arc, and maybe my favorite bar.", choices: [
          { text: "What is your favorite memory on this ship?", effect: 'best', love: 15 },
          { text: "How do you handle the busy days?", effect: 'good', love: 8 },
          { text: "Let's just take some quick photos.", effect: 'bad', love: -5 }
        ] },
      ],
      ending: "Koen rigs a private swing on the promenade. As you sway under the stars, he humbly admits, 'I build things to make people smile. And you—you're the best design I've ever made.'"
    }
  }
};

export function getCharacterById(id) {
  return CHARACTERS[id] || null;
}

export function getAllCharacters() {
  return Object.values(CHARACTERS);
}

export function getUnlockedCharacters(day) {
  return Object.values(CHARACTERS).filter(c => c.unlockDay <= day);
}

export function getDateableCharacters(state) {
  return Object.entries(CHARACTERS)
    .filter(([id, c]) => {
      const cs = state.characters[id];
      return cs && cs.met && cs.love >= 20 && c.unlockDay <= state.day;
    })
    .map(([id, c]) => c);
}

export function getInterviewableCharacters(state) {
  return Object.entries(CHARACTERS)
    .filter(([id, c]) => {
      const cs = state.characters[id];
      if (!cs || !cs.met) return false;
      if (cs.lastInterviewDay === state.day) return false; // Once per day
      if (id === 'captain') return state.day >= 2;
      return true;
    })
    .map(([id, c]) => c);
}

export function getDialogueForLevel(charId, love) {
  const c = CHARACTERS[charId];
  if (!c) return null;
  if (love >= 61) return c.dialogues.date_high;
  if (love >= 41) return c.dialogues.date_mid;
  if (love >= 21) return c.dialogues.date_low;
  return c.dialogues.first_meet;
}
