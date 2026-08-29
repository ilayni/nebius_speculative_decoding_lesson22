Input: 'once upon'
Generating 30 tokens with gamma=5...

========== STEP 1 ==========

--- DRAFT PHASE (small model generates 5 tokens) ---
  Draft token 1: ' a' (id=496, top-prob=0.980)
  Draft token 2: ' time' (id=990, top-prob=0.973)
  Draft token 3: ',' (id=236764, top-prob=0.365)
  Draft token 4: ' in' (id=528, top-prob=0.188)
  Draft token 5: ' the' (id=506, top-prob=0.400)

--- VERIFY PHASE (big model scores all 5 draft tokens in 1 pass) ---
  Position 1: draft=' a' | p=0.977 q=0.980 accept_prob=1.00 | big model preferred=' a'
  Position 2: draft=' time' | p=0.973 q=0.973 accept_prob=1.00 | big model preferred=' time'
  Position 3: draft=',' | p=0.264 q=0.365 accept_prob=0.72 | big model preferred=','
  Position 4: draft=' in' | p=0.087 q=0.061 accept_prob=1.00 | big model preferred=' there'
  Position 5: draft=' the' | p=0.176 q=0.400 accept_prob=0.44 | big model preferred=' a'

--- ACCEPTANCE / REJECTION ---
  Token 1 ' a': ACCEPTED (r=0.78 <= threshold=1.00)
  Token 2 ' time': ACCEPTED (r=0.74 <= threshold=1.00)
  Token 3 ',': ACCEPTED (r=0.04 <= threshold=0.72)
  Token 4 ' in': ACCEPTED (r=0.32 <= threshold=1.00)
  Token 5 ' the': ACCEPTED (r=0.41 <= threshold=0.44)
  All 5 accepted! Bonus token from big model: ' town'

  >> Tokens added this step: [' a', ' time', ',', ' in', ' the', ' town']
========== STEP 2 ==========

--- DRAFT PHASE (small model generates 5 tokens) ---
  Draft token 1: ' of' (id=529, top-prob=0.801)
  Draft token 2: ' K' (id=751, top-prob=0.054)
  Draft token 3: 'ites' (id=3705, top-prob=0.026)
  Draft token 4: ' Ridge' (id=28372, top-prob=0.312)
  Draft token 5: ' in' (id=528, top-prob=0.750)

--- VERIFY PHASE (big model scores all 5 draft tokens in 1 pass) ---
  Position 1: draft=' of' | p=0.926 q=0.801 accept_prob=1.00 | big model preferred=' of'
  Position 2: draft=' K' | p=0.006 q=0.017 accept_prob=0.34 | big model preferred=' the'
  Position 3: draft='ites' | p=0.000 q=0.006 accept_prob=0.08 | big model preferred='.'
  Position 4: draft=' Ridge' | p=0.000 q=0.001 accept_prob=0.10 | big model preferred=','
  Position 5: draft=' in' | p=0.014 q=0.079 accept_prob=0.18 | big model preferred=','

--- ACCEPTANCE / REJECTION ---
  Token 1 ' of': ACCEPTED (r=0.67 <= threshold=1.00)
  Token 2 ' K': REJECTED (r=0.90 > threshold=0.34) → resampled from corrected dist: ' heaven'

  >> Tokens added this step: [' of', ' heaven']
========== STEP 3 ==========

--- DRAFT PHASE (small model generates 5 tokens) ---
  Draft token 1: ',' (id=236764, top-prob=0.664)
  Draft token 2: ' every' (id=1418, top-prob=0.326)
  Draft token 3: ' days' (id=2668, top-prob=0.109)
  Draft token 4: '.' (id=236761, top-prob=0.141)
  Draft token 5: ' So' (id=1593, top-prob=0.211)

--- VERIFY PHASE (big model scores all 5 draft tokens in 1 pass) ---
  Position 1: draft=',' | p=0.668 q=0.664 accept_prob=1.00 | big model preferred=','
  Position 2: draft=' every' | p=0.001 q=0.002 accept_prob=0.37 | big model preferred=' there'
  Position 3: draft=' days' | p=0.000 q=0.000 accept_prob=1.00 | big model preferred=' day'
  Position 4: draft='.' | p=0.002 q=0.011 accept_prob=0.21 | big model preferred=' was'
  Position 5: draft=' So' | p=0.000 q=0.001 accept_prob=0.43 | big model preferred='
'

--- ACCEPTANCE / REJECTION ---
  Token 1 ',': ACCEPTED (r=0.36 <= threshold=1.00)
  Token 2 ' every': ACCEPTED (r=0.25 <= threshold=0.37)
  Token 3 ' days': ACCEPTED (r=0.86 <= threshold=1.00)
  Token 4 '.': REJECTED (r=0.32 > threshold=0.21) → resampled from corrected dist: ' of'

  >> Tokens added this step: [',', ' every', ' days', ' of']
========== STEP 4 ==========

--- DRAFT PHASE (small model generates 5 tokens) ---
  Draft token 1: ' the' (id=506, top-prob=0.609)
  Draft token 2: ' week' (id=2069, top-prob=0.449)
  Draft token 3: ' the' (id=506, top-prob=0.247)
  Draft token 4: ' church' (id=8268, top-prob=0.113)
  Draft token 5: ' was' (id=691, top-prob=0.264)

--- VERIFY PHASE (big model scores all 5 draft tokens in 1 pass) ---
  Position 1: draft=' the' | p=0.730 q=0.609 accept_prob=1.00 | big model preferred=' the'
  Position 2: draft=' week' | p=0.645 q=0.449 accept_prob=1.00 | big model preferred=' week'
  Position 3: draft=' the' | p=0.007 q=0.020 accept_prob=0.34 | big model preferred=' was'
  Position 4: draft=' church' | p=0.013 q=0.017 accept_prob=0.74 | big model preferred=' sun'
  Position 5: draft=' was' | p=0.268 q=0.232 accept_prob=1.00 | big model preferred=' was'

--- ACCEPTANCE / REJECTION ---
  Token 1 ' the': ACCEPTED (r=0.07 <= threshold=1.00)
  Token 2 ' week': ACCEPTED (r=0.22 <= threshold=1.00)
  Token 3 ' the': ACCEPTED (r=0.13 <= threshold=0.34)
  Token 4 ' church': ACCEPTED (r=0.59 <= threshold=0.74)
  Token 5 ' was': ACCEPTED (r=0.54 <= threshold=1.00)
  All 5 accepted! Bonus token from big model: ' founded'

  >> Tokens added this step: [' the', ' week', ' the', ' church', ' was', ' founded']
========== STEP 5 ==========

--- DRAFT PHASE (small model generates 5 tokens) ---
  Draft token 1: ' by' (id=684, top-prob=0.338)
  Draft token 2: ' ' (id=236743, top-prob=0.264)
  Draft token 3: '1' (id=236770, top-prob=0.215)
  Draft token 4: '5' (id=236810, top-prob=0.221)
  Draft token 5: ' families' (id=7965, top-prob=0.305)

--- VERIFY PHASE (big model scores all 5 draft tokens in 1 pass) ---
  Position 1: draft=' by' | p=0.132 q=0.125 accept_prob=1.00 | big model preferred='.'
  Position 2: draft=' ' | p=0.014 q=0.012 accept_prob=1.00 | big model preferred=' the'
  Position 3: draft='1' | p=0.189 q=0.189 accept_prob=1.00 | big model preferred='1'
  Position 4: draft='5' | p=0.047 q=0.056 accept_prob=0.84 | big model preferred='0'
  Position 5: draft=' families' | p=0.008 q=0.009 accept_prob=0.89 | big model preferred='0'

--- ACCEPTANCE / REJECTION ---
  Token 1 ' by': ACCEPTED (r=0.74 <= threshold=1.00)
  Token 2 ' ': ACCEPTED (r=0.86 <= threshold=1.00)
  Token 3 '1': ACCEPTED (r=0.45 <= threshold=1.00)
  Token 4 '5': ACCEPTED (r=0.45 <= threshold=0.84)
  Token 5 ' families': ACCEPTED (r=0.20 <= threshold=0.89)
  All 5 accepted! Bonus token from big model: ','

  >> Tokens added this step: [' by', ' ', '1', '5', ' families', ',']
========== STEP 6 ==========

--- DRAFT PHASE (small model generates 5 tokens) ---
  Draft token 1: ' and' (id=532, top-prob=0.127)
  Draft token 2: ' in' (id=528, top-prob=0.176)
  Draft token 3: ' that' (id=600, top-prob=0.387)
  Draft token 4: ' time' (id=990, top-prob=0.309)
  Draft token 5: ' nothing' (id=5017, top-prob=0.192)

--- VERIFY PHASE (big model scores all 5 draft tokens in 1 pass) ---
  Position 1: draft=' and' | p=0.151 q=0.127 accept_prob=1.00 | big model preferred=' and'
  Position 2: draft=' in' | p=0.037 q=0.035 accept_prob=1.00 | big model preferred=' the'
  Position 3: draft=' that' | p=0.038 q=0.052 accept_prob=0.73 | big model preferred=' the'
  Position 4: draft=' time' | p=0.124 q=0.309 accept_prob=0.40 | big model preferred=' church'
  Position 5: draft=' nothing' | p=0.001 q=0.001 accept_prob=1.00 | big model preferred=','

--- ACCEPTANCE / REJECTION ---
  Token 1 ' and': ACCEPTED (r=0.35 <= threshold=1.00)
  Token 2 ' in': ACCEPTED (r=0.97 <= threshold=1.00)
  Token 3 ' that': REJECTED (r=0.92 > threshold=0.73) → resampled from corrected dist: ' their'

  >> Tokens added this step: [' and', ' in', ' their']
========== STEP 7 ==========

--- DRAFT PHASE (small model generates 3 tokens) ---
  Draft token 1: ' weekly' (id=16789, top-prob=0.047)
  Draft token 2: ' labors' (id=154234, top-prob=0.130)
  Draft token 3: ' they' (id=901, top-prob=0.471)

--- VERIFY PHASE (big model scores all 3 draft tokens in 1 pass) ---
  Position 1: draft=' weekly' | p=0.002 q=0.004 accept_prob=0.39 | big model preferred=' church'
  Position 2: draft=' labors' | p=0.000 q=0.001 accept_prob=0.03 | big model preferred=' meeting'
  Position 3: draft=' they' | p=0.318 q=0.471 accept_prob=0.68 | big model preferred=' they'

--- ACCEPTANCE / REJECTION ---
  Token 1 ' weekly': REJECTED (r=0.48 > threshold=0.39) → resampled from corrected dist: ' communities'

  >> Tokens added this step: [' communities']
========== STEP 8 ==========

--- DRAFT PHASE (small model generates 2 tokens) ---
  Draft token 1: ',' (id=236764, top-prob=0.297)
  Draft token 2: ' where' (id=1298, top-prob=0.120)

--- VERIFY PHASE (big model scores all 2 draft tokens in 1 pass) ---
  Position 1: draft=',' | p=0.243 q=0.297 accept_prob=0.82 | big model preferred=' they'
  Position 2: draft=' where' | p=0.006 q=0.027 accept_prob=0.22 | big model preferred=' they'

--- ACCEPTANCE / REJECTION ---
  Token 1 ',': ACCEPTED (r=0.56 <= threshold=0.82)
  Token 2 ' where': REJECTED (r=0.45 > threshold=0.22) → resampled from corrected dist: ' they'

  >> Tokens added this step: [',', ' they']

========================================
DONE — 30 tokens in 0.85s
Acceptance rate: 77.14%

Generated text:
once upon a time, in the town of heaven, every days of the week the church was founded by 15 families, and in their communities, they

Click to add a cell.