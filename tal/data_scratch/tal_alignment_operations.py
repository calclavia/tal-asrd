# Extend episode 683 by 0.5 starting with ix 16
def extend_utterance(utts: dict, ix: int, amount: float):
    # Extend the given episode
    utts[ix]['utterance_end'] += amount
    utts[ix]['duration'] += amount
    # Move every other episode afterwads
    for i in range(ix + 1, len(utts)):
        utts[i]['utterance_start'] += amount
        utts[i]['utterance_end'] += amount
    return utts

def truncate_utterance(utts: dict, ix: int, amount: float):
    # Extend the given episode
    utts[ix]['utterance_end'] -= amount
    utts[ix]['duration'] -= amount
    # Move every other episode afterwads
    for i in range(ix + 1, len(utts)):
        utts[i]['utterance_start'] -= amount
        utts[i]['utterance_end'] -= amount
    return utts

def push_utterance(utts: dict, ix: int, amount: float):
    # Extend the given episode
    utts[ix]['utterance_start'] += amount
    utts[ix]['utterance_end'] += amount
    # Move every other episode afterwads
    for i in range(ix + 1, len(utts)):
        utts[i]['utterance_start'] += amount
        utts[i]['utterance_end'] += amount
    return utts


# Valid: 441
u_by_e['ep-441'] = push_utterance(u_by_e['ep-441'], 0, -41.0)
u_by_e['ep-441'] = extend_utterance(u_by_e['ep-441'], 90, 0.62)
u_by_e['ep-441'] = push_utterance(u_by_e['ep-441'], 183, -39.0)
u_by_e['ep-441'] = push_utterance(u_by_e['ep-441'], 276, -14.0)

# Test: 416 - Gets mucked up around foreign language translation
u_by_e['ep-416'] = extend_utterance(u_by_e['ep-416'], 196, 4.5)
u_by_e['ep-416'] = extend_utterance(u_by_e['ep-416'], 197, 1.2)
u_by_e['ep-416'] = extend_utterance(u_by_e['ep-416'], 198, 1.5)
u_by_e['ep-416'] = extend_utterance(u_by_e['ep-416'], 200, 4.0)
u_by_e['ep-416'] = extend_utterance(u_by_e['ep-416'], 202, 2.0)
u_by_e['ep-416'] = extend_utterance(u_by_e['ep-416'], 204, 2.0)
u_by_e['ep-416'] = extend_utterance(u_by_e['ep-416'], 206, 0.5)
u_by_e['ep-416'] = truncate_utterance(u_by_e['ep-416'], 207, 0.5)

# Test: ep-683
u_by_e['ep-683'] = extend_utterance(u_by_e['ep-683'], 16, 0.5)
u_by_e['ep-683'] = truncate_utterance(u_by_e['ep-683'], 149, 1.5)
u_by_e['ep-683'] = extend_utterance(u_by_e['ep-683'], 157, 2.15)
u_by_e['ep-683'] = truncate_utterance(u_by_e['ep-683'], 159, 0.25)
u_by_e['ep-683'] = truncate_utterance(u_by_e['ep-683'], 163, 0.5)
u_by_e['ep-683'] = extend_utterance(u_by_e['ep-683'], 215, 0.25)
u_by_e['ep-683'] = extend_utterance(u_by_e['ep-683'], 245, 1.5)

# Test: ep-403
u_by_e['ep-403'] = extend_utterance(u_by_e['ep-403'], 100, 0.5)
u_by_e['ep-403'][122]['utterance'] = "Yeah. And it was really exciting. What got me was the fact that they had a cross bolt, and they stopped the line to repair it. That's Rick Madrid, the worker who brought a thermos full of vodka and OJ to work every day at GM. What he saw in Japan was a kind of a bolt, a cross bolt that they put in wrong. And they stopped the line, and repaired it. Which is take the bolt off, ream the hole, put the bolt back in, instead of sending it on and putting all the other junk on top of it, so you have to take it off and repair it. And whoever puts it back isn't skilled in putting trim back, so they're going to mess that up."
u_by_e['ep-403'] = extend_utterance(u_by_e['ep-403'], 122, 13.2)
u_by_e['ep-403'] = truncate_utterance(u_by_e['ep-403'], 145, 0.5)
u_by_e['ep-403'] = extend_utterance(u_by_e['ep-403'], 164, 0.5)
u_by_e['ep-403'] = extend_utterance(u_by_e['ep-403'], 212, 0.5)
u_by_e['ep-403'] = extend_utterance(u_by_e['ep-403'], 213, 1.0)
u_by_e['ep-403'] = extend_utterance(u_by_e['ep-403'], 215, 1.0)
u_by_e['ep-403'] = extend_utterance(u_by_e['ep-403'], 217, 1.0)
u_by_e['ep-403'] = extend_utterance(u_by_e['ep-403'], 220, 0.5)
u_by_e['ep-403'] = truncate_utterance(u_by_e['ep-403'], 230, 0.755)
u_by_e['ep-403'] = extend_utterance(u_by_e['ep-403'], 234, 1.5)
u_by_e['ep-403'] = extend_utterance(u_by_e['ep-403'], 240, 1.0)
u_by_e['ep-403'] = extend_utterance(u_by_e['ep-403'], 265, 1.0)
u_by_e['ep-403'] = truncate_utterance(u_by_e['ep-403'], 277, 31.0)
u_by_e['ep-403'] = push_utterance(u_by_e['ep-403'], 278, 34.0)
u_by_e['ep-403'] = extend_utterance(u_by_e['ep-403'], 278, 1.0)
u_by_e['ep-403'][279]['utterance'] = "Our website, thisamericanlife.org, where our online store is now back up in operation, and where you can find the new update of our iPhone app. This American Life is distributed by Public Radio International. Support for This American Life comes from Kohler. Dedicated to helping people save water without sacrificing performance and great design. For more information on how you can save water and support habitat for humanity visit savewateramerica.com. WBEZ management oversight for our program by our boss, Mr. Torey Malatia. I overheard him in the hallway telling someone how surprised he was at the quality of our shows this year."
u_by_e['ep-403'] = extend_utterance(u_by_e['ep-403'], 279, 10.0)
u_by_e['ep-403'] = truncate_utterance(u_by_e['ep-403'], 281, 70.0)

# Valid: 559
u_by_e['ep-559'] = truncate_utterance(u_by_e['ep-559'], 0, 1.0)
u_by_e['ep-559'] = extend_utterance(u_by_e['ep-559'], 1, 0.6)
u_by_e['ep-559'] = push_utterance(u_by_e['ep-559'], 6, 1.5)
u_by_e['ep-559'] = extend_utterance(u_by_e['ep-559'], 32, 0.5)
u_by_e['ep-559'] = truncate_utterance(u_by_e['ep-559'], 124, 2.5)
u_by_e['ep-559'] = extend_utterance(u_by_e['ep-559'], 126, 0.3)
u_by_e['ep-559'] = truncate_utterance(u_by_e['ep-559'], 128, 10.0)
u_by_e['ep-559'] = push_utterance(u_by_e['ep-559'], 129, 10.0)
u_by_e['ep-559'] = extend_utterance(u_by_e['ep-559'], 141, 1.5)
u_by_e['ep-559'] = extend_utterance(u_by_e['ep-559'], 144, 1.0)
u_by_e['ep-559'][145]['utterance'] = "The girl guide in that story, Mary Crevity, has spent twenty years tracking down the soldiers who rescued her, and she's meeting the final one on her list in China next week."
u_by_e['ep-559'] = truncate_utterance(u_by_e['ep-559'], 145, 6.0)
u_by_e['ep-559'] = push_utterance(u_by_e['ep-559'], 147, 0.5)
u_by_e['ep-559'] = extend_utterance(u_by_e['ep-559'], 148, 0.5)
u_by_e['ep-559'] = extend_utterance(u_by_e['ep-559'], 199, 0.5)
u_by_e['ep-559'] = truncate_utterance(u_by_e['ep-559'], 253, 9.0)
u_by_e['ep-559'] = truncate_utterance(u_by_e['ep-559'], 264, 15.0)
u_by_e['ep-559'] = push_utterance(u_by_e['ep-559'], 265, 10.0)
u_by_e['ep-559'] = truncate_utterance(u_by_e['ep-559'], 280, 31.5)
u_by_e['ep-559'] = push_utterance(u_by_e['ep-559'], 281, 61.5)
u_by_e['ep-559'] = extend_utterance(u_by_e['ep-559'], 283, 1.0)

# TRAIN: ep-695
u_by_e['ep-695'] = push_utterance(u_by_e['ep-695'], 0, 158)
u_by_e['ep-695'] = truncate_utterance(u_by_e['ep-695'], 0, 9.0)
u_by_e['ep-695'] = push_utterance(u_by_e['ep-695'], 75, 80.0)
u_by_e['ep-695'] = truncate_utterance(u_by_e['ep-695'], 75, 3.0)
u_by_e['ep-695'] = push_utterance(u_by_e['ep-695'], 149, 28.5)

# TRAIN: ep-700
u_by_e['ep-700'] = push_utterance(u_by_e['ep-700'], 0, 35.5)
u_by_e['ep-700'] = push_utterance(u_by_e['ep-700'], 93, 65.0)

# TRAIN: ep-698
u_by_e['ep-698'] = push_utterance(u_by_e['ep-698'], 0, 48.0)
u_by_e['ep-698'] = truncate_utterance(u_by_e['ep-698'], 0, 2.0)
u_by_e['ep-698'] = extend_utterance(u_by_e['ep-698'], 13, 5.06)
u_by_e['ep-698'] = truncate_utterance(u_by_e['ep-698'], 15, 0.5)
u_by_e['ep-698'] = truncate_utterance(u_by_e['ep-698'], 16, 1.8)
u_by_e['ep-698'] = truncate_utterance(u_by_e['ep-698'], 17, 1.8)
u_by_e['ep-698'] = truncate_utterance(u_by_e['ep-698'], 18, 1.0)
u_by_e['ep-698'] = push_utterance(u_by_e['ep-698'], 150, 0.5)
u_by_e['ep-698'] = extend_utterance(u_by_e['ep-698'], 150, 2.0)
u_by_e['ep-698'] = truncate_utterance(u_by_e['ep-698'], 185, 3.0)
u_by_e['ep-698'] = push_utterance(u_by_e['ep-698'], 189, 82)
u_by_e['ep-698'] = truncate_utterance(u_by_e['ep-698'], 189, 3.5)

# TRAIN: ep-696
u_by_e['ep-696'] = push_utterance(u_by_e['ep-696'], 0, 35.0)
u_by_e['ep-696'] = push_utterance(u_by_e['ep-696'], 193, 65.0)
u_by_e['ep-696'] = push_utterance(u_by_e['ep-696'], 445, 18.0)

# TRAIN: ep-520
u_by_e['ep-520'] = push_utterance(u_by_e['ep-520'], 0, -7.0)
u_by_e['ep-520'][14]['utterance'] = 'You have to remember, at the time, Edmonton was it. They had the Edmonton Oilers, the most famous hockey team on the planet-- Wayne Gretzky. They won four Stanley Cups.'
u_by_e['ep-520'] = truncate_utterance(u_by_e['ep-520'], 14, 13.5)
u_by_e['ep-520'] = truncate_utterance(u_by_e['ep-520'], 40, 2.5)

# TRAIN: ep-577
u_by_e['ep-577'] = push_utterance(u_by_e['ep-577'], 0, -38.0)
u_by_e['ep-577'] = u_by_e['ep-577'][:179] + u_by_e['ep-577'][182:]
u_by_e['ep-577'] = push_utterance(u_by_e['ep-577'], 179, -71.5)
u_by_e['ep-577'] = truncate_utterance(u_by_e['ep-577'], 179, 1.0)

# TRAIN: ep-692
u_by_e['ep-692'] = push_utterance(u_by_e['ep-692'], 0, 45.0)
u_by_e['ep-692'] = push_utterance(u_by_e['ep-692'], 217, 78.0)

# TRAIN: ep-697
u_by_e['ep-697'] = push_utterance(u_by_e['ep-697'], 0, 35.5)
u_by_e['ep-697'] = push_utterance(u_by_e['ep-697'], 259, 65.0)

# TRAIN: ep-447
u_by_e['ep-447'] = push_utterance(u_by_e['ep-447'], 15, 6.5)
u_by_e['ep-447'] = extend_utterance(u_by_e['ep-447'], 42, 1.5)
u_by_e['ep-447'] = push_utterance(u_by_e['ep-447'], 66, 5.75)
u_by_e['ep-447'] = push_utterance(u_by_e['ep-447'], 69, 3.0)

# TRAIN: ep-699
u_by_e['ep-699'] = push_utterance(u_by_e['ep-699'], 0, 45.0)
u_by_e['ep-699'] = push_utterance(u_by_e['ep-699'], 183, 77.0)

# TRAIN: ep-687
u_by_e['ep-687'] = push_utterance(u_by_e['ep-687'], 170, -2.0)
u_by_e['ep-687'] = truncate_utterance(u_by_e['ep-687'], 170, 0.5)

# TRAIN: ep-556
u_by_e['ep-556'] = push_utterance(u_by_e['ep-556'], 25, 1.0)
u_by_e['ep-556'] = push_utterance(u_by_e['ep-556'], 50, 0.8)

