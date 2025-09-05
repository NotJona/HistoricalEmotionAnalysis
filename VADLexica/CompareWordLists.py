"""
Let us compare the different VAD Lexica.
As mentioned in Chapter 3, the words (not VAD scores) of Anew are included in Warriner and the words of Warriner in NRC_VAD.
Is that really the case?

Nope! ANEW has 1034 words in it, 1033 are included in Warriner and 1024 in NRC_VAD
Warriner has 13,915 words in it, 13,855 are included in NRC_VAD
NRC_VAD has 19,971 words in it - though the paper said 20,007???

Next, let us see how big the overlap between the COHA words and the VAD Lexica words is.
COHA (filtered) has 21,784 unique lemmata. Overlap with:
ANEW: 862
Warriner: 8519
NRC_VAD: 11003
okay, now we know the maximum number of seed-words we can increase to.

What about the goldstandard words?
There are 100 GoldStandard words (English). All of them are in the filtered COHA. However, 18 of them are in the ANEW,
and 97 in both Warriner and the NRC_VAD. These should be excluded as seed words. Cause otherwise they would be assigned
a contemporary VAD value and what is the use of that?
So words usable as seed-words in the VAD lexica:
ANEW: 844
Warriner: 8422
NRC_VAD: 10906
"""

ANEW = []
Warriner = []
NRC_VAD = []
Lemmata_in_COHA = []

with open('ANEW_refined') as f:
    for line in f:
        ANEW.append(line.split('\t')[0])
with open('Warriner_refined') as f:
    for line in f:
        Warriner.append(line.split('\t')[0])
with open('NRC_VAD_refined') as f:
    for line in f:
        NRC_VAD.append(line.split('\t')[0])
with open('AAA_word_list.txt') as f:
    for line in f:
        Lemmata_in_COHA.append(line.strip())

print(f'Number of words in ANEW {len(ANEW)}')
print(f'Number of words in Warriner {len(Warriner)}')
print(f'Number of words in NRC_VAD {len(NRC_VAD)}')
print(f'Number of words from ANEW that are in Warriner {len([word for word in ANEW if word in Warriner])}')
print(f'Number of words from ANEW that are in NRC_VAD {len([word for word in ANEW if word in NRC_VAD])}')
print(f'Number of words from Warriner that are in NRC_VAD {len([word for word in Warriner if word in NRC_VAD])}')

print(f'Number of lemmata in the (filtered) COHA {len(Lemmata_in_COHA)}')
print(f'Number of words from COHA that are in ANEW {len([word for word in Lemmata_in_COHA if word in ANEW])}')
print(f'Number of words from COHA that are in Warriner {len([word for word in Lemmata_in_COHA if word in Warriner])}')
print(f'Number of words from COHA that are in NRC_VAD {len([word for word in Lemmata_in_COHA if word in NRC_VAD])}')

GOLDSTANDARD = []
with open('goldEN.vad') as f:
    for line in f:
        GOLDSTANDARD.append(line.split('\t')[0])
print(f'Number of words from GoldStandard that are in COHA {len([word for word in GOLDSTANDARD if word in Lemmata_in_COHA])}')
print(f'Number of words from GoldStandard that are in ANEW {len([word for word in GOLDSTANDARD if word in ANEW])}')
print(f'Number of words from GoldStandard that are in Warriner {len([word for word in GOLDSTANDARD if word in Warriner])}')
print(f'Number of words from GoldStandard that are in NRC_VAD {len([word for word in GOLDSTANDARD if word in NRC_VAD])}')
print(f'Number of words in GoldStandard {len(GOLDSTANDARD)}')

ANEW_COHA_noGOLD = []
for word in ANEW:
    if word in Lemmata_in_COHA:
        if word not in GOLDSTANDARD:
            ANEW_COHA_noGOLD.append(word)
Warriner_COHA_noGOLD = []
for word in Warriner:
    if word in Lemmata_in_COHA:
        if word not in GOLDSTANDARD:
            Warriner_COHA_noGOLD.append(word)
NRC_COHA_noGOLD = []
for word in NRC_VAD:
    if word in Lemmata_in_COHA:
        if word not in GOLDSTANDARD:
            NRC_COHA_noGOLD.append(word)
for word in GOLDSTANDARD:
    if word in NRC_COHA_noGOLD:
        print(word)
print(f'Overlap ANEW and COHA minus Goldstandard words {len(ANEW_COHA_noGOLD)}')
print(f'Overlap Warriner and COHA minus Goldstandard words {len(Warriner_COHA_noGOLD)}')
print(f'Overlap NRCVAD and COHA minus Goldstandard words {len(NRC_COHA_noGOLD)}')
