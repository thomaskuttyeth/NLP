
# downloading the wordnet corpus
# nltk.download('wordnet')

'''
Understanding WordNet
=============================================
Wordnet uses the term synset to denote the set of synonyms. Each of synonym has a definition that explains what the set of synonym represents. Synonyms contained within a synset are called lemmas.
'''
# importing the wordnet corpus
from nltk.corpus import wordnet as wn

# retriving all the available synsets of the word car
word = 'car'
car_syns = wn.synsets(word)

# getting the definitions for each synset
for i in range(len(car_syns)):
    print(car_syns[i].definition())
    print()

# get lemmas for the first synset
car_lemma = car_syns[0].lemmas()

'''
### Associations between synsets
======================================
The associations between synsets  are of two categories;
 * is -a relationship
 * is-made-of relationship

For a given synset, there exist two categories of relations;
 * Hypernyms - synsets which carry a general (high-level) meaning of the considered synsets
 * Hyponyms - synsets which carry specific meaning of the considered synsets.
'''

# taking first synset
syn = car_syns[0]
syn.hypernyms()[0].name()

# getting the hyponyms
syn.hyponyms()
hyponyms = [hypo.name() for hypo in syn.hyponyms()]


'''
Holonyms and Metronyms
================================================
Holonyms of a synset are the group of synsets that represents the whole entity of the
 considered synset. For example holonym of tire is the cars synset.
 Metronyms are an is-made-of category and represents the opposite of holonyms
'''

syn = car_syns[2]  # specific subclass
print('syn =', syn)
[holo.name() for holo in syn.part_holonyms()]

'''
Problems with the wordnet
============================================
1. missing nuances. Nuances is subjective. For example, the words want and need have
   similar meanings, but one of them(need) is more assertive.
2. wordnet is subjective in itself
3. issue of maintainning wordnet, which is labor-intensive
4. developing wordnet for other languages can be costly.
'''
