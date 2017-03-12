from operator import attrgetter

class ConceptsList:
    dictionary = {}
    concept_list = []
    sorted = False

    def contains(self, name):
        if name in self.dictionary.keys():
            return True

    def toString(self):
        str = ''
        for (k,v) in self.dictionary.iteritems():
            str += (self.dictionary[k]).toString() + '\n'
        return str

    def toStringShort(self):
        str = ''
        for (index, c) in enumerate(self.concept_list):
            str += c.toStringShort() + '\n'
        return str

    def sortList(self):
        if len(self.concept_list) == 0:
            self.createList()

        self.concept_list = sorted(self.concept_list, key=attrgetter('numberOfShots'), reverse=True)
        self.sorted = True

    def createList(self):
        self.concept_list = [v for v in self.dictionary.values()]

    def copyNConcepts(self, ids):
        listings = ConceptsList()

        listings.dictionary = {}
        for i in ids:
            listings.dictionary[i] = self.dictionary[i]

        listings.sortList()

        return listings
