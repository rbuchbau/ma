class Concept:
    name = ''
    shots = []
    videos = []
    numberOfShots = 0

    def toString(self):
        string = self.name + ' ' + str(len(self.shots)) + ' ' + str(len(self.videos)) + ' '
        for i, s in enumerate(self.shots):
            string += s
            if i < self.numberOfShots-1:
                string += ','

        return string

    def toStringShort(self):
        string = self.name + ' ' + str(len(self.shots)) + ' ' + str(len(self.videos))

        return string