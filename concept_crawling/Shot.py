class Shot:
    name = ''
    timestamp = ''
    video = ''

    def __init__(self, name, timestamp, isNew):
        self.name = name
        self.video = name.split('shot')[1].split('_')[0]
        if isNew:
            ts = ''
            splits = timestamp.split(':')
            splits[0] = splits[0][1:]
            for i, s in enumerate(splits):
                if i == len(splits)-1:
                    part = s.split('F')[0]
                    ts += '.' + part
                else:
                    ts += ':' + s

            self.timestamp = ts[1:]
        else:
            self.timestamp = timestamp

    def toString(self):
        return self.name + ' ' + self.timestamp