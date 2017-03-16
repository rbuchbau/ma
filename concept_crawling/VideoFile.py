class VideoFile:
    id = ''
    filename = ''
    source = ''
    filepath = ''

    def toString(self):
        return self.id + ' ' + self.filename + ' ' + self.source + ' ' + self.filepath
