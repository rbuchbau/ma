class ConceptMeasurements:
    id = ''
    name = ''

    list_of_true_positives = []
    list_of_all_detected_images = []
    list_of_all_corpus_images = []

    precision = 0
    recall = 0
    f_measure = 0

    def toString(self):
        return self.name + ' ' + str(self.precision) + ' ' + str(self.recall) + ' ' + str(self.f_measure) + ' ' \
               + str(len(self.list_of_true_positives)) + ' ' + str(len(self.list_of_all_detected_images)) + ' ' \
               + str(len(self.list_of_all_corpus_images))

    def calc_precision(self):
        if len(self.list_of_all_detected_images) > 0:
            self.precision = float(len(self.list_of_true_positives)) / len(self.list_of_all_detected_images)
        else:
            self.precision = 0

    def calc_recall(self):
        if len(self.list_of_all_corpus_images) > 0:
            self.recall = float(len(self.list_of_true_positives)) / len(self.list_of_all_corpus_images)
        else:
            self.recall = 0


    def calc_f_measure(self):
        if (self.precision + self.recall) > 0:
            self.f_measure = 2 * float(self.precision * self.recall) / (self.precision + self.recall)
        else:
            self.f_measure = 0
