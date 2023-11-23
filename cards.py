class Card:
    def __init__(self, w0, w1, w2, w3):
        self.words = [w0, w1, w2, w3]

    def __getitem__(self, index):
        return self.words[index]
