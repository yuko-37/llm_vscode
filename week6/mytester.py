import matplotlib.pyplot as plt
import random
import math


class Tester:
    def __init__(self, data, size=250):
        self.data = data
        self.size = min(size, len(self.data))
        self.guesses = []
        self.truths = []
        self.errors = 0
        self.sles = 0
        self.colors = []

    def run(self, i, guess):
        item = self.data[i]
        truth = item.price
        error = abs(truth-guess)
        log_error = math.log(truth+1) - math.log(guess+1)
        sqr_log_error = log_error**2
        
        self.guesses.append(guess)
        self.truths.append(truth)
        self.errors += error
        self.sles += sqr_log_error

        if error < 40 or error/truth < 0.2:
            color_code = 92
            self.colors.append('green')
        elif error < 80 or error/truth < 0.4:
            color_code = 93
            self.colors.append('orange')
        else:
            color_code = 91
            self.colors.append('red')

        title = item.title if len(item.title) < 50 else item.title[:50] + '...'
        text = f"Guess = ${guess:.2f}, Truth = ${truth:.2f}, Error = ${error:.2f} for Item: {title}"
        print(f"\033[{color_code}m{text}\033[0m")


    def chart(self, title):
        plt.figure(figsize=(12, 8))
        plt.title(title)
        max_val = max(max(self.guesses), max(self.truths))
        plt.plot([0, max_val], [0, max_val], color='blue', lw=1, alpha=0.3)
        plt.scatter(self.truths, self.guesses, s=3, c=self.colors)
        plt.xlabel("Ground Truth")
        plt.ylabel("Model Estimate")
        plt.xlim(0, max_val)
        plt.ylim(0, max_val)
        plt.show()
    
    def report(self, name):
        average_error = self.errors / self.size
        RMSLE = math.sqrt(self.sles / self.size)
        hits = sum(1 for color in self.colors if color=='green') / self.size * 100
        title = f"{name}[{len(self.guesses)}] Avr Error = ${average_error:.2f} RMSLE = {RMSLE:.2f} Hits = {hits:.2f}%"
        self.chart(title)

    def clean(self):
        self.guesses = []
        self.truths = []
        self.errors = 0
        self.sles = 0
        self.colors = []

    def test(self, predictor):
        self.clean()
        for i in range(0, self.size):
            test_item = self.data[i]
            guess = predictor(test_item)
            self.run(i, guess)        
        
        name = predictor.__name__.replace("_", " ").title()
        self.report(name)

