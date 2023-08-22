'''
다음은 Calculator, UpgradeCalculator, UpgradeCalculator2 클래스이다.
아래 코드를 보고, 예상 출력값을 쓰세요. 총 3문제
'''


class Calculator:
    def __init__(self, val=0):
        self.value = val

    def add(self, val):
        self.value += val

    def minus(self, val):
        self.value -= val


class UpgradeCalculator(Calculator):
    def __init__(self):
        super().__init__()
        self.value = 10

    def add(self, val):
        self.value += 2 * val

    def minus(self, val):
        self.value -= 3 * val


class UpgradeCalculator2(Calculator):
    def add(self, val=100):
        self.value += 4 * val

    def minus(self, val):
        self.value -= 5 * val


cal = UpgradeCalculator()
cal.add(10)
cal.minus(7)

# Question 1
print(cal.value)

cal = UpgradeCalculator2()
cal.add()
cal.add(30)
cal.minus(15)

# Question 2
print(cal.value)

cal = Calculator(15)
cal.add(5)
cal.minus(3)

# Question 3
print(cal.value)

