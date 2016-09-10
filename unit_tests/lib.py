class matchedFilter:
    SampleRate = 1e6
    PulseWight = 50e-6
    FRF = 1e4
    SweepBandwight = 1e5
    NumSamples = 100
    NumPulses = 1

    def __init__(self):
        self.SampleRate = 1e6
        self.PulseWight = 50e-6
        self.FRF = 1e4
        self.SweepBandwight = 1e5
        self.NumSamples = 100
        self.NumPulses = 1

    def set_rate(self, v):
        self.SampleRate = v
    def set_pulse(self, p):
        self.PulseWight = p
    def set_prf(self, prf):
        self.FRF = prf
    def self(self, d):
        self.SweepBandwight = d
    def show(self):
        print("SampleRate", self.SampleRate)

FM1 = matchedFilter()

FM2 = matchedFilter()
print(FM1.SampleRate, FM2.SampleRate)
FM1.SampleRate = 33
print(matchedFilter.SampleRate, matchedFilter.SampleRate)
print(matchedFilter.set_prf)


