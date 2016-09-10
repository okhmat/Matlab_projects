import scipy.io
import numpy.testing
from scipy import signal
import numpy as np
import matplotlib.pyplot as plt

import unittest


class MatchedFilterTest(unittest.TestCase):
    def test_matched_filter(self):
        in_data = scipy.io.loadmat("matched_filter_variables.mat")

        in_data_FM = scipy.io.loadmat("linear_waveform_param.mat")
        sampleRate = in_data_FM["SampleRate"]
        pulseWidth = in_data_FM["PulseWidth"]
        PRF = in_data_FM["PRF"]
        sweepBandWidth = in_data_FM["SweepBandWidth"]
        y_data_matlab_FM = in_data_FM["wav"]
        y_data_matlab_FM = np.reshape(y_data_matlab_FM, (len(y_data_matlab_FM), ))
        print(y_data_matlab_FM.shape)
        x_data = in_data["x"]
        coeffs = in_data["coeffs"]
        y_data_matlab = in_data["y"]


        megaMatchedFilter = myMatchedFilter(coeffs)     #creation of a prototipe of the class
        y_data_python = megaMatchedFilter.step(x_data)      #application step method to matlab's data

        megaLinearFMWaweform = phasedLinearFMWaveform(sampleRate, pulseWidth, PRF, \
                                                      sweepBandWidth)
        y_data_python_FM = megaLinearFMWaweform.step()
        #plt.plot(np.real(y_data_python_FM))

        #plt.plot(y_data_matlab_FM)
        #plt.plot(y_data_matlab_FM)
        plt.plot(y_data_matlab_FM - y_data_python_FM)
        #print("y data in matlab \n", y_data_matlab)
        #print("y data in python \n", y_data_python)
        print(y_data_matlab.shape)
        print(y_data_python.shape)
        #plt.plot(y_data_python)
        #plt.plot(y_data_matlab)
        plt.show()

        np.testing.assert_allclose(y_data_python, y_data_matlab, atol=1e-12)
        np.testing.assert_allclose(y_data_python_FM, y_data_matlab_FM, atol=1e-10, rtol=1e-3)


class myMatchedFilter:
    def __init__(self, coeffs):
        self.coeffs = coeffs

    def step(self, x):
        conv = signal.convolve(x, self.coeffs, mode = 'full')
        conv = np.resize(conv, (len(x), 1))
        return conv

class phasedLinearFMWaveform:
    def __init__(self, sampleRate, pulseWigth, PRF, sweepBandWidth):
        self.sampleRate = sampleRate
        self.pulseWidth = pulseWigth
        self.PRF = PRF
        self.sweepBandWidth = sweepBandWidth

    def step(self):
        T = []  #T == [10. 0]
        signal = np.array([])
        #array = np.array()
        #shape = signal.shape
        #print(shape)
        endOfCycle = (self.sampleRate)/(self.PRF)
        for i in range(0, int(endOfCycle)):
            T.append(i/self.sampleRate)
            if T[i] < self.pulseWidth:
                signal = np.append(signal, np.exp(np.pi * np.power(T[i], 2) * \
                                                  (self.sweepBandWidth/self.pulseWidth) * 1j))
            else:
                signal = np.append(signal, 0)
            i+1
        #signal = np.reshape(signal, (len(signal), ))
        return signal



if __name__ == '__main__':
    unittest.main()
