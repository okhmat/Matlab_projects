hw = phased.LinearFMWaveform('PulseWidth',1e-4,'PRF',5e3);
x = step(hw);
hmf = phased.MatchedFilter(...
    'Coefficients',getMatchedFilter(hw));
coeffs = getMatchedFilter(hw);
y = step(hmf,x);
subplot(211),plot(real(x));
xlabel('Samples'); ylabel('Amplitude');
title('Input Signal');
subplot(212),plot(real(y));
xlabel('Samples'); ylabel('Amplitude');
title('Matched Filter Output');

save('matched_filter_variables.mat');