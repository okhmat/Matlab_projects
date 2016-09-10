pd = 0.9;            % Probability of detection
pfa = 1e-6;          % Probability of false alarm
max_range = 5000;    % Maximum unambiguous range
tgt_rcs = 1;         % Required target radar cross section
int_pulsenum = 10;   % Number of pulses to integrate

load BasicMonostaticRadarExampleData;

fc = hradiator.OperatingFrequency;  % Operating frequency (Hz)
v = hradiator.PropagationSpeed;     % Wave propagation speed (m/s)
lambda = v/fc;                      % Wavelength (m)
fs = hwav.SampleRate;               % Sampling frequency (Hz)
prf = hwav.PRF;                     % Pulse repetition frequency (Hz)

harray = phased.URA('Element',hant,...
    'Size',[30 30],'ElementSpacing',[lambda/2, lambda/2]);
% Configure the antenna elements such that they only transmit forward
harray.Element.BackBaffled = true;

% Visualize the response pattern.
pattern(harray,fc,'PropagationSpeed',physconst('LightSpeed'),...
    'Type','powerdb');

hradiator.Sensor = harray;
hcollector.Sensor = harray;

% We need to set the WeightsInputPort property to true to enable it to
% accept transmit beamforming weights
hradiator.WeightsInputPort = true;

% Calculate the array gain
hag = phased.ArrayGain('SensorArray',harray,'PropagationSpeed',v);
ag = step(hag,fc,[0;0]);

% Use the radar equation to calculate the peak power
snr_min = albersheim(pd, pfa, int_pulsenum);
peak_power = radareqpow(lambda,max_range,snr_min,hwav.PulseWidth,...
    'RCS',tgt_rcs,'Gain',htx.Gain + ag);

% Set the peak power of the transmitter
htx.PeakPower = peak_power;

initialAz = 45; endAz = -45;
volumnAz = initialAz - endAz;

% Calculate 3-dB beamwidth
theta = radtodeg(sqrt(4*pi/db2pow(ag)));

scanstep = -6;
scangrid = initialAz+scanstep/2:scanstep:endAz;
numscans = length(scangrid);
pulsenum = int_pulsenum*numscans;

% Calculate revisit time
revisitTime = pulsenum/prf;

htarget{1} = phased.RadarTarget(...
    'MeanRCS',1.6,...
    'OperatingFrequency',fc);
htargetplatform{1} = phased.Platform(...
    'InitialPosition',[3532.63; 800; 0],...
    'Velocity',[-100; 50; 0]);

% Calculate the range, angle, and speed of the first target
[tgt1_rng,tgt1_ang] = rangeangle(htargetplatform{1}.InitialPosition,...
        hantplatform.InitialPosition);
tgt1_speed = radialspeed(htargetplatform{1}.InitialPosition,...
        htargetplatform{1}.Velocity,hantplatform.InitialPosition);

htarget{2} = phased.RadarTarget(...
    'MeanRCS',1.2,...
    'OperatingFrequency',fc);

htargetplatform{2} = phased.Platform(...
    'InitialPosition',[2020.66; 0; 0],...
    'Velocity',[60; 80; 0]);

% Calculate the range, angle, and speed of the second target
[tgt2_rng,tgt2_ang] = rangeangle(htargetplatform{2}.InitialPosition,...
        hantplatform.InitialPosition);
tgt2_speed = radialspeed(htargetplatform{2}.InitialPosition,...
        htargetplatform{2}.Velocity,hantplatform.InitialPosition);

numtargets = length(htarget);




% Create the steering vector for transmit beamforming
hsv = phased.SteeringVector('SensorArray',harray,'PropagationSpeed',v);

% Create the receiving beamformer
hbf = phased.PhaseShiftBeamformer('SensorArray',harray,...
    'OperatingFrequency',fc,'PropagationSpeed',v,...
    'DirectionSource','Input port');

% Define propagation channel for each target
for n = numtargets:-1:1
    htargetchannel{n} = phased.FreeSpace(...
        'SampleRate',fs,...
        'TwoWayPropagation',true,...
        'OperatingFrequency',fc);
end

fast_time_grid = unigrid(0, 1/fs, 1/prf, '[)');
rx_pulses = zeros(numel(fast_time_grid),pulsenum);  % Pre-allocate
tgt_ang = zeros(2,numtargets);                      % Target angle

for m = 1:pulsenum

    x = step(hwav);                              % Generate pulse
    [s, tx_status] = step(htx,x);                % Transmit pulse
    [ant_pos,ant_vel] = step(hantplatform,1/prf);% Update antenna position

    % Calculate the steering vector
    scanid = floor((m-1)/int_pulsenum) + 1;
    sv = step(hsv,fc,scangrid(scanid));
    w = conj(sv);

    rsig = zeros(length(s),numtargets);
    for n = numtargets:-1:1                      % For each target
        [tgt_pos,tgt_vel] = step(...
            htargetplatform{n},1/prf);           % Update target position
        [~,tgt_ang(:,n)] = rangeangle(tgt_pos,...% Calculate range/angle
            ant_pos);
        tsig = step(hradiator,s,tgt_ang(:,n),w); % Radiate toward target
        tsig = step(htargetchannel{n},...        % Propagate pulse
            tsig,ant_pos,tgt_pos,ant_vel,tgt_vel);
        rsig(:,n) = step(htarget{n},tsig);       % Reflect off target
    end
    rsig = step(hcollector,rsig,tgt_ang);        % Collect all echoes
    rsig = step(hrx,rsig,~(tx_status>0));        % Receive signal
    rsig = step(hbf,rsig,[scangrid(scanid);0]);  % Beamforming
    rx_pulses(:,m) = rsig;                       % Form data matrix

end





% Matched filtering
matchingcoeff = getMatchedFilter(hwav);
hmf = phased.MatchedFilter(...
    'Coefficients',matchingcoeff,...
    'GainOutputPort',true);
[mf_pulses, mfgain] = step(hmf,rx_pulses);
mf_pulses = reshape(mf_pulses,[],int_pulsenum,numscans);

matchingdelay = size(matchingcoeff,1)-1;
sz_mfpulses = size(mf_pulses);
mf_pulses = [mf_pulses(matchingdelay+1:end) zeros(1,matchingdelay)];
mf_pulses = reshape(mf_pulses,sz_mfpulses);

% Pulse integration
int_pulses = pulsint(mf_pulses,'noncoherent');
int_pulses = squeeze(int_pulses);

% Visualize
r = v*fast_time_grid/2;
X = r'*cosd(scangrid); Y = r'*sind(scangrid);
clf;
pcolor(X,Y,pow2db(abs(int_pulses).^2));
axis equal tight
shading interp
axis off
text(-800,0,'Array');
text((max(r)+10)*cosd(initialAz),(max(r)+10)*sind(initialAz),...
    [num2str(initialAz) '^o']);
text((max(r)+10)*cosd(endAz),(max(r)+10)*sind(endAz),...
    [num2str(endAz) '^o']);
text((max(r)+10)*cosd(0),(max(r)+10)*sind(0),[num2str(0) '^o']);
colorbar;

range_gates = v*fast_time_grid/2;
htvg = phased.TimeVaryingGain(...
    'RangeLoss',2*fspl(range_gates,lambda),...
    'ReferenceLoss',2*fspl(max(range_gates),lambda));
tvg_pulses = step(htvg,mf_pulses);

% Pulse integration
int_pulses = pulsint(tvg_pulses,'noncoherent');
int_pulses = squeeze(int_pulses);

% Calculate the detection threshold

% sample rate is twice the noise bandwidth in the design system
noise_bw = hrx.SampleRate/2;
npower = noisepow(noise_bw,...
    hrx.NoiseFigure,hrx.ReferenceTemperature);
threshold = npower * db2pow(npwgnthresh(pfa,int_pulsenum,'noncoherent'));
% Increase the threshold by the matched filter processing gain
threshold = threshold * db2pow(mfgain);

N = 51;
clf;
surf(X(N:end,:),Y(N:end,:),...
    pow2db(abs(int_pulses(N:end,:)).^2));
hold on;
mesh(X(N:end,:),Y(N:end,:),...
    pow2db(threshold*ones(size(X(N:end,:)))),'FaceAlpha',0.8);
view(0,56);
axis off

for m = 2:-1:1
    [p, f] = periodogram(mf_pulses(I(m),:,J(m)),[],256,prf, ...
                'power','centered');
    speed_vec = dop2speed(f,lambda)/2;

    spectrum_data = p/max(p);
    [~,dop_detect1] = findpeaks(pow2db(spectrum_data),'MinPeakHeight',-5);
    sp(m) = speed_vec(dop_detect1);
end