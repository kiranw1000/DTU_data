EEGBASEPATH = './data/DTU/EEG';           % Find EEG files here
WAVBASEPATH = './AUDIO';         % Find AUDIO wav files here 
MATBASEPATH = '.';               % Save preprocessed data files here



for ss = 1:18
    clear data data_noise
    fprintf('Processing subject: %s\n', num2str(ss));
    
    %% Load data
    load(fullfile(EEGBASEPATH,['S' num2str(ss) '.mat']))

    writetable(expinfo,fullfile(EEGBASEPATH,['S' num2str(ss) '.csv']))
end

