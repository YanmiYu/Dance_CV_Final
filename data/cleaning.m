%% AIST++ Video Filter — Universal Script
% Configure the filter parameters at the top, then just run.
% Filters AIST++ videos by dance genre, situation, camera, dancer, music, and choreography.
% Date: 2026-04-21

clear; clc;   % Clear workspace and command window

%% ============================================================
%  USER CONFIGURATION — only edit this block
%  Set any field to "" (empty string) to skip that filter
%  ============================================================

% --- File paths ---
dataFolder = '/Users/dsteven/Desktop/Dance_CV_Final/data';
inputFile  = 'refined_2M_sBM_url.xlsx';   % Can be .xlsx or .csv

% --- Filter criteria (leave "" to ignore that filter) ---
filter.genre        = "gBR";   % Dance Genre:  gBR/gPO/gLO/gMH/gLH/gHO/gWA/gKR/gJS/gJB  ("" = any)
filter.situation    = "sBM";   % Situation:    sBM (basic) / sFM (advanced) / sMM (moving cam) etc.  ("" = any)
filter.camera       = "c09";   % Camera ID:    c01–c09                                  ("" = any)
filter.dancer       = "";      % Dancer ID:    e.g. "d04"                               ("" = any)
filter.music        = "";      % Music ID:     e.g. "mBR0"                              ("" = any)
filter.choreography = "";      % Choreo ID:    e.g. "ch01"                              ("" = any)

% --- Output options ---
saveFilteredTable = true;      % Save a csv with all matched rows
saveUrlList       = true;      % Save a txt with just the URLs (for wget/curl)

%% ============================================================
%  END OF USER CONFIGURATION — no need to edit below
%  ============================================================

%% 1. Build input/output paths
inputPath = fullfile(dataFolder, inputFile);

% Auto-generate output filename from active filter values
% e.g. filtered_gBR_sBM_c09.csv
tags = [filter.genre, filter.situation, filter.camera, ...
        filter.dancer, filter.music, filter.choreography];
tags = tags(strlength(tags) > 0);   % Drop empty tags
if isempty(tags)
    tagStr = "all";                 % No filters means "all"
else
    tagStr = strjoin(tags, "_");    % Join active tags with underscore
end

outputCsv = fullfile(dataFolder, sprintf('filtered_%s.csv', tagStr));
outputTxt = fullfile(dataFolder, sprintf('urls_%s.txt', tagStr));

%% 2. Verify the input file exists
fprintf('Reading file: %s\n', inputPath);

if ~isfile(inputPath)
    error('File not found: %s', inputPath);
end

%% 3. Read the file (readtable auto-detects xlsx vs csv)
% 'TextType','string' makes string columns easier to process later
videoTable = readtable(inputPath, 'TextType', 'string');

fprintf('Loaded %d total records\n', height(videoTable));
fprintf('Columns detected: %s\n\n', strjoin(videoTable.Properties.VariableNames, ', '));

%% 4. Identify the URL/filename column
% Try common column names; fall back to first column.
candidateNames = {'URL', 'url', 'Url', 'video_url', 'VideoURL', 'link', 'Link'};
urlColName = '';
for i = 1:length(candidateNames)
    if ismember(candidateNames{i}, videoTable.Properties.VariableNames)
        urlColName = candidateNames{i};
        break;
    end
end

if isempty(urlColName)
    % Fallback: use the first column
    urlColName = videoTable.Properties.VariableNames{1};
    fprintf('No standard URL column found — using first column "%s"\n\n', urlColName);
else
    fprintf('Using URL column: "%s"\n\n', urlColName);
end

urlColumn = videoTable.(urlColName);   % Dynamic field access by column name

%% 5. Build the filter mask
% Start with all-true mask, then AND each active filter into it
mask = true(height(videoTable), 1);

% List of (field name, human-readable label) pairs for logging
filterFields = {
    'genre',        'Genre';
    'situation',    'Situation';
    'camera',       'Camera';
    'dancer',       'Dancer';
    'music',        'Music';
    'choreography', 'Choreography'
};

fprintf('Active filters:\n');
anyActive = false;
for i = 1:size(filterFields, 1)
    fieldName = filterFields{i, 1};
    label     = filterFields{i, 2};
    value     = filter.(fieldName);

    % Skip empty filters
    if strlength(value) == 0
        continue;
    end
    anyActive = true;
    fprintf('  - %-15s = %s\n', label, value);

    % AND this condition into the running mask
    mask = mask & contains(urlColumn, value);
end

if ~anyActive
    fprintf('  (no filters set — returning all records)\n');
end

%% 6. Apply the mask
filteredTable = videoTable(mask, :);
nMatches = height(filteredTable);

fprintf('\nMatched %d records out of %d\n', nMatches, height(videoTable));

if nMatches == 0
    warning('No records matched. Check your filter values or column selection.');
    return;
end

%% 7. Preview first few results
fprintf('\nFirst %d results:\n', min(5, nMatches));
disp(head(filteredTable, 5));

%% 8. Save outputs
if saveFilteredTable
    writetable(filteredTable, outputCsv);
    fprintf('\n✅ Filtered table saved to:\n   %s\n', outputCsv);
end

if saveUrlList
    filteredUrls = filteredTable.(urlColName);
    writelines(filteredUrls, outputTxt);
    fprintf('📄 URL list saved to:\n   %s\n', outputTxt);
    fprintf('\nBatch download command (run in terminal):\n');
    fprintf('   cd %s && wget -i %s\n', dataFolder, sprintf('urls_%s.txt', tagStr));
end

fprintf('\nDone.\n');