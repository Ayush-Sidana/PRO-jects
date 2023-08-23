% Loading the video
videoFile = 'PGP_VID.mp4';
videoReader = VideoReader(videoFile);

% Cell arrays for key frames and non-key frames in each GOP
keyFramesCell = {};
nonKeyFramesCell = {};

% Generate SBHE matrix
GOPSize = 5; % Number of frames per GOP
CR_K = 0.6; % Compression ratio for key frames (K frames)
CR_NK = 0.3; % Compression ratio for non-key frames (NK frames)
B = 16; % Size of the Hadamard matrix

% Loop over the video frames
GOPIndex = 0; % Counter for GOPs

while hasFrame(videoReader)
    frame = readFrame(videoReader);
    
    % Check if it's the first frame in a new GOP
    if mod(GOPIndex, GOPSize) == 0
        isKeyFrame = true;
        GOPIndex = GOPIndex + 1;
        
        % Create new cells for key and non-key frames in this GOP
        keyFramesCell{end+1} = frame;
        nonKeyFramesCell{end+1} = [];
    else
        isKeyFrame = false;
        
        % Add non-key frame to the current GOP cell
        GOPCellIndex = numel(nonKeyFramesCell);
        nonKeyFramesCell{GOPCellIndex} = [nonKeyFramesCell{GOPCellIndex}, frame];
    end
    
    % Check if we have reached the end of the current GOP
    if GOPIndex == GOPSize
        GOPIndex = 0;
    end
    
    % Increment GOPIndex
    GOPIndex = GOPIndex + 1;
end

% Constructing the SBHE matrix for key frames
PN_K = pnScrambling(numel(keyFramesCell), A_key_frames);
% Assuming B is a square matrix of size B
x_1 = numel(keyFramesCell); % Replace with the desired size of the block diagonal matrix

% Generate the Hadamard matrix of size B
hadamard_matrix_B = hadamard(B);

% Create a block diagonal matrix with Hadamard matrix of size B as diagonal blocks
W_K = kron(eye(x_1/B), hadamard_matrix_B);

SBHEMatrix_K = constructSBHEMatrix(W_K, PN_K, CR_K);

% Constructing the SBHE matrix for non-key frames
PN_NK = pnScrambling(numel([nonKeyFramesCell{:}]), A_non_key_frames);
% Assuming B is a square matrix of size B
x_2 = numel([nonKeyFramesCell{:}]); % Replace with the desired size of the block diagonal matrix

% Generate the Hadamard matrix of size B
hadamard_matrix_B = hadamard(B);

% Create a block diagonal matrix with Hadamard matrix of size B as diagonal blocks
W_NK = kron(eye(x_2/B), hadamard_matrix_B);
SBHEMatrix_NK = constructSBHEMatrix(W_NK, PN_NK, CR_NK);

% Calculate the total number of frames in the video
totalFrames = numel(keyFramesCell) + numel([nonKeyFramesCell{:}]);

% Calculate the value of M for key frames and non-key frames
M_K = round(CR_K * numel(keyFramesCell));
M_NK = round(CR_NK * numel([nonKeyFramesCell{:}]));

% Apply the SBHE matrix to key frames and non-key frames vectors
keyFramesVector = cellfun(@(frame) double(SBHEMatrix_K) .* double(frame), keyFramesCell, 'UniformOutput', false);
nonKeyFramesVector = cellfun(@(frame) double(SBHEMatrix_NK) .* double(frame), [nonKeyFramesCell{:}], 'UniformOutput', false);

% Convert keyFramesVector and nonKeyFramesVector to arrays
keyFramesVector = cat(4, keyFramesVector{:});
nonKeyFramesVector = cat(4, nonKeyFramesVector{:});

% Print the dimensions of keyFramesVector and nonKeyFramesVector
disp(['Dimensions of keyFramesVector: ', num2str(size(keyFramesVector))]);
disp(['Dimensions of nonKeyFramesVector: ', num2str(size(nonKeyFramesVector))]);

% Function for PN scrambling operator
function PN = pnScrambling(N, A)
    PN = mod((0:N-1) + A - 1, N) + 1;
end

% Function for picking random rows from W*PN matrix
function QM = pickRandomRows(WPN, M)
    [N, ~] = size(WPN);
    randomIndices = randperm(N, M);
    QM = WPN(randomIndices, :);
end

% Function for constructing the SBHE matrix with desired compression rate
function SBHEMatrix = constructSBHEMatrix(W, PN, CR)
    [~, ~] = size(W);
    WPN = W * PN;
    M = round(CR * size(WPN, 1));
    QM = pickRandomRows(WPN, M);
    SBHEMatrix = QM * WPN;
end
