% Parameters for the image processing step
tau_hard = 2500;               % Distance threshold τhard
lambda_hard_sigma = 108;       % Threshold λhardσ (for σ ≤ 40, use λhardσ = 0)
sigma_squared = 40^2;          % Variance of the zero-mean Gaussian noise
nhard = 39;                    % Size of the 3D block for collaborative filtering
khards = 8;                    % Size of the reference block P (khards x khards)
kwien = khards;                % Size of the reference block P (kwien x kwien)
Nhard = 16;                    % Number of similar patches to keep . (Nhard is always chosen as a power of 2)
Nwien = 32;                    % Number of similar patches to keep 
tau_wien = 2500;               % Distance threshold τwien
sigma = 40;

% Image processing for each frame in keyFramesCell
processedKeyFramesCell = cell(size(keyFramesCell));
for i = 1:numel(keyFramesCell)
    frame = keyFramesCell{i};
    processedFrame = processFrame(frame, nhard, tau_hard, lambda_hard_sigma, sigma_squared, khards, Nhard);
    processedKeyFramesCell{i} = processedFrame;
end

% Image processing function using the described steps
function processedFrame = processFrame(frame, nhard, tau_hard, lambda_hard_sigma, sigma_squared, khards, Nhard)
    % Convert RGB frame to YUV space
    yuvFrame = rgb2ycbcr(frame);
    
    % Get the Y channel (luminance)
    Y = double(yuvFrame(:, :, 1));
    
    % First Denoising Step  to the Y channel
     processedY = processBlock(Y, nhard, tau_hard, lambda_hard_sigma, sigma_squared, khards, Nhard);

    % Second Denoising Step to the processedY.
     processedFrame = Second_Denoising_Step(processedY,Y, tau_wien, lambda_hard_sigma,nhard, khards, Nwien,Nhard);

end

% Function of First Denoising Step  on Y channel with patch search
function processedY = processBlock(Y, nhard, tau_hard, lambda_hard_sigma, ~, khards, Nhard)
    % Define the hard thresholding operator gamma_prime based on
    % lambda_hard_sigma
    gamma_prime = @(x) (abs(x) >= lambda_hard_sigma) .* x;
    
    % Pad the Y channel to handle boundary cases
    paddedY = padarray(Y, [khards/2, khards/2], 'symmetric', 'both');
    
    % Initialize the processed Y channel
    processedY = zeros(size(Y, 1), size(Y, 2), Nhard);
    
    % Get the size of the Y channel
    [height, width] = size(paddedY);

    % Calculate the number of blocks of nhard*nhard size in the padded Y channel
    nyards = floor(height / nhard);
    nxards = floor(width / nhard);

    offset= (khards)/2;
    
    % Loop over the padded Y channel in blocks of nhard*nhard size
    for y = 1:nyards
        for x = 1:nxards
            % Extract the search window
            searchWindow = paddedY((y - 1) * nhard + 1:(y - 1) * nhard + nhard, (x - 1) * nhard + 1:(x - 1) * nhard + nhard);

            % Extract the reference block P from the center of the search window, with some offset
            blockP = searchWindow((nhard - 1)/2 + 1 - offset:(nhard - 1)/2 + khards - offset, (nhard - 1)/2 + 1 - offset:(nhard - 1)/2 + khards - offset);

            % Calculate the normalized quadratic distance between P and each patch Q in the search window
            distances = zeros(nhard, nhard);
            for dy = 0:nhard - 1
                for dx = 0:nhard - 1
                    blockQ = searchWindow(dy:(dy + khards - 1), dx:(dx + khards - 1));
                    distances(dy + 1, dx + 1) = sqrt(sum(sum((gamma_prime(blockP) - gamma_prime(blockQ)).^2)) / (khards^2));
                end
            end
            
            % Convert the 2D distances array to a 1D vector
            distanceVector = distances(:);

            % Find the set of similar patches to P based on the distance threshold tau_hard
            similarPatchesIndices = find(distanceVector <= tau_hard);

            % Sort the patches in P(P) according to their distance to P
            [~, sortedIndices] = sort(distanceVector(similarPatchesIndices));
            sortedSimilarPatchesIndices = similarPatchesIndices(sortedIndices);

            % Keep the first Nhard patches in P(P).  
            NhardIndices = sortedSimilarPatchesIndices(0:(Nhard-1));

            % Extract the '3Dgroup' named set with Nhard number of patches with that indices with 'distanceVector <= tau_hard'
            similarPatches = searchWindow(NhardIndices, :);

            % Store the '3Dgroup' named set in the processed Y channel
            processedY(y, x, :) = similarPatches;
         end
    end
 
    % Apply Collaborative Filtering and Aggregation.
    for y = 1:nyards
        for x = 1:nxards
            
            % Apply Collaborative Filtering
            [processedPatches, transformed_signal_shrunk] = Collaborative_Filtering(processedY(y,x,:), sigma_squared);
    
            % After Collaborative Filtering Aggregation is to be done.
            
            % Initialize buffers
            numeratorBuffer = zeros(size(processedPatches));
            denominatorBuffer = zeros(size(processedPatches));
    
            % Iterate through each pixel in the processedPatches region
            for py = 1:khards
                for px = 1:khards
                    % Extract the indices of similar patches for the current pixel
                    similarIndices = processedY(y + py - 1, x + px - 1, :);
    
                    % Extract the similar patches for the current pixel
                    %similarPatches = paddedY(y + py - 1:y + py + khards - 2, x + px - 1:x + px + khards - 2, similarIndices);
    
                    % Iterate through each similar patch
                    for idx = 1:length(similarIndices)
                        % Extract the current similar patch
                        currentPatch_2 = processedPatches(:, :, idx); % This should be u_Q,P_hard(x)

                         % Apply Kaiser window
                        kaiserWindow = kaiser(khards, 6.28); % You can adjust the beta parameter as needed
                        currentPatch = currentPatch_2 .* kaiserWindow;
    
                        % Calculate the number of retained coefficients N_P_hard
                        N_P_hard = sum(abs(transformed_signal_shrunk(:)) > 0);
    
                        % Calculate the weight w_P_hard
                        if N_P_hard >= 1
                            w_P_hard = 1 / N_P_hard;
                        else
                            w_P_hard = 1;
                        end
    
                        % Extract the indicator function χ_Q(x) for the current pixel
                        indicator = abs(currentPatch) > 0;  % χ_Q(x) for current Q
    
                        % Update numerator and denominator buffers using indicator function
                        numeratorBuffer(:, :, idx) = numeratorBuffer(:, :, idx) + w_P_hard * currentPatch .* indicator;
                        denominatorBuffer(:, :, idx) = denominatorBuffer(:, :, idx) + w_P_hard * indicator;
                    end
                end
            end
    
            % Calculate the final basic estimate u_basic(x)
            u_basic = zeros(size(processedPatches(:, :, 1)));
            for idx = 1:length(similarIndices)
                % Calculate u_basic for each pixel x
                u_basic = numeratorBuffer(:, :, idx) ./ denominatorBuffer(:, :, idx);
            end
    
            % Update the processedY region with the final basic estimate
            processedY(y:y + khards - 1, x:x + khards - 1, similarIndices) = u_basic;
    
        end
    end
end

function [processedPatches, transformed_signal_shrunk] = Collaborative_Filtering(patches, ~)
    [patch_size, ~, num_patches] = size(patches);
    
    % Apply 2D Bior1.5 transform on each patch
    transformed_patches = zeros(patch_size, patch_size, num_patches);
    for idx = 1:num_patches
        patch = patches(:, :, idx);
        
        % Apply 2D Bior1.5 transform (Forward Transform)
        transformed_patch = apply2DBior15Transform(patch);
        
        transformed_patches(:, :, idx) = transformed_patch;
    end
    
    % Apply 1D Walsh-Hadamard transform along the third dimension
    transformed_patches_walsh = zeros(patch_size, patch_size, num_patches);
    for row = 1:patch_size
        for col = 1:patch_size
            % Extract the 1D signal along the third dimension
            signal = squeeze(transformed_patches(row, col, :));              
            % Apply 1D Walsh-Hadamard transform
            transformed_signal = apply1DWalshTransform(signal);
            
            % Store the transformed signal
            transformed_patches_walsh(row, col, :) = transformed_signal;
        end
    end
    
    % Apply shrinkage to the transform spectrum 
    for idx = 1:num_patches
        transformed_signal = transformed_patches_walsh(:, :, idx);
        
        % Apply hard thresholding shrinkage
        transformed_signal_shrunk = gamma(transformed_signal, lambda_hard_sigma);
        
        transformed_patches_walsh(:, :, idx) = transformed_signal_shrunk;
    end
    
    % Apply inverse 1D Walsh-Hadamard transform
    transformed_patches_inv = zeros(patch_size, patch_size, num_patches);
    for row = 1:patch_size
        for col = 1:patch_size
            % Extract the transformed signal along the third dimension
            transformed_signal = squeeze(transformed_patches_walsh(row, col, :));
            
            % Apply inverse 1D Walsh-Hadamard transform
            inv_transformed_signal = applyInverse1DWalshTransform(transformed_signal);
            
            % Store the inverse transformed signal
            transformed_patches_inv(row, col, :) = inv_transformed_signal;
        end
    end
    
    % Apply inverse 2D Bior1.5 transform on each patch
    processedPatches = zeros(patch_size, patch_size, num_patches);
    for idx = 1:num_patches
        transformed_patch_inv = transformed_patches_inv(:, :, idx);
        
        % Apply inverse 2D Bior1.5 transform
        processed_patch = applyInverse2DBior15Transform(transformed_patch_inv);
        
        processedPatches(:, :, idx) = processed_patch;
    end
end
%__
function transformed_patch = apply2DBior15Transform(patch)
    % Apply 2D Bior1.5 transform using built-in MATLAB functions
    [loD, hiD, ~, ~] = biorwavf('bior1.5');
    transformed_patch = wavedec2(patch, 1, loD, hiD);
end
function transformed_signal = apply1DWalshTransform(signal)
    % Apply 1D Walsh-Hadamard transform using built-in MATLAB function
    transformed_signal = fwht(signal);
end

function inv_transformed_signal = applyInverse1DWalshTransform(transformed_signal)
    % Apply inverse 1D Walsh-Hadamard transform using built-in MATLAB function
    inv_transformed_signal = ifwht(transformed_signal);
end

function processed_patch = applyInverse2DBior15Transform(transformed_patch_inv)
    % Apply inverse 2D Bior1.5 transform using built-in MATLAB functions
    [~, ~, loR, hiR] = biorwavf('bior1.5');
    processed_patch = waverec2(transformed_patch_inv, loR, hiR);
end

function output = gamma(x, threshold)
    output = (abs(x) > threshold) .* x;
end
%__

%STEP_2

%__

function processedY_basic = Second_Denoising_Step(processedY,Y, tau_wien, lambda_hard_sigma,nhard, khards, Nwien,Nhard)
    % Get the size of the processedY channel
    [height, width, ~] = size(processedY);

    % Define the hard thresholding operator gamma_prime based on
    % lambda_hard_sigma
    gamma_prime = @(x) (abs(x) >= lambda_hard_sigma) .* x;
    
    % Pad the processedY channel to handle boundary cases
    padded_processedY = padarray(processedY, [khards/2, khards/2], 'symmetric', 'both');
    
    % Initialize the processed Y basic channel
    processedY_basic = zeros(size(processedY, 1), size(processedY, 2), Nwien);
    
    % Get the size of the Y channel
    %[height, width] = size(paddedY);
    % Calculate the number of blocks of nhard*nhard size in the padded Y channel
    nyards = floor(height / nhard);
    nxards = floor(width / nhard);

    offset= (khards)/2;
    
    % Loop over the pixels in the Y channel
    for y = 1:nyards    
        for x = 1:nxards
             % Extract the search window
            searchWindow = padded_processedY((y - 1) * nhard + 1:(y - 1) * nhard + nhard, (x - 1) * nhard + 1:(x - 1) * nhard + nhard);
            
             % Extract the reference block P from the center of the search window, with some offset
            blockP = searchWindow((nhard - 1)/2 + 1 - offset:(nhard - 1)/2 + khards - offset, (nhard - 1)/2 + 1 - offset:(nhard - 1)/2 + khards - offset);

            % Calculate the normalized quadratic distance between P and each patch Q in the neighborhood
            distances = zeros(nhard, nhard);
            for dy = 0:nhard-1
                for dx = 0:nhard-1
                    blockQ = searchWindow(dy:(dy + khards - 1), dx:(dx + khards - 1));
                    distances(dy+1, dx+1) = sqrt(sum(sum((gamma_prime(blockP) - gamma_prime(blockQ)).^2)) /(khards^2)) ;  
                end 
            end
            
            % Convert 2D distances to a vector for sorting
            distanceVector = distances(:);
            
            % Find the set of similar patches to P based on the distance threshold tau_hard
            similarPatchesIndices = find(distanceVector <= tau_wien);
            
            % Sort the patches in P(P) according to their distance to P
            [~, sortedIndices] = sort(distanceVector(similarPatchesIndices));
            sortedSimilarPatchesIndices = similarPatchesIndices(sortedIndices);
            
            % Keep the first Nwien patches in P(P) .  
            NwienIndices = sortedSimilarPatchesIndices(0:(Nwien-1)); 

            similarPatches = searchWindow(NwienIndices, :);
            
            % Store the sorted similar patches indices in the processed Y channel
            processedY_basic(y, x, :) = similarPatches;

        end
    end

            %___Calculation_Of_P(P)_____%

    % Pad the Y channel to handle boundary cases
    paddedY = padarray(Y, [khards/2, khards/2], 'symmetric', 'both');
    
    % Initialize the processed Y channel
    processed_Y = zeros(size(Y, 1), size(Y, 2), Nhard);
    
    % Get the size of the Y channel
    [height, width] = size(paddedY);

    % Calculate the number of blocks of nhard*nhard size in the padded Y channel
    nyards = floor(height / nhard);
    nxards = floor(width / nhard);

    offset= (khards)/2;
    
    % Loop over the pixels in the Y channel
    for j = 1:nyards    
        for i = 1:nxards
             % Extract the search window
             searchWindow = paddedY((y - 1) * nhard + 1:(y - 1) * nhard + nhard, (x - 1) * nhard + 1:(x - 1) * nhard + nhard);
             % Extract the reference block P from the center of the search window, with some offset
            blockP = searchWindow((nhard - 1)/2 + 1 - offset:(nhard - 1)/2 + khards - offset, (nhard - 1)/2 + 1 - offset:(nhard - 1)/2 + khards - offset);
            
            % Calculate the normalized quadratic distance between P and each patch Q in the neighborhood
            distances = zeros(nhard, nhard);
            for dj = 0:nhard-1
                for di = 0:nhard-1
                    blockQ =  searchWindow(dj:(dj + khards - 1), di:(di + khards - 1));
                    distances(dj+1, di+1) = sqrt(sum(sum((gamma_prime(blockP) - gamma_prime(blockQ)).^2)) /(khards^2)) ;  
                end 
            end
            
            % Convert 2D distances to a vector for sorting
            distanceVector = distances(:);
            
            % Find the set of similar patches to P based on the distance threshold tau_hard
            similarPatchesIndices = find(distanceVector <= tau_hard);
            
            % Sort the patches in P(P) according to their distance to P
            [~, sortedIndices] = sort(distanceVector(similarPatchesIndices));
            sortedSimilarPatchesIndices = similarPatchesIndices(sortedIndices);
            
            % Keep the first Nhard patches in P(P) .  
            Nwien_Indices = sortedSimilarPatchesIndices(0:(Nwien-1)); 

              % Extract the '3Dgroup' named set with Nhard number of patches with that indices with 'distanceVector <= tau_hard'
            similarPatches = searchWindow(Nwien_Indices, :);
            
            % Store the sorted similar patches indices in the processed Y channel
            processed_Y(j, i, :) = similarPatches;
         end
    end
      
    
   
     % Apply Colaaborative_Filtering_2
    for y = 1:nyards
        for x = 1:nxards
            % Extract the indices of similar patches
            %similarIndices_Pbasic = processedY_basic(y, x, :);
            %similarIndices_P_P = processed_Y(y, x, :);
    
            % Extract the similar patches
            %similarPatches_Pbasic= padded_processedY(y:y + khards - 1, x:x + khards - 1, similarIndices_Pbasic(P));
            %similarPatches_P_P = paddedY(y:y + khards - 1, x:x + khards - 1, similarIndices_P_P);
    
            % Apply 3D transform and shrinkage
            [processedPatches_2,wiener_coeff] = Collaborative_Filtering_2(processedY_basic(y, x, :),processed_Y(y, x, :), sigma);

               % After Collaborative Filtering
            
            % Initialize buffers
            numeratorBuffer = zeros(size(processedPatches_2));
            denominatorBuffer = zeros(size(processedPatches_2));
    
            % Iterate through each pixel in the processedPatches_2 region
            for py = 1:kwien
                for px = 1:kwien
                    % Extract the indices of similar patches for the current pixel
                    similarIndices = processedY_basic(y + py - 1, x + px - 1, :);
    
                    % Extract the similar patches for the current pixel
                    %similarPatches = padded_processedY(y + py - 1:y + py + khards - 2, x + px - 1:x + px + khards - 2, similarIndices);

    
                    % Iterate through each similar patch
                    for idx = 1:length(similarIndices)
                        % Extract the current similar patch
                        currentPatch_1 = processedPatches_2(:, :, idx); % This should be u_Q,P_wien(x)

                         % Apply Kaiser window
                        kaiserWindow = kaiser(kwien, 6.28); % You can adjust the beta parameter as needed
                        currentPatch = currentPatch_1 .* kaiserWindow;
    
                       
                        w_P_wien =  1 ./ (norm(wiener_coeff)^2);

                        
    
                        % Extract the indicator function χ_Q(x) for the current pixel
                        indicator = abs(currentPatch) > 0;  % χ_Q(x) for current Q
    
                        % Update numerator and denominator buffers using indicator function
                        numeratorBuffer(:, :, idx) = numeratorBuffer(:, :, idx) + w_P_wien * currentPatch .* indicator;
                        denominatorBuffer(:, :, idx) = denominatorBuffer(:, :, idx) + w_P_wien * indicator;
                    end
                end
            end
    
            % Calculate the final basic estimate u_basic(x)
            u_final = zeros(size(processedPatches_2(:, :, 1)));
            for idx = 1:length(similarIndices)
                % Calculate u_basic for each pixel x
                u_final = numeratorBuffer(:, :, idx) ./ denominatorBuffer(:, :, idx);
            end
    
            % Update the processedY region with the final basic estimate
            processedY(y:y + khards - 1, x:x + khards - 1, similarIndices) = u_final;
    
        end
    end
end

%___Collaborative_Filtering_Step_2___%


function [processedPatches_2,wiener_coeff] = Collaborative_Filtering_2(similarPatches_Pbasic,similarPatches_P_P, sigma)
    

    [patch_size_1, ~, num_patches1] = size(similarPatches_Pbasic);
    [patch_size_2, ~, num_patches2] = size(similarPatches_P_P);
    
    % Initialize denoised patches
    processedPatches_2 = zeros(size(similarPatches_Pbasic));
    
    % Loop over each 3D patch Pbasic(P)

    transformed_patches = zeros(patch_size_1, patch_size_1, num_patches1);
    for idx = 1:num_patches1
        patch = similarPatches_Pbasic(:, :, idx);
        
        % Apply 2D DCT transform
        transformed_patch = dct2(patch);
        
        transformed_patches(:, :, idx) = transformed_patch;
    end
   % Apply 1D Walsh-Hadamard transform
   transformed_patch1 = zeros(patch_size_1, patch_size_1, num_patches1);
    for row = 1:patch_size_1
        for col = 1:patch_size_1
            % Extract the 1D signal along the third dimension
            signal = squeeze(transformed_patches(row, col, :));              
            % Apply 1D Walsh-Hadamard transform
            transformed_signal = apply1DWalshTransform(signal);
            
            % Store the transformed signal
            transformed_patch1(row, col, :) = transformed_signal;
        end
    end

   % Compute Wiener coefficients
   wiener_coeff_patches = zeroes(patch_size_1, patch_size_1, num_patches1);
   % Loop over each denoised patch for computing Wiener coefficients
    for idx = 1:num_patches1
        % Extract the corresponding transformed signals
        transformed_patch1_current = transformed_patch1(:, :, idx);
        
        % Compute Wiener coefficients
        wiener_coeff = abs(transformed_patch1_current).^2 ./ (abs(transformed_patch1_current).^2 + sigma^2);
        
        % Store Wiener coefficients for the current patch
        wiener_coeff_patches(:, :, idx) = wiener_coeff;
    end
        
   % Loop over each 3D patch P(P)


    transformed_patches = zeros(patch_size_2, patch_size_2, num_patches2);
    for idx = 1:num_patches2
        patch = similarPatches_Pbasic(:, :, idx);
        
        % Apply 2D DCT transform
        transformed_patch = dct2(patch);
        
        transformed_patches(:, :, idx) = transformed_patch;
    end
   % Apply 1D Walsh-Hadamard transform
   transformed_patch2 = zeros(patch_size_2, patch_size_2, num_patches2);
    for row = 1:patch_size_2
        for col = 1:patch_size_2
            % Extract the 1D signal along the third dimension
            signal = squeeze(transformed_patches(row, col, :));              
            % Apply 1D Walsh-Hadamard transform
            transformed_signal = apply1DWalshTransform(signal);
            
            % Store the transformed signal
            transformed_patch2(row, col, :) = transformed_signal;
        end
    end

 % Wiener collaborative filtering

   % Loop over each denoised patch for Wiener collaborative filtering
for idx = 1:num_patches2
    % Extract the corresponding transformed signals
    transformed_patch2_current = transformed_patch2(:, :, idx);
    
    % Extract Wiener coefficients for the current patch
    wiener_coeff = wiener_coeff_patches(:, :, idx);
    
    % Wiener collaborative filtering
    denoised_patch = wiener_coeff .* transformed_patch2_current;

    % Apply inverse 1D Walsh-Hadamard transform

   transformed_patches_inv = zeros(patch_size_2, patch_size_2, num_patches2);
    for row = 1:patch_size_2
        for col = 1:patch_size_2
            % Extract the transformed signal along the third dimension
            transformed_signal = squeeze(denoised_patch(row, col, :));
            
            % Apply inverse 1D Walsh-Hadamard transform
            inv_transformed_signal = applyInverse1DWalshTransform(transformed_signal);
            
            % Store the inverse transformed signal
            transformed_patches_inv(row, col, :) = inv_transformed_signal;
        end
    end
            
   % Apply inverse 2D DCT transform
   for idx1 = 1:num_patches2
        transformed_patch_inv = transformed_patches_inv(:, :, idx1); 
        % Apply inverse 2D DCT transform
        processed_patch = idct2(transformed_patch_inv);
        processedPatches_2(:, :, idx1) = processed_patch;
   end
end
end
            
   

            
