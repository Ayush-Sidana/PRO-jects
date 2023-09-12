
LCUSize = 64; % Size of Large Coding Units (LCUs)
CUSizes = [32, 16, 8,4]; % Sizes of Coding Units (CUs) for quadtree partitioning
currentCUSizeIdx = 1; % Initialize the current CU size index
blockPosition = [x, y]; % Define your block position here

% Update the current CU size index
   if currentCUSizeIdx < numel(CUSizes)
       currentCUSizeIdx = currentCUSizeIdx + 1;
   else
       currentCUSizeIdx = 1; % Reset to the first CU size
   end

SI=generateSIForFrames(processedKeyFramesCell, nonKeyFramesVector, LCUSize, CUSizes);


function SIFrame = generateSIForFrames(processedKeyFramesCell, nonKeyFramesVector, LCUSize, CUSizes)
    % Initialize the SI frame
    SIFrame = zeros(size(nonKeyFramesVector{1})); % Assuming all frames have the same size

    % Loop through the frames in nonKeyFramesVector
    for frameIdx = 1:numel(nonKeyFramesVector)
        currentFrame = nonKeyFramesVector{frameIdx};
        referenceFrames = processedKeyFramesCell{frameIdx};
        % Generate SI frame using recursive motion estimation
        SIFrame = recursiveMotionEstimation(referenceFrames, currentFrame, LCUSize, CUSizes, SIFrame);
         
    end
end

function SIFrame = recursiveMotionEstimation(referenceFrames, currentFrame, LCUSize, CUSizes, SIFrame)
    % Divide the current frame into LCU blocks
    LCUBlocks = divideFrameIntoLCUBlocks(currentFrame, LCUSize);

    % Loop through the LCU blocks
    for i = 1:size(LCUBlocks, 1)       %(LCUBlocks,1) will come or not as LCUBlocks size has to be taken.
        LCU = LCUBlocks{i};

        % Perform motion estimation for the LCU block .
        % This has to be implemented and make sure it is needed or not. 
        %estimatedMV = motionEstimation(LCU, referenceFrames, searchRange);

        % Determine whether the LCU block should be divided into smaller blocks
        shouldDivide = shouldDivideLCUBlock(LCU,referenceFrames,blockPosition, CUSizes);

        if shouldDivide
            % Initialize the current CU size
            currentCUSizeIdx = 1;

            % Divide the LCU block into smaller blocks with varying CU sizes
            while shouldDivide && currentCUSizeIdx <= numel(CUSizes)
                cuSize = CUSizes(currentCUSizeIdx);
                CUBlocks = divideLCUBlock(LCU, cuSize);

                % Recurse on the smaller blocks
                for j = 1:numel(CUBlocks)
                    subLCU = CUBlocks{j};
                    
                    [estimatedMV_forward, estimatedMV_backward] = motionEstimation(subLCU,referenceFrames);
                    combineCompensated=combineCompensatedLCUs(subLCU, estimatedMV_forward, estimatedMV_backward);
                    SIFrame(i,j) = combineCompensated;

                    % Update the shouldDivide condition
                    shouldDivide = shouldDivideLCUBlock(subLCU,referenceFrames,blockPosition, CUSizes);
                end

                % Move to the next CU size
                currentCUSizeIdx = currentCUSizeIdx + 1;
            end
        else
            % Compensate the LCU block using the motion vector
           [estimatedMV_forward, estimatedMV_backward] = motionEstimation(LCU,referenceFrames);
           combineCompensated=combineCompensatedLCUs(LCU, estimatedMV_forward, estimatedMV_backward);

            % Add the compensated LCU block to the SI frame
            SIFrame(i) = combineCompensated;
        end
    end
end



function shouldDivide = shouldDivideLCUBlock(LCU,referenceFrames,blockPosition, CUSizes)
    bestSATD = inf;
    searchRange = ((CUSizes) / 2) - 1;

    for y = -searchRange:(searchRange+2)
        for x = -searchRange:(searchRange+2)
            refY = blockPosition(2) + y;
            refX = blockPosition(1) + x;

            if refY > 0 && refY <=   size(referenceFrames{1}, 1) - size(LCU, 1)  + 1 && ...
               refX > 0 && refX <=  size(referenceFrames{1}, 2) - size(LCU, 2)   + 1

                refBlock = referenceFrames(refY:refY+size(LCU, 1)-1, refX:refX+size(LCU, 2)-1);

                % Calculate SATD
                SATD = calculateSATD(LCU, refBlock);

                if SATD < bestSATD
                    bestSATD = SATD;   %LCU_SATD
                    %bestMVs_1 = [x, y];
                    
                end
            end
        end
    end

    % Check for quadtree partitioning
    if shouldDivideLCUBlock(LCU,referenceFrames,blockPosition, CUSizes)
        % Divide LCU into CUs
        CUBlocks = divideLCUBlock(LCU, CUSizes);

        % Initialize best motion vectors and SATD
        bestMVs_2 = cell(size(CUBlocks));
        bestSATDs = inf(size(CUBlocks));

        % Loop through CUs
        for cuIdx = 1:numel(CUBlocks)
            % Perform SATD motion estimation on CU
            %[bestMVs{cuIdx}, bestSATDs(cuIdx)] = SATDMotionEstimation(CUBlocks{cuIdx}, referenceFrames, blockPosition, searchRange, CUSizes);
            for y = -searchRange:searchRange
                for x = -searchRange:searchRange
                    refY = blockPosition(2) + y;
                    refX = blockPosition(1) + x;
        
                    if refY > 0 && refY <=  size(referenceFrames{1}, 1) - size(CUBlocks{cuIdx}, 1) + 1 && ...
                       refX > 0 && refX <=  size(referenceFrames{1}, 1) - size(CUBlocks{cuIdx}, 2) + 1
        
                        refBlock = referenceFrames(refY:refY+size(LCU, 1)-1, refX:refX+size(LCU, 2)-1);
        
                        % Calculate SATD
                        SATD = calculateSATD(LCU, refBlock);
        
                        if SATD < bestSATDs{cuIdx}
                            bestSATDs{cuIdx} = SATD;   %LCU_SATD
                            bestMVs_2{cuIdx} = [x, y];
                        end
                    end
                end
             end
        end
    end

    % Sum SATD values of CUs
        sumSATD = sum(bestSATDs);    %CUs_SATD

        % If the SATD value of the LCU is smaller, then the LCU is not divided
        if sumSATD < bestSATD
            bestSATD = sumSATD;
            
        end
 

    % Return true if the sum of the SAD values is greater than or equal to the threshold
        shouldDivide = sumSATD < bestSATD;
end

function LCUBlocks = divideFrameIntoLCUBlocks(currentFrame, LCUSize)
    % Get the size of the current frame
    [frameHeight, frameWidth] = size(currentFrame);

    % Calculate the number of LCU blocks in both dimensions
    numLCUsVertical = frameHeight / LCUSize;
    numLCUsHorizontal = frameWidth / LCUSize;

    % Initialize the cell array to store LCU blocks
    LCUBlocks = cell(numLCUsVertical, numLCUsHorizontal);

    % Loop through the LCUs
    for i = 1:numLCUsVertical
        for j = 1:numLCUsHorizontal
            % Calculate the coordinates of the current LCU block
            x0 = (j - 1) * LCUSize + 1;
            y0 = (i - 1) * LCUSize + 1;
            x1 = x0 + LCUSize - 1;
            y1 = y0 + LCUSize - 1;

            % Extract the current LCU block from the current frame
            LCU = currentFrame(y0:y1, x0:x1);

            % Store the LCU block in the cell array
            LCUBlocks{i, j} = LCU;
        end
    end
end


function CUBlocks = divideLCUBlock(LCU, CUSizes)
    % Calculate the number of CU blocks in the LCU
    numCUsH = size(LCU, 1) / CUSizes;
    numCUsW = size(LCU, 2) / CUSizes;
    
    % Initialize the CU blocks cell array
    CUBlocks = cell(numCUsH, numCUsW);     % cell to be size of numCUs only according to me.

    % Loop through the CU blocks
    
    for y = 1:numCUsH
        for x = 1:numCUsW
            x0 = (x - 1) * CUSizes + 1;
            y0 = (y - 1) * CUSizes + 1;
            x1 = x0 + CUSizes - 1;
            y1 = y0 + CUSizes - 1;

             % Extract the current LCU block from the current frame
            CU = LCU(y0:y1, x0:x1);

            % Store the LCU block in the cell array
            CUBlocks{y,x} = CU;
            
        end
    end
end


function SATD = calculateSATD(LCU, refBlock)
    Q_ij = double(LCU) - double(refBlock);
    H = hadamard(size(Q_ij,1));
    S_ij = H * Q_ij * H';
    SATD = sum(abs(S_ij(:)));
end

function [estimatedMV_forward, estimatedMV_backward] = motionEstimation(LCU, referenceFrames)
    % Initialize variables
    estimatedMV_forward = [0, 0];
    estimatedMV_backward = [0, 0];
    bestSATD_forward = inf;
    bestSATD_backward = inf;
    searchRange = ((LCU) / 2) - 1;

    % Loop through the search range
    for y = -searchRange : (searchRange + 2)
        for x = -searchRange : (searchRange + 2)
            % Calculate reference block positions for forward motion estimation
            refY_forward = blockPosition(2) + y;
            refX_forward = blockPosition(1) + x;

            % Calculate reference block positions for backward motion estimation
            refY_backward = blockPosition(2) - y;
            refX_backward = blockPosition(1) - x;

            if refY_forward > 0 && refY_forward <= size(referenceFrames{1}, 1) - size(LCU, 1) + 1 && ...
                    refX_forward > 0 && refX_forward <= size(referenceFrames{1}, 2) - size(LCU, 2) + 1
                % Forward motion estimation
                refBlock_forward = referenceFrames{frameIdx}(refY_forward : refY_forward + size(LCU, 1) - 1, ...
                                                              refX_forward : refX_forward + size(LCU, 2) - 1);
                SATD_forward = calculateSATD(LCU, refBlock_forward);

                if SATD_forward < bestSATD_forward
                    bestSATD_forward = SATD_forward;
                    estimatedMV_forward = [x, y];
                end
            end

            if refY_backward > 0 && refY_backward <= size(referenceFrames{1}, 1) - size(LCU, 1) + 1 && ...
                    refX_backward > 0 && refX_backward <= size(referenceFrames{1}, 2) - size(LCU, 2) + 1
                % Backward motion estimation
                refBlock_backward = referenceFrames{frameIdx}(refY_backward : refY_backward + size(LCU, 1) - 1, ...
                                                              refX_backward : refX_backward + size(LCU, 2) - 1);
                SATD_backward = calculateSATD(LCU, refBlock_backward);

                if SATD_backward < bestSATD_backward
                    bestSATD_backward = SATD_backward;
                    estimatedMV_backward = [-x, -y]; % Negative values for backward motion
                end
            end
        end
    end
end

function compensatedLCU = combineCompensatedLCUs(LCU, estimatedMV_forward, estimatedMV_backward)
    % Combine the compensated LCUs using motion vectors from forward and backward motion estimation
    [h, w] = size(LCU);
    compensatedLCU = zeros(h,w);
    

    % Apply forward motion vector
    compensatedLCU_forward = applyMotionVector(LCU, estimatedMV_forward);

    % Apply backward motion vector
    compensatedLCU_backward = applyMotionVector(LCU, estimatedMV_backward);
    for y = 1:h
            for x = 1:w
                % Combine the two compensated LCUs, e.g., by averaging
                compensatedLCU(y,x) = ((compensatedLCU_forward(y,x) + compensatedLCU_backward(y,x))) / 2;
    
                
             end
            
     end
    
end

function compensatedLCU = applyMotionVector(LCU, motionVector)
    % Perform motion compensation and return the compensated LCU
    [h, w] = size(LCU);
    compensatedLCU = zeros(h, w);

    % Apply motion vector to each pixel in the LCU
    for y = 1:h
        for x = 1:w
            refY = y + motionVector(2);
            refX = x + motionVector(1);

            % Check if the reference coordinates are within bounds
            if refY >= 1 && refY <= h && refX >= 1 && refX <= w
                compensatedLCU(y, x) = LCU(refY, refX);
            end
        end
    end
end




