% Stone Preston, Truong Dinh, and Faranak Abri
% ECE 5367
% Project 4: Assessing the effectiveness of SIFT and SURF features for object recognition(traffic sign).

clc
clear
close all

[imgSet, I_ref_color_padded, I_ref_padded] = initialize();

% Let user choose which method to use
methodsList = {'SIFT', 'SURF'};
[indx,tf] = listdlg('ListString', methodsList, 'SelectionMode', 'single', 'PromptString', 'Select a feature matching method:', 'ListSize',[300,100]);

selectedMethod = methodsList(indx);

if (strcmp(selectedMethod{1}, 'SURF'))
    
   runSurf(imgSet, I_ref_padded, I_ref_color_padded);
   
end

if (strcmp(selectedMethod{1}, 'SIFT'))
    
   runSift(imgSet, I_ref_padded, I_ref_color_padded);
   
end

 function runSift(imgSet, I_ref_padded, I_ref_color_padded)
 
    wb = waitbar(0,'Initializing reference image ...');
    I_ref_single = single(I_ref_padded);
    [f_ref, d_ref] = vl_sift(I_ref_single);
    
    
    close(wb)
    
    for i = 1:imgSet.Count
        
        
        wb = waitbar(.15, 'Extracting SIFT features ...');
        I = rgb2gray(read_rotate(imgSet, i));
        I_color = read_rotate(imgSet, i);

        % Convert to single matrices
        I_single = single(I);

        %extracting SIFT features
        
        [f_I, d_I] = vl_sift(I_single);

        
        waitbar(.65,wb,'Matching features ...');

        %matching SIFT features
        [matches, scores] = vl_ubcmatch(d_ref, d_I);
        x_ref = f_ref(1,matches(1,:)) ;
        x_I = f_I(1,matches(2,:)) + size(I_ref_single,2) ;
        y_ref = f_ref(2,matches(1,:)) ;
        y_I = f_I(2,matches(2,:)) ;


        %showing the results for SIFT features
        waitbar(1,wb,'Showing results for SIFT features ...');
        %accuracy based on the # of matched features to the total # of features
        %reference image
        f_ref_no=size(f_ref);
        matches_no=size(matches);
        accuracy=(matches_no(1,2)/f_ref_no(1,2))*100;
        imshow(cat(2, I_ref_color_padded, I_color)) ;
        title(['(',num2str(i),'/',num2str(imgSet.Count),')  ','SIFT Accuracy is ',num2str(accuracy),'%']);
        xlabel('Press any key to continue ...');

        hold on ;
        %show all the features in reference image  
        h1 = vl_plotframe(f_ref) ;
        set(h1,'color','y','linewidth',2) ;


        h = line([x_ref ; x_I], [y_ref ; y_I]) ;
        set(h,'linewidth', 1, 'color', 'b') ;

        vl_plotframe(f_ref(:,matches(1,:))) ;
        f_I(1,:) = f_I(1,:) + size(I_ref_single,2) ;



        vl_plotframe(f_I(:,matches(2,:))) ;
        % axis image off
        close(wb);
        
        pause;
        
    end
 
 end
 
 function runSurf(imgSet, I_ref_padded, I_ref_color_padded)
    
    
    wb = waitbar(0,'Initializing reference image ...');

    %extracting surf features
    ref_points = detectSURFFeatures(I_ref_padded);
    [ref_sf, ref_vpts] = extractFeatures(I_ref_padded, ref_points);
    
    close(wb)
    
    h = figure;
    set(h, 'Visible', 'off');
    ax = axes;
    
    for i = 1:imgSet.Count
        
        wb = waitbar(.15, 'Extracting SURF features ...');
        I = rgb2gray(read_rotate(imgSet, i));
        I_color = read_rotate(imgSet, i);
        
        I_points = detectSURFFeatures(I);
        [I_sf,I_vpts] = extractFeatures(I,I_points);


        waitbar(.65,wb,'Matching features ...');
        
        indexPairs = matchFeatures(ref_sf,I_sf) ;
        ref_matchedPoints = ref_vpts(indexPairs(:,1));
        I_matchedPoints = I_vpts(indexPairs(:,2));
        
        %show the results for SURF features
        waitbar(1, wb, 'Showing results for SURF features ...');
        %accuracy based on the # of matched features to the total # of features
        %reference image
        sf_ref_no=size(ref_sf,1);
        sf_matches_no=size(indexPairs,1);
        sf_accuracy=(sf_matches_no/sf_ref_no)*100;

        %show matched surf features 
        
        set(h, 'Visible', 'on');
        showMatchedFeatures(I_ref_color_padded, I_color,ref_matchedPoints, I_matchedPoints, 'montage', 'PlotOptions', {'mo','g+','b-'});
        title(ax, ['(',num2str(i),'/',num2str(imgSet.Count),')  ','SURF Accuracy is ',num2str(sf_accuracy),'%']);
        xlabel('Press any key to continue ...');
        
        close(wb);
        pause;
        

    end
 
 end
 
 
 function [imgSet, I_ref_color_padded, I_ref_padded] = initialize()
 
    run('vlfeat-0.9.21/toolbox/vl_setup')
    vl_version verbose

    % Select test image directory
    % selpath = uigetdir(path, title);

    % Create imageSet object using directory
    imgSet = imageSet('test_images');

    % Get initial image so we can resize reference
    I = rgb2gray(read_rotate(imgSet, 1));

    % Set reference image
    I_ref = rgb2gray(imread('reference.png'));
    I_ref_color = imread('reference.png');

    % Pad reference to match captured image size
    height_I = size(I,1); 
    width_I = size(I,2); 
    height_ref = size(I_ref,1); 
    width_ref = size(I_ref,2);

    I_ref_padded = padarray(I_ref,[(height_I-height_ref)/2, (width_I-width_ref)/2], 255); 
    I_ref_color_padded = padarray(I_ref_color,[(height_I-height_ref)/2, (width_I-width_ref)/2], 255);
 end

function I = read_rotate(imgSet, index)

    original = read(imgSet, index);
    I = imrotate(original, -90);

end

