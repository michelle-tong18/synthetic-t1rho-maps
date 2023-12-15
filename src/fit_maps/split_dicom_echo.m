function split_dicom_echo(examDir,input_folder,series_num,dest_folder,T1rho_idx,T2_idx)
%function split_dicom_echo(examDir,input_folder,series_num,dest_folder,T1rho_idx,T2_idx)
% Split dicom images from MAPSS into diferent echo folder, T1rho/e1, T2/e2,..
% Re-implented by Misung Han 
% Apr 14, 2020
% originated from Radhika's codes.
	num_ver=version('-release')
	matlab_ver=str2num(num_ver(1:4));
	
	T1rho_folder=sprintf('%s/%s/T1rho',examDir,dest_folder);
	T2_folder=sprintf('%s/%s/T2',examDir,dest_folder);
	cart1_folder=sprintf('%s/%s/T1rho/reg',examDir,dest_folder);
   cart2_folder=sprintf('%s/%s/T2/reg',examDir,dest_folder);
   
	if(length(T1rho_idx)>0)

	    mkdir(T1rho_folder);
		mkdir(cart1_folder);
	end
	if(length(T2_idx)>0)
		mkdir(T2_folder);
		mkdir(cart2_folder);
	end
	importDir = strcat(examDir,'/',input_folder);
    importFile=sprintf('%s/*.DCM',importDir);
	files_contra=dir(importFile);
	if(size(files_contra,1)==0)
		importFile=sprintf('%s/*.dcm',importDir);
		files_contra=dir(importFile);
		disp(['legnth dcm: ' num2str(length(files_contra))]);
		
	end	
    files = files_contra;
for i=1:length(T1rho_idx)
		foldernames=strcat('e',num2str(T1rho_idx(i)));
		mkdir(T1rho_folder,foldernames);
	end
	for i=1:length(T2_idx)
		foldernames=strcat('e',num2str((i)));
		mkdir(T2_folder,foldernames);
	end
	
	for i=1:length(files)
		disp([importDir,'/',files(i).name]);
        info_all=dicominfo([importDir,'/',files(i).name]);
		if(strcmp(info_all.Manufacturer,'SIEMENS'))
			 E_num_all(i)=info_all.AcquisitionNumber;
		else
			if(matlab_ver<2021)
       			 E_num_all(i)=info_all.EchoNumber;
			else
        			E_num_all(i)=info_all.EchoNumbers;
			end
		end
    end
    % which is the highest echo number? This way i know how many echoes
    % there are

    echo_num_max=max(E_num_all);
    % this means there are echo_num_max+1 echoes, since echo 1 is shated
    % between t1rho and t2
    %importDir = dualdir;
    for i=1:length(files)
       [isT1rho,idx]= ismember(E_num_all(i),T1rho_idx);
	   if(isT1rho)
			source=[importDir,'/', files(i).name];
		 	destination = [T1rho_folder,'/e',num2str(idx),'/',files(i).name];
           	system(['rsync ',source,' ',destination]);
   	   end 
	   [isT2,idx]= ismember(E_num_all(i),T2_idx);
		if(isT2)
			source=[importDir, '/',files(i).name];
		 	destination = [T2_folder,'/e',num2str(idx),'/',files(i).name];
         	system(['rsync ',source,' ',destination]);
  	   end 
   
  end
