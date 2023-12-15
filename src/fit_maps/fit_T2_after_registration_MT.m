function fit_T2_after_registration_MT(examDir,input_folder,series_num,dest_folder,T1rho_idx,T2_idx, reg_on, scale,write_dicom,is_knee,is_spine,t)
curr_dir = pwd;
fprintf('check1\n');

% Misung Han June 2019
% T1 T2 echo splitting than 
% MAPPS echo registration (if reg_on=1)
% if write_dicom=1 write dicom files 
%fit_T2_after_registration('/data/bigbone/hmisung/mapps/E8866_Mar19_2019','15',15,'15_reg',[1 2 3 4],[1 5 6 7], regon,scale,write_dicom);
if(nargin<9)
	write_dicom=0;
end
if(nargin<8)
	scale=1;
end

if(nargin<7)
	reg_on=0;
end

addpath('/home/hmisung/matlab/basic');
%addpath('/home/hmisung/matlab/quantification/radhika_mqir_codes');
addpath('/home/hmisung/matlab/quantification'); %Directory permission denied
addpath('/data/knee_mri6/mwtong/imageSynthesis/code_py/src'); 
fprintf('check2\n');

nT1rho=length(T1rho_idx);
nT2=length(T2_idx);
if(nT1rho>0 & nT2>0)
	if(T1rho_idx(1)==T2_idx(1))
		shared=1;
	else
		shared=0;
	end
end



% Generate int2 files from echo images
mapss_folder=strcat(examDir,'/',input_folder);
output_name=strcat(examDir,'/',input_folder,'/','Echo');
importCommand = ['/netopt/bin/local/old/import --no_volume_check ',mapss_folder,' ',output_name];

system(importCommand);
%
if(reg_on)
	if(nT1rho>1)
		echo1_dir=strcat(examDir,'/',dest_folder,'/T1rho','/','e1');
	else
		echo1_dir=strcat(examDir,'/',dest_folder,'/T2','/','e1');
	end
else
	echo1_dir=strcat(examDir,'/',input_folder);

end


if(reg_on)
	analDir=sprintf('%s/%s',examDir,dest_folder);
	mkdir(analDir);
	% Split Dicom files to T1rho/T2 folders
	split_dicom_echo(examDir,input_folder,series_num,dest_folder,T1rho_idx,T2_idx);


	if(nT1rho>0)
		folderT1rhofit=strcat(analDir,'/T1rho','/reg');
		mkdir(folderT1rhofit);
		%Registration
		%importT1rho_echo1=['/netopt/bin/local/old/import --no_volume_check ',echo1_dir,' ',folderT1rhofit,'/Echo_e1'];
		%system(importT1rho_echo1);

		importT1rho_echo1=['/netopt/bin/local/old/import --no_volume_check ',echo1_dir,' ',folderT1rhofit,'/Echo_e1'];
    	system(importT1rho_echo1);
  

		for echo_num = 2:length(T1rho_idx)
			register2echo1(analDir,echo_num,'T1rho', echo1_dir);
		end
	end
	if(nT2>0)
		folderT2fit=strcat(analDir,'/T2','/reg');
		mkdir(folderT2fit);
	
		for echo_num=1:length(T2_idx)
			register2echo1(analDir,echo_num, 'T2',echo1_dir);
		end
	end
else
	split_int2_echo(mapss_folder,T1rho_idx,T2_idx);
end
fprintf('check3\n');
dicom_folder=sprintf('%s/*.DCM',echo1_dir);
Ts=sorttx_(mapss_folder,series_num,echo1_dir,is_knee)

dicom_file_list=dir(dicom_folder);
dicom_name=dicom_file_list(1).name;

[C matches]=strsplit(dicom_name,'S');
exam_name=C{1};
dicom_series=sprintf('%s/%sS%.1dI',echo1_dir,exam_name,series_num);
disp('dicom series');
disp(dicom_series)

fprintf('check4\n');
% T1rho/T2 fitting
cd(echo1_dir)
disp(dicom_name);
dicom_info=dicominfo(dicom_name);

corr_factor = (1-(1/34))/2;
if is_spine == 1
    Trefocus = dicom_info.Private_0019_10ac;
    Ts(5) = dicom_info.Private_0019_10b1*(Trefocus-4*0.406*corr_factor);
    Ts(6) = dicom_info.Private_0019_10b2*(Trefocus-4*0.406*corr_factor);
    Ts(7) = dicom_info.Private_0019_10b3*(Trefocus-4*0.406*corr_factor);
end
Ts

% Get TSL Times from dicom header
if is_knee == 1
    nTSL=sscanf(char(dicom_info.Private_0019_10a8),'%f');
    nTE=sscanf(char(dicom_info.Private_0019_10aa),'%f');
    if isempty(nTSL) == 1
        nTSL = dicom_info.Private_0019_10a8;
    end
    if isempty(nTE) == 1
        nTE = dicom_info.Private_0019_10aa;
    end
else
    nTSL = dicom_info.Private_0019_10a8;
    nTE = dicom_info.Private_0019_10aa;
end
nechoes=nTSL+nTE;
disp(nTSL);
orig_SeriesDescription=dicom_info.SeriesDescription;


%TSL1=dicom_info.Private_0019_10ad;
%TSL2=dicom_info.Private_0019_10ae;
%TSL3=dicom_info.Private_0019_10af;
%TSL4=dicom_info.Private_0019_10b0;
%Trefocus=dicom_info.Private_0019_10ac;

%corr_factor=(1-(1/34))/2;
% Calculate TE times using dicom header and T1 correction values and make t2_time_points.txt
%if(shared==1)
%	TE1=TSL1;
%	TE2=dicom_info.Private_0019_10b1*(Trefocus-4*0.406*corr_factor);
%	TE3=dicom_info.Private_0019_10b2*(Trefocus-4*0.406*corr_factor);
%	TE4=dicom_info.Private_0019_10b3*(Trefocus-4*0.406*corr_factor);
%else
%	TE1=dicom_info.Private_0019_10b1*(Trefocus-4*0.406*corr_factor);
%	TE2=dicom_info.Private_0019_10b2*(Trefocus-4*0.406*corr_factor);
%	TE3=dicom_info.Private_0019_10b3*(Trefocus-4*0.406*corr_factor);
%	TE4=dicom_info.Private_0019_10b4*(Trefocus-4*0.406*corr_factor);
%end

ismapps=1;

if(nT1rho>0)
	if(reg_on)
		T1rho_dir=sprintf('%s/%s/T1rho/reg',examDir,dest_folder);
		fname_T1rho_time=sprintf('%s/%s/T1rho/reg/t2_time_points.txt',examDir,dest_folder);
	else
		T1rho_dir=sprintf('%s/%s/T1rho',examDir,input_folder);
		fname_T1rho_time=sprintf('%s/%s/T1rho/t2_time_points.txt',examDir,input_folder);
	end
	cd(T1rho_dir);

	fid=fopen(fname_T1rho_time,'w');
	for m=1:nT1rho
		fprintf(fid,'%f\n',Ts(T1rho_idx(m)));
	end

%fprintf(fid,'%f\n',TSL1*scale);
%fprintf(fid,'%f\n',TSL2*scale);
%fprintf(fid,'%f\n',TSL3*scale);
%fprintf(fid,'%f\n',TSL4*scale);
	fclose(fid);

% use t2 for T1rho fitting
	cmd_t2_fit=['t2 -M 0 -s ' num2str(scale) ' -f -t ' num2str(t) ' -c -k Map -v Echo_e . T1rho'];
	system(cmd_t2_fit);

% Write dicom images of T1rho maps
	series_name=sprintf('%s T1rho, scale:x%d',orig_SeriesDescription,scale);
	if(write_dicom)
		make_int2_dicom('T1rho_Map',echo1_dir,'T1rhoDicom',series_num*100,ismapps,nechoes,series_name);
	end
end
fprintf('check5\n');
if(nT2>0)
	if(reg_on)


		T2_dir=sprintf('%s/%s/T2/reg',examDir,dest_folder);
		fname_T2_time=sprintf('%s/%s/T2/reg/t2_time_points.txt',examDir,dest_folder);
	else
		T2_dir=sprintf('%s/%s/T2',examDir,input_folder);
		fname_T2_time=sprintf('%s/%s/T2/t2_time_points.txt',examDir,input_folder);
	end

	cd(T2_dir)
	fid=fopen(fname_T2_time,'w');

	for m=1:nT2
		fprintf(fid,'%f\n',Ts(T2_idx(m)));
	end


%fprintf(fid,'%f\n',TE1*scale);
%fprintf(fid,'%f\n',TE2*scale);
%fprintf(fid,'%f\n',TE3*scale);
%fprintf(fid,'%f\n',TE4*scale);
	fclose(fid);
% use t2 for T2 fitting
	cmd_t2_fit=['t2 -M 0 -s ' num2str(scale) ' -f -t ' num2str(t) ' -c -k Map -v Echo_e . T2'];
	system(cmd_t2_fit);


	series_name=sprintf('%s T2, scale:x%d',orig_SeriesDescription,scale);
	if(write_dicom)
		make_int2_dicom('T2_Map',echo1_dir,'T2Dicom',series_num*100+1,ismapps,nechoes,series_name);
	end
end
fprintf('check6\n');
cd(curr_dir);
