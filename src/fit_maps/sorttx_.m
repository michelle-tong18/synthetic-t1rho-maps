function Ts = sorttx_(mapps_folder,series_num,echo1_dir,is_knee)
%
%   -----------------------------------------------------------------------
%   K. Subburaj, PhD
%   Dept. of Radiology and Biomedical Imaging
%   MQIR, University of California San Francisco
%   Sep 2013
%   subburaj@radiology.ucsf.edu
%   -----------------------------------------------------------------------

% get rootname for looking for files
dicom_folder=sprintf('%s/*.DCM',mapps_folder);
dicom_file_list=dir(dicom_folder);
dicom_name=dicom_file_list(1).name;
disp(['dicom_name:' dicom_name])

[C matches]=strsplit(dicom_name,'S');
exam_name=C{1};
dicom_series=sprintf('%sS%.1d',exam_name,series_num);

disp(dicom_series);
cd(mapps_folder);

dicom_info=dicominfo(dicom_name);
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
    nTE  = dicom_info.Private_0019_10aa;
end
nEchoes=nTSL+nTE;
Ts=[];
for slice = 1:nEchoes
	dicom_image=sprintf('IM%.1d.DCM',slice);
    % get info    
    %[pathname mainnamee sprintf('%d',slice) ext]
    info = dicominfo(dicom_image);
    % get echo number
%    EchoNumber = info.EchoNumber;
    ts = info.EchoTime;
    % if this echo is a new one, record the new total number of
    % echoes and add the echo time to the list
    Ts = [Ts ts];
    % get out of time-consuming file loop if all echoes are accounted
    % for.  Doesn't save time for combined t1rho/t2 or other
    % instances of only 4 echoes
end
%
