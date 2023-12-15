function make_int2_dicom(fname, orig_dicom_folder,output_dicom_dir,series_num,ismapps,nechoes,SeriesName);
%function make_int2_dicom(fname, orig_dicom_folder,output_dicom_dir,series_num,ismapps,nechoes,SeriesName);

%make_int2_dicom('T2_Map','s1','T2Dicom',2100,0,6);For cones
% make_int2_dicom('T2_Map','.','T2Dicom',900,1,6); for mapss

%if sag L->R, the slices should be reversed. 
if(nargin<6)
	nechoes=1;
end

if(nargin<5)
ismapps=0;% cones UTE
end

dicom_file_list=dir(sprintf('%s/*.DCM',orig_dicom_folder));
dicom_name=dicom_file_list(1).name;

[C matches]=strsplit(dicom_name,'S');
exam_name=C{1};
[C matches]=strsplit(C{2},'I');
%series_num=str2num(C{1});

fname_idf=sprintf('%s.idf',fname);
fname_map=sprintf('%s.int2',fname);
txt=sprintf('mkdir -p %s',output_dicom_dir);
system(txt);

%[nixx,npixy,nslices,pixelsizex,pixelsizey,pixelsizez,fslice,lslice,gap,toplc,cosines,fovx,fovy,center, examid] = readidf(fname_idf);

%idf_info=readidf_file(fname);
dat=read_idf_image(fname);


input_im=dat.img;
file_info=dat.idf;

%
%if(file_info.firstread<file_info.lastread)
	
input_im=input_im(:,:,end:-1:1);%L->R
%end
input_im=permute(input_im,[2 1 3]);
nslices=file_info.npix(3);
metadata=dicominfo([orig_dicom_folder '/' dicom_name]);
orig_snumber=metadata.SeriesNumber;
orig_SeriesDescription=metadata.SeriesDescription;
if(nargin<4)
	NewSeriesNumber = orig_snumber*100;
else
	NewSeriesNumber = series_num;
end
if(nargin<7)
	NewSeriesDescription=orig_SeriesDescription
else
	NewSeriesDescription = SeriesName;
end
[C matches]=strsplit(dicom_name,'I');
prefix_dicom=C{1};

for sindex=1:nslices
	if(ismapps)
		fname_sl=sprintf('%s/%sI%d.DCM',orig_dicom_folder,prefix_dicom,(sindex-1)*nechoes+1);%first echo 
	else
		fname_sl=sprintf('%s/%sI%d.DCM',orig_dicom_folder,prefix_dicom,sindex);%first echo 
	end	
	
	metadata=dicominfo(fname_sl);

	outfname=sprintf('%s/im.%03d.DCM',output_dicom_dir,sindex);
	metadata.SeriesNumber=NewSeriesNumber;
	
	metadata.SeriesDescription=NewSeriesDescription;
	dicomwrite(int16(input_im(:,:,sindex)),outfname,metadata,'CreateMode','copy');
end

%T2map=zeros(nrows,ncols,nslices);
%pdmap=zeros(nrows,ncols,nslices);
%for sindex=1:nslices
%	disp(sindex);
%	for rindex =1:nrows
%		for cindex=1:ncols
%			echo_sig=double(squeeze(tot_im(rindex,cindex,:,sindex)));
%			if(sum(echo_sig)>100)
%				[T2 pd]=	fitT2(TE', echo_sig);
%				T2map(rindex,cindex,sindex)=T2;
%				pdmap(rindex,cindex,sindex)=pd;
%	
%			end
%		end
%	end
%%end

