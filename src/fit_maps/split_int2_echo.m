function split_int2_echo(mapss_folder,T1rho_idx,T2_idx)
%function split_int2_echo(mapss_folder,T1rho_idx,T2_idx)
% Split int2 files under mapss_folder on to T1rho, T2 folder 
% Re-implented by Misung Han 
% Apr 14, 2020
	T1rho_folder=sprintf('%s/T1rho',mapss_folder);
	T2_folder=sprintf('%s/T2',mapss_folder);
	cd(mapss_folder);
    mkdir(T1rho_folder);
	mkdir(T2_folder);
	for idx=1:length(T1rho_idx)
		cp_command=sprintf('cp -f %s/Echo_e%.1d* %s/',mapss_folder,T1rho_idx(idx), T1rho_folder);	
		system(cp_command);
	end
	for idx=1:length(T2_idx)
		cp_command=sprintf('cp %s/Echo_e%.1d* %s/',mapss_folder,T2_idx(idx), T2_folder);	
		system(cp_command);
	end
	rm_command=sprintf('rm %s/Echo_e*',mapss_folder);
	%system(rm_command);	
