function register2echo1(anal_dir, echo_num,etype, base_dir)

commandFile='/data/bigbone/kneeACL/commandline.txt';
cd(anal_dir);

directory_for_echo_rho=base_dir;%strcat(anal_dir,'/','T1rho/e1');

echo_num_dir=strcat(anal_dir,'/',etype,'/','e',num2str(echo_num));
echo_num_dir_reg=strcat(echo_num_dir,'/reg');
mkdir(echo_num_dir_reg);

importFile = strcat(anal_dir,'/',etype, '/', 'reg/','Echo');
disp(importFile)

system(['run_Rview ' directory_for_echo_rho blanks(1) echo_num_dir ' <' commandFile]);

system(['Apply ' directory_for_echo_rho blanks(1) echo_num_dir ' -dof RviewTransform.dof ' echo_num_dir_reg ' Bspline']);

importCommand = ['/netopt/bin/local/old/import --no_volume_check ',echo_num_dir_reg,' ',importFile,'_e',num2str(echo_num)];

system(importCommand);
cd(directory_for_echo_rho)
% if exist('reg','dir')
%     rmdir reg
% end
disp('done')
end




